import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "FacebookAI/xlm-roberta-large"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "models_xlm_roberta_large"
REPORTS_DIR = OUTPUT_DIR / "reports"


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    train_csv: Path
    eval_csv: Path


TASKS = {
    "tipo": TaskConfig(
        task_name="tipo",
        train_csv=DATASETS_DIR / "dataset-tipo-train.csv",
        eval_csv=DATASETS_DIR / "dataset-tipo-evaluation.csv",
    ),
    "produto": TaskConfig(
        task_name="produto",
        train_csv=DATASETS_DIR / "dataset-produto-train.csv",
        eval_csv=DATASETS_DIR / "dataset-produto-evaluation.csv",
    ),
    "assunto": TaskConfig(
        task_name="assunto",
        train_csv=DATASETS_DIR / "dataset-assunto-train.csv",
        eval_csv=DATASETS_DIR / "dataset-assunto-evaluation.csv",
    ),
}


def set_seed(seed: int) -> None:
    # Reprodutibilidade entre execuções.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv(path: Path) -> pd.DataFrame:
    # Lê e valida o formato mínimo esperado para treino.
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    expected = {"text", "target"}
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV invalido ({path}). Esperado colunas: {expected}, obtido: {set(df.columns)}")
    df = df[["text", "target"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["target"] != "")]
    return df


def compute_metrics(eval_pred):
    # Métricas usadas para seleção do melhor checkpoint.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def train_one_task(
    cfg: TaskConfig,
    tokenizer,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict:
    # Treina uma tarefa por vez (tipo/produto/assunto) com XLM-R Large.
    train_df = load_csv(cfg.train_csv)
    eval_df = load_csv(cfg.eval_csv)

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["target"])
    eval_df["label"] = label_encoder.transform(eval_df["target"])

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    eval_ds = eval_ds.map(tokenize_batch, batched=True)

    task_output = OUTPUT_DIR / cfg.task_name
    task_output.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=str(task_output / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # Mantém batch efetivo maior sem estourar memória de GPU.
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=0,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # Persistência de artefatos para uso em avaliação/inferência.
    trainer.save_model(str(task_output))
    tokenizer.save_pretrained(str(task_output))

    classes_path = task_output / "classes.json"
    classes_path.write_text(json.dumps(label_encoder.classes_.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")

    pred = trainer.predict(eval_ds)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_path = REPORTS_DIR / f"{cfg.task_name}_confusion_matrix.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(cm_path, encoding="utf-8")

    metrics = {
        "task": cfg.task_name,
        "model_name": MODEL_NAME,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "num_labels": int(len(label_encoder.classes_)),
        "effective_train_batch_size": int(batch_size * gradient_accumulation_steps),
        "max_length": int(max_length),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "classes_path": str(classes_path),
        "confusion_matrix_path": str(cm_path),
    }
    metrics_path = REPORTS_DIR / f"{cfg.task_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treino com FacebookAI/xlm-roberta-large para tipo/produto/assunto.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS.keys()), default=["tipo", "produto", "assunto"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def merge_summary(summary_path: Path, current_results: list[dict]) -> list[dict]:
    # Atualiza resumo mantendo ordem padrão das tarefas.
    existing_by_task = {}
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if isinstance(item, dict) and "task" in item:
                        existing_by_task[item["task"]] = item
        except Exception:
            existing_by_task = {}

    for item in current_results:
        existing_by_task[item["task"]] = item

    order = ["tipo", "produto", "assunto"]
    merged = [existing_by_task[t] for t in order if t in existing_by_task]
    merged.extend([v for k, v in existing_by_task.items() if k not in order])
    return merged


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Loop principal de treino multi-tarefa.
    summary = []
    for task in args.tasks:
        print(f"\n=== Treinando tarefa: {task} ===")
        result = train_one_task(
            cfg=TASKS[task],
            tokenizer=tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
        )
        summary.append(result)
        print(f"[{task}] concluido. F1 macro: {result['metrics'].get('eval_f1_macro')}")

    summary_path = REPORTS_DIR / "training_summary.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    merged_summary = merge_summary(summary_path, summary)
    summary_path.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
