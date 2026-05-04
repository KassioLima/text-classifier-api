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


MODEL_NAME = "neuralmind/bert-large-portuguese-cased"
ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_BASE_DIR = ROOT_DIR / "models_neuralmind_bert_large"


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv(path: Path) -> pd.DataFrame:
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
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_one_task(
    cfg: TaskConfig,
    tokenizer,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
    model_name: str,
    output_dir: Path,
    reports_dir: Path,
) -> dict:
    train_df = load_csv(cfg.train_csv)
    eval_df = load_csv(cfg.eval_csv)

    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["target"])
    eval_df["label"] = label_encoder.transform(eval_df["target"])
    class_counts = np.bincount(train_df["label"], minlength=len(label_encoder.classes_))
    safe_class_counts = np.clip(class_counts, 1, None)
    class_weights = len(train_df) / (len(label_encoder.classes_) * safe_class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize_batch, batched=True)
    eval_ds = eval_ds.map(tokenize_batch, batched=True)

    task_output = output_dir / cfg.task_name
    task_output.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=str(task_output / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=0,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(task_output))
    tokenizer.save_pretrained(str(task_output))

    classes_path = task_output / "classes.json"
    classes_path.write_text(json.dumps(label_encoder.classes_.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")

    pred = trainer.predict(eval_ds)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_path = reports_dir / f"{cfg.task_name}_confusion_matrix.csv"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(cm_path, encoding="utf-8")

    metrics = {
        "task": cfg.task_name,
        "model_name": model_name,
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "num_labels": int(len(label_encoder.classes_)),
        "class_weights": [float(v) for v in class_weights.tolist()],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "classes_path": str(classes_path),
        "confusion_matrix_path": str(cm_path),
    }
    metrics_path = reports_dir / f"{cfg.task_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treino baseline com neuralmind/bert-large-portuguese-cased para tipo/produto/assunto.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS.keys()), default=["tipo", "produto", "assunto"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-subdir", type=str, default="", help="Subpasta dentro de models_neuralmind_bert_large para separar execucoes.")
    return parser.parse_args()


def merge_summary(summary_path: Path, current_results: list[dict]) -> list[dict]:
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
    selected_model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(selected_model_name, use_fast=True)
    output_dir = OUTPUT_BASE_DIR / args.output_subdir if args.output_subdir else OUTPUT_BASE_DIR
    reports_dir = output_dir / "reports"

    summary = []
    for task in args.tasks:
        print(f"\n=== Treinando tarefa: {task} ===")
        result = train_one_task(
            cfg=TASKS[task],
            tokenizer=tokenizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            seed=args.seed,
            model_name=selected_model_name,
            output_dir=output_dir,
            reports_dir=reports_dir,
        )
        summary.append(result)
        print(f"[{task}] concluido. F1 macro: {result['metrics'].get('eval_f1_macro')}")

    summary_path = reports_dir / "training_summary.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    merged_summary = merge_summary(summary_path, summary)
    summary_path.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
