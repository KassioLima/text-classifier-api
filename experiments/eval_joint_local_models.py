import argparse
import json
from pathlib import Path
from collections import Counter
import re
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "dataset_train_ready.json"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "reports" / "joint_eval_lr2e5_bs4_ep4_tipo30_produto30_assunto50.json"


TASK_SPECS = {
    "tipo": {"label_field": "TipoDeDemanda", "subdir": "tipo"},
    "produto": {"label_field": "produto", "subdir": "produto"},
    "assunto": {"label_field": "assunto", "subdir": "assunto"},
}


def parse_args():
    # Parâmetros para reproduzir avaliação conjunta com modelos locais exportados.
    parser = argparse.ArgumentParser(description="Avalia acerto conjunto (3/3) para tipo, produto e assunto.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--text-field", type=str, default="DetalhesDaDemanda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="0 = usa todos os registros")
    parser.add_argument("--max-error-examples", type=int, default=30)
    parser.add_argument("--top-k-errors", type=int, default=20)
    parser.add_argument("--enable-tipo-postrule", action="store_true", help="Ativa regra de desambiguacao para tipo 10/20.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescreve o arquivo de saida se ele ja existir.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_label(value) -> str:
    return str(value).strip()


def load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON esperado em formato de lista de objetos.")
    return data


def load_model_bundle(model_dir: Path, device: torch.device):
    # Carrega tokenizer/model/classes.json de uma dimensão específica.
    classes_path = model_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.json nao encontrado em: {classes_path}")
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError(f"classes.json invalido em: {classes_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    id_to_class = [normalize_label(v) for v in classes]
    return tokenizer, model, id_to_class, set(id_to_class)


@torch.inference_mode()
def predict_labels_with_confidence(
    texts: list[str],
    tokenizer,
    model,
    id_to_class: list[str],
    device: torch.device,
    max_length: int,
) -> tuple[list[str], list[float]]:
    # Predição em batch com confiança (softmax) para reduzir custo de inferência.
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)
    confs, pred_ids = torch.max(probs, dim=-1)
    pred_ids = pred_ids.tolist()
    conf_pct = (confs * 100.0).tolist()
    return [id_to_class[i] for i in pred_ids], [float(v) for v in conf_pct]


def build_filtered_dataset(records: list[dict], text_field: str, class_sets: dict[str, set[str]], limit: int) -> tuple[list[dict], dict]:
    # Filtra registros inválidos e remove classes fora do vocabulário dos modelos.
    accepted = []
    dropped_missing = 0
    dropped_empty_text = 0
    dropped_unknown = {task: 0 for task in TASK_SPECS}

    for row in records:
        if text_field not in row:
            dropped_missing += 1
            continue
        text = str(row.get(text_field, "")).strip()
        if not text:
            dropped_empty_text += 1
            continue

        mapped = {}
        valid = True
        for task, spec in TASK_SPECS.items():
            field = spec["label_field"]
            if field not in row:
                valid = False
                dropped_missing += 1
                break
            gold = normalize_label(row[field])
            if gold not in class_sets[task]:
                valid = False
                dropped_unknown[task] += 1
                break
            mapped[task] = gold

        if not valid:
            continue

        accepted.append(
            {
                "text": text,
                "gold": mapped,
            }
        )
        if limit > 0 and len(accepted) >= limit:
            break

    stats = {
        "input_rows": len(records),
        "accepted_rows": len(accepted),
        "dropped_missing_or_invalid_fields": dropped_missing,
        "dropped_empty_text": dropped_empty_text,
        "dropped_unknown_class_by_task": dropped_unknown,
    }
    return accepted, stats


SOLICITACAO_PATTERNS = [
    r"\bsolicit(?:o|amos|acao|a[cç][aã]o)\b",
    r"\bsolicito\b",
    r"\bsolicitamos\b",
    r"\bfavor\b",
    r"\bpor favor\b",
    r"\bpedimos\b",
    r"\bpe[cç]o\b",
    r"\bprazo\b",
    r"\bprevis[aã]o\b",
    r"\burgente\b",
    r"\bprioriza(?:r|cao|ç[aã]o)\b",
    r"\bajust(?:e|ar)\b",
    r"\bregulariza(?:cao|ç[aã]o|r)\b",
    r"\breativa(?:r|cao|ç[aã]o)\b",
    r"\bremanejamento\b",
    r"\bmudan[cç]a\b",
    r"\bativac(?:ao|a[cç][aã]o)\b",
    r"\bdesativa(?:cao|ç[aã]o|r)\b",
]

INFORMACAO_PATTERNS = [
    r"\bd[uú]vida\b",
    r"\bduvidas\b",
    r"\binforma[cç][aã]o\b",
    r"\binformar\b",
    r"\besclarec(?:er|imento)\b",
    r"\bentender\b",
    r"\bqual\b",
    r"\bcomo\b",
    r"\bquando\b",
    r"\bstatus\b",
    r"\bposi[cç][aã]o\b",
    r"\bretorno\b",
]


def score_patterns(text: str, patterns: list[str]) -> int:
    score = 0
    for pattern in patterns:
        if re.search(pattern, text):
            score += 1
    return score


def apply_tipo_postrule(text: str, pred_tipo: str) -> str:
    # Regra opcional de desambiguação entre tipo 10 e 20 baseada em padrões de texto.
    if pred_tipo not in {"10", "20"}:
        return pred_tipo
    normalized = text.lower()
    score_solic = score_patterns(normalized, SOLICITACAO_PATTERNS)
    score_info = score_patterns(normalized, INFORMACAO_PATTERNS)

    # Regra conservadora: so troca quando ha vantagem clara do outro sinal.
    if pred_tipo == "10" and (score_solic - score_info) >= 2:
        return "20"
    if pred_tipo == "20" and (score_info - score_solic) >= 2:
        return "10"
    return pred_tipo


def evaluate(
    records: list[dict],
    bundles: dict[str, tuple],
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_error_examples: int,
    top_k_errors: int,
    enable_tipo_postrule: bool,
):
    # Avaliação principal: calcula acurácia por tarefa, acerto conjunto e análise de erros.
    total = len(records)
    task_correct = {task: 0 for task in TASK_SPECS}
    joint_3 = 0
    joint_2 = 0
    joint_1 = 0
    joint_0 = 0
    error_examples = []
    wrong_task_in_exactly_2 = Counter()
    per_task_error_pairs = {task: Counter() for task in TASK_SPECS}
    combo_error_pairs = Counter()
    confidence_agg = {
        task: {"sum": 0.0, "count": 0, "sum_correct": 0.0, "count_correct": 0, "sum_incorrect": 0.0, "count_incorrect": 0}
        for task in TASK_SPECS
    }
    record_results = []

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        texts = [r["text"] for r in batch]
        preds_by_task = {}
        for task in TASK_SPECS:
            tokenizer, model, id_to_class, _ = bundles[task]
            labels, confs = predict_labels_with_confidence(texts, tokenizer, model, id_to_class, device, max_length)
            preds_by_task[task] = {"labels": labels, "confidences": confs}

        for idx, row in enumerate(batch):
            correct_count = 0
            pred_row = {}
            confidence_row = {}
            for task in TASK_SPECS:
                pred = preds_by_task[task]["labels"][idx]
                conf = preds_by_task[task]["confidences"][idx]
                gold = row["gold"][task]
                if task == "tipo" and enable_tipo_postrule:
                    pred = apply_tipo_postrule(row["text"], pred)
                pred_row[task] = pred
                confidence_row[task] = round(conf, 4)
                confidence_agg[task]["sum"] += conf
                confidence_agg[task]["count"] += 1
                if pred == gold:
                    task_correct[task] += 1
                    correct_count += 1
                    confidence_agg[task]["sum_correct"] += conf
                    confidence_agg[task]["count_correct"] += 1
                else:
                    per_task_error_pairs[task][(gold, pred)] += 1
                    confidence_agg[task]["sum_incorrect"] += conf
                    confidence_agg[task]["count_incorrect"] += 1

            record_results.append(
                {
                    "gold": row["gold"],
                    "pred": pred_row,
                    "confidence_pct": confidence_row,
                    "correct_count": correct_count,
                    "text_preview": row["text"][:280],
                }
            )

            if correct_count == 3:
                joint_3 += 1
            elif correct_count == 2:
                joint_2 += 1
                for task in TASK_SPECS:
                    if pred_row[task] != row["gold"][task]:
                        wrong_task_in_exactly_2[task] += 1
                        break
            elif correct_count == 1:
                joint_1 += 1
            else:
                joint_0 += 1

            if correct_count < 3:
                combo_error_pairs[
                    (
                        row["gold"]["tipo"],
                        pred_row["tipo"],
                        row["gold"]["produto"],
                        pred_row["produto"],
                        row["gold"]["assunto"],
                        pred_row["assunto"],
                    )
                ] += 1

            if correct_count < 3 and len(error_examples) < max_error_examples:
                error_examples.append(
                    {
                        "gold": row["gold"],
                        "pred": pred_row,
                        "confidence_pct": confidence_row,
                        "correct_count": correct_count,
                        "text_preview": row["text"][:280],
                    }
                )

    def pct(v: int) -> float:
        return (100.0 * v / total) if total else 0.0

    return {
        "total_evaluated": total,
        "accuracy_by_task": {task: {"correct": task_correct[task], "pct": pct(task_correct[task])} for task in TASK_SPECS},
        "joint_accuracy": {
            "all_3_correct": {"count": joint_3, "pct": pct(joint_3)},
            "exactly_2_correct": {"count": joint_2, "pct": pct(joint_2)},
            "exactly_1_correct": {"count": joint_1, "pct": pct(joint_1)},
            "none_correct": {"count": joint_0, "pct": pct(joint_0)},
        },
        "exactly_2_wrong_task_distribution": dict(wrong_task_in_exactly_2),
        "top_error_pairs_by_task": {
            task: [
                {"gold": gp[0], "pred": gp[1], "count": cnt}
                for gp, cnt in per_task_error_pairs[task].most_common(top_k_errors)
            ]
            for task in TASK_SPECS
        },
        "top_joint_error_combinations": [
            {
                "gold_tipo": key[0],
                "pred_tipo": key[1],
                "gold_produto": key[2],
                "pred_produto": key[3],
                "gold_assunto": key[4],
                "pred_assunto": key[5],
                "count": count,
            }
            for key, count in combo_error_pairs.most_common(top_k_errors)
        ],
        "confidence_summary_by_task": {
            task: {
                "avg_confidence_pct": round((v["sum"] / v["count"]) if v["count"] else 0.0, 4),
                "avg_confidence_when_correct_pct": round((v["sum_correct"] / v["count_correct"]) if v["count_correct"] else 0.0, 4),
                "avg_confidence_when_incorrect_pct": round((v["sum_incorrect"] / v["count_incorrect"]) if v["count_incorrect"] else 0.0, 4),
            }
            for task, v in confidence_agg.items()
        },
        "record_results": record_results,
        "error_examples": error_examples,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_json_records(args.dataset)

    bundles = {}
    class_sets = {}
    for task, spec in TASK_SPECS.items():
        model_dir = args.model_root / spec["subdir"]
        tokenizer, model, id_to_class, class_set = load_model_bundle(model_dir, device)
        bundles[task] = (tokenizer, model, id_to_class, class_set)
        class_sets[task] = class_set

    filtered_records, filter_stats = build_filtered_dataset(records, args.text_field, class_sets, args.limit)
    if not filtered_records:
        raise RuntimeError("Nenhum registro elegivel para avaliacao conjunta apos filtros.")

    metrics = evaluate(
        records=filtered_records,
        bundles=bundles,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_error_examples=args.max_error_examples,
        top_k_errors=args.top_k_errors,
        enable_tipo_postrule=args.enable_tipo_postrule,
    )

    # Relatório final consolidado para análise posterior.
    output = {
        "dataset": str(args.dataset),
        "model_root": str(args.model_root),
        "text_field": args.text_field,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "enable_tipo_postrule": args.enable_tipo_postrule,
        "device": str(device),
        "filter_stats": filter_stats,
        "metrics": metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_output = args.output
    if final_output.exists() and not args.overwrite:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = final_output.with_name(f"{final_output.stem}_{ts}{final_output.suffix}")

    final_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Relatorio salvo em: {final_output}")
    print(f"Total avaliado: {metrics['total_evaluated']}")
    print(f"Acerto conjunto 3/3: {metrics['joint_accuracy']['all_3_correct']['pct']:.2f}%")
    print(f"Acerto 2/3: {metrics['joint_accuracy']['exactly_2_correct']['pct']:.2f}%")
    print(f"Acerto 1/3: {metrics['joint_accuracy']['exactly_1_correct']['pct']:.2f}%")
    print(f"Acerto 0/3: {metrics['joint_accuracy']['none_correct']['pct']:.2f}%")


if __name__ == "__main__":
    main()
