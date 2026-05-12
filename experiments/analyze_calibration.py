import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT_DIR / "datasets" / "reports" / "joint_eval_full_without_postrule_with_confidence.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "datasets" / "reports"
TASKS = ("tipo", "produto", "assunto")


def parse_args():
    parser = argparse.ArgumentParser(description="Calcula ECE e plota reliability diagram por tarefa.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def load_records(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("metrics", {}).get("record_results", [])
    if not records:
        raise ValueError("Nenhum registro encontrado em metrics.record_results")
    return records


def get_task_data(records, task: str):
    conf = []
    correct = []
    for item in records:
        confidence = float(item["confidence_pct"][task]) / 100.0
        is_correct = 1.0 if str(item["gold"][task]) == str(item["pred"][task]) else 0.0
        conf.append(confidence)
        correct.append(is_correct)
    return np.array(conf), np.array(correct)


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(conf)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append((lo, hi, 0, None, None, None))
            continue

        conf_mean = float(conf[mask].mean())
        acc_mean = float(correct[mask].mean())
        gap = abs(acc_mean - conf_mean)
        ece += (count / n) * gap
        rows.append((lo, hi, count, conf_mean, acc_mean, gap))

    return ece, rows


def plot_reliability(task: str, rows, output: Path):
    centers = []
    accs = []
    counts = []
    for lo, hi, count, conf_mean, acc_mean, _ in rows:
        if count > 0:
            centers.append((lo + hi) / 2.0)
            accs.append(acc_mean)
            counts.append(count)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal (y=x)")
    ax.plot(centers, accs, marker="o", color="#1f77b4", label="Acuracia observada")

    for x, y, c in zip(centers, accs, counts):
        ax.text(x, y + 0.015, str(c), ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Reliability Diagram - {task.capitalize()}")
    ax.set_xlabel("Confianca (centro da faixa)")
    ax.set_ylabel("Acuracia observada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    records = load_records(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for task in TASKS:
        conf, correct = get_task_data(records, task)
        ece, rows = compute_ece(conf, correct, args.bins)
        out = output_dir / f"calibration_reliability_{task}.png"
        plot_reliability(task, rows, out)
        summary[task] = {"ece": ece, "image": str(out), "rows": rows}

    summary_path = output_dir / "calibration_summary.json"
    serializable = {}
    for task, payload in summary.items():
        serializable[task] = {
            "ece": payload["ece"],
            "image": payload["image"],
            "bins": [
                {
                    "range": [float(lo), float(hi)],
                    "count": int(count),
                    "confidence_mean": None if conf_mean is None else float(conf_mean),
                    "accuracy_mean": None if acc_mean is None else float(acc_mean),
                    "gap_abs": None if gap is None else float(gap),
                }
                for lo, hi, count, conf_mean, acc_mean, gap in payload["rows"]
            ],
        }
    summary_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Resumo salvo em: {summary_path}")
    for task in TASKS:
        print(f"{task}: ECE={summary[task]['ece']:.6f} | imagem={summary[task]['image']}")


if __name__ == "__main__":
    main()
