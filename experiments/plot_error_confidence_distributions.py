import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "reports" / "joint_eval_full_without_postrule_with_confidence_2of3_only.json"
DEFAULT_OUTPUT_HIST = PROJECT_ROOT / "datasets" / "reports" / "confidence_error_distribution_2of3_hist.png"
DEFAULT_OUTPUT_BOX = PROJECT_ROOT / "datasets" / "reports" / "confidence_error_distribution_2of3_box.png"
DEFAULT_OUTPUT_BINNED = PROJECT_ROOT / "datasets" / "reports" / "confidence_error_distribution_2of3_binned.png"
DEFAULT_OUTPUT_DIR_PER_TASK = PROJECT_ROOT / "datasets" / "reports"


TASKS = ("tipo", "produto", "assunto")


def parse_args():
    parser = argparse.ArgumentParser(description="Plota distribuicao de confianca para classificacoes erradas no grupo 2/3.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-hist", type=Path, default=DEFAULT_OUTPUT_HIST)
    parser.add_argument("--output-box", type=Path, default=DEFAULT_OUTPUT_BOX)
    parser.add_argument("--output-binned", type=Path, default=DEFAULT_OUTPUT_BINNED)
    parser.add_argument("--output-dir-per-task", type=Path, default=DEFAULT_OUTPUT_DIR_PER_TASK)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--bin-width", type=int, default=5, help="Largura da faixa em pontos percentuais para os graficos por faixa.")
    parser.add_argument(
        "--mode",
        choices=("errors", "all"),
        default="errors",
        help="errors: usa apenas classificacoes erradas por tarefa; all: usa todas as classificacoes da tarefa.",
    )
    parser.add_argument(
        "--correct-count-filter",
        type=int,
        default=None,
        help="Filtra registros por correct_count (ex.: 2 para 2/3, 3 para 3/3).",
    )
    parser.add_argument("--tag", type=str, default="2of3", help="Tag usada no nome dos arquivos de saida.")
    parser.add_argument(
        "--per-task-only",
        action="store_true",
        help="Gera apenas os graficos por classe (binned e zoom 90-100), sem histograma/box/agregado.",
    )
    return parser.parse_args()


def load_confidences(path: Path, mode: str = "errors", correct_count_filter: int | None = None) -> dict[str, list[float]]:
    # Carrega confiança por tarefa e permite filtrar por acertos/erros e faixa 2/3, 3/3 etc.
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {task: [] for task in TASKS}

    items = payload.get("items", [])
    if not items and "metrics" in payload and "record_results" in payload["metrics"]:
        items = payload["metrics"]["record_results"]

    for item in items:
        if correct_count_filter is not None and int(item.get("correct_count", -1)) != correct_count_filter:
            continue
        gold = item.get("gold", {})
        pred = item.get("pred", {})
        conf = item.get("confidence_pct", {})
        for task in TASKS:
            is_error = str(gold.get(task)) != str(pred.get(task))
            if mode == "all" or is_error:
                values[task].append(float(conf[task]))
    return values


def plot_histograms(values: dict[str, list[float]], output: Path, bins: int) -> None:
    # Distribuição (histograma + KDE) da confiança por tarefa.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    palette = {"tipo": "#1f77b4", "produto": "#2ca02c", "assunto": "#d62728"}

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        data = values[task]
        sns.histplot(data, bins=bins, kde=True, stat="density", color=palette[task], ax=ax)
        ax.set_title(f"{task.capitalize()} (erros)\nN={len(data)}")
        ax.set_xlabel("Certeza (%)")
        ax.set_xlim(0, 100)
        if idx == 0:
            ax.set_ylabel("Densidade")
        else:
            ax.set_ylabel("")

    fig.suptitle("Distribuicao da Certeza nas Classificacoes Erradas (grupo 2/3)", fontsize=14, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_box_violin(values: dict[str, list[float]], output: Path) -> None:
    # Resumo visual por tarefa para comparar dispersão da confiança.
    sns.set_theme(style="whitegrid")
    x = []
    y = []
    for task in TASKS:
        x.extend([task.capitalize()] * len(values[task]))
        y.extend(values[task])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x=x, y=y, inner="quartile", cut=0, ax=ax, palette=["#1f77b4", "#2ca02c", "#d62728"])
    sns.stripplot(x=x, y=y, ax=ax, color="black", alpha=0.12, size=2, jitter=0.25)
    ax.set_title("Resumo da Certeza nas Classificacoes Erradas (grupo 2/3)")
    ax.set_xlabel("Classe errada")
    ax.set_ylabel("Certeza (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_bins(start: int, end: int, width: int) -> list[tuple[int, int]]:
    bins = []
    current = start
    while current < end:
        nxt = min(current + width, end)
        bins.append((current, nxt))
        current = nxt
    return bins


def assign_to_bins(data: list[float], bins: list[tuple[int, int]]) -> list[int]:
    counts = [0] * len(bins)
    for value in data:
        for idx, (lo, hi) in enumerate(bins):
            in_bin = lo <= value < hi
            if idx == len(bins) - 1:
                in_bin = lo <= value <= hi
            if in_bin:
                counts[idx] += 1
                break
    return counts


def plot_grouped_binned_bars(values: dict[str, list[float]], output: Path, bin_width: int) -> None:
    # Barras agrupadas por faixa de confiança para comparação direta entre tarefas.
    sns.set_theme(style="whitegrid")
    bins = build_bins(0, 100, bin_width)
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    counts_by_task = {task: assign_to_bins(values[task], bins) for task in TASKS}

    import numpy as np
    x = np.arange(len(bins))
    width = 0.26
    palette = {"tipo": "#1f77b4", "produto": "#2ca02c", "assunto": "#d62728"}

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, counts_by_task["tipo"], width=width, color=palette["tipo"], label="Tipo")
    ax.bar(x, counts_by_task["produto"], width=width, color=palette["produto"], label="Produto")
    ax.bar(x + width, counts_by_task["assunto"], width=width, color=palette["assunto"], label="Assunto")

    ax.set_title("Erros por Faixa de Certeza (grupo 2/3)")
    ax.set_xlabel("Faixa de certeza (%)")
    ax.set_ylabel("Quantidade de erros")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_binned(
    values: dict[str, list[float]],
    output_dir: Path,
    bin_width: int,
    tag: str,
    label_kind: str = "casos",
) -> list[Path]:
    # Gera gráficos dedicados por tarefa (mais fáceis para leitura operacional).
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = build_bins(0, 100, bin_width)
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    palette = {"tipo": "#1f77b4", "produto": "#2ca02c", "assunto": "#d62728"}
    outputs: list[Path] = []

    for task in TASKS:
        counts = assign_to_bins(values[task], bins)
        import numpy as np
        x = np.arange(len(bins))

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(x, counts, color=palette[task], width=0.8)
        ax.set_title(f"{task.capitalize()}: {label_kind} por faixa de certeza ({bin_width} em {bin_width}%)")
        ax.set_xlabel("Faixa de certeza (%)")
        ax.set_ylabel(f"Quantidade de {label_kind}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        offset = max(1, int(max(counts) * 0.01)) if counts else 1
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        out = output_dir / f"confidence_{label_kind}_distribution_{tag}_{task}_binned_{bin_width}pct.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)

    return outputs


def plot_per_task_zoom_90_100(
    values: dict[str, list[float]],
    output_dir: Path,
    tag: str,
    label_kind: str = "casos",
) -> list[Path]:
    # Zoom na faixa 90-100 para inspecionar overconfidence em alta confiança.
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = build_bins(90, 100, 1)
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    palette = {"tipo": "#1f77b4", "produto": "#2ca02c", "assunto": "#d62728"}
    outputs: list[Path] = []

    filtered = {task: [v for v in values[task] if 90 <= v <= 100] for task in TASKS}
    for task in TASKS:
        counts = assign_to_bins(filtered[task], bins)
        import numpy as np
        x = np.arange(len(bins))

        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(x, counts, color=palette[task], width=0.85)
        ax.set_title(f"{task.capitalize()}: zoom 90-100 (faixas de 1%)")
        ax.set_xlabel("Faixa de certeza (%)")
        ax.set_ylabel(f"Quantidade de {label_kind}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        offset = max(1, int(max(counts) * 0.01)) if counts else 1
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        out = output_dir / f"confidence_{label_kind}_distribution_{tag}_{task}_zoom_90_100_1pct.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)

    return outputs


def main():
    args = parse_args()
    values = load_confidences(args.input, mode=args.mode, correct_count_filter=args.correct_count_filter)

    if not args.per_task_only:
        plot_histograms(values, args.output_hist, args.bins)
        plot_box_violin(values, args.output_box)
        plot_grouped_binned_bars(values, args.output_binned, args.bin_width)
    label_kind = "acertos" if args.mode == "all" and args.correct_count_filter == 3 else "erros"
    per_task_outputs = plot_per_task_binned(values, args.output_dir_per_task, args.bin_width, args.tag, label_kind=label_kind)
    zoom_outputs = plot_per_task_zoom_90_100(values, args.output_dir_per_task, args.tag, label_kind=label_kind)

    if not args.per_task_only:
        print(f"Histograma salvo em: {args.output_hist}")
        print(f"Violin/box salvo em: {args.output_box}")
        print(f"Barras por faixa salvas em: {args.output_binned}")
    for out in per_task_outputs:
        print(f"Por classe salvo em: {out}")
    for out in zoom_outputs:
        print(f"Zoom 90-100 salvo em: {out}")
    for task in TASKS:
        if values[task]:
            avg = sum(values[task]) / len(values[task])
            print(f"{task}: N={len(values[task])} media={avg:.4f}% min={min(values[task]):.4f}% max={max(values[task]):.4f}%")


if __name__ == "__main__":
    main()
