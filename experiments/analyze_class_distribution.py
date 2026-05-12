import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"

TASK_FILES = {
    "tipo": DATASETS_DIR / "dataset-tipo-train.csv",
    "produto": DATASETS_DIR / "dataset-produto-train.csv",
    "assunto": DATASETS_DIR / "dataset-assunto-train.csv",
}


def load_counts(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    df = pd.read_csv(path, encoding="utf-8", usecols=["target"])
    df["target"] = df["target"].astype(str).str.strip()
    df = df[df["target"] != ""]
    return df["target"].value_counts().sort_values(ascending=False)


def summarize_task(task: str, min_samples: int) -> None:
    counts = load_counts(TASK_FILES[task])
    kept = counts[counts >= min_samples]
    dropped = counts[counts < min_samples]

    print(f"\n=== {task} ===")
    print(f"classes totais: {len(counts)}")
    print(f"linhas totais: {int(counts.sum())}")
    print(f"min_samples: {min_samples}")
    print(f"classes mantidas: {len(kept)}")
    print(f"linhas mantidas: {int(kept.sum())}")
    print(f"classes removidas: {len(dropped)}")
    print(f"linhas removidas: {int(dropped.sum())}")

    print("\nDistribuicao completa:")
    for label, count in counts.items():
        status = "mantem" if count >= min_samples else "remove"
        print(f"{label}\t{int(count)}\t{status}")


def parse_args():
    parser = argparse.ArgumentParser(description="Mostra distribuicao de classes nos datasets de treino.")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_FILES.keys()), default=sorted(TASK_FILES.keys()))
    parser.add_argument("--min-samples", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    for task in args.tasks:
        summarize_task(task, args.min_samples)


if __name__ == "__main__":
    main()
