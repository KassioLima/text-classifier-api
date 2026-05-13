import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
PROJECT_ROOT = Path(__file__).resolve().parents[2]


BASE_DIR = PROJECT_ROOT / "datasets"
JSON_DATASET_NAME = BASE_DIR / "dataset_train_ready.json"

DATASET_TIPO_NAME = BASE_DIR / "dataset-tipo.csv"
DATASET_PRODUTO_NAME = BASE_DIR / "dataset-produto.csv"
DATASET_ASSUNTO_NAME = BASE_DIR / "dataset-assunto.csv"

TEST_DIR = BASE_DIR / "teste"
REPORTS_DIR = BASE_DIR / "reports"

VALIDATION_PERCENT = 0.2
TEST_SAMPLE_SIZE = 100
RANDOM_STATE = 42
MIN_CLASS_SAMPLES = 2
MIN_CLASS_SAMPLES_FOR_EVAL = int(1 / VALIDATION_PERCENT)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    text_column: str
    target_column: str
    output_base: Path
    source_columns_to_preprocess: tuple[str, ...]


def preprocess_text(text: str) -> str:
    # Mantem compatibilidade com o pipeline existente, que trabalha com CSV escapado.
    return json.dumps(text.strip())[1:-1].replace('"', '\\"')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_train_name(path: Path) -> Path:
    return path.with_name(f"{path.stem}-train{path.suffix}")


def to_evaluation_name(path: Path) -> Path:
    return path.with_name(f"{path.stem}-evaluation{path.suffix}")


def to_test_name(path: Path) -> Path:
    return TEST_DIR / f"{path.stem}-teste{path.suffix}"


def _read_source_df(json_source: Path, text_column: str, target_column: str) -> pd.DataFrame:
    # Lê o JSON bruto e normaliza apenas as colunas necessárias para o treino.
    df = pd.read_json(json_source)
    required_columns = {text_column, target_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes no dataset fonte: {sorted(missing)}")

    df = df[[text_column, target_column]].copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df[target_column] = df[target_column].astype(str).str.strip()
    df = df[(df[text_column] != "") & (df[target_column] != "")]
    return df


def _drop_rare_classes(df: pd.DataFrame, min_class_samples: int) -> tuple[pd.DataFrame, list[str]]:
    # Remove classes com pouca representatividade para evitar split estratificado inválido.
    counts = df["target"].value_counts()
    keep_labels = counts[counts >= min_class_samples].index
    removed_labels = counts[counts < min_class_samples].index.tolist()
    filtered = df[df["target"].isin(keep_labels)].copy()
    return filtered, removed_labels


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")


def _validate_dataset(path: Path) -> None:
    # Validação defensiva para garantir que o dataset gerado está pronto para treino.
    df = pd.read_csv(path, encoding="utf-8")
    if list(df.columns) != ["text", "target"]:
        raise ValueError(f"Colunas invalidas em {path}: {list(df.columns)}")
    if df.isnull().any().any():
        raise ValueError(f"DataFrame contem nulos: {path}")
    if df.duplicated().any():
        raise ValueError(f"DataFrame contem duplicados: {path}")
    if (df["text"].astype(str).str.strip() == "").any():
        raise ValueError(f"DataFrame contem texto vazio: {path}")
    if (df["target"].astype(str).str.strip() == "").any():
        raise ValueError(f"DataFrame contem target vazio: {path}")


def _class_report(df: pd.DataFrame, label: str) -> dict:
    counts = df["target"].value_counts()
    return {
        "label": label,
        "rows": int(len(df)),
        "classes": int(counts.shape[0]),
        "min_class_size": int(counts.min()) if len(counts) else 0,
        "max_class_size": int(counts.max()) if len(counts) else 0,
        "class_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
    }


def _save_report(config_name: str, reports: Iterable[dict], extra: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"dataset-{config_name}-report.json"
    payload = {
        "config": config_name,
        "validation_percent": VALIDATION_PERCENT,
        "random_state": RANDOM_STATE,
        "min_class_samples": MIN_CLASS_SAMPLES,
        "min_class_samples_for_eval": MIN_CLASS_SAMPLES_FOR_EVAL,
        "extra": extra,
        "reports": list(reports),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def montarDataSetTrainEvaluation(config: DatasetConfig, teste: bool = False) -> None:
    # Gera os pares train/evaluation para uma dimensão (tipo/produto/assunto),
    # incluindo modo teste (amostra reduzida) para smoke tests rápidos.
    source_df = _read_source_df(JSON_DATASET_NAME, config.text_column, config.target_column)
    source_df = source_df.rename(columns={config.text_column: "text", config.target_column: "target"})

    # Evita vazamento por texto duplicado entre treino e validacao.
    source_df = source_df.drop_duplicates(subset=["text", "target"]).copy()

    if teste:
        n = min(TEST_SAMPLE_SIZE, len(source_df))
        source_df = source_df.sample(n=n, random_state=RANDOM_STATE).copy()

    for text_col in config.source_columns_to_preprocess:
        if text_col == config.text_column:
            source_df["text"] = source_df["text"].apply(preprocess_text)

    min_required = max(MIN_CLASS_SAMPLES, MIN_CLASS_SAMPLES_FOR_EVAL)
    source_df, removed_labels = _drop_rare_classes(source_df, min_required)

    if source_df.empty:
        raise ValueError(f"Dataset vazio para '{config.name}' apos filtros de classe.")

    train_df, eval_df = train_test_split(
        source_df,
        test_size=VALIDATION_PERCENT,
        stratify=source_df["target"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Garante que nenhuma classe do eval fique ausente no treino.
    eval_classes = set(eval_df["target"].unique())
    train_df = train_df[train_df["target"].isin(eval_classes)].copy()

    if set(train_df["target"].unique()) != eval_classes:
        raise ValueError(f"Split inconsistente para '{config.name}'.")

    # Remove possiveis duplicados residuais.
    train_df = train_df.drop_duplicates()
    eval_df = eval_df.drop_duplicates()

    base_name = config.output_base
    if teste:
        train_path = to_test_name(to_train_name(base_name))
        eval_path = to_test_name(to_evaluation_name(base_name))
    else:
        train_path = to_train_name(base_name)
        eval_path = to_evaluation_name(base_name)

    _write_csv(train_df, train_path)
    _write_csv(eval_df, eval_path)
    _validate_dataset(train_path)
    _validate_dataset(eval_path)

    report_path = _save_report(
        config_name=f"{config.name}{'-teste' if teste else ''}",
        reports=[
            _class_report(source_df, "source_after_filters"),
            _class_report(train_df, "train"),
            _class_report(eval_df, "evaluation"),
        ],
        extra={
            "removed_rare_labels": removed_labels,
            "train_path": str(train_path),
            "evaluation_path": str(eval_path),
        },
    )

    print(f"\n[{config.name}{' teste' if teste else ''}]")
    print(f"Train: {train_path}")
    print(f"Evaluation: {eval_path}")
    print(f"Report: {report_path}")
    print(
        f"Classes train/eval: {train_df['target'].nunique()} / {eval_df['target'].nunique()} | "
        f"Rows train/eval: {len(train_df)} / {len(eval_df)}"
    )
    if removed_labels:
        print(f"Classes removidas por baixa frequencia (<{min_required}): {removed_labels}")


def datasetsConfigs() -> list[DatasetConfig]:
    # Mapeia como cada dimensão deve ser extraída do dataset fonte.
    return [
        DatasetConfig(
            name="tipo",
            text_column="DetalhesDaDemanda",
            target_column="TipoDeDemanda",
            output_base=DATASET_TIPO_NAME,
            source_columns_to_preprocess=("DetalhesDaDemanda",),
        ),
        DatasetConfig(
            name="produto",
            text_column="DetalhesDaDemanda",
            target_column="produto",
            output_base=DATASET_PRODUTO_NAME,
            source_columns_to_preprocess=("DetalhesDaDemanda",),
        ),
        DatasetConfig(
            name="assunto",
            text_column="DetalhesDaDemanda",
            target_column="assunto",
            output_base=DATASET_ASSUNTO_NAME,
            source_columns_to_preprocess=("DetalhesDaDemanda",),
        ),
    ]


def montarDatasets() -> None:
    # Ponto de entrada para reproduzir geração dos datasets de todas as tarefas.
    print(f"Dataset fonte: {JSON_DATASET_NAME}")

    for cfg in datasetsConfigs():
        montarDataSetTrainEvaluation(cfg, teste=False)
        montarDataSetTrainEvaluation(cfg, teste=True)


if __name__ == "__main__":
    montarDatasets()
