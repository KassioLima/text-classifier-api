import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "datasets" / "demandas_unificado_sem_duplicados.json"
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "demandas_unificado_train_ready.json"
DEFAULT_REPORT = BASE_DIR / "datasets" / "reports" / "prepare_unified_report.json"

MIN_TEXT_LEN = 30
MAX_TEXT_LEN = 50000


REPLY_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^-{2,}\s*ORIGINAL MESSAGE\s*-{2,}$", re.IGNORECASE),
    re.compile(r"^DE:\s", re.IGNORECASE),
    re.compile(r"^FROM:\s", re.IGNORECASE),
    re.compile(r"^ENVIADO:\s", re.IGNORECASE),
    re.compile(r"^SENT:\s", re.IGNORECASE),
    re.compile(r"^PARA:\s", re.IGNORECASE),
    re.compile(r"^TO:\s", re.IGNORECASE),
    re.compile(r"^ASSUNTO:\s", re.IGNORECASE),
    re.compile(r"^SUBJECT:\s", re.IGNORECASE),
)

SIGNATURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^att[.,! ]*$", re.IGNORECASE),
    re.compile(r"^atenciosamente[.,! ]*$", re.IGNORECASE),
    re.compile(r"^obter o outlook para", re.IGNORECASE),
    re.compile(r"^enviado do meu", re.IGNORECASE),
)


def _normalize_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_history_and_signature(text: str) -> str:
    lines = _normalize_text(text).split("\n")
    kept: list[str] = []

    for line in lines:
        raw = line.strip()
        if not raw:
            kept.append("")
            continue

        if raw.startswith(">"):
            continue

        if any(p.search(raw) for p in REPLY_BOUNDARY_PATTERNS):
            break

        if any(p.search(raw) for p in SIGNATURE_PATTERNS):
            break

        kept.append(raw)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise KeyError(f"Colunas ausentes: {sorted(missing)}")


def prepare_unified_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> Path:
    df = pd.read_json(input_path)
    required_cols = ["DetalhesDaDemanda", "TipoDeDemanda", "produto", "assunto"]
    _validate_columns(df, required_cols)

    work = df[required_cols].copy()
    rows_before = len(work)

    # Padroniza tipos e remove nulos.
    for col in ["TipoDeDemanda", "produto", "assunto"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
    work = work.dropna(subset=["TipoDeDemanda", "produto", "assunto"]).copy()

    # Limpa texto.
    work["DetalhesDaDemanda_original"] = work["DetalhesDaDemanda"].astype(str)
    work["DetalhesDaDemanda"] = work["DetalhesDaDemanda_original"].map(_strip_history_and_signature)

    # Filtros de tamanho util.
    work["text_len"] = work["DetalhesDaDemanda"].str.len()
    work = work[(work["text_len"] >= MIN_TEXT_LEN) & (work["text_len"] <= MAX_TEXT_LEN)].copy()

    # Dedup exato por texto + labels.
    rows_before_dedup = len(work)
    work = work.drop_duplicates(subset=["DetalhesDaDemanda", "TipoDeDemanda", "produto", "assunto"]).copy()
    rows_after_dedup = len(work)

    # Salva no formato esperado do pipeline.
    out = work[["DetalhesDaDemanda", "TipoDeDemanda", "produto", "assunto"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(output_path, orient="records", force_ascii=False, indent=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_before": int(rows_before),
        "rows_after_null_filter": int(rows_before_dedup),
        "rows_after_dedup": int(rows_after_dedup),
        "rows_removed_dedup": int(rows_before_dedup - rows_after_dedup),
        "min_text_len": MIN_TEXT_LEN,
        "max_text_len": MAX_TEXT_LEN,
        "text_length_stats": {
            "min": int(out["DetalhesDaDemanda"].str.len().min()) if len(out) else 0,
            "p50": float(out["DetalhesDaDemanda"].str.len().quantile(0.5)) if len(out) else 0.0,
            "p90": float(out["DetalhesDaDemanda"].str.len().quantile(0.9)) if len(out) else 0.0,
            "p95": float(out["DetalhesDaDemanda"].str.len().quantile(0.95)) if len(out) else 0.0,
            "max": int(out["DetalhesDaDemanda"].str.len().max()) if len(out) else 0,
            "mean": float(out["DetalhesDaDemanda"].str.len().mean()) if len(out) else 0.0,
        },
        "class_counts": {
            "TipoDeDemanda": {str(k): int(v) for k, v in out["TipoDeDemanda"].value_counts().to_dict().items()},
            "produto": {str(k): int(v) for k, v in out["produto"].value_counts().to_dict().items()},
            "assunto": {str(k): int(v) for k, v in out["assunto"].value_counts().to_dict().items()},
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = prepare_unified_dataset()
    print(f"Dataset preparado: {out}")
