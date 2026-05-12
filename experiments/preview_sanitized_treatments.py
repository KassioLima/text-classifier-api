import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from experiments.sanitize_text import sanitize_text


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "datasets" / "tbl_perm_acionamento_demanda.json"
OUTPUT_PATH = ROOT_DIR / "datasets" / "reports" / "sanitized_treatments_preview.json"


def parse_date(value):
    if not value:
        return None
    text = str(value)
    for candidate in (text[:26], text[:19]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    return None


def latest_month(rows):
    months = Counter()
    for row in rows:
        dt = parse_date(row.get("DataCriacao"))
        if dt:
            months[dt.strftime("%Y-%m")] += 1
    if not months:
        raise ValueError("Nenhuma DataCriacao valida encontrada.")
    return max(months)


def main():
    parser = argparse.ArgumentParser(description="Gera preview local de DetalhesDoTratamentoDaDemanda sanitizado.")
    parser.add_argument("--month", help="Mes no formato YYYY-MM. Se omitido, usa o mes mais recente do dataset.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    month = args.month or latest_month(rows)

    preview = []
    total = 0
    nonempty = 0
    changed = 0
    for row in rows:
        dt = parse_date(row.get("DataCriacao"))
        if not dt or dt.strftime("%Y-%m") != month:
            continue

        total += 1
        original = str(row.get("DetalhesDoTratamentoDaDemanda") or "").strip()
        sanitized = sanitize_text(original)
        if original:
            nonempty += 1
        if original != sanitized:
            changed += 1

        if len(preview) < args.limit:
            preview.append(
                {
                    "ID": row.get("ID"),
                    "DataCriacao": row.get("DataCriacao"),
                    "original": original,
                    "sanitized": sanitized,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "month": month,
        "total_rows": total,
        "nonempty_treatments": nonempty,
        "changed_by_sanitizer": changed,
        "preview": preview,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "preview"}, ensure_ascii=False, indent=2))
    print(f"Preview salvo em: {args.output}")


if __name__ == "__main__":
    main()
