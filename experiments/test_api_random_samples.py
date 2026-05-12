import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.label_mappings import ASSUNTO_LABELS, PRODUTO_LABELS, TIPO_DEMANDA_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa exemplos aleatorios no endpoint /classify e compara com gabarito."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "demandas_unificado_train_ready.json",
        help="Caminho do dataset JSON.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/classify",
        help="URL do endpoint de classificacao.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Quantidade de exemplos aleatorios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para amostragem reproducivel. Se omitido, sorteia diferente a cada execucao.",
    )
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def label_for(field: str, class_id: int | None) -> str:
    if class_id is None:
        return "DESCONHECIDO"
    if field == "tipo":
        return TIPO_DEMANDA_LABELS.get(class_id, f"ID {class_id}")
    if field == "produto":
        return PRODUTO_LABELS.get(class_id, f"ID {class_id}")
    if field == "assunto":
        return ASSUNTO_LABELS.get(class_id, f"ID {class_id}")
    return f"ID {class_id}"


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset)
    if not dataset_path.exists():
        print(f"ERRO: dataset nao encontrado: {dataset_path}")
        return 1

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print("ERRO: dataset vazio ou formato invalido.")
        return 1

    valid_rows = [
        row
        for row in rows
        if row.get("DetalhesDaDemanda")
        and row.get("TipoDeDemanda") is not None
        and row.get("produto") is not None
        and row.get("assunto") is not None
    ]
    if not valid_rows:
        print("ERRO: nenhum registro valido encontrado.")
        return 1

    sample_size = min(args.samples, len(valid_rows))
    if args.seed is None:
        selected = random.sample(valid_rows, sample_size)
    else:
        rnd = random.Random(args.seed)
        selected = rnd.sample(valid_rows, sample_size)

    total_hits = 0
    total_possible = sample_size * 3
    hit_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    error_by_field = {"tipo": 0, "produto": 0, "assunto": 0}

    print(f"Testando {sample_size} exemplo(s) em {args.api_url}")
    print("-" * 80)

    for i, row in enumerate(selected, start=1):
        prompt = row["DetalhesDaDemanda"]
        gold = {
            "tipo": as_int(row.get("TipoDeDemanda")),
            "produto": as_int(row.get("produto")),
            "assunto": as_int(row.get("assunto")),
        }

        try:
            response = post_json(args.api_url, {"prompt": prompt})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[{i:02d}] ERRO HTTP {exc.code}: {body}")
            print("-" * 80)
            continue
        except Exception as exc:
            print(f"[{i:02d}] ERRO DE REQUISICAO: {exc}")
            print("-" * 80)
            continue

        data = response.get("data", {})
        pred = {
            "tipo": as_int(data.get("tipo", {}).get("classId")),
            "produto": as_int(data.get("produto", {}).get("classId")),
            "assunto": as_int(data.get("assunto", {}).get("classId")),
        }

        field_hits = 0
        wrong_fields = []
        for field in ("tipo", "produto", "assunto"):
            if pred[field] == gold[field]:
                field_hits += 1
            else:
                wrong_fields.append(field)
                error_by_field[field] += 1

        total_hits += field_hits
        hit_distribution[field_hits] += 1
        status = "ACERTOU 3/3" if field_hits == 3 else f"ACERTOU {field_hits}/3"

        print(f"[{i:02d}] {status}")
        print(
            f"  tipo:    ERC={label_for('tipo', gold['tipo'])} | "
            f"IA={data.get('tipo', {}).get('label', label_for('tipo', pred['tipo']))}"
        )
        print(
            f"  produto: ERC={label_for('produto', gold['produto'])} | "
            f"IA={data.get('produto', {}).get('label', label_for('produto', pred['produto']))}"
        )
        print(
            f"  assunto: ERC={label_for('assunto', gold['assunto'])} | "
            f"IA={data.get('assunto', {}).get('label', label_for('assunto', pred['assunto']))}"
        )
        if wrong_fields:
            print("  errou em:", ", ".join(wrong_fields))
        print("-" * 80)

    pct = (total_hits / total_possible) * 100 if total_possible else 0.0
    print(f"Resumo: {total_hits}/{total_possible} campos corretos ({pct:.2f}%).")
    print("Distribuicao por exemplo:")
    for hits in (3, 2, 1, 0):
        count = hit_distribution[hits]
        share = (count / sample_size) * 100 if sample_size else 0.0
        print(f"  {hits}/3: {count}/{sample_size} ({share:.2f}%)")
    total_errors = total_possible - total_hits
    print(f"Erros totais (campos): {total_errors}")
    if total_errors > 0:
        for field in ("tipo", "produto", "assunto"):
            count = error_by_field[field]
            share = (count / total_errors) * 100
            print(f"  {field}: {count} ({share:.2f}%)")
    else:
        print("  tipo: 0 (0.00%)")
        print("  produto: 0 (0.00%)")
        print("  assunto: 0 (0.00%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
