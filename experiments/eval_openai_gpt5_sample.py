import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.label_mappings import ASSUNTO_LABELS, PRODUTO_LABELS, TIPO_DEMANDA_LABELS, labels_prompt_block
from experiments.sanitize_text import sanitize_text


DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train_ready.json"
REPORTS_DIR = PROJECT_ROOT / "datasets" / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "openai_gpt5_sample_eval.json"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


CLASSIFICATION_RULES = """
Regras de decisao da taxonomia interna:

TipoDeDemanda:
- Use "Solicitacao" quando o texto pede uma acao operacional: abrir OS/chamado, alterar, cancelar, ativar, agendar, enviar, corrigir, providenciar, priorizar, migrar, instalar, mudar endereco ou liberar acesso.
- Use "Informacao / Duvida" quando o texto pede retorno, status, confirmacao, esclarecimento, validacao ou informacao, sem exigir execucao direta de uma atividade.
- Use "Reclamacao" quando houver falha, indisponibilidade, demora, cobranca de retorno, insatisfacao, reincidencia, servico inoperante ou problema nao resolvido.
- Use "Sugestao" apenas quando o texto propuser melhoria.
- Use "Elogio" apenas quando o texto elogiar atendimento/servico.

Produto:
- Evite "NAO SE APLICA" quando houver qualquer pista de produto. Use "NAO SE APLICA" somente se o texto nao permite inferir nenhum produto.
- Se mencionar link, circuito, dados, internet, banda, Mbps, GNA, TCS, CTA, rede privativa, MPLS, IP, roteador, switch, AP, OS de entrega de circuito ou mudanca de endereco de link/circuito, classifique como "DADOS".
- Se mencionar linha telefonica fixa, telefone fixo, par metalico, linha muda, linha inoperante ou numero fixo sem indicacao de UC4X/Voice Net/Digitronco, classifique como "OI FIXO (THT)".
- Se mencionar UC4X, NRES ou Voice Net no contexto de telefonia corporativa, classifique como "UC4X (NRES / VOICE NET)", exceto quando o texto disser explicitamente PABX Virtual/Voice Net como produto.
- Se mencionar PABX Virtual ou Voice Net de forma direta, classifique como "VOICE NET / PABX VIRTUAL".
- Se mencionar ISDN, R2, E1, digitronco ou sinalizacao, classifique como "VOZ AVANCADA - DIGITRONCO/ISDN".
- Se mencionar 193, tridigito ou numero de emergencia tridigito, classifique como "TRIDIGITO", mesmo que tambem apareca 0800 no texto.
- Se mencionar 0800/NUN sem relacao com 193/tridigito, classifique como "0800/NUN".
- Se mencionar celular, movel, SIM card ou linha movel, classifique como "OI MOVEL".
- Se mencionar cloud communication explicitamente, classifique como "CLOUD COMMUNICATION".
- Se mencionar FTTH/fibra residencial explicitamente, classifique como "FTTH".
- Se mencionar Anti-DDoS explicitamente, classifique como "ANTI-DDOS".

Assunto:
- Use "ENTREGA" para ativacao, instalacao, implantacao, agendamento, OS de entrega, mudanca de endereco de link/circuito, migracao tecnica ou etapa de execucao de pedido.
- Use "REPARO" para falha tecnica, indisponibilidade, servico inoperante, linha muda, telefone nao toca, lentidao/problema de desempenho tecnico ou abertura/acompanhamento de reparo.
- Use "CONTA" para fatura, detalhamento de fatura, cobranca, pagamento, prorrogacao de vencimento, contestacao financeira ou titulo.
- Use "CONTRATO" para criacao, renovacao, aceite, alteracao contratual, distrato ou questoes formais do contrato.
- Use "VENDAS" para pedido comercial novo, proposta, cotacao, venda, nova contratacao ou internalizacao/ativacao tratada como venda.
- Use "DESEMPENHO" apenas para performance/qualidade/degradacao de servico sem falha total.
- Use "RELACIONAMENTO DE POS-VENDA" para acompanhamento, cancelamento ou alteracao de servico ja existente, suporte administrativo, retorno a cliente, alinhamento e demandas de relacionamento que nao sejam claramente Conta, Entrega, Reparo, Contrato, Vendas ou Desempenho.

Prioridades:
- Primeiro identifique produto por palavras-chave tecnicas. Depois identifique assunto pela natureza da demanda. Por fim identifique tipo pela intencao do solicitante.
- Quando houver conflito entre "0800" e "193/tridigito", priorize "TRIDIGITO".
- Quando houver conflito entre "cancelamento de servico existente" e "contrato", priorize "RELACIONAMENTO DE POS-VENDA", salvo se o texto falar explicitamente de contrato formal.
""".strip()


CLASSIFICATION_RULES_V3 = """
Regras de decisao da taxonomia interna:

Principio geral:
- A classificacao deve reproduzir o historico operacional, nao apenas o sentido comum das palavras.
- Evite classes genericas quando houver pista operacional. "NAO SE APLICA" e "RELACIONAMENTO DE POS-VENDA" devem ser usados com cuidado.

TipoDeDemanda:
- Use "Solicitacao" quando o texto pede apoio, providencia, verificacao, retorno operacional, prioridade, abertura de OS/chamado/VIP, envio de documento, correcao, cancelamento, ativacao, migracao, remanejamento, agendamento, instalacao ou alteracao.
- Frases como "algum retorno?", "consegue verificar?", "solicitamos apoio", "favor informar andamento", "pedimos prioridade" geralmente sao "Solicitacao" quando estao ligadas a uma demanda operacional existente.
- Use "Informacao / Duvida" quando o objetivo principal for obter esclarecimento, confirmar informacao, validar dados, saber status sem pedido de providencia, ou quando o texto relata uma informacao sem cobrar acao.
- Use "Reclamacao" somente quando houver insatisfacao clara, reincidencia, atraso cobrado, "sem retorno", "continua inoperante", "nao tivemos posicionamento", "prazo vencido", impacto prolongado, cobranca dura ou problema nao resolvido.
- Falha tecnica por si so nao torna a demanda "Reclamacao"; se o texto pede abertura/apoio de reparo, normalmente e "Solicitacao".
- Use "Elogio" apenas quando o proposito principal for elogiar. "Obrigado", "validado" ou "teste com sucesso" nao bastam para Elogio.
- Use "Sugestao" apenas quando houver proposta de melhoria.

Produto:
- "NAO SE APLICA" e ultima opcao. Use apenas quando nao houver pista de produto, contrato, linha, circuito, fatura, servico, portal ou tecnologia.
- Se houver termos de infraestrutura, circuitos, links, banda, internet, rede, roteador, switch, AP, fibra como circuito corporativo, GNA, TCS, CTA, PAE, SPO, RJO, BSA, designacao tecnica, status report, remanejamento de link, mudanca de endereco de circuito, entrega/baixa de OS de circuito ou prazos de circuito, classifique como "DADOS".
- Se o texto for curto/generico mas mencionar fatura, prorrogação, planta, detalhamento, status report, pendencia, portal, OS ou retorno relacionado a telecom corporativo sem indicar voz/linha/0800/UC4X, prefira "DADOS" em vez de "NAO SE APLICA".
- Se mencionar linha telefonica fixa, telefone fixo, linha muda, reparo de linha, par metalico, numero fixo, VIP de linha ou defeito em linha sem indicar Voice Net/UC4X/Digitronco, classifique como "OI FIXO (THT)".
- Se mencionar "linha" mas o contexto for ouvidoria, ramais corporativos, PABX, Voice Net, chamadas corporativas ou linhas de servico gerenciadas, prefira "VOICE NET / PABX VIRTUAL" em vez de "OI FIXO (THT)".
- Se mencionar PABX Virtual, Voice Net, ramais, ouvidoria, retorno das linhas da ouvidoria, linhas de laboratorio/regional em lote ou reparos associados ao produto Voice Net, classifique como "VOICE NET / PABX VIRTUAL".
- Se mencionar UC4X, NRES, logins/senhas UC4X, migracao UC4X ou telefonia em nuvem UC4X, classifique como "UC4X (NRES / VOICE NET)".
- Se mencionar ISDN, R2, E1, digitronco, sinalizacao, tronco, DDR ou SIP trunk no contexto de voz avancada, classifique como "VOZ AVANCADA - DIGITRONCO/ISDN".
- Se mencionar LDN/LDI, voz avancada sem digitronco/ISDN, chamadas de longa distancia ou servicos de voz avancada genericos, classifique como "VOZ AVANCADA".
- Se mencionar 193, 190, tridígito, tridigito ou numero de emergencia tridigito, classifique como "TRIDIGITO", mesmo que tambem apareca 0800.
- Se mencionar 0800 ou NUN sem relacao com tridigito/emergencia, classifique como "0800/NUN".
- Se mencionar celular, movel, SIM card, chip ou linha movel, classifique como "OI MOVEL".
- Se mencionar MSS, switches/APs sobressalentes ou solucao gerenciada MSS explicitamente, classifique como "MSS".
- Se mencionar Wi-Fi/WIFI como produto, classifique como "WIFI".
- Se mencionar GIS como portal/produto, classifique como "GIS".

Assunto:
- Evite usar "RELACIONAMENTO DE POS-VENDA" como classe geral quando houver sinal claro de Entrega, Reparo, Conta, Contrato ou Vendas.
- Use "ENTREGA" para ativacao, implantacao, instalacao, agendamento, OS de entrega, baixa/devolucao de OS de entrega, mudanca de endereco, remanejamento, migracao, troca de link, novo circuito, validacao tecnica para virada/faturamento de circuito ou andamento de pedido/OS.
- "Algum retorno?", "andamento", "previsao", "cronograma" ou "prioridade" ligados a ativacao, migracao, remanejamento, instalacao, OS ou circuito novo devem ser "ENTREGA".
- Use "REPARO" para falha tecnica, indisponibilidade, linha muda, telefone nao toca, sem internet, link offline, defeito, abertura/acompanhamento de reparo, VIP de reparo, ticket/chamado de defeito ou normalizacao de servico.
- Use "CONTA" para fatura, detalhamento de fatura, conta, cobranca, pagamento, vencimento, prorrogacao de vencimento, contestacao financeira, titulo ou disponibilizacao de fatura.
- Use "CONTRATO" para retirada/cancelamento formal de circuitos/linhas em contrato, alteracao contratual, renovacao, prorrogacao contratual, distrato, termo, aceite contratual ou divergencia formal de contrato.
- Use "VENDAS" para nova contratacao, proposta, cotacao, pedido comercial novo, venda, internalizacao comercial ou ativacao tratada explicitamente como venda.
- Use "DESEMPENHO" apenas para performance/degradacao/lentidao/qualidade sem indisponibilidade total.
- Use "RELACIONAMENTO DE POS-VENDA" para acompanhamento administrativo de servico existente, alinhamento, retorno, cadastro, pendencia nao financeira, suporte de relacionamento, cancelamento simples sem formalidade contratual, ou quando nenhuma das classes especificas acima se aplica.

Prioridades:
- Produto antes de assunto: identifique tecnologia/produto pelas pistas do texto.
- Assunto depois: identifique se a natureza e entrega, reparo, conta, contrato, vendas, desempenho ou pos-venda.
- Tipo por ultimo: diferencie pedido de acao ("Solicitacao") de pedido de esclarecimento puro ("Informacao / Duvida") e de cobranca/insatisfacao ("Reclamacao").
- Se o texto contem "solicito", "solicitamos", "favor", "gentileza", "precisamos", "pedimos", isso tende a "Solicitacao", salvo se o foco for somente tirar duvida.
""".strip()


def label_for(labels: dict[int, str], value: Any) -> str:
    try:
        key = int(value)
    except (TypeError, ValueError):
        return f"UNKNOWN_ID_{value}"
    return labels.get(key, f"UNKNOWN_ID_{key}")


def expected_from_record(record: dict[str, Any]) -> dict[str, Any]:
    # Constrói o gabarito esperado (IDs + labels) a partir do dataset.
    return {
        "tipo_id": int(record["TipoDeDemanda"]),
        "tipo_label": label_for(TIPO_DEMANDA_LABELS, record["TipoDeDemanda"]),
        "produto_id": int(record["produto"]),
        "produto_label": label_for(PRODUTO_LABELS, record["produto"]),
        "assunto_id": int(record["assunto"]),
        "assunto_label": label_for(ASSUNTO_LABELS, record["assunto"]),
    }


def build_prompt(record_id: int, sanitized_text: str, prompt_version: str) -> str:
    # Monta prompt versionado para permitir comparação entre estratégias de instrução.
    if prompt_version == "v2":
        rules = f"\n\n{CLASSIFICATION_RULES}"
    elif prompt_version == "v3":
        rules = f"\n\n{CLASSIFICATION_RULES_V3}"
    else:
        rules = ""
    return (
        "Classifique a demanda abaixo usando exclusivamente as opcoes listadas.\n"
        "Responda em JSON conforme o schema, usando os IDs numericos e os nomes correspondentes.\n\n"
        f"{labels_prompt_block()}"
        f"{rules}\n\n"
        f"Demanda ID: {record_id}\n"
        f"DetalhesDaDemanda sanitizado:\n{sanitized_text}"
    )


def response_schema() -> dict[str, Any]:
    # Esquema estrito para forçar saída JSON com IDs válidos da taxonomia.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tipo_id": {"type": "integer", "enum": sorted(TIPO_DEMANDA_LABELS)},
            "tipo_label": {"type": "string", "enum": [TIPO_DEMANDA_LABELS[k] for k in sorted(TIPO_DEMANDA_LABELS)]},
            "produto_id": {"type": "integer", "enum": sorted(PRODUTO_LABELS)},
            "produto_label": {"type": "string", "enum": [PRODUTO_LABELS[k] for k in sorted(PRODUTO_LABELS)]},
            "assunto_id": {"type": "integer", "enum": sorted(ASSUNTO_LABELS)},
            "assunto_label": {"type": "string", "enum": [ASSUNTO_LABELS[k] for k in sorted(ASSUNTO_LABELS)]},
        },
        "required": [
            "tipo_id",
            "tipo_label",
            "produto_id",
            "produto_label",
            "assunto_id",
            "assunto_label",
        ],
    }


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                chunks.append(content["text"])
            elif content.get("type") == "refusal":
                raise RuntimeError(f"Modelo recusou a classificacao: {content.get('refusal')}")
    if not chunks:
        raise RuntimeError(f"Nao foi possivel extrair output_text da resposta: {response}")
    return "".join(chunks)


def call_openai(prompt: str, model: str, timeout: int, max_output_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    # Chamada HTTP direta para Responses API com saída estruturada.
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Defina a variavel de ambiente OPENAI_API_KEY antes de executar sem --dry-run.")

    payload = {
        "model": model,
        "store": False,
        "reasoning": {"effort": "minimal"},
        "instructions": (
            "Voce e um classificador de demandas. Use apenas as opcoes fornecidas. "
            "Nao invente IDs ou labels. Retorne somente o JSON estruturado."
        ),
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "demand_classification",
                "strict": True,
                "schema": response_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
    }

    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API retornou HTTP {exc.code}: {body}") from exc

    api_response = json.loads(raw)
    output_text = extract_output_text(api_response)
    return json.loads(output_text), api_response


def compare(expected: dict[str, Any], predicted: dict[str, Any]) -> dict[str, bool]:
    return {
        "tipo": expected["tipo_id"] == predicted.get("tipo_id"),
        "produto": expected["produto_id"] == predicted.get("produto_id"),
        "assunto": expected["assunto_id"] == predicted.get("assunto_id"),
    }


def sample_records(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    # Amostragem aleatória reproduzível do dataset para benchmark rápido.
    eligible = [
        record
        for record in records
        if str(record.get("DetalhesDaDemanda") or "").strip()
        and record.get("TipoDeDemanda") is not None
        and record.get("produto") is not None
        and record.get("assunto") is not None
        and int(record["TipoDeDemanda"]) in TIPO_DEMANDA_LABELS
        and int(record["produto"]) in PRODUTO_LABELS
        and int(record["assunto"]) in ASSUNTO_LABELS
    ]
    if len(eligible) < sample_size:
        raise ValueError(f"Amostra solicitada ({sample_size}) maior que registros elegiveis ({len(eligible)}).")
    rng = random.Random(seed)
    return rng.sample(eligible, sample_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia gpt-5 em 10 demandas aleatorias com prompt detalhado e texto sanitizado."
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="Nao chama a API; apenas gera a amostra e os prompts.")
    args = parser.parse_args()

    records = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    selected = sample_records(records, args.sample_size, args.seed)
    skipped_unmapped = sum(
        1
        for record in records
        if record.get("TipoDeDemanda") is not None
        and record.get("produto") is not None
        and record.get("assunto") is not None
        and (
            int(record["TipoDeDemanda"]) not in TIPO_DEMANDA_LABELS
            or int(record["produto"]) not in PRODUTO_LABELS
            or int(record["assunto"]) not in ASSUNTO_LABELS
        )
    )

    # Loop de avaliação item a item, com comparação contra o gabarito.
    results = []
    for index, record in enumerate(selected, start=1):
        record_id = int(record.get("ID", index)) if str(record.get("ID", "")).isdigit() else index
        original_text = str(record.get("DetalhesDaDemanda") or "").strip()
        sanitized_text = sanitize_text(original_text)
        expected = expected_from_record(record)
        prompt = build_prompt(record_id, sanitized_text, args.prompt_version)

        item = {
            "sample_index": index,
            "record_id": record.get("ID"),
            "sanitized_text": sanitized_text,
            "expected": expected,
            "prompt": prompt,
        }

        if args.dry_run:
            item["predicted"] = None
            item["matches"] = None
            item["api_usage"] = None
        else:
            predicted, api_response = call_openai(
                prompt=prompt,
                model=args.model,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
            )
            matches = compare(expected, predicted)
            item["predicted"] = predicted
            item["matches"] = matches
            item["all_match"] = all(matches.values())
            item["api_response_id"] = api_response.get("id")
            item["api_usage"] = api_response.get("usage")
            print(
                f"[{index}/{args.sample_size}] "
                f"tipo={matches['tipo']} produto={matches['produto']} assunto={matches['assunto']}"
            )

        results.append(item)

    summary = {
        "model": args.model,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "prompt_version": args.prompt_version,
        "dataset": str(DATASET_PATH),
        "text_field": "DetalhesDaDemanda",
        "skipped_unmapped_records": skipped_unmapped,
        "results": results,
    }

    if not args.dry_run:
        summary["accuracy"] = {
            "tipo": sum(1 for r in results if r["matches"]["tipo"]) / len(results),
            "produto": sum(1 for r in results if r["matches"]["produto"]) / len(results),
            "assunto": sum(1 for r in results if r["matches"]["assunto"]) / len(results),
            "all": sum(1 for r in results if r["all_match"]) / len(results),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatorio salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
