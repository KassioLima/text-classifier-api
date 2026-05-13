TIPO_LABELS = {
    10: "INFORMAÇÃO / DÚVIDA",
    20: "SOLICITAÇÃO",
    25: "SUGESTÃO",
    30: "RECLAMAÇÃO",
    40: "ELOGIO",
}

PRODUTO_LABELS = {
    4: "OI FIXO (THT)",
    5: "VELOX",
    6: "DADOS",
    7: "VOZ AVANÇADA",
    8: "OI MÓVEL",
    9: "TI",
    10: "VOICE NET / PABX VIRTUAL",
    11: "VOZ AVANÇADA - DIGITRONCO/ISDN",
    12: "VOZ AVANÇADA - DEMAIS PRODUTOS",
    13: "WIFI",
    14: "GIS",
    15: "MSS",
    16: "ANTI-DDOS",
    17: "TI - DEMAIS PRODUTOS",
    18: "OI GESTOR",
    19: "TRIDÍGITO",
    20: "NÃO SE APLICA",
    21: "UC4X (NRES / VOICE NET)",
    22: "CLOUD COMMUNICATION",
    23: "FTTH",
    24: "0800/NUN",
}

ASSUNTO_LABELS = {
    1: "RELACIONAMENTO COM CLIENTE",
    2: "CONTA",
    3: "ENTREGA",
    4: "COBRANÇA",
    5: "REPARO",
    6: "VENDAS",
    7: "ADMINISTRAÇÃO DE CONTRATOS",
    8: "DESEMPENHO",
    9: "PRODUTOS",
}


def format_label_options(labels: dict[int, str]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(labels.items()))


def labels_prompt_block() -> str:
    return (
        "Tipos:\n"
        f"{format_label_options(TIPO_LABELS)}\n\n"
        "Produtos:\n"
        f"{format_label_options(PRODUTO_LABELS)}\n\n"
        "Assuntos:\n"
        f"{format_label_options(ASSUNTO_LABELS)}"
    )
