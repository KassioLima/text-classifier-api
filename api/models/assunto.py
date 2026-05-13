from pathlib import Path

from shared.label_mappings import ASSUNTO_LABELS

from .base import AiModelBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"


class AiModelAssunto(AiModelBase):
    # Configuração da dimensão "assunto".
    # O labelMap traduz classId numérico para nome amigável no retorno da API.
    # Coloque seu modelo local ou do Hugging Face na lista abaixo para usá-lo.
    # Certifique-se de que "AiModelAssunto.init(modelIndex=0)" está apontando para o índice certo do modelo que deseja usar
    modelOptions = [
        {
            "path": str(MODEL_ROOT / "assunto"),
            "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-assunto",
            "tokens": 512,
            "task": "sentiment-analysis",
            "labelMap": ASSUNTO_LABELS,
        },
    ]
