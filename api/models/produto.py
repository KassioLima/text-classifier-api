from pathlib import Path

from shared.label_mappings import PRODUTO_LABELS

from .base import AiModelBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"


class AiModelProduto(AiModelBase):
    # Configuração da dimensão "produto".
    # O labelMap traduz classId numérico para nome amigável no retorno da API.
    # Coloque seu modelo local ou do Hugging Face na lista abaixo para usá-lo.
    # Certifique-se de que "AiModelProduto.init(modelIndex=0)" está apontando para o índice certo do modelo que deseja usar
    modelOptions = [
        {
            "path": str(MODEL_ROOT / "produto"),
            "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-produto",
            "tokens": 512,
            "task": "sentiment-analysis",
            "labelMap": PRODUTO_LABELS,
        },
    ]
