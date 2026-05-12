from pathlib import Path

from shared.label_mappings import TIPO_DEMANDA_LABELS

from .base import AiModelBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"


class AiModelTipoDemanda(AiModelBase):
    modelOptions = [
        {
            "path": str(MODEL_ROOT / "tipo"),
            "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-tipo",
            "tokens": 512,
            "task": "sentiment-analysis",
            "labelMap": TIPO_DEMANDA_LABELS,
        },
    ]
