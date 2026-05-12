from pathlib import Path

from shared.label_mappings import ASSUNTO_LABELS

from .base import AiModelBase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"


class AiModelAssunto(AiModelBase):
    modelOptions = [
        {
            "path": str(MODEL_ROOT / "assunto"),
            "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-assunto",
            "tokens": 512,
            "task": "sentiment-analysis",
            "labelMap": ASSUNTO_LABELS,
        },
    ]
