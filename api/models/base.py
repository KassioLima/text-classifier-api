import json
from abc import ABC
from pathlib import Path

from transformers import AutoModelForSequenceClassification as AutoModel
from transformers import AutoTokenizer, pipeline

from api.printUtils import greenText


class AiModelBase(ABC):
    modelOptions = []
    model_index = None
    model = None
    tokenizer = None
    classes = None
    classifier = None

    @classmethod
    def getModelAttr(cls, attr: str):
        return cls.modelOptions[cls.model_index][attr]

    @classmethod
    def init(cls, modelIndex: int):
        cls.model_index = modelIndex

        cls.model = AutoModel.from_pretrained(cls.getModelAttr("path"))
        print(greenText(f'INFO:     Modelo carregado ("{cls.getModelAttr("name")}")'))

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.getModelAttr("path"))
        print(greenText("INFO:     Tokenizer carregado"))

        cls.classifier = pipeline(
            cls.getModelAttr("task"),
            model=cls.model,
            tokenizer=cls.tokenizer,
        )
        print(greenText("INFO:     Pipeline carregado"))

        classes_path = Path(cls.getModelAttr("path")) / "classes.json"
        if classes_path.exists():
            cls.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            print(greenText("INFO:     Classes carregadas"))
        else:
            cls.classes = None
