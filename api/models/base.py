import json
from abc import ABC
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification as AutoModel
from transformers import AutoTokenizer, pipeline

from api.printUtils import greenText

load_dotenv()


class AiModelBase(ABC):
    # Classe base reutilizada por tipo/produto/assunto.
    # Cada classe filha define apenas modelOptions (path, nome, task, labelMap, tokens).
    modelOptions = []
    model_index = None
    model = None
    tokenizer = None
    classes = None
    classifier = None

    @classmethod
    def getModelAttr(cls, attr: str, default=None):
        return cls.modelOptions[cls.model_index].get(attr, default)

    @classmethod
    def init(cls, modelIndex: int):
        # modelIndex permite alternar rapidamente entre opções de modelo
        # (ex.: local em disco vs Hugging Face Hub), sem mudar código de inferência.
        cls.model_index = modelIndex
        model_path = cls.getModelAttr("path")
        # O token já vem resolvido no modelOptions (campo ACCESS_TOKEN).
        token = cls.getModelAttr("ACCESS_TOKEN")
        pretrained_kwargs = {"token": token} if token else {}

        cls.model = AutoModel.from_pretrained(model_path, **pretrained_kwargs)
        print(greenText(f'INFO:     Modelo carregado ("{cls.getModelAttr("name")}")'))

        cls.tokenizer = AutoTokenizer.from_pretrained(model_path, **pretrained_kwargs)
        print(greenText("INFO:     Tokenizer carregado"))

        cls.classifier = pipeline(cls.getModelAttr("task"), model=cls.model, tokenizer=cls.tokenizer)
        print(greenText("INFO:     Pipeline carregado"))

        # classes.json mapeia LABEL_X -> classId original de negócio.
        # Se path for local, lê direto do disco; se for Hub, baixa o arquivo do repositório.
        classes_path = None
        local_path = Path(model_path)
        if local_path.exists():
            classes_path = local_path / "classes.json"
        else:
            try:
                downloaded_path = hf_hub_download(repo_id=model_path, filename="classes.json", token=token)
                classes_path = Path(downloaded_path)
            except Exception:
                classes_path = None

        if classes_path and classes_path.exists():
            cls.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            print(greenText("INFO:     Classes carregadas"))
        else:
            cls.classes = None
