import asyncio
import json
import time
from pathlib import Path

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification as AutoModel
from api.constraints import Constraints
from api.printUtils import greenText
from experiments.label_mappings import TIPO_DEMANDA_LABELS, PRODUTO_LABELS, ASSUNTO_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models_neuralmind_bert_large" / "lr2e5_bs4_ep4_tipo30_produto30_assunto50"

class AiModelTipoDemanda:
    modelOptions = [
        {"path": str(MODEL_ROOT / "tipo"), "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-tipo", "tokens": 512, "task": "sentiment-analysis", "labelMap": TIPO_DEMANDA_LABELS},
    ]
    
    model_index = None
    model = None
    tokenizer = None
    classes = None
    classifier = None
    
    @staticmethod
    def getModelAttr(attr: str):
        return AiModelTipoDemanda.modelOptions[AiModelTipoDemanda.model_index][attr]
    
    @staticmethod
    def init(modelIndex: int):
        AiModelTipoDemanda.model_index = modelIndex
        
        AiModelTipoDemanda.model = AutoModel.from_pretrained(AiModelTipoDemanda.getModelAttr('path'))
        print(greenText("INFO:     Modelo carregado (\"" + AiModelTipoDemanda.getModelAttr('name') + "\")"))
        AiModelTipoDemanda.tokenizer = AutoTokenizer.from_pretrained(AiModelTipoDemanda.getModelAttr('path'))
        print(greenText("INFO:     Tokenizer carregado"))
        AiModelTipoDemanda.classifier = pipeline(
            AiModelTipoDemanda.getModelAttr("task"),
            model=AiModelTipoDemanda.model,
            tokenizer=AiModelTipoDemanda.tokenizer,
        )
        print(greenText("INFO:     Pipeline carregado"))
        classes_path = Path(AiModelTipoDemanda.getModelAttr('path')) / "classes.json"
        if classes_path.exists():
            AiModelTipoDemanda.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            print(greenText("INFO:     Classes carregadas"))
        else:
            AiModelTipoDemanda.classes = None

class AiModelProduto:
    
    modelOptions = [
        {"path": str(MODEL_ROOT / "produto"), "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-produto", "tokens": 512, "task": "sentiment-analysis", "labelMap": PRODUTO_LABELS},
    ]
    
    model_index = None
    model = None
    tokenizer = None
    classes = None
    classifier = None
    
    @staticmethod
    def getModelAttr(attr: str):
        return AiModelProduto.modelOptions[AiModelProduto.model_index][attr]
    
    @staticmethod
    def init(modelIndex: int):
        AiModelProduto.model_index = modelIndex
        
        AiModelProduto.model = AutoModel.from_pretrained(AiModelProduto.getModelAttr('path'))
        print(greenText("INFO:     Modelo carregado (\"" + AiModelProduto.getModelAttr('name') + "\")"))
        AiModelProduto.tokenizer = AutoTokenizer.from_pretrained(AiModelProduto.getModelAttr('path'))
        print(greenText("INFO:     Tokenizer carregado"))
        AiModelProduto.classifier = pipeline(
            AiModelProduto.getModelAttr("task"),
            model=AiModelProduto.model,
            tokenizer=AiModelProduto.tokenizer,
        )
        print(greenText("INFO:     Pipeline carregado"))
        classes_path = Path(AiModelProduto.getModelAttr('path')) / "classes.json"
        if classes_path.exists():
            AiModelProduto.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            print(greenText("INFO:     Classes carregadas"))
        else:
            AiModelProduto.classes = None

class AiModelAssunto:
    modelOptions = [
        {"path": str(MODEL_ROOT / "assunto"), "name": "lr2e5_bs4_ep4_tipo30_produto30_assunto50-assunto", "tokens": 512, "task": "sentiment-analysis", "labelMap": ASSUNTO_LABELS},
    ]
    
    model_index = None
    model = None
    tokenizer = None
    classes = None
    classifier = None
    
    @staticmethod
    def getModelAttr(attr: str):
        return AiModelAssunto.modelOptions[AiModelAssunto.model_index][attr]
    
    @staticmethod
    def init(modelIndex: int):
        AiModelAssunto.model_index = modelIndex
        
        AiModelAssunto.model = AutoModel.from_pretrained(AiModelAssunto.getModelAttr('path'))
        print(greenText("INFO:     Modelo carregado (\"" + AiModelAssunto.getModelAttr('name') + "\")"))
        AiModelAssunto.tokenizer = AutoTokenizer.from_pretrained(AiModelAssunto.getModelAttr('path'))
        print(greenText("INFO:     Tokenizer carregado"))
        AiModelAssunto.classifier = pipeline(
            AiModelAssunto.getModelAttr("task"),
            model=AiModelAssunto.model,
            tokenizer=AiModelAssunto.tokenizer,
        )
        print(greenText("INFO:     Pipeline carregado"))
        classes_path = Path(AiModelAssunto.getModelAttr('path')) / "classes.json"
        if classes_path.exists():
            AiModelAssunto.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            print(greenText("INFO:     Classes carregadas"))
        else:
            AiModelAssunto.classes = None

class AiModelService:
    promptGigante = None
    testingMode = False
    
    @staticmethod
    def init(testing: bool = False):
        AiModelService.testingMode = testing
        
        if AiModelService.testingMode:
            AiModelService.promptGigante = Constraints.promptTeste
            print(greenText("INFO:     Prompt de teste carregado."))
        
    @staticmethod
    def _extract_class_id(raw_label: str, classes):
        if not raw_label or classes is None:
            return None
        if not raw_label.startswith("LABEL_"):
            return None
        try:
            index = int(raw_label.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
        if index < 0 or index >= len(classes):
            return None
        return classes[index]

    @staticmethod
    async def classifyByAiModel(prompt, AiModel):
        tempo_inicio = time.time()
        num_tokens_before_truncation = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.model.config.max_position_embeddings, truncation=False, return_tensors="pt")["input_ids"].shape[1]
        
        tokens = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.getModelAttr('tokens'), truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        
        decoded_prompt = AiModel.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        results = (await asyncio.to_thread(AiModel.classifier, decoded_prompt))
        tempo_fim = time.time()
        
        result = results[0]
        raw_label = result.get("label")
        class_id = AiModelService._extract_class_id(raw_label, AiModel.classes)
        if class_id is not None:
            result["rawLabel"] = raw_label
            result["classId"] = class_id
            label_map = AiModel.getModelAttr("labelMap")
            result["label"] = label_map.get(int(class_id), str(class_id))
        
        result['task'] = AiModel.getModelAttr('task')
        result['model'] = AiModel.getModelAttr('name')
        result['maxTokens'] = AiModel.getModelAttr('tokens')
        result['characteresBeforeTruncation'] = len(prompt)
        result['tokensBeforeTruncation'] = num_tokens_before_truncation
        result['processedCharacteres'] = len(decoded_prompt)
        result['processedTokens'] = num_tokens
        result['milliseconds'] = (tempo_fim - tempo_inicio) * 1000
        
        return result
    
    @staticmethod
    async def classify(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        resultTipo = (await AiModelService.classifyByAiModel(prompt, AiModelTipoDemanda))
        resultProduto = (await AiModelService.classifyByAiModel(prompt, AiModelProduto))
        resultAssunto = (await AiModelService.classifyByAiModel(prompt, AiModelAssunto))
        
        return {"tipo": resultTipo, "produto": resultProduto, "assunto": resultAssunto}
    
    @staticmethod
    async def classifyTipoDemanda(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        result = (await AiModelService.classifyByAiModel(prompt, AiModelTipoDemanda))
        
        return {"tipo": result}
    
    @staticmethod
    async def classifyProduto(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        result = (await AiModelService.classifyByAiModel(prompt, AiModelProduto))
        
        return {"produto": result}
    
    @staticmethod
    async def classifyAssunto(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        result = (await AiModelService.classifyByAiModel(prompt, AiModelAssunto))
        
        return {"assunto": result}
