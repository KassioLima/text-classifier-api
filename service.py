import asyncio
import json
import time

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification as AutoModel
from constraints import Constraints
from printUtils import greenText


class AiModelProdutoAssunto:
    modelOptions = [
        {"path": "./models/autotrain-rac-11679-cardiffnlp-roberta-base", "name": "local-rac-produto-assunto-roberta-11679-512", "tokens": 512, "task": "sentiment-analysis"},
        {"path": "facebook/bart-large-mnli", "name": "facebook-bart-1024", "tokens": 1024, "task": "sentiment-analysis"}
    ]
    
    model_index = None
    model = None
    tokenizer = None
    
    @staticmethod
    def getModelAttr(attr: str):
        return AiModelProdutoAssunto.modelOptions[AiModelProdutoAssunto.model_index][attr]
    
    @staticmethod
    def init(modelIndex: int):
        AiModelProdutoAssunto.model_index = modelIndex
        
        AiModelProdutoAssunto.model = AutoModel.from_pretrained(AiModelProdutoAssunto.getModelAttr('path'))
        print(greenText("INFO:     Modelo carregado (\"" + AiModelProdutoAssunto.getModelAttr('name') + "\")"))
        AiModelProdutoAssunto.tokenizer = AutoTokenizer.from_pretrained(AiModelProdutoAssunto.getModelAttr('path'))
        print(greenText("INFO:     Tokenizer carregado"))

class AiModelProduto:
    
    modelOptions = [
        {"path": "./models/autotrain-rac-8801-produto-cardiffnlp-roberta-base", "name": "local-rac-produto-roberta-8801-512", "tokens": 512, "task": "sentiment-analysis"},
        {"path": "facebook/bart-large-mnli", "name": "facebook-bart-1024", "tokens": 1024, "task": "sentiment-analysis"}
    ]
    
    model_index = None
    model = None
    tokenizer = None
    
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

class AiModelAssunto:
    modelOptions = [
        {"path": "./models/autotrain-rac-8801-assunto-cardiffnlp-roberta-base", "name": "local-rac-assunto-roberta-8801-512", "tokens": 512, "task": "sentiment-analysis"},
        {"path": "facebook/bart-large-mnli", "name": "facebook-bart-1024", "tokens": 1024, "task": "sentiment-analysis"}
    ]
    
    model_index = None
    model = None
    tokenizer = None
    
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
    async def classifyByAiModel(prompt, AiModel):
        tempo_inicio = time.time()
        num_tokens_before_truncation = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.model.config.max_position_embeddings, truncation=False, return_tensors="pt")["input_ids"].shape[1]
        
        tokens = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.getModelAttr('tokens'), truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        
        decoded_prompt = AiModel.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        classifier = pipeline(AiModel.getModelAttr('task'), model=AiModel.model, tokenizer=AiModel.tokenizer)
        results = (await asyncio.to_thread(classifier, decoded_prompt))
        tempo_fim = time.time()
        
        result = results[0]
        
        result['task'] = AiModel.getModelAttr('task')
        result['model'] = AiModel.getModelAttr('name')
        result['maxTokens'] = AiModel.getModelAttr('tokens')
        result['characteresBeforeTruncation'] = len(prompt)
        result['tokensBeforeTruncation'] = num_tokens_before_truncation
        result['processedCharacteres'] = len(decoded_prompt)
        result['processedTokens'] = num_tokens
        result['milliseconds'] = (tempo_fim - tempo_inicio) * 1000
        
        # if 'autotrain-rac' in AiModel.modelOptions[AiModel.model_index]['path']:
        #     result['label'] = json.loads(result['label'].replace("'", '"'))
        
        return result
    
    @staticmethod
    async def classify(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        # Inicializou o modelo AiModelProdutoAssunto
        if AiModelProdutoAssunto.model_index is not None:
            resultProdutoAssunto = (await AiModelService.classifyByAiModel(prompt, AiModelProdutoAssunto))
            return resultProdutoAssunto
            
        resultProduto = (await AiModelService.classifyByAiModel(prompt, AiModelProduto))
        resultAssunto = (await AiModelService.classifyByAiModel(prompt, AiModelAssunto))
        
        return {"produto": resultProduto, "assunto": resultAssunto}