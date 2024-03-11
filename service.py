import asyncio
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification as AutoModel
from constraints import Constraints
from printUtils import greenText


# TODO: TREINAR UM MODELO APENAS PARA CLASSIFICAR PRODUTOS [IN PROGRESS]
# TODO: TREINAR UM MODELO APENAS PARA CLASSIFICAR ASSUNTOS [IN PROGRESS]
# TODO: USAR AMBOS OS MODELOS PROPOSTOS ACIMA SIMULTANEAMENTE [IN PROGRESS]


class AiModelProduto:
    
    modelOptions = [
        {"path": "KassioMaminfo/autotrain-rac-11679-cardiffnlp-roberta-base", "name": "rac-roberta-11679-512", "tokens": 512, "task": "sentiment-analysis"},
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
        {"path": "KassioMaminfo/autotrain-rac-11679-cardiffnlp-roberta-base", "name": "rac-roberta-11679-512", "tokens": 512, "task": "sentiment-analysis"},
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
    async def classify(prompt):
        if AiModelService.promptGigante is not None:
            prompt = AiModelService.promptGigante
        
        num_tokens_before_truncation = AiModelProduto.tokenizer.encode_plus(prompt, max_length=AiModelProduto.model.config.max_position_embeddings, truncation=False, return_tensors="pt")["input_ids"].shape[1]
        
        tokens = AiModelProduto.tokenizer.encode_plus(prompt, max_length=AiModelProduto.getModelAttr('tokens'), truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        
        decoded_prompt = AiModelProduto.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        classifier = pipeline(AiModelProduto.getModelAttr('task'), model=AiModelProduto.model, tokenizer=AiModelProduto.tokenizer)
        results = (await asyncio.to_thread(classifier, decoded_prompt))
        
        result = results[0]
        
        result['task'] = AiModelProduto.getModelAttr('task')
        result['model'] = AiModelProduto.getModelAttr('name')
        result['maxTokens'] = AiModelProduto.getModelAttr('tokens')
        result['characteresBeforeTruncation'] = len(prompt)
        result['tokensBeforeTruncation'] = num_tokens_before_truncation
        result['processedCharacteres'] = len(decoded_prompt)
        result['processedTokens'] = num_tokens
        
        if 'autotrain-rac' in AiModelProduto.modelOptions[AiModelProduto.model_index]['path']:
            result['label'] = json.loads(result['label'].replace("'", '"'))
        
        return result