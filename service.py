import asyncio
import json
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification as AutoModel
from constraints import Constraints
from printUtils import greenText


class AiModel:
    
    # TODO: TREINAR UM MODELO APENAS PARA CLASSIFICAR PRODUTOS [IN PROGRESS]
    # TODO: TREINAR UM MODELO APENAS PARA CLASSIFICAR ASSUNTOS [IN PROGRESS]
    # TODO: USAR AMBOS OS MODELOS PROPOSTOS ACIMA SIMULTANEAMENTE [PENDING]
    
    modelOptions = [
        {"path": "KassioMaminfo/autotrain-rac-11679-cardiffnlp-roberta-base", "name": "rac-roberta-11679-512", "tokens": 512, "task": "sentiment-analysis"},
        {"path": "facebook/bart-large-mnli", "name": "facebook-bart-1024", "tokens": 1024, "task": "sentiment-analysis"}
    ]
    
    model_index = None
    model = None
    tokenizer = None
    
    @staticmethod
    def getModelAttr(attr: str):
        return AiModel.modelOptions[AiModel.model_index][attr]
    
    @staticmethod
    def init(modelIndex: int):
        AiModel.model_index = modelIndex
        
        AiModel.model = AutoModel.from_pretrained(AiModel.getModelAttr('path'))
        print(greenText("INFO:     Modelo carregado (\"" + AiModel.getModelAttr('name') + "\")"))
        AiModel.tokenizer = AutoTokenizer.from_pretrained(AiModel.getModelAttr('path'))
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
        
        num_tokens_before_truncation = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.model.config.max_position_embeddings, truncation=False, return_tensors="pt")["input_ids"].shape[1]
        
        tokens = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.getModelAttr('tokens'), truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        
        decoded_prompt = AiModel.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
        
        classifier = pipeline(AiModel.getModelAttr('task'), model=AiModel.model, tokenizer=AiModel.tokenizer)
        results = (await asyncio.to_thread(classifier, decoded_prompt))
        
        result = results[0]
        
        result['task'] = AiModel.getModelAttr('task')
        result['model'] = AiModel.getModelAttr('name')
        result['maxTokens'] = AiModel.getModelAttr('tokens')
        result['characteresBeforeTruncation'] = len(prompt)
        result['tokensBeforeTruncation'] = num_tokens_before_truncation
        result['processedCharacteres'] = len(decoded_prompt)
        result['processedTokens'] = num_tokens
        
        if 'autotrain-rac' in AiModel.modelOptions[AiModel.model_index]['path']:
            result['label'] = json.loads(result['label'].replace("'", '"'))
        
        return result