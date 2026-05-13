import asyncio
import time

from api.constraints import Constraints
from api.models import AiModelAssunto, AiModelProduto, AiModelTipoDemanda
from api.printUtils import greenText


class AiModelService:
    promptTeste = None
    testingMode = False

    @staticmethod
    def init(testing: bool = False):
        # Quando testing=True, todas as rotas usam um prompt controlado para validação local.
        AiModelService.testingMode = testing

        if AiModelService.testingMode:
            AiModelService.promptTeste = Constraints.promptTeste
            print(greenText("INFO:     Prompt de teste carregado."))

    @staticmethod
    def _extract_class_id(raw_label: str, classes):
        # Converte labels do Transformers (ex.: LABEL_3) para classId real do classes.json.
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
        # Pipeline de inferência de uma única dimensão:
        # 1) mede tokens antes/depois de truncamento
        # 2) executa classificador
        # 3) enriquece saída com classId/label e métricas de execução
        
        tempo_inicio = time.time()
        num_tokens_before_truncation = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.model.config.max_position_embeddings, truncation=False, return_tensors="pt")["input_ids"].shape[1]

        tokens = AiModel.tokenizer.encode_plus(prompt, max_length=AiModel.getModelAttr("tokens"), truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]

        decoded_prompt = AiModel.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        results = await asyncio.to_thread(AiModel.classifier, decoded_prompt)
        tempo_fim = time.time()

        result = results[0]
        raw_label = result.get("label")
        class_id = AiModelService._extract_class_id(raw_label, AiModel.classes)
        if class_id is not None:
            result["rawLabel"] = raw_label
            result["classId"] = class_id
            label_map = AiModel.getModelAttr("labelMap")
            result["label"] = label_map.get(int(class_id), str(class_id))

        result["task"] = AiModel.getModelAttr("task")
        result["model"] = AiModel.getModelAttr("name")
        result["maxTokens"] = AiModel.getModelAttr("tokens")
        result["characteresBeforeTruncation"] = len(prompt)
        result["tokensBeforeTruncation"] = num_tokens_before_truncation
        result["processedCharacteres"] = len(decoded_prompt)
        result["processedTokens"] = num_tokens
        result["milliseconds"] = (tempo_fim - tempo_inicio) * 1000

        return result

    @staticmethod
    async def classify(prompt):
        # Classificação completa (3 dimensões). Mantido sequencial para facilitar rastreio.
        if AiModelService.promptTeste is not None:
            prompt = AiModelService.promptTeste

        resultTipo = await AiModelService.classifyByAiModel(prompt, AiModelTipoDemanda)
        resultProduto = await AiModelService.classifyByAiModel(prompt, AiModelProduto)
        resultAssunto = await AiModelService.classifyByAiModel(prompt, AiModelAssunto)

        return {"tipo": resultTipo, "produto": resultProduto, "assunto": resultAssunto}

    @staticmethod
    async def classifyTipoDemanda(prompt):
        if AiModelService.promptTeste is not None:
            prompt = AiModelService.promptTeste

        result = await AiModelService.classifyByAiModel(prompt, AiModelTipoDemanda)

        return {"tipo": result}

    @staticmethod
    async def classifyProduto(prompt):
        if AiModelService.promptTeste is not None:
            prompt = AiModelService.promptTeste

        result = await AiModelService.classifyByAiModel(prompt, AiModelProduto)

        return {"produto": result}

    @staticmethod
    async def classifyAssunto(prompt):
        if AiModelService.promptTeste is not None:
            prompt = AiModelService.promptTeste

        result = await AiModelService.classifyByAiModel(prompt, AiModelAssunto)

        return {"assunto": result}


__all__ = [
    "AiModelService",
    "AiModelTipoDemanda",
    "AiModelProduto",
    "AiModelAssunto",
]
