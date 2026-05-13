import traceback
from api.CustomResponse import Response
from fastapi import FastAPI

from api.printUtils import greenText
from api.service import AiModelService as AMS, AiModelProduto, AiModelAssunto, AiModelTipoDemanda

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Carrega os três modelos na subida da API para evitar latência alta
    # na primeira requisição de classificação.
    AiModelTipoDemanda.init(modelIndex=0)
    AiModelProduto.init(modelIndex=0)
    AiModelAssunto.init(modelIndex=0)
    # Modo de teste permite forçar prompt fixo (ver service.py).
    AMS.init(testing=False)
    print(greenText("INFO:     API PRONTA PARA USO!"))

@app.get("/apiStatus")
async def apiStatus():
    return Response("API funcionando corretamente")

@app.post("/classify/tipoDemanda")
async def classifyTipoDemanda(request: dict):
    try:
        result = await AMS.classifyTipoDemanda(request['prompt'])
        return Response(result)
    except:
        return Response(traceback.format_exc(), 500)
    
@app.post("/classify/produto")
async def classifyProduto(request: dict):
    try:
        result = await AMS.classifyProduto(request['prompt'])
        return Response(result)
    except:
        return Response(traceback.format_exc(), 500)

@app.post("/classify/assunto")
async def classifyAssunto(request: dict):
    try:
        result = await AMS.classifyAssunto(request['prompt'])
        return Response(result)
    except:
        return Response(traceback.format_exc(), 500)

@app.post("/classify")
async def classify(request: dict):
    try:
        # Endpoint agregado: retorna tipo + produto + assunto no mesmo payload.
        result = await AMS.classify(request['prompt'])
        return Response(result)
    except:
        return Response(traceback.format_exc(), 500)
