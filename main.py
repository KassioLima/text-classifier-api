import traceback
from CustomResponse import Response
from fastapi import FastAPI

from printUtils import greenText
from service import AiModelService as AMS, AiModelProduto, AiModelAssunto

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    AiModelProduto.init(modelIndex=0)
    AiModelAssunto.init(modelIndex=0)
    AMS.init(testing=False)
    print(greenText("INFO:     API PRONTA PARA USO!"))

@app.get("/apiStatus")
async def apiStatus():
    return Response("API funcionando corretamente")

@app.post("/classify")
async def classify(request: dict):
    try:
        result = await AMS.classify(request['prompt'])
        return Response(result)
    except:
        return Response(traceback.format_exc(), 500)