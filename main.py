import traceback
from CustomResponse import Response
from fastapi import FastAPI
from service import AiModelService as AMS, AiModel

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    AiModel.init()

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