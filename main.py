from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fish_sugg.fish_sugg import predict_fish
from dd.dd import predict_diseases

app = FastAPI()


class PondMetrics(BaseModel):
    metric: list[float]


@app.get("/")
async def root():
    return "AgroSmartAI API is running!"


@app.get("/ping")
async def ping():
    return "Pong"


@app.post("/fish/")
async def fish(data: PondMetrics):
    return predict_fish(data.metric)


@app.post("/dd")
async def dd(file: UploadFile):
    open('./temp/temp.jpg', 'wb').write(file.file.read())
    return predict_diseases('./temp/temp.jpg')
