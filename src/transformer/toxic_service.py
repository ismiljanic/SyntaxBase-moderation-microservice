from fastapi import FastAPI
from pydantic import BaseModel
from src.transformer.toxic_predict import toxicbert_predict

app = FastAPI(title="ToxicBERT Service")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: TextRequest):
    try:
        result = toxicbert_predict(req.text)
        return result
    except Exception as e:
        return {"error": str(e)}