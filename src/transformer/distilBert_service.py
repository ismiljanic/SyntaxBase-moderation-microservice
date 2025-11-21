from fastapi import FastAPI
from pydantic import BaseModel
from src.model_loader import load_bert
from src.config import DEVICE

app = FastAPI(title="Distil BERT Model Service")

# ----- LOAD BERT MODEL -----
bert_tokenizer, bert_model = load_bert("models/saved/bert")

LABEL_MAP = {
    0: "safe",
    1: "mild",
    2: "toxic",
    3: "severe"
}

# ----- PREDICTION FUNCTION -----
import torch

def bert_predict_with_confidence(text: str):
    enc = bert_tokenizer([text], truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        outputs = bert_model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return LABEL_MAP.get(pred_idx, str(pred_idx)), float(probs[pred_idx])

# ----- API SCHEMA -----
class TextRequest(BaseModel):
    text: str

# ----- API ROUTE -----
@app.post("/predict")
def predict(req: TextRequest):
    try:
        label, confidence = bert_predict_with_confidence(req.text)
        return {"label": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}