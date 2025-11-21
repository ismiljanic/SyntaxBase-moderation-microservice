from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import hstack
import numpy as np
from src.model_loader import load_classical
from src.utils import make_numeric_features_list

app = FastAPI(title="Classical Model Service")

# ----- LOAD CLASSICAL MODEL -----
xgb_model, vectorizer, label_encoder = load_classical()

# ----- PREDICTION FUNCTION -----
def classical_predict_single(text: str):
    """
    Takes raw text, vectorizes it, adds numeric features, and predicts the label.
    """
    X_vec = vectorizer.transform([text])
    X_num = make_numeric_features_list([text])
    X_combined = hstack([X_vec, X_num])
    pred_enc = xgb_model.predict(X_combined)[0]
    return label_encoder.inverse_transform([pred_enc])[0]

# ----- API SCHEMA -----
class Comment(BaseModel):
    text: str

# ----- API ROUTE -----
@app.post("/predict")
def predict(comment: Comment):
    try:
        label = classical_predict_single(comment.text)
        return {"label": label}
    except Exception as e:
        return {"error": str(e)}