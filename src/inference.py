# src/inference.py
import torch
import numpy as np
import requests
from pathlib import Path
from scipy.sparse import hstack

from src.model_loader import load_classical, load_bert
from src.utils import make_numeric_features_list
from src.config import BERT_CONFIDENCE_THRESHOLD, DEVICE, LLM_MODEL_NAME
from .llm.llm_moderation_client import LLMModerationClient

# -------------------------
# Load classical and BERT models
# -------------------------
xgb_model, vectorizer, label_encoder = load_classical()
bert_tokenizer, bert_model = load_bert("models/saved/bert")

LABEL_MAP = {
    0: "safe",
    1: "mild",
    2: "toxic",
    3: "severe"
}

# -------------------------
# Project paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "src/llm/prompts/api_call.prompt.txt"

# -------------------------
# Classical prediction
# -------------------------
def classical_predict_single(text: str):
    X_vec = vectorizer.transform([text])
    X_num = make_numeric_features_list([text])
    X_combined = hstack([X_vec, X_num])
    pred_enc = xgb_model.predict(X_combined)[0]
    return label_encoder.inverse_transform([pred_enc])[0]

# -------------------------
# BERT prediction
# -------------------------
def bert_predict_with_confidence(text: str):
    enc = bert_tokenizer([text], truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        outputs = bert_model(**enc)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return LABEL_MAP.get(pred_idx, str(pred_idx)), float(probs[pred_idx])

# -------------------------
# Initialize LLM client (local)
# -------------------------
# llm_client = LLMModerationClient(
#     base_url="http://localhost:1234/v1/chat/completions",
#     model_name=LLM_MODEL_NAME,
#     prompt_path=str(PROMPT_PATH)
# )

# -------------------------
# Initialize LLM client (Docker)
# -------------------------
llm_client = LLMModerationClient(
    base_url="http://host.docker.internal:1234/v1/chat/completions",
    model_name=LLM_MODEL_NAME,
    prompt_path=str(PROMPT_PATH)
)

# -------------------------
# LLM classification
# -------------------------
def llm_classify_via_lmstudio(text: str):
    """
    Calls LM Studio via LLMModerationClient and returns (label, reasoning)
    """
    result = llm_client.classify_comment(text)
    return result["label"], result["reasoning"]

# -------------------------
# Hybrid classification
# -------------------------
def hybrid_classify(text: str):
    classical_label = classical_predict_single(text)

    try:
        bert_label, bert_conf = bert_predict_with_confidence(text)
    except Exception as e:
        return {
            "final_label": classical_label,
            "pipeline": ["classical-fallback"],
            "classical_label": classical_label,
            "bert_error": str(e)
        }

    if bert_conf >= BERT_CONFIDENCE_THRESHOLD:
        return {
            "final_label": bert_label,
            "pipeline": ["classical", "bert"],
            "classical_label": classical_label,
            "bert_label": bert_label,
            "bert_confidence": bert_conf
        }

    llm_label, llm_reasoning = llm_classify_via_lmstudio(text)
    return {
        "final_label": llm_label,
        "pipeline": ["classical", "bert", "llm"],
        "classical_label": classical_label,
        "bert_label": bert_label,
        "bert_confidence": bert_conf,
        "llm_label": llm_label,
        "llm_reasoning": llm_reasoning
    }

# -------------------------
# API wrapper
# -------------------------
def classify_text_api(text: str):
    return hybrid_classify(text)