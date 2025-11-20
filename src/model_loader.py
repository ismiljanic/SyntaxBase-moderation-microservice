import joblib
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import (
    CLASSICAL_XGB_PATH, CLASSICAL_VEC_PATH, CLASSICAL_LE_PATH,
    BERT_PATH, TOXICBERT_PATH, LLM_MODEL_NAME, DEVICE
)

# ---------- classical (XGBoost + vectorizer + label encoder) ----------
classical_model = None
vectorizer = None
label_encoder = None

def load_classical():
    global classical_model, vectorizer, label_encoder
    if classical_model is None:
        classical_model = joblib.load(CLASSICAL_XGB_PATH)
        vectorizer = joblib.load(CLASSICAL_VEC_PATH)
        label_encoder = joblib.load(CLASSICAL_LE_PATH)
    return classical_model, vectorizer, label_encoder

# ---------- transformer (DistilBERT/ToxicBERT) ----------
bert_tokenizer = None
bert_model = None
bert_label_map = None

def load_bert(model_dir: Path):
    global bert_tokenizer, bert_model, bert_label_map
    if bert_model is None:
        bert_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        bert_model.eval()
    return bert_tokenizer, bert_model


# ---------- LLM loader (if local HF-style model usable directly) ----------
llm_tokenizer = None
llm_model = None

def load_llm(name=LLM_MODEL_NAME):
    global llm_tokenizer, llm_model
    if llm_model is None:
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(name)
            llm_model = AutoModelForSequenceClassification.from_pretrained(name)
            llm_model.eval()
        except Exception:
            llm_tokenizer, llm_model = None, None
    return llm_tokenizer, llm_model
