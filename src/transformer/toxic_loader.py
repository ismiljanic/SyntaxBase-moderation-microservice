import joblib
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from pathlib import Path

MODEL_DIR = Path("models/saved/toxic_bert/checkpoint-48000")

_tokenizer = None
_model = None
_id2label = None


def load_toxic_bert():
    global _tokenizer, _model, _id2label

    if _model is None:
        _tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
        _model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()

        id2label_path = MODEL_DIR.parent / "id2labelToxicBERT.pkl"
        _id2label = joblib.load(id2label_path)

    return _tokenizer, _model, _id2label
