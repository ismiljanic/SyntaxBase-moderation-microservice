from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

CLASSICAL_XGB_PATH = ROOT / "models" / "saved" / "classical" / "xgboost.pkl"
CLASSICAL_VEC_PATH = ROOT / "models" / "saved" / "classical" / "vectorizer.pkl"
CLASSICAL_LE_PATH = ROOT / "models" / "saved" / "classical" / "label_encoder.pkl"

BERT_PATH = ROOT / "models" / "saved" / "bert"
TOXICBERT_PATH = ROOT / "models" / "saved" / "toxic_bert"

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-4b-thinking-2507")

BERT_CONFIDENCE_THRESHOLD = float(os.getenv("BERT_CONFIDENCE_THRESHOLD", 0.6))

DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")
