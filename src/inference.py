import requests
from src.config import BERT_CONFIDENCE_THRESHOLD, TOXICBERT_CONFIDENCE_THRESHOLD

# -------------------------
# Microservice endpoints
# -------------------------
CLASSICAL_URL = "http://classical:7001/predict"
BERT_URL = "http://bert:7002/predict"
TOXICBERT_URL = "http://toxicbert:7003/predict"
LLM_URL = "http://llm:7004/predict"

# local testing
# CLASSICAL_URL = "http://127.0.0.1:7001/predict"
# BERT_URL = "http://127.0.0.1:7002/predict"
# TOXICBERT_URL = "http://127.0.0.1:7003/predict"
# LLM_URL = "http://127.0.0.1:7004/predict"

# -------------------------
# Service call wrappers
# -------------------------
def call_classical(text: str):
    resp = requests.post(CLASSICAL_URL, json={"text": text})
    resp.raise_for_status()
    return resp.json()["label"]

def call_bert(text: str):
    resp = requests.post(BERT_URL, json={"text": text})
    resp.raise_for_status()
    data = resp.json()
    return data["label"], float(data["confidence"])

def call_toxicbert(text: str):
    resp = requests.post(TOXICBERT_URL, json={"text": text})
    resp.raise_for_status()
    data = resp.json()
    return data["label"], float(data["confidence"])

def call_llm(text: str):
    resp = requests.post(LLM_URL, json={"text": text})
    resp.raise_for_status()
    data = resp.json()
    return data["label"], data.get("reasoning", "")

# -------------------------
# Hybrid classification
# -------------------------
def hybrid_classify(text: str):
    # Phase 1: Classical ML
    classical_label = call_classical(text)

    # Phase 2: BERT
    try:
        bert_label, bert_conf = call_bert(text)
    except Exception as e:
        return {
            "final_label": classical_label,
            "pipeline": ["classical-fallback"],
            "classical_label": classical_label,
            "bert_error": str(e)
        }

    # Phase 3: ToxicBERT
    try:
        toxic_label, toxic_conf = call_toxicbert(text)
    except Exception as e:
        toxic_label, toxic_conf = None, 0.0

    # Check if BERT and ToxicBERT agree
    disagree = (toxic_label is not None) and (bert_label != toxic_label)

    result = {
        "classical_label": classical_label,
        "bert_label": bert_label,
        "bert_confidence": bert_conf,
        "toxic_label": toxic_label,
        "toxic_confidence": toxic_conf,
    }

    if disagree:
        # Phase 4: LLM only if BERT and ToxicBERT disagree
        try:
            llm_label, llm_reasoning = call_llm(text)
        except Exception as e:
            llm_label, llm_reasoning = "error", str(e)

        result.update({
            "final_label": llm_label,
            "llm_label": llm_label,
            "llm_reasoning": llm_reasoning,
            "pipeline": ["classical", "bert", "toxic_bert", "llm"]
        })
    else:
        # If they agree or ToxicBERT missing, pick BERT if confident
        final_label = bert_label if bert_conf >= BERT_CONFIDENCE_THRESHOLD else toxic_label
        result.update({
            "final_label": final_label,
            "pipeline": ["classical", "bert", "toxic_bert"]
        })

    return result

# -------------------------
# API wrapper
# -------------------------
def classify_text_api(text: str):
    return hybrid_classify(text)