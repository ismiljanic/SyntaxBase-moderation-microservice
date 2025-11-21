import torch
import torch.nn.functional as F
from src.transformer.toxic_loader import load_toxic_bert

CONFIDENCE_THRESHOLD = 0.8

tokenizer, model, id2label = load_toxic_bert()


def toxicbert_predict(text: str):
    """Runs ToxicBERT inference and returns label + confidence."""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    pred_idx = int(torch.argmax(probs))
    pred_label = id2label[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label": pred_label,
        "confidence": confidence,
        "threshold_passed": confidence >= CONFIDENCE_THRESHOLD
    }