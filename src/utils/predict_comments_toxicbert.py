import re
import torch
import torch.nn.functional as F
import joblib
from datetime import datetime
from transformers import BertTokenizerFast, BertForSequenceClassification
import csv
import os

# ==============================
# CONFIG
# ==============================
CHECKPOINT = os.path.abspath("models/saved/toxic_bert/checkpoint-48000")
LOW_CONF_FILE = "data/utils/test_low_confidence_toxicbert.csv"
CONFIDENCE_THRESHOLD = 0.8

# ==============================
# LOAD MODEL & TOKENIZER
# ==============================
tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)
model = BertForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()

# ==============================
# LOAD LABEL MAPS
# ==============================
label2id = joblib.load("models/saved/toxic_bert/label2idToxicBERT.pkl")
id2label = joblib.load("models/saved/toxic_bert/id2labelToxicBERT.pkl")

# ==============================
# TEXT CLEANING
# ==============================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# PREP CSV FILE (for low confidence)
# ==============================
existing_comments = set()
if os.path.exists(LOW_CONF_FILE):
    with open(LOW_CONF_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_comments.add(row["cleaned_comment"])

if not os.path.exists(LOW_CONF_FILE):
    os.makedirs(os.path.dirname(LOW_CONF_FILE), exist_ok=True)
    with open(LOW_CONF_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "original_comment", "cleaned_comment",
            "predicted_label", "confidence", "all_probabilities"
        ])

# ==============================
# INTERACTIVE LOOP
# ==============================
print("ToxicBERT Comment Classification (checkpoint-48000)")
print("Type your comment and press Enter. Type 'exit' to quit.\n")

while True:
    comment = input("Your comment: ")
    if comment.lower() == "exit":
        break

    comment_clean = clean_text(comment)

    if comment_clean in existing_comments:
        print("Comment already saved for review. Skipping.\n")
        continue

    # Tokenize
    inputs = tokenizer(
        comment_clean,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()

    pred_id = torch.argmax(probs).item()
    pred_label = id2label[pred_id]
    confidence = probs[pred_id].item()

    # Save low-confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        timestamp = datetime.now().isoformat()
        with open(LOW_CONF_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, comment, comment_clean,
                pred_label, f"{confidence:.3f}",
                {id2label[idx]: float(prob) for idx, prob in enumerate(probs)}
            ])
        existing_comments.add(comment_clean)
        print(f"Low confidence ({confidence:.3f}) — saved for later review.\n")
    else:
        print(f"Predicted label: {pred_label} (confidence: {confidence:.3f})")

    print("Class probabilities:")
    for idx, prob in enumerate(probs):
        label = id2label[idx]
        print(f"  {label}: {prob.item():.3f}")
    print()