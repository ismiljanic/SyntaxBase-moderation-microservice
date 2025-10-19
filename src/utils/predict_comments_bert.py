import re
import torch
import joblib
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datetime import datetime
import csv
import os

CONFIDENCE_THRESHOLD = 0.8
CHECKPOINT = "models/saved/bert/checkpoint-17500"
LOW_CONF_FILE = "data/utils/test_low_confidence_comments.csv"

# -----------------------------
# Load tokenizer & model
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(CHECKPOINT)
model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT)
model.eval()

# -----------------------------
# Load label maps
# -----------------------------
label2id = joblib.load(f"models/saved/bert/label2id.pkl")
id2label = joblib.load(f"models/saved/bert/id2label.pkl")

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Track existing comments to avoid duplicates
# -----------------------------
existing_comments = set()
if os.path.exists(LOW_CONF_FILE):
    with open(LOW_CONF_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_comments.add(row["cleaned_comment"])

if not os.path.exists(LOW_CONF_FILE):
    with open(LOW_CONF_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "original_comment", "cleaned_comment",
            "predicted_label", "confidence", "all_probabilities"
        ])

# -----------------------------
# Interactive loop
# -----------------------------
print("BERT Comment Moderation Tool (checkpoint-17500)")
print("Type your comment and press Enter. Type 'exit' to quit.\n")

while True:
    comment = input("Your comment: ")
    if comment.lower() == "exit":
        break

    comment_clean = clean_text(comment)

    if comment_clean in existing_comments:
        print("Comment already saved for review. Skipping.\n")
        continue

    inputs = tokenizer(
        comment_clean,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()
    
    pred_id = torch.argmax(probs).item()
    pred_label = id2label[pred_id]
    confidence = probs[pred_id].item()

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
        print(f"Low confidence comment saved for LLM review. (confidence: {confidence:.3f})\n")
    else:
        print(f"Predicted label: {pred_label} (confidence: {confidence:.3f})")
        
    print("All class probabilities:")
    for idx, prob in enumerate(probs):
        label = id2label[idx]
        print(f"  {label}: {prob.item():.3f}")
    print()