#!/usr/bin/env python3
"""
evaluate_both_models.py

Usage:
    python src/utils/evaluate_both_models.py --csv data/test/forum_test_dataset.csv

This version considers `acceptable_labels` for flexible evaluation. A prediction is
correct if it matches any of the acceptable labels for that comment.
"""
import argparse
import os
import joblib
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import BertTokenizerFast, BertForSequenceClassification, DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TOXICBERT_CHECKPOINT = os.path.abspath("models/saved/toxic_bert/checkpoint-48000")
TOXICBERT_ID2LABEL = "models/saved/toxic_bert/id2labelToxicBERT.pkl"

DISTILBERT_CHECKPOINT = os.path.abspath("models/saved/bert/checkpoint-17500")
DISTILBERT_ID2LABEL = "models/saved/bert/id2label.pkl"

BATCH_SIZE = 8
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def clean_text(text):
    if not text:
        return ""
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_toxicbert():
    tokenizer = BertTokenizerFast.from_pretrained(TOXICBERT_CHECKPOINT)
    model = BertForSequenceClassification.from_pretrained(TOXICBERT_CHECKPOINT)
    model.to(DEVICE)
    model.eval()
    id2label = joblib.load(TOXICBERT_ID2LABEL)
    return tokenizer, model, id2label

def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_CHECKPOINT)
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_CHECKPOINT)
    model.to(DEVICE)
    model.eval()
    id2label = joblib.load(DISTILBERT_ID2LABEL)
    return tokenizer, model, id2label

def batch_predict(tokenizer, model, texts, id2label):
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()
            probs = F.softmax(logits, dim=1).numpy()
        for row in probs:
            pred_id = int(np.argmax(row).item())
            pred_label = id2label[pred_id]
            pred_prob = float(row[pred_id])
            probs_dict = {id2label[idx]: float(p) for idx, p in enumerate(row)}
            results.append((pred_label, pred_prob, probs_dict))
    return results

def unify_labelset(*id2label_dicts):
    union = []
    for d in id2label_dicts:
        for idx, lab in d.items():
            if lab not in union:
                union.append(lab)
    mapping = []
    for d in id2label_dicts:
        map_i = {int(k): union.index(v) for k, v in d.items()}
        mapping.append(map_i)
    return union, mapping

def plot_confusion(y_true, y_pred, labels, outpath, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

def main(args):
    df = pd.read_csv(args.csv)
    if "cleaned_comment" not in df.columns:
        df["cleaned_comment"] = df["original_comment"].astype(str).apply(clean_text)

    # Get acceptable labels list
    df["acceptable_labels_list"] = df["acceptable_labels"].astype(str).apply(lambda x: [label.strip() for label in x.split(",")])

    texts = df["cleaned_comment"].tolist()
    expected = df["expected_label"].tolist()
    acceptable_labels_list = df["acceptable_labels_list"].tolist()

    print("Loading ToxicBERT...")
    t_tokenizer, t_model, t_id2label = load_toxicbert()
    print("Loading DistilBERT...")
    d_tokenizer, d_model, d_id2label = load_distilbert()

    print("Predicting with ToxicBERT...")
    t_results = batch_predict(t_tokenizer, t_model, texts, t_id2label)
    print("Predicting with DistilBERT...")
    d_results = batch_predict(d_tokenizer, d_model, texts, d_id2label)

    rows = []
    correct_t, correct_d = 0, 0
    for i, text in enumerate(texts):
        t_label, t_conf, t_probs = t_results[i]
        d_label, d_conf, d_probs = d_results[i]
        acceptable = acceptable_labels_list[i]

        if t_label in acceptable:
            correct_t += 1
        if d_label in acceptable:
            correct_d += 1

        rows.append({
            "original_comment": df.loc[i,"original_comment"],
            "cleaned_comment": text,
            "expected_label": expected[i],
            "acceptable_labels": ",".join(acceptable),
            "toxicbert_pred": t_label,
            "toxicbert_conf": t_conf,
            "toxicbert_probs": t_probs,
            "distilbert_pred": d_label,
            "distilbert_conf": d_conf,
            "distilbert_probs": d_probs
        })

    out_df = pd.DataFrame(rows)

    # ------------------------
    # Output directory per CSV
    # ------------------------
    csv_base = os.path.basename(args.csv)
    csv_name, _ = os.path.splitext(csv_base)  
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../results/comparisons/bert/{csv_name}"))
    os.makedirs(output_dir, exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(output_dir, f"results_toxic_eval_{timestamp}.csv")
    out_df.to_csv(results_csv, index=False)
    print("Saved predictions ->", results_csv)

    # ------------------------
    # Confusion matrices
    # ------------------------
    union_labels, mapping = unify_labelset(t_id2label, d_id2label)
    y_true = [expected[i] for i in range(len(expected))]
    y_pred_t = out_df["toxicbert_pred"].tolist()
    y_pred_d = out_df["distilbert_pred"].tolist()

    tox_cm_path = plot_confusion(
    y_true, y_pred_t, union_labels,
    os.path.join(output_dir, f"toxicbert_confusion_{csv_name}.png"),
    "ToxicBERT Confusion Matrix"
    )
    dist_cm_path = plot_confusion(
        y_true, y_pred_d, union_labels,
        os.path.join(output_dir, f"distilbert_confusion_{csv_name}.png"),
        "DistilBERT Confusion Matrix"
    )

    # ------------------------
    # Classification reports
    # ------------------------

    report_filename = f"classification_report_{csv_name}.txt"
    report_path = os.path.join(output_dir, report_filename)

    report_t = classification_report(y_true, y_pred_t, labels=union_labels, zero_division=0)
    report_d = classification_report(y_true, y_pred_d, labels=union_labels, zero_division=0)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== ToxicBERT ===\n")
        f.write(report_t + "\n\n")
        f.write("=== DistilBERT ===\n")
        f.write(report_d + "\n")

    print(f"Saved classification reports in {report_path}")
    # ------------------------
    # Accuracy based on acceptable_labels
    # ------------------------
    acc_t = correct_t / len(texts)
    acc_d = correct_d / len(texts)
    print(f"ToxicBERT accuracy (acceptable labels): {acc_t:.2%}")
    print(f"DistilBERT accuracy (acceptable labels): {acc_d:.2%}")

    print("Saved confusion matrices and reports in", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to test csv")
    args = parser.parse_args()
    main(args)