import os
import pandas as pd
import numpy as np
import joblib
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
import evaluate

# -----------------------------
# Paths
# -----------------------------
CHECKPOINT_DIR = "models/saved/bert"
DATA_PATH = "data/processed/jigsaw_multilevel_features.csv"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])

labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

test_df = df.sample(frac=0.2, random_state=42)
test_ds = Dataset.from_pandas(test_df)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

test_ds = test_ds.map(tokenize, batched=True)
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
test_ds = test_ds.rename_column("label_id", "labels")

# -----------------------------
# Metrics
# -----------------------------
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

# -----------------------------
# Find all checkpoints
# -----------------------------
checkpoints = [os.path.join(CHECKPOINT_DIR, d) 
               for d in os.listdir(CHECKPOINT_DIR) 
               if d.startswith("checkpoint")]

checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

# -----------------------------
# Evaluate each checkpoint
# -----------------------------
results_list = []

for ckpt in checkpoints:
    print(f"\nEvaluating {ckpt} ...")
    model = DistilBertForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    trainer = Trainer(
        model=model,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    results = trainer.evaluate()
    results_list.append({"checkpoint": ckpt, **results})
    print(f"Results: {results}")

# -----------------------------
# Save results to CSV
# -----------------------------
results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.join(CHECKPOINT_DIR, "checkpoints_eval_results.csv"), index=False)
print("\nAll checkpoint evaluations saved to checkpoints_eval_results.csv")