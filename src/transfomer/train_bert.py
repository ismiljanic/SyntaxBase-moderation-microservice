import os
import pandas as pd
import numpy as np
import joblib
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# -----------------------------
# Paths & checkpoint
# -----------------------------
DATA_PATH = "data/processed/jigsaw_multilevel_features.csv"
OUTPUT_DIR = "models/saved/bert"
LATEST_CHECKPOINT = "models/saved/bert/checkpoint-19000"

# -----------------------------
# Load preprocessed dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])

# Encode labels
labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

train_ds = train_ds.rename_column("label_id", "labels")
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
# Model: resume if checkpoint exists
# -----------------------------
if LATEST_CHECKPOINT and os.path.exists(LATEST_CHECKPOINT):
    print(f"Resuming training from checkpoint: {LATEST_CHECKPOINT}")
    model = DistilBertForSequenceClassification.from_pretrained(
        LATEST_CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
else:
    print("Starting training from scratch...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

# -----------------------------
# Training Arguments
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    report_to=[]
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -----------------------------
# Train (resume if checkpoint)
# -----------------------------
if LATEST_CHECKPOINT and os.path.exists(LATEST_CHECKPOINT):
    print(f"Resuming full training state from checkpoint: {LATEST_CHECKPOINT}")
    trainer.train(resume_from_checkpoint=LATEST_CHECKPOINT)
else:
    trainer.train()

# -----------------------------
# Evaluate & Save
# -----------------------------
results = trainer.evaluate(test_ds)
print("Evaluation results:", results)

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

joblib.dump(label2id, os.path.join(OUTPUT_DIR, "label2id.pkl"))
joblib.dump(id2label, os.path.join(OUTPUT_DIR, "id2label.pkl"))

print("Training complete. Model and label maps saved.")