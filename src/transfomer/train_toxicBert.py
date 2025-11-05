import os
import pandas as pd
import numpy as np
import joblib
import torch
from datasets import Dataset
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import evaluate
from transformers import AutoConfig, AutoModelForSequenceClassification

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/jigsaw_multilevel_features.csv"
OUTPUT_DIR = "models/saved/toxic_bert"
LATEST_CHECKPOINT = "models/saved/toxic_bert/checkpoint-42000"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Device setup: MPS (Apple Metal) or CPU
# -----------------------------
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print("Using device:", device)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])

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
MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])

train_ds = train_ds.rename_column("label_id", "labels")
test_ds = test_ds.rename_column("label_id", "labels")

# -----------------------------
# Compute class weights
# -----------------------------
class_counts = df['label'].value_counts().sort_index().values
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = class_weights.to(device)

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
# Model: load ToxicBERT safely for 4 labels
# -----------------------------
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    problem_type="single_label_classification"
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True
)
model.to(device)

# -----------------------------
# Custom loss for class weights
# -----------------------------
def compute_weighted_loss(model, inputs):
    labels = inputs.get("labels")
    outputs = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device))
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(logits, labels.to(device))
    return loss

# -----------------------------
# Training arguments
# -----------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True,
    report_to=[],
    fp16=False,
    lr_scheduler_type="cosine"
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
    compute_metrics=compute_metrics,
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

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(label2id, os.path.join(OUTPUT_DIR, "label2idToxicBERT.pkl"))
joblib.dump(id2label, os.path.join(OUTPUT_DIR, "id2labelToxicBERT.pkl"))

print("Training complete. Model and label maps saved.")