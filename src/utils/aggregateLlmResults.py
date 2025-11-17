import json
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_llm_batch_summaries(model_name: str):
    """
    Loads all LLM batch summary JSONs from:
        results/logs/llm/<model_name>/forum_test_dataset_summary_batch_*.json
    """
    base_dir = Path("results/logs/llm") / model_name
    files = sorted(
        base_dir.glob("forum_test_dataset_summary_batch_*.json"),
        key=lambda p: int(p.stem.split("_")[-1].replace("batch_", ""))
    )

    if not files:
        raise RuntimeError(f"No batch summaries found in {base_dir}")

    summaries = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    return summaries


# --------------------------
# MAIN AGGREGATION LOGIC
# --------------------------

# MODEL_NAME = "meta-llama-3.1-8b-instruct"
# MODEL_NAME = "qwen3-4b-thinking-2507"
MODEL_NAME = "phi-4-reasoning-plus"

batches = load_llm_batch_summaries(MODEL_NAME)

all_labels = set()
agg_confusion = defaultdict(lambda: defaultdict(int))

for batch in batches:
    cm = batch["confusion_matrix"]
    for true_label, preds in cm.items():
        all_labels.add(true_label)
        for pred_label, count in preds.items():
            all_labels.add(pred_label)
            agg_confusion[true_label][pred_label] += count

all_labels = sorted(all_labels)

y_true = []
y_pred = []
for true_label in all_labels:
    for pred_label in all_labels:
        count = agg_confusion[true_label].get(pred_label, 0)
        y_true.extend([true_label] * count)
        y_pred.extend([pred_label] * count)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=all_labels, zero_division=0
)
macro_f1 = f1.mean()
per_class_f1 = {label: f for label, f in zip(all_labels, f1)}

# Output directory
RESULT_DIR = Path(f"results/comparisons/llm/{MODEL_NAME}/")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Save summary JSON
summary = {
    "model": f"Local_LLM - {MODEL_NAME}",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "per_class_f1": per_class_f1,
    "confusion_matrix": agg_confusion
}

json_file = RESULT_DIR / "summary.json"
with json_file.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# Classification report
report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)
report_file = RESULT_DIR / "classification_report.txt"
with report_file.open("w", encoding="utf-8") as f:
    f.write(f"=== Local_LLM ({MODEL_NAME}) ===\n")
    f.write(report)

# Confusion matrix heatmap
cm_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_matrix,
    annot=True,
    fmt="d",
    xticklabels=all_labels,
    yticklabels=all_labels,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.tight_layout()
cm_file = RESULT_DIR / f"{MODEL_NAME}_confusion_matrix.png"
plt.savefig(cm_file)
plt.close()

print(f"JSON summary saved to: {json_file}")
print(f"Classification report saved to: {report_file}")
print(f"Confusion matrix PNG saved to: {cm_file}")