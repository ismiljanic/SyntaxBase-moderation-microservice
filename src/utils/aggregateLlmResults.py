import json
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

batches = [
    batch_1 := {
        "confusion_matrix": {
            "safe": {"safe": 3, "mild": 4},
            "mild": {"mild": 12},
            "toxic": {"toxic": 1},
            "severe": {"severe": 1}
        }
    },
    batch_2 := {
        "confusion_matrix": {
            "safe": {"mild": 3, "safe": 5},
            "mild": {"mild": 10},
            "toxic": {"toxic": 1},
            "severe": {"severe": 1}
        }
    },
    batch_3 := {
        "confusion_matrix": {
            "mild": {"mild": 9},
            "safe": {"mild": 4, "safe": 1},
            "toxic": {"toxic": 6},
            "severe": {"severe": 1}
        }
    },
    batch_4 := {
        "confusion_matrix": {
            "toxic": {"toxic": 4, "severe": 1, "mild": 1},
            "safe": {"mild": 7, "safe": 6},
            "mild": {"mild": 10}
        }
    },
    batch_5 := {
        "confusion_matrix": {
            "mild": {"mild": 8},
            "safe": {"safe": 5, "mild": 5},
            "severe": {"severe": 1},
            "toxic": {"toxic": 5}
        }
    }
]

all_labels = set()
agg_confusion = defaultdict(lambda: defaultdict(int))
for batch in batches:
    cm = batch["confusion_matrix"]
    for true_label, pred_counts in cm.items():
        all_labels.add(true_label)
        for pred_label, count in pred_counts.items():
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

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=all_labels, zero_division=0
)
macro_f1 = f1.mean()
per_class_f1 = {label: f for label, f in zip(all_labels, f1)}

RESULT_DIR = Path("results/comparisons/llm/llama-3.1-8b-instruct/")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "model": "Local_LLM - meta-llama-3.1-8b-instruct",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "per_class_f1": per_class_f1,
    "confusion_matrix": agg_confusion
}
json_file = RESULT_DIR / "local_llm_summary.json"
with json_file.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)
report_file = RESULT_DIR / "local_llm_classification_report.txt"
with report_file.open("w", encoding="utf-8") as f:
    f.write("=== Local_LLM (meta-llama-3.1-8b-instruct) ===\n")
    f.write(report)

cm_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_matrix, annot=True, fmt="d", xticklabels=all_labels, yticklabels=all_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - meta-llama-3.1-8b-instruct")
plt.tight_layout()
cm_file = RESULT_DIR / "local_llm_confusion_matrix.png"
plt.savefig(cm_file)
plt.close()

print(f"JSON summary saved to: {json_file}")
print(f"Classification report saved to: {report_file}")
print(f"Confusion matrix PNG saved to: {cm_file}")