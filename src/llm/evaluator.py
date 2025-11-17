import csv
import json
from pathlib import Path
from collections import defaultdict
from llm_moderation_client import LLMModerationClient

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# INPUT = Path("data/test/llm/forum_test_dataset_for_llm_batch_1.csv")
# OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_results_batch_1.csv")
# SUMMARY_OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_summary_batch_1.json")

# INPUT = Path("data/test/llm/forum_test_dataset_for_llm_batch_2.csv")
# OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_results_batch_2.csv")
# SUMMARY_OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_summary_batch_2.json")

# INPUT = Path("data/test/llm/forum_test_dataset_for_llm_batch_3.csv")
# OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_results_batch_3.csv")
# SUMMARY_OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_summary_batch_3.json")

# INPUT = Path("data/test/llm/forum_test_dataset_for_llm_batch_4.csv")
# OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_results_batch_4.csv")
# SUMMARY_OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_summary_batch_4.json")

INPUT = Path("data/test/llm/forum_test_dataset_for_llm_batch_5.csv")
OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_results_batch_5.csv")
SUMMARY_OUTPUT = Path("results/logs/llm/qwen3-4b-thinking-2507/forum_test_dataset_summary_batch_5.json")

def run_evaluation():
    client = LLMModerationClient()

    rows_out = []
    y_true_exact = []      # exact expected label
    y_true_acceptable = [] # acceptable-aware label
    y_pred = []

    with INPUT.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            comment = row["cleaned_comment"]
            expected = row["expected_label"]
            acceptable = [x.strip() for x in row["acceptable_labels"].split(",") if x.strip()]

            result = client.classify_comment(comment)
            predicted = result["label"]

            # Exact vs Acceptable-aware y_true
            y_true_exact.append(expected)
            if predicted in acceptable:
                y_true_acceptable.append(predicted)
            else:
                y_true_acceptable.append(expected)

            y_pred.append(predicted)

            rows_out.append({
                "comment": comment,
                "expected": expected,
                "acceptable_labels": row["acceptable_labels"],
                "predicted": predicted,
                "is_correct": predicted == expected,
                "is_acceptable": predicted in acceptable,
                "reasoning": result["reasoning"],
            })

    # write per-comment CSV
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    # compute metrics using acceptable-aware y_true
    labels = sorted(list(set(y_true_exact + y_pred)))
    accuracy = accuracy_score(y_true_acceptable, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_acceptable, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1.mean()
    per_class_f1 = {label: f for label, f in zip(labels, f1)}

    # confusion matrix (acceptable-aware)
    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true_acceptable, y_pred):
        confusion[t][p] += 1

    summary = {
        "model": "Local_LLM - qwen3-4b-thinking-2507",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion
    }

    # write summary JSON
    with SUMMARY_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # print human-readable
    print(f"Strict accuracy: {sum(r['is_correct'] for r in rows_out)}/{len(rows_out)}")
    print(f"Acceptable accuracy: {sum(r['is_acceptable'] for r in rows_out)}/{len(rows_out)}")
    print("\nSummary JSON written to:", SUMMARY_OUTPUT)
    # print("\nConfusion Matrix:")
    # for true_label in labels:
    #     row = "\t".join(str(confusion[true_label].get(pred_label, 0)) for pred_label in labels)
    #     print(f"{true_label}:\t{row}")

if __name__ == "__main__":
    run_evaluation()