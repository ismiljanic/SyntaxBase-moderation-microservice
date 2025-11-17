import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_batches_for_model(model_name: str):
    """Load all batch summaries for a given LLM model."""
    base_dir = Path("results/logs/llm") / model_name
    files = sorted(base_dir.glob("forum_test_dataset_summary_batch_*.json"))

    if not files:
        print(f"[WARN] No batch files found for model: {model_name}")
        return None

    summaries = []
    for f in files:
        with f.open("r", encoding="utf-8") as fp:
            summaries.append(json.load(fp))
    return summaries


def aggregate_confusion(summaries):
    """Aggregate confusion matrices across batches."""
    agg = defaultdict(lambda: defaultdict(int))
    all_labels = set()

    for batch in summaries:
        cm = batch["confusion_matrix"]
        for true_label, preds in cm.items():
            all_labels.add(true_label)
            for pred_label, count in preds.items():
                all_labels.add(pred_label)
                agg[true_label][pred_label] += count

    return agg, sorted(all_labels)


def flatten_confusion(agg_confusion, all_labels):
    """Convert aggregated confusion structure into y_true / y_pred."""
    y_true, y_pred = [], []
    for t in all_labels:
        for p in all_labels:
            count = agg_confusion[t].get(p, 0)
            y_true.extend([t] * count)
            y_pred.extend([p] * count)
    return y_true, y_pred


def evaluate_model(model_name, summaries):
    """Produce all metrics + plots for one LLM model."""
    agg_confusion, labels = aggregate_confusion(summaries)
    y_true, y_pred = flatten_confusion(agg_confusion, labels)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1.mean()
    per_class_f1 = {label: f for label, f in zip(labels, f1)}

    # Save results
    out_dir = Path(f"results/comparisons/llm/{model_name}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": agg_confusion
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # classification_report
    clf_rep = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"=== {model_name} ===\n\n")
        f.write(clf_rep)

    # confusion matrix heatmap
    cm_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    # plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    return summary


def generate_leaderboard(all_summaries):
    """Generate aggregated leaderboard across all LLMs."""
    leaderboard = sorted(
        all_summaries,
        key=lambda x: x["macro_f1"],
        reverse=True
    )

    # JSON output
    lb_json = Path("results/comparisons/llm/leaderboard.json")
    lb_json.parent.mkdir(parents=True, exist_ok=True)
    with lb_json.open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=4)

    # Markdown table
    lb_md = Path("results/comparisons/llm/leaderboard.md")
    with lb_md.open("w", encoding="utf-8") as f:
        f.write("# LLM Toxicity Classification Leaderboard\n\n")
        f.write("| Rank | Model | Accuracy | Macro F1 |\n")
        f.write("|------|--------|-----------|-----------|\n")
        for i, row in enumerate(leaderboard, 1):
            f.write(f"| {i} | {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |\n")

    # Macro F1 bar plot
    plt.figure(figsize=(10, 5))
    models = [m["model"] for m in leaderboard]
    f1_scores = [m["macro_f1"] for m in leaderboard]
    plt.bar(models, f1_scores)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1")
    plt.title("LLM Comparison – Macro F1 Scores")
    plt.tight_layout()
    plt.savefig("results/comparisons/llm/leaderboard_macro_f1.png")
    plt.close()

    print("Leaderboard generated.")
    return leaderboard


if __name__ == "__main__":
    base = Path("results/logs/llm/")
    model_names = [d.name for d in base.iterdir() if d.is_dir()]

    print(f"Detected LLM models: {model_names}")

    all_results = []
    for model in model_names:
        print(f"\nProcessing {model} ...")
        summaries = load_batches_for_model(model)
        if summaries:
            result = evaluate_model(model, summaries)
            all_results.append(result)

    print("\nGenerating leaderboard...")
    generate_leaderboard(all_results)

    print("\nDone.")