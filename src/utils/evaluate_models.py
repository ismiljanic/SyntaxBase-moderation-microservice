from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import json

def evaluate_model(model_name, y_true, y_pred, save_path="results/metrics.json"):
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    result = {
        "model": model_name,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm
    }
    with open(save_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    return result