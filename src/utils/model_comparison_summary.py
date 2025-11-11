import pandas as pd

data = [
    ["Classical_XGBoost", "2025-10-10", "2025-10-10", 0.8758, 0.6393, 0.8938, "3m24s", "Mac M1 Max"],
    ["DistilBERT", "2025-10-21", "2025-10-23", 0.9546, 0.6837, None, "~3h", "Mac M1 Max (MPS)"],
    ["ToxicBERT", "2025-10-23", "2025-11-10", 0.9555, 0.7359, None, "~9.5h", "Mac M1 Max (MPS)"]
]

df_summary = pd.DataFrame(data, columns=["model","date_start","date_end","accuracy","macro_f1","weighted_f1","runtime","hardware"])
df_summary.to_csv("results/comparisons/model_comparison_summary.csv", index=False)