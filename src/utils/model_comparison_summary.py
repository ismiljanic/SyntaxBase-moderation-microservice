import pandas as pd

data = [
    ["Classical_XGBoost", "2025-10-10", "2025-10-10", 0.8758, 0.6393, 0.8938, "3m24s", "Mac M1 Max"],
    ["DistilBERT", "2025-10-21", "2025-10-23", 0.9546, 0.6837, None, "~3h", "Mac M1 Max (MPS)"],
    ["ToxicBERT", "2025-10-23", "2025-11-10", 0.9555, 0.7359, None, "~9.5h", "Mac M1 Max (MPS)"],
    ["meta-llama-3.1-8b-instruct", "2025-11-15", "2025-11-15", 0.7826, 0.8179, None, "~1min", "Mac M1 Max"],
    ["qwen3-4b-thinking-2507", "2025-11-17", "2025-11-17", 0.9565, 0.9512, None, "~1min", "Mac M1 Max"],
    ["phi-4-reasoning-plus", "2025-11-17", "2025-11-17", 0.9739, 0.9429, None, "~1min", "Mac M1 Max"]
]

df_summary = pd.DataFrame(data, columns=["model","date_start","date_end","accuracy","macro_f1","weighted_f1","runtime","hardware"])
df_summary.to_csv("results/comparisons/model_comparison_summary.csv", index=False)