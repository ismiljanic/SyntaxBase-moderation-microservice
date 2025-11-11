import pandas as pd

data = [
    ["Classical_XGBoost", "3m24s", 8, 1000, 0.1],
    ["DistilBERT", "~3h", 12, 200, 0.5],
    ["ToxicBERT", "~9.5h", 16, 180, 0.55]
]
df_tradeoff = pd.DataFrame(data, columns=["model","runtime","memory","throughput","latency"])
df_tradeoff.to_csv("results/comparisons/runtime_memory_tradeoff.csv", index=False)