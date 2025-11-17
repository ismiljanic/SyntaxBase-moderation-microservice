import pandas as pd

data = [
    ["Classical_XGBoost", "3m24s", 8, 1000, 0.1],
    ["DistilBERT", "~3h", 12, 200, 0.5],
    ["ToxicBERT", "~9.5h", 16, 180, 0.55],
    ["meta-llama-3.1-8b-instruct", "~1min", 8, None, 0.5],
    ["qwen3-4b-thinking-2507","~1min", 6, None, 0.5],
    ["phi-4-reasoning-plus", "~1min", 10, None, 0.5]
]
df_tradeoff = pd.DataFrame(data, columns=["model","runtime","memory in GB","throughput","latency in s"])
df_tradeoff.to_csv("results/comparisons/runtime_memory_tradeoff.csv", index=False)