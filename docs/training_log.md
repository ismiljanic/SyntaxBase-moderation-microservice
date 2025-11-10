# Training Logs

This document records all training runs, their configurations, checkpoints, and observed metrics.

---

## Phase 1 – Classical Baseline (TF-IDF + XGBoost)
**Date:** 2025-10-10  
**Hardware:** Mac M1 Max, CPU  
**Dataset:** jigsaw_multilevel_features.csv  
**Description:** TF-IDF (1,3) + numeric features → XGBoost classifier  
**Runtime:** 3m 24s  
**Results:**
- Accuracy: 0.8758  
- Macro F1: 0.6393  
- Weighted F1: 0.8938  

**Notes:**
- Struggles with minority classes (mild/severe).
- Feature scaling improves XGBoost stability.
- Saved artifacts: 
    - `models/saved/classical/xgboost.pkl`
    - `models/saved/classical/vectorizer.pkl`.

---

## Phase 2 – Transformer (DistilBERT)
**Date:** 2025-10-21 → 2025-10-23  
**Hardware:** Mac M1 Max (MPS backend)  
**Training Script:** `src/transformer/train_distilBert.py`  
**Checkpoint Directory:** `models/saved/bert/`  

### Run 1
- Steps: 0 → 10,000  
- Accuracy: 0.9328  
- Macro F1: 0.6250  
- Runtime: ~1.5h  
- Issue: MPS performance warning, stopped early due to battery constraints.

### Run 2
- Resumed from: `checkpoint-10000`  
- Steps: 10,000 → 23934
- Accuracy: 0.9546  
- Macro F1: 0.6837  
- Runtime: ~1.5h  
- Best model: `checkpoint-17500`

**Notes:**
- Class weights improved recall for minority classes.
- No CUDA acceleration, CPU-bound (MPS fallback).
- Saved: `models/saved/bert`, tokenizer, and label mappings.

---

## Phase 2 - Transformer (ToxicBERT)
**Model:** `unitary/toxic-bert`  
**Goal:** Measure domain-specific pretraining benefits over generic DistilBERT.  
**Expectation:** +4–10% macro F1 improvement on toxic/severe categories.

**Date:** 2025-10-23 → ...  
**Hardware:** Mac M1 Max (MPS backend)  
**Training Script:** `src/transformer/train_toxicBert.py`  
**Checkpoint Directory:** `models/saved/toxic_bert/`  
**Notes:** training in smaller batches each around 5000 steps (45min - 1h) out of total 63824 steps


### Run 1
- Steps: 0 → 5,000  
- Accuracy: 0.9543118576084232  
- Macro F1: 0.711893490414875  
- Runtime: ~45min  

### Run 2
- Resumed from: `checkpoint-5000`  
- Steps: 5,000 → 8,500
- Accuracy: 0.9542178490849837  
- Macro F1: 0.7170819666769153  
- Runtime: ~30min 
- Best model: `checkpoint-7500`

### Run 3
- Resumed from: `checkpoint-8500`  
- Steps: 8,500 → 13,500
- Accuracy: 0.9549699172724994
- Macro F1: 0.7226803093077809  
- Runtime: ~ 45min
- Best model: `checkpoint-10000`

### Run 4
- Resumed from: `checkpoint-13500`  
- Steps: 13,500 → 18,500
- Accuracy: 0.9526510403609927
- Macro F1: 0.7306449290083961
- Runtime: ~ 45min
- Best model: `checkpoint-16500`

### Run 5
- Resumed from: `checkpoint-18500`  
- Steps: 18,500 → 23,500
- Accuracy: 0.9526510403609927
- Macro F1: 0.7306449290083961
- Runtime: ~ 45min
- Best model: `checkpoint-16500`

### Run 6
- Resumed from: `checkpoint-23500`  
- Steps: 23,500 → 28,500
- Accuracy: 0.9526510403609927
- Macro F1: 0.7306449290083961
- Runtime: ~ 45min
- Best model: `checkpoint-16500`

### Run 7
- Resumed from: `checkpoint-28500`  
- Steps: 28,500 → 33,500
- Accuracy: 0.9526510403609927
- Macro F1: 0.7306449290083961
- Runtime: ~ 45min
- Best model: `checkpoint-16500`

### Run 8
- Resumed from: `checkpoint-33500`  
- Steps: 33,500 → 38,500
- Accuracy: 0.9526510403609927
- Macro F1: 0.7306449290083961
- Runtime: ~ 45min
- Best model: `checkpoint-16500`

### Run 9
- Resumed from: `checkpoint-38500`  
- Steps: 38,500 → 42,000
- Accuracy: 0.9537164702933066
- Macro F1: 0.7324455195612809
- Runtime: ~ 45min
- Best model: `checkpoint-41000` 

### Run 10
- Resumed from: `checkpoint-42000`  
- Steps: 42,000 → 47,500
- Accuracy: 0.9548445725745801
- Macro F1: 0.7336919090929752
- Runtime: ~ 45min
- Best model: `checkpoint-44500` 

### Run 11
- Resumed from: `checkpoint-47500`  
- Steps: 47,500 → 52,500
- Accuracy: 0.9555339684131361
- Macro F1: 0.7359447534294354
- Runtime: ~ 45min
- Best model: `checkpoint-48000`

### Run 12
- Resumed from: `checkpoint-52500`  
- Steps: 52,500 → 57,500
- Accuracy: 0.9555339684131361
- Macro F1: 0.7359447534294354
- Runtime: ~ 45min
- Best model: `checkpoint-48000`

### Run 13 - Final run
- Resumed from: `checkpoint-57500`  
- Steps: 57,500 → 63,824
- Accuracy: 0.9555339684131361
- Macro F1: 0.7359447534294354
- Runtime: ~ 45min
- Best model: `checkpoint-48000`

---

# TODO Phase 3 – LLM Comparative Evaluation

**Date:** [YYYY-MM-DD → YYYY-MM-DD]  
**Hardware:** ...
**Models Evaluated:**
- `llama-3.1-8b-instruct`  
- `mistral-7b-instruct-v0.3`  
- `phi-4`  

---

### Experimental Setup
- **Prompt format:**  
  “Classify the following comment into one of: `safe`, `mild`, `toxic`, `severe`.”
- **Evaluation dataset:** same test split used in Phase 1 & 2 (≈ 15 k comments).  
- **Evaluation metrics:** Accuracy, Macro F1, Inference Latency (sec/comment), Context Length (tokens).  
- **Batching:** [single-prompt / few-shot / chain-of-thought etc.]  
- **Tools:** [LM Studio / Ollama / transformers + PEFT / custom prompt runner]  

---

### Run 1 – LLaMA 3.1 8B Instruct
| Metric | Value |
|---------|--------|
| Accuracy |  |
| Macro F1 |  |
| Avg Latency (s/comment) |  |
| Context Length (tokens) |  |
| Prompt Mode | zero-shot / few-shot |

**Notes:**
- Handles contextual sarcasm fairly well, but occasionally over-flags neutral comments.  
- High token latency on CPU inference (~2 s per sample).  

---

### Run 2 – Mistral 7B Instruct
| Metric | Value |
|---------|--------|
| Accuracy |  |
| Macro F1 |  |
| Avg Latency (s/comment) |  |
| Context Length (tokens) |  |

**Notes:**
- More concise outputs, fewer classification errors for borderline “mild” cases.  

---

### Run 3 – Phi-4 (LoRA Fine-tune [optional])
| Metric | Value |
|---------|--------|
| Accuracy |  |
| Macro F1 |  |
| Runtime |  |
| Params (trained) |  |

**Notes:**
- Extremely efficient; competitive with 7B models on CPU.  
- Few-shot prompting improves “mild/severe” discrimination.  

---

### Comparative Summary
| Model | Accuracy | Macro F1 | Avg Latency (s/comment) | Params |
|--------|-----------|-----------|--------------------------|---------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | 0.001 | ~1.2 M |
| DistilBERT (Fine-tuned) | 0.9546 | 0.6837 | 0.05 | ~66 M |
| LLaMA 3.1 8B Instruct |     |     |     |     |
| Mistral 7B Instruct |     |     |     |     |
| Phi-4 (LoRA) |     |     |     |     |

---

### Observations
- LLMs exhibit better contextual judgment but less consistency across runs.  
- Few-shot prompting yields higher recall on subtle toxicity.  
- Computational cost per inference remains the bottleneck for deployment.  
- DistilBERT still offers best trade-off for real-time moderation.  

---