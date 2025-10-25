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

## Next Phase (Planned)
**Model:** `unitary/toxic-bert`  
**Goal:** Measure domain-specific pretraining benefits over generic DistilBERT.  
**Expectation:** +4–10% macro F1 improvement on toxic/severe categories.

---

## Observations
- DistilBERT substantially reduces false negatives for “mild” toxicity.
- Batch training (resume from checkpoint) works reliably.
- MPS backend stable but ~5× slower than CUDA.

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