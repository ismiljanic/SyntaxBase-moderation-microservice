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
**Runtime:** ~3h  
**Results:**
- Accuracy: 0.9546  
- Macro F1: 0.6837  

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

**Date:** 2025-10-23 → 2025-11-10  
**Hardware:** Mac M1 Max (MPS backend)  
**Training Script:** `src/transformer/train_toxicBert.py`  
**Checkpoint Directory:** `models/saved/toxic_bert/`  
**Runtime:** ~9.5h 
**Results:**
- Accuracy: 0.9555339684131361  
- Macro F1: 0.7359447534294354  

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


## Phase 3 – LLM reasoning
### Phase 3.1 - meta-llama-3.1-8b-instruct  
**Date:** 2025-11-15  
**Hardware:** Mac M1 Max, 32GB RAM, CPU only  
**Dataset:** `forum_test_dataset.csv` (~115 comments across 5 batches)  
**Description:** Prompt-based classification with meta-llama-3.1-8b-instruct  
**Runtime:** ~1 minute  

**Results:**
- Accuracy: 0.7826  
- Macro F1: 0.8179  
- Weighted F1: 0.77  

**Per-class performance:**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.67      | 1.00   | 0.80     | 49      |
| safe   | 1.00      | 0.47   | 0.63     | 43      |
| severe | 0.80      | 1.00   | 0.89     | 4       |
| toxic  | 1.00      | 0.89   | 0.94     | 19      |

**Notes:**
- Model occasionally hallucinates and repeats reasoning.  
- Context window ~36k tokens is fully utilized during batch processing.  
- Struggles with balanced recall for `safe` class due to LLM output variance.  
- Fast execution: <1 minute for 115 comments.  
- Hardware constraints: ~10GB RAM, CPU-only processing.  

---

## Phase 3.2 – qwen3-4b-thinking-2507

**Date:** 2025-11-17
**Hardware:** Mac M1 Max, 32GB RAM, CPU-only  
**Dataset:** `forum_test_dataset.csv` (~115 comments across 5 batches)  
**Description:** Prompt-based toxicity classification using qwen3-4b-thinking-2507  
**Runtime:** ~1 minute total  

### Results
- **Accuracy:** 0.9565  
- **Macro F1:** 0.9512  
- **Weighted F1:** 0.96  

### Per-class performance

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.91      | 0.98   | 0.94     | 42      |
| safe   | 0.98      | 0.98   | 0.98     | 56      |
| severe | 1.00      | 1.00   | 1.00     | 3       |
| toxic  | 1.00      | 0.79   | 0.88     | 14      |

### Observations
- Strong precision across all classes, with perfect scores for `severe` and `toxic`.  
- Recall for the `toxic` class dips (0.79), indicating mild under-detection of toxicity.  
- Excellent performance on the majority `safe` class with 0.98 precision/recall symmetry.  

### Notes
- Fully utilizes qwen’s expanded 36k-token context window during multi-batch runs.  
- CPU-only performance remains highly efficient: ~10–12 seconds per batch.  
- Memory footprint stable around ~6-8GB throughout execution.

## Phase 3.3 – phi-4-reasoning-plus

**Date:** 2025-11-17
**Hardware:** Mac M1 Max, 32GB RAM, CPU-only  
**Dataset:** `forum_test_dataset.csv` (~115 comments across 5 batches)  
**Description:** Prompt-based toxicity classification using phi-4-reasoning-plus  
**Runtime:** ~1 minute total  

### Results
- **Accuracy:** 0.9739  
- **Macro F1:** 0.9429  
- **Weighted F1:** 0.96  

### Per-class performance

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.97      | 0.97   | 0.97     | 32      |
| safe   | 0.98      | 1.00   | 0.99     | 57      |
| severe | 1.00      | 0.75   | 0.86     | 4       |
| toxic  | 0.95      | 0.95   | 0.94     | 22      |

### Observations
- Strong precision across all classes, with perfect scores for `severe` and `toxic`.  
- Recall for the `toxic` class dips (0.75), indicating mild under-detection of toxicity.  
- Excellent performance on the majority `safe` and `toxic` class with 0.98/1.00 and 0.95/0.95 precision/recall symmetry.  

### Notes
- Fully utilizes qwen’s expanded 36k-token context window during multi-batch runs.  
- CPU-only performance remains highly efficient: ~10–12 seconds per batch.  
- Memory footprint stable around ~8-10GB throughout execution.

---

### Experimental Setup
- **Prompt format:**  
  You are a Comment Moderation AI. Classify a single forum comment. Return **JSON only**, no extra text.

  Labels:  
  - safe: no offensive content  
  - mild: minor insults, slightly rude  
  - toxic: strong insults, harassment  
  - severe: extreme aggression or identity-based hate  

  Rules:  
  - One label per comment.  
  - Consider literal impact of sarcasm or humor.  
  - Profanity alone ≠ severe, use toxic.  

  Output format:  
  ```json
  { "label": "<safe|mild|toxic|severe>", "reasoning": "<short explanation>" }
  ```

---

### Run 1 – LLaMA 3.1 8B Instruct
| Metric | Value |
|---------|--------|
| Accuracy | 0.78 |
| Macro F1 | 0.82 |
| Runtime | ~1min |
| Avg Latency (s/comment) | ~0.5 |
| Context Length (tokens) | ~36k |
| Prompt Mode | single-prompt, zero-shot |

**Notes:**
- Handles contextual sarcasm fairly well; occasional hallucinations or repeated outputs.  
- Dataset: `forum_test_dataset.csv`, 115 comments split into 5 batches (~25 comments each).  
- CPU inference only; RAM ~10GB, context window fully utilized.  
- Full dataset processed in <1 minute.  

---

### Run 2 - qwen3-4b-thinking-2507
| Metric | Value |
|---------|--------|
| Accuracy | 0.9565 |
| Macro F1 | 0.9512 |
| Runtime | ~1min |
| Avg Latency (s/comment) | ~0.5 |
| Context Length (tokens) | ~36k |
| Prompt Mode | single-prompt, zero-shot |


---

### Run 3 – phi-4-reasoning-plus
| Metric | Value |
|---------|--------|
| Accuracy | 0.9739   |
| Macro F1 | 0.9429   |
| Runtime | ~1min |
| Avg Latency (s/comment) | ~0.5 |
| Context Length (tokens) | ~36k |
| Prompt Mode | single-prompt, zero-shot |

## Comparative Summaries

### Classical vs BERT (full dataset)
**Dataset:** Full training/test Jigsaw-toxic-comments dataset (~159k comments)  

| Model | Accuracy | Macro F1 | Avg Latency (s/comment) | Params |
|--------|-----------|-----------|--------------------------|---------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | 0.001 | ~1.2 M |
| DistilBERT (Fine-tuned) | 0.9546 | 0.6837 | 0.05 | ~66 M |
| ToxicBERT | 0.955533 | 0.735944 | 0.05 | ~110 M |

**Notes:**
- Classical model is fast but struggles on minority classes (`mild`/`toxic`).  
- BERT models improve contextual understanding and recall for minority classes.  
- Training a local LLM on the full 159k comment dataset is impractical and almost impossible on standard hardware due to memory and compute constraints so it is excluded from metric above.

---

### All Models Comparison (forum_test_dataset.csv, 115 comments)
**Dataset:** forum_test_dataset.csv, split into 5 batches (~25 comments each)  

| Model | Accuracy | Macro F1 | Avg Latency (s/comment) | Params |
|--------|-----------|-----------|--------------------------|---------|
| TF-IDF + XGBoost | 0.7304 | 0.7053 | 0.001 | ~1.2 M |
| DistilBERT (Fine-tuned) | 0.7100 | 0.6900 | 0.05 | ~66 M |
| ToxicBERT | 0.7300 | 0.7500 | 0.05 | ~110 M |
| LLaMA 3.1 8B Instruct | 0.78 | 0.82 | ~0.5 | 8B |
| qwen3-4b-thinking-2507 | 0.9565 | 0.9512 | ~0.5 | 4B |
| phi-4-reasoning-plus | 0.9739 | 0.9429 | ~0.5 | 8B |


**Per-class performance (TF-IDF + XGBoost):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| mild  | 0.6667    | 0.4444 | 0.5333   | 36      |
| safe  | 0.7308    | 0.9828 | 0.8382   | 58      |
| severe| 1.0000    | 0.7500 | 0.8571   | 4       |
| toxic | 0.8000    | 0.4706 | 0.5926   | 17      |

**Per-class performance (DistilBERT):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| mild  | 0.62      | 0.36   | 0.46     | 36      |
| safe  | 0.71      | 1.00   | 0.83     | 58      |
| severe| 1.00      | 0.75   | 0.86     | 4       |
| toxic | 0.89      | 0.47   | 0.62     | 17      |

**Per-class performance (ToxicBERT):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| mild  | 0.75      | 0.33   | 0.46     | 36      |
| safe  | 0.68      | 1.00   | 0.81     | 58      |
| severe| 1.00      | 1.00   | 1.00     | 4       |
| toxic | 1.00      | 0.59   | 0.74     | 17      |

**Per-class performance (LLaMA 3.1):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| mild  | 0.67      | 1.00   | 0.80     | 49      |
| safe  | 1.00      | 0.47   | 0.63     | 43      |
| severe| 0.80      | 1.00   | 0.89     | 4       |
| toxic | 1.00      | 0.89   | 0.94     | 19      |

**Per-class performance (qwen3):**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.91      | 0.98   | 0.94     | 42      |
| safe   | 0.98      | 0.98   | 0.98     | 56      |
| severe | 1.00      | 1.00   | 1.00     | 3       |
| toxic  | 1.00      | 0.79   | 0.88     | 14      |

**Per-class performance (phi4):**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.97      | 0.97   | 0.97     | 32      |
| safe   | 0.98      | 1.00   | 0.99     | 57      |
| severe | 1.00      | 0.75   | 0.86     | 4       |
| toxic  | 0.95      | 0.95   | 0.94     | 22      |


**Notes:**
- LLM models evaluated on a smaller testing dataset (forum_test_dataset.csv).  
- Handles minority classes (`severe`/`toxic`) better than BERT for this dataset.  
- CPU-only inference; ~6-10GB RAM, context window ~36k tokens.  
- Full evaluation (~115 comments) completed in <1 minute.  
- Occasional hallucination and repeated outputs; zero-shot single-prompt setup.  

---


### Observations
- LLMs exhibit better contextual judgment but less consistency across runs.  
- Few-shot prompting yields higher recall on subtle toxicity.  
- Computational cost per inference remains the bottleneck for deployment.  
- DistilBERT still offers best trade-off for real-time moderation.  

---