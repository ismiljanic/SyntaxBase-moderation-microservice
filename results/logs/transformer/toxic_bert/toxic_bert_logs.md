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

---

## Observations
- DistilBERT substantially reduces false negatives for “mild” toxicity.
- Batch training (resume from checkpoint) works reliably.
- MPS backend stable but ~5× slower than CUDA.