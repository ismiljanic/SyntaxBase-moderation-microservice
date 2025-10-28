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