## Phase 3 – LLM reasoning with phi-4-reasoning-plus
**Date:** 2025-11-17
**Hardware:** Mac M1 Max, 32GB RAM, CPU only  
**Dataset:** Combined forum comment batches (~115 comments)  
**Description:** Prompt-based classification with phi-4-reasoning-plus
**Runtime:** ~1 minute  

**Results:**
- Accuracy: 0.9739  
- Macro F1: 0.9429  
- Weighted F1: 0.97  

**Per-class performance:**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.97      | 0.97   | 0.97     | 32      |
| safe   | 0.98      | 1.00   | 0.99     | 57      |
| severe | 1.00      | 0.75   | 0.86     | 4       |
| toxic  | 0.95      | 0.95   | 0.94     | 22      |

**Notes:**
- Context window ~36k tokens is fully utilized during batch processing.  
- Fast execution: <1 minute for 115 comments.  
- Hardware constraints: ~8-10GB RAM, CPU-only processing.  

---