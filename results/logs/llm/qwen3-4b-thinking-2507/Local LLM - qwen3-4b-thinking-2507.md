## Phase 3 – LLM reasoning with qwen3-4b-thinking-2507
**Date:** 2025-11-17  
**Hardware:** Mac M1 Max, 32GB RAM, CPU only  
**Dataset:** Combined forum comment batches (~115 comments)  
**Description:** Prompt-based classification with qwen3-4b-thinking-2507  
**Runtime:** ~1 minute  

**Results:**
- Accuracy: 0.9565  
- Macro F1: 0.9512  
- Weighted F1: 0.96  

**Per-class performance:**

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.91      | 0.98   | 0.94     | 42      |
| safe   | 0.98      | 0.98   | 0.98     | 56      |
| severe | 1.00      | 1.00   | 1.00     | 3       |
| toxic  | 1.00      | 0.79   | 0.88     | 14      |

**Notes:**
- Context window ~36k tokens is fully utilized during batch processing.  
- Fast execution: <1 minute for 115 comments.  
- Hardware constraints: ~8–10GB RAM, CPU-only processing.  
