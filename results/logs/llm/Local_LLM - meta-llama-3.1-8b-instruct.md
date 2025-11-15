## Phase 3 – LLM reasoning with meta-llama-3.1-8b-instruct  
**Date:** 2025-11-15  
**Hardware:** Mac M1 Max, 32GB RAM, CPU only  
**Dataset:** Combined forum comment batches (~115 comments)  
**Description:** Prompt-based classification with meta-llama-3.1-8b-instruct  
**Runtime:** ~1 minute  

**Results:**
- Accuracy: 0.78  
- Macro F1: 0.82  
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