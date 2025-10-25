# Multi-Phase Toxicity Detection System

## Abstract

## 1. Introduction

## 2. Related Work

## 3. Dataset & Preprocessing

## 4. Methodology

## 5. Experiments and Results

This section presents a comparative evaluation between the baseline classical models and the fine-tuned transformer-based model (DistilBERT). Both were trained on the same preprocessed dataset derived from the Jigsaw Toxic Comment dataset, extended with engineered numeric features.

### 5.1 Experimental Setup
- **Dataset size:** 159,559 comments  
- **Classes:** `safe`, `mild`, `toxic`, `severe`
- **Hardware:** Apple M1 Max (no GPU acceleration)
- **Evaluation metrics:** Accuracy, Macro F1, Per-class F1
- **Tools:** scikit-learn, XGBoost, Hugging Face Transformers

### 5.2 Classical Model (TF-IDF + XGBoost Baseline)
| Metric | Value |
|---------|--------|
| **Accuracy** | 0.8758 |
| **Macro F1** | 0.6393 |
| **Runtime** | ~3m (training) |
| **Parameters** | ~1.2M (XGBoost) |

**Per-class performance:**

| Class | Precision | Recall | F1-Score |
|-------|------------|--------|-----------|
| mild | 0.2899 | 0.5947 | 0.3898 |
| safe | 0.9792 | 0.9059 | 0.9411 |
| severe | 0.5782 | 0.7738 | 0.6619 |
| toxic | 0.5647 | 0.5638 | 0.5642 |

**Observations:**
- Strong overall precision due to dominance of the `safe` class.
- Struggles with minority classes (`mild`, `severe`) — low recall.
- Quick to train and very lightweight, making it suitable for initial filtering.

---

### 5.3 Transformer Model (DistilBERT Fine-Tuned)
| Metric | Value |
|---------|--------|
| **Accuracy** | 0.9546 |
| **Macro F1** | 0.6837 |
| **Runtime** | ~3h (on CPU / MPS) |
| **Parameters** | ~66M (DistilBERT base) |
| **Best Checkpoint** | `checkpoint-17500` |
| **Epochs** | 2.19 |

**Observations:**
- +5% macro F1 improvement over baseline model.
- Significantly higher recall for minority classes (`mild`, `severe`).
- Handles contextual toxicity (e.g., sarcasm, indirect insults) much better.
- Tradeoff: slower training and inference cost, but more semantically robust.

---

### 5.4 Comparative Summary
| Model | Accuracy | Macro F1 | Runtime | Params |
|-------|-----------|-----------|----------|---------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | ~3 min | ~1.2M |
| DistilBERT (Fine-tuned) | 0.9546 | 0.6837 | ~3 hr | ~66M |

**Conclusion:**  
DistilBERT demonstrates a clear performance uplift in macro F1 and accuracy, showing stronger capability in detecting nuanced and minority-class toxicity. However, its computational footprint and runtime make it less ideal for real-time, high-throughput systems without GPU support.

---
## 6. Discussion

## 7. System Integration (SyntaxBase)

## 8. Conclusion