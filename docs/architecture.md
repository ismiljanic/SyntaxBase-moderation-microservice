# System Architecture – Multi-Phase Toxicity Detection System

This document describes the architecture of the Multi-Phase Toxicity Detection System, detailing the model pipeline, integration with the SyntaxBase forum, and the hybrid moderation setup.

---

## 1. High-Level Overview

The system consists of three main layers:

1. **Classical ML Layer**
   - TF-IDF + numeric features
   - XGBoost / Logistic Regression
   - Fast and lightweight; serves as first-pass filter

2. **Transformer Layer**
   - Fine-tuned DistilBERT / ToxicBERT
   - Semantic and contextual understanding
   - Handles nuanced toxicity missed by classical models

3. **LLM Layer (Optional / Phase 3)**
   - Zero-shot or few-shot classification
   - Context-aware moderation and edge cases
   - Acts as a verification layer for uncertain predictions

---

## 2. Data Flow
### Overall Model Pipeline

```mermaid
flowchart TD
    %% User Input
    A[User Comment] --> B[Preprocessing Pipeline]
    B --> C[Classical ML Model TF-IDF + XGBoost]
    
    %% Classical Output
    C --> D[Primary Label & Confidence Score]
    
    %% Transformer Verification
    D --> E[Transformer Model Verification DistilBERT / ToxicBERT]
    E --> F{Transformer Confidence >= 0.9?}
    
    %% Decision Path
    F -->|Yes| G[Pass Label]
    F -->|No| H[LLM Verification]
    
    %% Final Output
    H --> G
    G --> I[Final Label]
    I --> J[SyntaxBase Forum / API]

    %% Artifacts & Metrics
    class B,C,E,H artifacts;

    B:::artifacts --> K[Data: tokenized text, numeric features]
    C:::artifacts --> L[Saved model: xgboost.pkl Vectorizer: vectorizer.pkl]
    E:::artifacts --> M[Saved checkpoint: Best checkpoint, Tokenizer + Label mapping]
    H:::artifacts --> N[LLM Prompt Templates + Optional LoRA weights]

    %% Metrics
    D --> O[Metrics: Accuracy, Macro F1, Per-class F1]
    E --> P[Metrics: Accuracy, Macro F1, Runtime, Params]
    H --> Q[Metrics: Accuracy, Macro F1, Inference Latency]
```

---

## 3. Microservice Integration

- **API:** FastAPI backend wrapping DistilBERT/ToxicBERT
- **Endpoint:** `/classify`
- **Input:** JSON comment
- **Output:** JSON label (`safe`, `mild`, `toxic`, `severe`)
- **Hybrid Mode:** Optional LLM verification if transformer confidence < threshold
- **Deployment:** Docker or MCP service (planned for production-scale testing)

```mermaid
sequenceDiagram
    participant User(admin)
    participant Forum
    participant Microservice
    participant Model
    participant LLM

    User->>Forum: Post Comment
    Forum->>Microservice: Send JSON comment
    Microservice->>Model: Tokenize & Predict
    Model-->>Microservice: Label + Confidence
    alt Confidence < Threshold
        Microservice->>LLM: Verify Comment
        LLM-->>Microservice: Verified Label
    end
    Microservice-->>Forum: Return Final Label
    Forum-->>User: Display Moderation Result

```
---

## 4. Artifact Storage

- **Models:** `models/saved/`
- **Metrics & Results:** `results/metrics`, `results/comparisons`
- **Training Logs:** `docs/training_logs.md`

---

## 5. Notes & Future Improvements

- Add real-time LLM fallback for edge cases
- Experiment with caching or batched inference for throughput
- Multi-lingual toxicity detection in future iterations