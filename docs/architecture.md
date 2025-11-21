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

3. **LLM Layer (Phase 3)**
   - Zero-shot or few-shot classification
   - Context-aware moderation and edge cases
   - Acts as a verification layer for uncertain predictions

---

## 2. Data Flow
### 2.1 Overall Model Pipeline

```mermaid
flowchart TD
    %% User Input
    A[User Comment] --> B[Preprocessing Pipeline]
    B --> C[Classical ML Model TF-IDF + XGBoost]
    
    %% Classical Output
    C --> D[Primary Label & Confidence Score]
    
    %% Transformer Verification
    D --> E[Transformer Model Verification DistilBERT / ToxicBERT]
    E --> F{Transformer Confidence >= 0.8?}
    
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
### 2.2 Decision flow

```mermaid
graph LR
    A[Start: Comment Received] --> B[Classical ML Prediction]
    B --> C{BERT/ToxicBERT Confidence >0.8?}
    C -->|Higher| D[Use Transformer Label]
    C -->|Lower or Disagreement| E[LLM Verification]
    E --> F[Final Label]
```

>**Note**: A disagreement between the transformer models occurs when DistilBERT and ToxicBERT produce different predicted labels, regardless of confidence score.

## 3. Microservice Integration

The SyntaxBase Moderation Microservice is fully containerized and modular. Each model/service runs in its own Docker container, orchestrated via Docker Compose. The API exposes a single endpoint for comment moderation, implementing a **hybrid multi-phase pipeline**:

1. **Classical ML:** Fast, rule-based moderation using numeric and TF-IDF features.
2. **Transformer Models:** DistilBERT and ToxicBERT provide semantic analysis with confidence scores.
3. **LLM Verification:** If BERT/ToxicBERT disagree or are uncertain, a preloaded LLM evaluates the comment for final moderation.

**API Service:**
- **Framework:** FastAPI
- **Endpoint:** `/classify`
- **Input:** JSON comment, e.g. `{"text": "Your comment here"}`
- **Output:** JSON with:
  - `final_label` (safe, mild, toxic, severe)
  - Intermediate labels and confidence from classical, BERT, and ToxicBERT
  - Optional `llm_reasoning` when LLM is invoked

### 3.1 Sequence Diagram

```mermaid
sequenceDiagram
    participant Admin
    participant Forum
    participant API
    participant Classical
    participant BERT
    participant ToxicBERT
    participant LLM

    User->>Forum: Post comment
    Forum->>API: Send JSON comment
    API->>Classical: Phase 1 prediction
    Classical-->>API: Classical label
    API->>BERT: Phase 2 prediction
    BERT-->>API: Label + Confidence
    API->>ToxicBERT: Phase 2b prediction
    ToxicBERT-->>API: Label + Confidence
    alt BERT & ToxicBERT disagree or uncertain
        API->>LLM: Phase 3 LLM verification
        LLM-->>API: Verified label + reasoning
    end
    API-->>Forum: Return final label with pipeline details
    Forum-->>Admin: Display moderation result
```

---

### 3.2 Deployment diagram

All services are Dockerized and networked together via Docker Compose.

Ports:
- `classical`: 7001
- `bert`: 7002
- `toxicbert`: 7003
- `llm`: 7004
- `api`: 8000

```mermaid 
graph LR

    subgraph UserFacing[Client + Forum]
        FORUM[SyntaxBase Forum Frontend + Backend]
        FORUM -->|HTTP POST /classify| API
    end

    subgraph Network[Docker Network]
        API[API Gateway FastAPI Port 8000]
        CLASSICAL[Classical Model Port 7001]
        BERT[DistilBERT Model Port 7002]
        TOXIC[ToxicBERT Model Port 7003]
        LLM[LLM Reasoning Port 7004]
        MON[Monitoring Future: MLflow/W&B]
    end

    subgraph Volumes[Docker Volumes]
        V1[(models:/models)]
        V2[(logs:/logs)]
    end

    API --> CLASSICAL
    API --> BERT
    API --> TOXIC
    API --> LLM

    CLASSICAL --- V1
    BERT --- V1
    TOXIC --- V1
    LLM --- V1
    API --> V2
```

---

### 3.3 Metrics and artifact flow

```mermaid 
flowchart LR
    Raw[Raw Comments] --> Preprocessing[Preprocessing Pipeline]
    Preprocessing --> Classical
    Classical --> Transformers
    Transformers --> LLM
    LLM --> API
    API --> Metrics[Metrics Storage: results/metrics, results/comparisons]
    API --> Models[Models Storage: models/saved]
```
>**Note:** This flowchart reflects the consolidated results, metrics, and comparative analyses obtained from the final evaluation phase.

---

## 4. Artifact Storage
- **Models:** `models/saved/`
- **Metrics & Results:** `results/metrics`, `results/comparisons`
- **Training Logs:** `docs/training_logs.md`
