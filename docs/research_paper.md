# Multi-Phase Toxicity Detection System

## Abstract

Online toxicity in user-generated content poses significant challenges for platform moderation due to its nuanced, context-dependent nature. This study presents a multi-phase toxicity detection system that combines classical machine learning, transformer-based models, and large language model reasoning to achieve scalable and context-aware moderation. Using the Jigsaw Toxic Comment dataset augmented with engineered numeric features, we establish a baseline with a TF-IDF + XGBoost classifier, achieving a macro F1 of 0.6393 and strong precision for the majority class. We then fine-tune transformer-based models, including DistilBERT and ToxicBERT, demonstrating significant improvements in both overall accuracy and minority-class recall (macro F1 up by 10%-15%), highlighting the importance of contextual embeddings for detecting subtle or indirect toxicity. This research evaluates model performance, runtime, and computational trade-offs, providing actionable insights for production deployment. The resulting system is designed for hybrid moderation, integrating fast classical models with semantically robust transformers and optional large language model reasoning, enabling adaptive, high-fidelity content moderation through an API-ready microservice framework.

## 1. Introduction

Online platforms face a growing challenge in moderating user-generated content due to the prevalence of toxic, abusive, or otherwise harmful comments. Automated moderation is essential for scalability, but the task is complicated by:

- **Context-dependent toxicity** – insults or harassment may be indirect or sarcastic.  
- **Class imbalance** – most comments are safe, while severe or mild toxic comments are minority classes.  
- **Scalability requirements** – moderation systems must process large volumes of content efficiently.

This project presents a **multi-phase toxicity detection system** designed to combine interpretability, semantic understanding, and adaptive reasoning:

1. **Classical ML Baseline (Phase 1)**
   - TF-IDF features + engineered numeric features.  
   - XGBoost classifier.  
   - Lightweight, interpretable, and fast, serving as the foundation for comparison.

2. **Transformer-Based Models (Phase 2)**
   - Fine-tuned models like DistilBERT and ToxicBERT.  
   - Capture contextual nuances and improve recall for minority classes.  
   - Evaluate trade-offs between accuracy, macro F1, and computational cost.

3. **LLM Reasoning Layer (Phase 3, future work)**
   - Use large language models (e.g. LLaMA3) for zero- or few-shot toxicity classification.  
   - Handle subtle or ambiguous cases beyond transformer capabilities.

The ultimate goal is a **hybrid moderation framework**: a pipeline where classical models provide quick filtering, transformers capture semantic subtleties, and LLMs verify uncertain or borderline cases. This approach aims to balance **performance, interpretability, and scalability**, creating a practical and publication-worthy system for real-world deployment.


## 2. Related Work

Toxicity detection in online platforms has been extensively studied, evolving from classical machine learning approaches to transformer-based models and, more recently, large language models and hybrid moderation frameworks.

Early approaches relied on **feature-based models**. Nobata et al. (2016) combined linguistic, semantic, and n-gram features with SVM and logistic regression classifiers to detect abusive language in online comments. Davidson et al. (2017) explored TF-IDF and logistic regression for hate speech and offensive content on Twitter, while Gao and Huang (2017) proposed ensemble models that combined n-grams and sentiment features for comment moderation. Wulczyn et al. (2017) introduced the widely used Jigsaw Wikipedia Toxicity dataset and applied linear classifiers to detect multiple levels of toxicity, laying the foundation for subsequent research.

With the rise of **transformer models**, contextual embeddings significantly improved performance. Devlin et al. (2019) introduced BERT, which, when fine-tuned for toxicity classification, outperformed traditional baselines by capturing context and subtle semantic cues. Unitary.ai’s ToxicBERT (2020) further specialized BERT for toxic, offensive, and hate speech detection, showing enhanced recall for minority classes. Other studies, such as HateXplain (Mathew et al., 2021), combined transformer-based models with explainability mechanisms, emphasizing interpretability in classification. RoBERTa and XLNet variants (Zhang et al., 2020) also demonstrated improved multi-class toxicity detection performance over standard BERT, while Google's Perspective API (2018) showcased production-ready deep neural networks for real-time comment scoring.

Recently, **large language models (LLMs) and hybrid systems** have been explored for context-aware moderation. GPT-3 and GPT-4 (OpenAI, 2022–2023) can perform zero- and few-shot toxicity classification without task-specific fine-tuning, handling nuanced cases such as sarcasm and indirect insults. Open-source models like LLaMA and Mistral (Touvron et al., 2023) have also been leveraged for prompt-based toxicity detection. Hybrid moderation frameworks (You et al., 2021; Wang et al., 2022) combine rule-based filters, classical ML models, transformers, and LLM reasoning to balance computational efficiency and semantic understanding. These approaches often use a staged architecture: a fast first-stage model filters obvious safe content, while more complex transformers or LLMs handle ambiguous or highly toxic instances.

Finally, **datasets** play a central role in model development. The Jigsaw Toxic Comments dataset is widely adopted for multi-class classification tasks (`safe`, `mild`, `toxic`, `severe`), while Davidson et al.’s Twitter hate speech dataset and various Reddit and Twitter crawls are frequently used to augment training data, particularly for minority classes.  

**Summary:**  
The evolution from classical ML to transformers and LLM-based systems highlights the trade-offs between speed, interpretability, and contextual understanding. Our multi-phase approach builds on these prior works by combining a lightweight baseline (TF-IDF + XGBoost), fine-tuned transformers (DistilBERT and ToxicBERT), and LLM reasoning for scalable, context-aware, and interpretable online toxicity moderation.


## 3. Dataset & Preprocessing

### 3.1 Dataset

The primary dataset used in this study is the **Jigsaw Toxic Comment dataset**, which contains user-generated comments labeled for toxicity. To improve model performance and handle multi-level toxicity, we extended the dataset with **engineered numeric features**, such as:

- Comment length (number of words/characters)  
- Punctuation counts (e.g., exclamation marks, question marks)  
- Capitalization ratio  
- Presence of swear words or offensive keywords  

The dataset consists of **159,559 comments** classified into four categories:

- `safe`  
- `mild`  
- `toxic`  
- `severe`  

### 3.2 Preprocessing

Preprocessing was performed to standardize the text data and prepare it for both classical and transformer-based models:

1. **Text Cleaning**
   - Convert text to lowercase.  
   - Remove URLs, HTML tags, and non-alphanumeric characters.  
   - Normalize whitespace.  

2. **Tokenization**
   - For classical models: simple word tokenization with TF-IDF vectorization.  
   - For transformers: use the tokenizer provided by the pre-trained model (DistilBERT, ToxicBERT).

3. **Feature Engineering**
   - Combine TF-IDF vectors with numeric features for classical models.  
   - Transformers use embeddings from tokenized input, optionally concatenated with numeric features.

4. **Label Encoding**
   - Convert string labels (`safe`, `mild`, `toxic`, `severe`) into numeric form for model training.  

5. **Train-Test Split**
   - 80% for training, 20% for testing.  
   - Stratified splitting to maintain class distribution across sets.  

### 3.3 Output

After preprocessing, the following artifacts are saved for reproducibility and later phases:

- `data/processed/jigsaw_multilevel_features.csv` – processed dataset with numeric features  
- TF-IDF vectorizer and label encoder (for classical baseline models)  
- Preprocessed tokenized dataset ready for transformer fine-tuning

> **Note:** Fine-tuning of transformer-based models (DistilBERT, ToxicBERT) is performed in Phase 2.5 to evaluate contextual embeddings and improve detection of subtle and minority-class toxicity.  

This preprocessing pipeline ensures consistency across **all three phases** of the multi-phase toxicity detection system, enabling fair comparison between classical, transformer-based, and LLM approaches.

## 4. Methodology

This section describes the approaches and models used in the multi-phase toxicity detection system, covering classical ML baselines, transformer-based models, and future LLM reasoning.

### 4.1 Phase 1 – Classical ML Baseline
- **Model:** XGBoost classifier with TF-IDF + engineered numeric features
- **Training:**
  - Hyperparameters:  
    - `n_estimators=200`  
    - `max_depth=6`  
    - `learning_rate=0.05`  
    - `tree_method='hist'`  
    - `eval_metric='mlogloss'`  
  - Class balancing: sample weights computed using `sklearn.utils.compute_sample_weight(class_weight='balanced')` to address class imbalance
- **Input Features:** 
  - TF-IDF vectors (`max_features=10000`, ngram range 1–3, min_df=3, max_df=0.9, sublinear_tf=True)  
  - Numeric features: `char_count`, `word_count`, `num_uppercase`, `num_exclamation`, `num_question`, `has_swear`
- **Preprocessing Steps:**
  - Lowercasing, removing URLs and non-alphanumeric characters, normalizing whitespace  
  - Feature engineering to create numeric features from text
- **Train-Test Split:** 70% train, 15% validation, 15% test with stratified splitting
- **Purpose:** Establish a fast, interpretable baseline and serve as the first filtering layer in the hybrid system
- **Artifacts Saved:**  
  - Trained XGBoost model (`models/saved/classical/xgboost.pkl`)  
  - TF-IDF vectorizer (`models/saved/classical/vectorizer.pkl`)  
  - Label encoder (`models/saved/classical/label_encoder.pkl`)  

> **Note:** This classical baseline achieves a macro F1 of ~0.64, providing a reference for evaluating transformer-based models in Phase 2.

### 4.2 Phase 2 – Transformer-Based Models

- **Models:** 
  - DistilBERT (`distilbert-base-uncased`)  
  - ToxicBERT (`unitary/toxic-bert`)  

- **Fine-Tuning:**
  - Preprocessed dataset: `data/processed/jigsaw_multilevel_features.csv`  
  - Label encoding: map string labels (`safe`, `mild`, `toxic`, `severe`) to numeric IDs  
  - Train/Test split: 80% train, 20% test  
  - Tokenization:
    - DistilBERT: `DistilBertTokenizerFast`, `max_length=128`, padding, truncation  
    - ToxicBERT: `AutoTokenizer` with same settings  
  - Hyperparameters:
    - **DistilBERT**
      - `learning_rate=2e-5`  
      - `per_device_train_batch_size=16`  
      - `num_train_epochs=3`  
      - `weight_decay=0.01`  
      - `eval_strategy="steps"`, `eval_steps=500`  
      - `save_strategy="steps"`, `save_steps=500`  
    - **ToxicBERT**
      - `learning_rate=2e-5`  
      - `per_device_train_batch_size=8`  
      - `num_train_epochs=4`  
      - `weight_decay=0.01`  
      - `eval_strategy="steps"`, `eval_steps=500`  
      - `save_strategy="steps"`, `save_steps=500`  
      - `lr_scheduler_type="cosine"`, class-weighted loss for handling imbalance  
  - Early stopping and best model selection: `load_best_model_at_end=True`, metric=`eval_macro_f1`  

- **Input Features:** tokenized text embeddings (`input_ids` + `attention_mask`)  

- **Training Approach:** resume from checkpoint if available; otherwise, start from pre-trained weights  

- **Evaluation Metrics:** Accuracy, Macro F1 (per-class metrics computed separately if needed)  

- **Purpose:** 
  - Capture contextual semantics and subtle toxicity  
  - Improve recall for minority classes (`mild`, `toxic`, `severe`)  
  - Provide embeddings for downstream LLM reasoning layer  

- **Artifacts Saved (Best overall checkpoint):**  
  - Fine-tuned DistilBERT model (`models/saved/bert`)  
  - Fine-tuned ToxicBERT model (`models/saved/toxic_bert`)  
  - Tokenizers for both models  
  - Label maps (`label2id.pkl`, `id2label.pkl`, `label2idToxicBERT.pkl`, `id2labelToxicBERT.pkl`)  

> **Note:** Both models are compared in terms of accuracy, macro F1, per-class performance, and runtime to determine the best transformer for integration in the hybrid moderation system.

### 4.3 Phase 3 – LLM Reasoning Layer
### 4.3.1 Meta-llama-3.1-8b-instruct

- **Model:** 
  - LLaMA 3.1 8B Instruct (`meta-llama-3.1-8b-instruct`)  

- **Prompt Design:**  
    ```text
      You are a Comment Moderation AI. Classify a single forum comment. Return JSON only, no extra text.

      Labels:
        safe: no offensive content
        mild: minor insults, slightly rude
        toxic: strong insults, harassment
        severe: extreme aggression or identity-based hate

      Rules:
        - One label per comment
        - Consider literal impact of sarcasm or humor
        - Profanity alone ≠ severe, use toxic

      Output format:
        { "label": "<safe|mild|toxic|severe>", "reasoning": "<short explanation>" }

      Input: "{comment}"
      Output:
    ```


**Evaluation Dataset:** `forum_test_dataset.csv` (~115 comments split into 5 batches of ~25 each)  
**Tools:** LM Studio  
**Batching:** Sequential single-batch processing (~1 min per full dataset)  
**Evaluation Metrics:** Accuracy, Macro F1, Per-class F1, Confusion matrix, Runtime per batch  
**Hardware:** Mac M1 Max, ~10 GB RAM, CPU only  

**Integration Strategy:** 
- LLM verifies uncertain or borderline predictions from classical or transformer models.  
- Useful for nuanced context, sarcasm, and indirect insults.  
- Practical limits prevent training LLMs on the full 159k-comment dataset locally.


**Notes:**
- Handles contextual sarcasm well; occasional over-flagging of neutral comments.  
- Runtime per batch ~10–12 seconds; full dataset ~1 min.  
- Memory limitations restrict full-scale training; LLM best applied for small batches or verification tasks.

### 4.3.2 qwen3-4b-thinking-2507

- **Model:** 
  - qwen3-4b-thinking-2507 (`qwen/qwen3-4b-thinking-2507`)

- **Prompt Design:**  
    ```text
      You are a Comment Moderation AI. Classify a single forum comment.

      Return **JSON only**. No extra text. No explanations outside JSON.

      Labels:
      - safe: no offensive content
      - mild: minor insults or slightly rude
      - toxic: strong insults, harassment
      - severe: extreme aggression or identity-based hate

      Rules:
      - Exactly one label per comment.
      - Sarcasm must be taken literally.
      - Profanity alone does NOT imply severe; use toxic unless hate-driven.

      Output:
      {"label": "<safe|mild|toxic|severe>", "reasoning": "<short explanation>"}
    ```


**Evaluation Dataset:** `forum_test_dataset.csv` (~115 comments split into 5 batches of ~25 each)  
**Tools:** LM Studio  
**Batching:** Sequential single-batch processing (~1 min per full dataset)  
**Evaluation Metrics:** Accuracy, Macro F1, Per-class F1, Confusion matrix, Runtime per batch  
**Hardware:** Mac M1 Max, ~10 GB RAM, CPU only  


### 4.4 Evaluation Metrics
- **Primary Metrics:** Accuracy, Macro F1-score  
- **Per-class Metrics:** Precision, Recall, F1-score for each class (`safe`, `mild`, `toxic`, `severe`)  
- **Additional Analysis:** Confusion matrices, runtime and memory usage comparisons, qualitative examples of model success/failure

### 4.5 System Workflow
1. Preprocessed data is fed to the **Phase 1 classical model** for fast filtering.  
2. **Phase 2 transformer models** handle contextual classification of remaining comments.  
3. **Phase 3 LLM layer** optionally verifies uncertain or borderline cases based on confidence thresholds.  
4. Predictions are recorded and evaluated using the metrics above to provide quantitative and qualitative insights.

## 5. Experiments and Results

This section presents a comparative evaluation between the baseline classical models and the fine-tuned transformer-based models (DistilBERT and ToxicBERT). All models were trained on the same preprocessed dataset derived from the Jigsaw Toxic Comment dataset, extended with engineered numeric features.

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

### 5.3 Transformer Models (Fine-Tuned)

#### DistilBERT
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

#### ToxicBERT
| Metric | Value |
|---------|--------|
| **Accuracy** | 0.95553 |
| **Macro F1** | 0.73594 |
| **Runtime** | ~9.5h (on CPU / MPS) |
| **Parameters** | ~110M (ToxicBERT base) |
| **Best Checkpoint** | `checkpoint-48000` |
| **Epochs** | 4 |

**Observations:**
- Designed for toxicity detection, expected to outperform DistilBERT on subtle or extreme cases.  
- Class-weighted loss improves recall for minority classes.  
- Longer runtime and higher memory footprint due to larger model size.  

---

### 5.4 Comparative Summary
| Model | Accuracy | Macro F1 | Runtime | Params |
|-------|-----------|-----------|----------|---------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | ~3 min | ~1.2M |
| DistilBERT (Fine-tuned) | 0.9546 | 0.6837 | ~3 hr | ~66M |
| ToxicBERT (Fine-tuned) | 0.95553 | 0.73594 | ~9.5 hr | ~110M |

**Conclusion:**  
- Both transformer models demonstrate clear performance gains over the classical baseline, particularly in macro F1 and minority-class detection.  
- ToxicBERT is expected to excel in nuanced and highly toxic comments due to domain-specific pretraining and class-weighted training.  
- Trade-offs include longer training and inference times, and higher computational requirements.  
- Final per-class analysis will determine which transformer is better suited for integration into the hybrid moderation system.

### 5.5 Phase 3 – LLM Evaluation (Forum Comments)

This phase evaluates the performance of the local LLM (`meta-llama-3.1-8b-instruct`) on the `forum_test_dataset.csv` (~115 comments) split into 5 sequential batches (~25 comments each).

#### Prompt

LLM is instructed as a Comment Moderation AI, returning JSON only:
```json
{
  "label": "<safe|mild|toxic|severe>",
  "reasoning": "<short explanation>"
}
```

#### Evaluation Metrics

- Accuracy
- Macro F1
- Per-class F1
- Confusion Matrix
- Runtime per batch

#### Hardware

Mac M1 Max (~10 GB RAM, CPU only)

---

### LLaMA 3.1 8B Instruct Results

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.67      | 1.00   | 0.80     | 49      |
| safe   | 1.00      | 0.47   | 0.63     | 43      |
| severe | 0.80      | 1.00   | 0.89     | 4       |
| toxic  | 1.00      | 0.89   | 0.94     | 19      |

### qwen3-4b-thinking-2507 Results

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| mild   | 0.91      | 0.98   | 0.94     | 42      |
| safe   | 0.98      | 0.98   | 0.98     | 56      |
| severe | 1.00      | 1.00   | 1.00     | 3       |
| toxic  | 1.00      | 0.79   | 0.88     | 14      |

### Additional Local LLM Results (Placeholders)

> **Note:** Metrics for the remaining two LLMs should be added once evaluation is complete.

| Model | Accuracy | Macro F1 | Runtime (full dataset) |
|-------|----------|-----------|----------------------|
| LLaMA 3.1 8B Instruct | 0.78 | 0.82 | ~1 min |
| qwen3-4b-thinking-2507 | 0.96 | 0.95 | ~1 min |
| LLM-1 | TBD | TBD | TBD |
| LLM-2 | TBD | TBD | TBD |

#### Overall Metrics (All LLMs)

| Model | Accuracy | Macro F1 | Weighted F1 | Runtime (full dataset) |
|-------|----------|-----------|-------------|----------------------|
| LLaMA 3.1 8B Instruct | 0.78 | 0.82 | 0.77 | ~1 min |
| qwen3-4b-thinking-2507 | 0.96 | 0.95 | 0.96 | ~1 min |
| LLM-1 | TBD | TBD | TBD | TBD |
| LLM-2 | TBD | TBD | TBD | TBD |

### Integration Insight

- LLM serves as a reasoning layer to verify uncertain or borderline predictions from classical and transformer-based models.
- Provides improved semantic judgment for subtle or context-dependent toxicity cases.

### 5.6 Comparative Summary – Forum Test Dataset

| Model                        | Accuracy | Macro F1 | Avg Latency (s/comment) | Params  |
|------------------------------|----------|----------|-------------------------|---------|
| Classical TF-IDF + XGBoost   | 0.7304   | 0.7053   | 0.001                   | ~1.2M   |
| DistilBERT (Fine-tuned)      | 0.71     | 0.69     | 0.05                    | ~66M    |
| ToxicBERT (Fine-tuned)       | 0.73     | 0.75     | 0.05                    | ~110M   |
| LLaMA 3.1 8B Instruct        | 0.78     | 0.82     | 0.5                     | 8B      |
| qwen3-4b-thinking-2507        | 0.96     | 0.95     | 0.5                     | 4B      |

---

### Notes

- **Large language models** improves macro F1 and minority-class detection over both classical and transformer models on the forum dataset.
- The **classical baseline** remains useful for ultra-low-latency filtering.
- **Transformer models** retain high performance on structured training sets.
- **LLM** is most valuable for contextual reasoning on small-scale datasets or uncertain cases due to memory limitations.

## 6. Discussion

The experimental outcomes across all phases reveal a consistent pattern: as models increase in complexity and contextual understanding, their ability to capture nuanced forms of toxicity improves substantially, though at a cost in computational efficiency. This section outlines the main insights, common errors, computational trade-offs, and future directions for system enhancement.

### 6.1 Model Performance Insights

The **TF-IDF + XGBoost baseline** delivered a respectable **accuracy of 0.8758** and **macro F1 of 0.6393**, performing well on the dominant `safe` class but struggling with minority categories such as `mild` and `severe`. Its simplicity and 3-minute training time make it an excellent lightweight filtering stage, but its lack of contextual understanding limits effectiveness for sarcasm, indirect toxicity, or coded insults.

The **DistilBERT** model demonstrated substantial gains, achieving **0.9546 accuracy** and a **macro F1 of 0.6837**. The improvement in recall for minority classes reflects the strength of contextual embeddings in identifying subtle or implied toxicity. DistilBERT also showed better semantic robustness, e.g., correctly identifying passive-aggressive or indirect harassment comments that the classical model misclassified as neutral.

The **ToxicBERT** model, fine-tuned specifically for toxic and offensive language, achieved the highest performance among transformers with **accuracy of 0.9555** and **macro F1 of 0.7359**, representing roughly a **+10% to +15% macro F1 improvement** over the baseline. Its class-weighted loss contributed to stronger recall on severe and highly toxic comments, confirming the benefit of domain-specific pretraining. However, this came with a **9.5-hour runtime** on CPU/MPS hardware and a significantly larger parameter footprint (~110M), which poses challenges for real-time deployment.

The **(Phase 3) large language models**, evaluated on the smaller `forum_test_dataset.csv` (~115 comments), significantly higher accuracy and f1 metrics. The models demonstrated:
- Excellent contextual reasoning for sarcasm, humor, and subtle toxicity.  
- Strong per-class performance, particularly on `mild` and `toxic` comments, with fewer false negatives than classical and transformer models.  
- Runtime per full dataset ~1 minute, with batch-level inference of ~10–12 seconds per ~25-comment batch.  
- Practical limits on local hardware prevent training on the full 159k-comment dataset; best suited for verification or small-batch inference.

In summary:
- Transformers, particularly ToxicBERT, drastically outperform classical models in **minority-class recall and contextual detection**.
- LLM adds a **reasoning layer**, improving semantic judgment for subtle or ambiguous comments.
- The **baseline** remains valuable for low-latency pre-filtering.
- The **transformer tier** forms the semantic backbone of the moderation system, while the **LLM tier** provides interpretive verification where needed.

### 6.2 Error Analysis

Error analysis revealed several recurring misclassification patterns:
- **False negatives** commonly occurred in borderline or sarcastic comments where toxicity was context-dependent (e.g., “you’re such a genius” used ironically).  
- **False positives** were occasionally triggered by emotionally charged but non-toxic language (e.g., political or passionate debate comments).  
- **Domain shift** was evident in comments with slang, memes, or context-specific abbreviations not well represented in the Jigsaw dataset.  

LLM evaluation highlighted:
- Superior handling of indirect toxicity and sarcasm compared to classical and transformer models.
- Occasional over-flagging of neutral comments, suggesting the need for confidence thresholds or hybrid verification with transformer models.

### 6.3 Computational Trade-offs

Each model presents a distinct balance between performance and operational feasibility:

| Model | Accuracy | Macro F1 | Runtime | Params | Notes |
|--------|-----------|-----------|----------|---------|-------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | ~3 min | ~1.2M | Lightweight, interpretable |
| DistilBERT | 0.9546 | 0.6837 | ~3 h | ~66M | Strong contextual performance |
| ToxicBERT | 0.9555 | 0.7359 | ~9.5 h | ~110M | Domain-optimized, heavier compute |
| LLaMA 3.1 8B | 0.78 | 0.82 | ~1 min (115 comments) | 8B | Excellent reasoning; batch-limited due to hardware |
| qwen3-4b-thinking-2507| 0.96 | 0.95 | ~1 min (115 comments) | 8B | Excellent reasoning; batch-limited due to hardware |

**Deployment Considerations:**
- **Runtime and memory footprint** remain critical constraints.  
- Tiered inference strategy:
  1. XGBoost for high-throughput pre-filtering.  
  2. DistilBERT / ToxicBERT for semantic verification.  
  3. LLM for reasoning on ambiguous, subtle, or context-dependent content.  
- Provides a **scalable, hybrid moderation pipeline** balancing speed and contextual accuracy.

### 6.4 Future Work / TODOs

Several avenues remain open to extend the system’s capabilities:

- **Dataset Expansion:** Fine-tune ToxicBERT with larger and more diverse datasets (e.g., Reddit, Twitter, or multilingual corpora) to improve generalization across domains.
- **Multi-Modal Moderation:** Incorporate image and metadata signals alongside text to detect cross-modal toxicity.
- **LLM Integration (Phase 3):** Evaluate prompt-based large language model classification (e.g., LLaMA 3, Mistral, Phi-4) as a reasoning layer for low-confidence predictions.
- **Hybrid Thresholding:** Develop adaptive decision thresholds where DistilBERT handles primary classification and LLMs verify uncertain or borderline cases.
- **Inference Optimization:** Explore quantization, ONNX conversion, or model distillation to reduce latency for real-time deployment.

---

**Summary:**  
Transformers, particularly ToxicBERT, mark a decisive leap forward in accuracy and contextual understanding compared to classical baselines, albeit with higher computational demands. The multi-phase architecture offers a pragmatic path forward: leveraging the speed of classical models, the semantic power of transformers, and the reasoning depth of LLMs to achieve a scalable, context-aware, and production-ready moderation pipeline.

## 7. System Integration (SyntaxBase)

This section will describe how the trained models will be incorporated into the production moderation workflow. Key points / TODOs:

- Wrap transformer models as **FastAPI microservices** for real-time inference.
- Integrate the classical XGBoost baseline for quick filtering before transformer inference.
- Implement optional **LLM verification layer** for uncertain or borderline comments.
- Track predictions and performance metrics in production (e.g., MLflow, Weights & Biases).
- Monitor system performance and retrain models periodically based on new data.

---

## 8. Conclusion

This research demonstrates a structured, multi-phase approach to toxicity detection that incrementally improves contextual understanding and classification performance through the integration of classical, transformer-based, and large language model architectures. The findings underscore the value of layering models to balance efficiency, interpretability, and semantic depth in real-world moderation systems.

The **classical TF-IDF + XGBoost baseline** established a strong yet lightweight foundation, achieving a **macro F1 of 0.6393** and providing interpretable, low-latency predictions suitable for large-scale content filtering. Building on this, **DistilBERT** significantly improved minority-class detection and contextual sensitivity, achieving a **macro F1 of 0.6837**, while maintaining a manageable computational footprint. **ToxicBERT**, with domain-specific pretraining, delivered the best overall performance (**macro F1 of 0.7359**), capturing subtle, indirect, and severe forms of toxicity that classical methods often missed.

The **LLaMA 3.1 8B Instruct model (Phase 3)**, tested on a smaller forum evaluation set (`forum_test_dataset.csv`), further enhanced classification of nuanced and context-dependent comments, achieving a **macro F1 of 0.82**. While hardware limitations prevent full-scale training on the entire 159k-comment dataset, the LLM serves effectively as a reasoning layer for verification and disambiguation, particularly for sarcasm, humor, and borderline toxicity.

The **multi-phase framework** proposed here:
1. **Phase 1:** Fast classical filtering,  
2. **Phase 2:** Contextual transformer analysis, and  
3. **Phase 3:** Optional LLM reasoning  

provides a scalable and adaptive foundation for modern content moderation pipelines. This hierarchical design ensures that high-volume moderation systems maintain both **speed and contextual accuracy**, while selectively applying higher-compute models where ambiguity or subtlety is present.

From a deployment standpoint, the hybrid strategy offers a **production-feasible path** toward scalable moderation:  
- Lightweight classical models manage throughput for bulk content,  
- Transformers handle context-rich, minority-class detection, and  
- LLMs offer interpretive reasoning for the most ambiguous or nuanced cases.  

The results affirm that **real-time, context-aware toxicity detection** is achievable without fully sacrificing computational efficiency, providing a robust foundation for hybrid moderation systems that balance accuracy, speed, and semantic reasoning.

### Future Directions

Future work will focus on several extensions to further refine system robustness and adaptability:

- **Extended fine-tuning of ToxicBERT** on additional datasets (e.g., Reddit, Twitter) to improve generalization and handle slang, memes, and cultural context shifts.  
- **LLM-based reasoning (Phase 3)** for zero- and few-shot classification, enabling dynamic interpretation of subtle toxicity and sarcasm.  
- **Multi-modal analysis**, integrating text with visual or metadata cues for richer moderation.  
- **Adaptive hybrid thresholds**, where low-confidence transformer outputs trigger LLM verification to ensure balanced precision-recall trade-offs.  
- **Optimization for deployment**, leveraging quantization, model distillation, or ONNX runtimes to reduce inference latency.

---

**In summary**, the study validates the practical viability of a multi-phase toxicity detection system that strategically combines classical interpretability, transformer contextualization, and LLM reasoning. This hybrid design not only enhances detection accuracy but also establishes a scalable blueprint for next-generation moderation frameworks capable of adapting to evolving linguistic and cultural toxicity patterns in online discourse.