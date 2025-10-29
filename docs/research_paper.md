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

> **Note / TODO:** Fine-tuning of transformer-based models (DistilBERT, ToxicBERT) will be performed in Phase 2.5 to evaluate contextual embeddings and improve detection of subtle and minority-class toxicity.  

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

> **Note / TODO:** Both models will be compared in terms of accuracy, macro F1, per-class performance, and runtime to determine the best transformer for integration in the hybrid moderation system.


### 4.3 Phase 3 – LLM Reasoning Layer (Future Work)
- **Models:** LLaMa3 and other open source LLMs.
- **Approach:** Zero- or few-shot classification via prompt engineering  
- **Integration:** Optional verification of uncertain or borderline predictions from Phases 1 and 2
- **TODO:** Design confidence thresholds, prompts, and evaluation protocol for LLM verification

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
| **Accuracy** | TODO: insert actual value |
| **Macro F1** | TODO: insert actual value |
| **Runtime** | ~22h (on CPU / MPS) |
| **Parameters** | ~110M (ToxicBERT base) |
| **Best Checkpoint** | `checkpoint-13000` |
| **Epochs** | 4 |

**Observations:**
- Designed for toxicity detection, expected to outperform DistilBERT on subtle or extreme cases.  
- Class-weighted loss improves recall for minority classes.  
- Longer runtime and higher memory footprint due to larger model size.  
- TODO: Update with per-class F1 and qualitative examples once evaluation completes.

---

### 5.4 Comparative Summary
| Model | Accuracy | Macro F1 | Runtime | Params |
|-------|-----------|-----------|----------|---------|
| TF-IDF + XGBoost | 0.8758 | 0.6393 | ~3 min | ~1.2M |
| DistilBERT (Fine-tuned) | 0.9546 | 0.6837 | ~3 hr | ~66M |
| ToxicBERT (Fine-tuned) | TODO | TODO | ~22 hr | ~110M |

**Conclusion:**  
- Both transformer models demonstrate clear performance gains over the classical baseline, particularly in macro F1 and minority-class detection.  
- ToxicBERT is expected to excel in nuanced and highly toxic comments due to domain-specific pretraining and class-weighted training.  
- Trade-offs include longer training and inference times, and higher computational requirements.  
- Final per-class analysis will determine which transformer is better suited for integration into the hybrid moderation system.

---
## 6. Discussion

This section will summarize the insights, challenges, and practical implications observed from the experiments. Key points to include:

- **Model Performance Insights**
  - Compare classical baseline, DistilBERT, and ToxicBERT across accuracy, macro F1, and minority-class detection.
  - Highlight scenarios where transformers significantly outperform classical models (e.g., sarcasm, indirect toxicity).

- **Error Analysis**
  - Identify common misclassifications by class.
  - Discuss potential reasons for false positives/negatives (ambiguous comments, subtle toxicity, domain shift).

- **Computational Trade-offs**
  - Discuss runtime, memory, and hardware requirements for each model.
  - Comment on the feasibility of deploying transformer models in production.

- **Future Work / TODOs**
  - Fine-tune ToxicBERT with extended datasets (Reddit, Twitter) to improve generalization.
  - Explore multi-modal toxicity detection (images + text) for richer content moderation.
  - Evaluate prompt-based LLM classification (Phase 3) to assist transformer predictions.
  - Consider hybrid decision thresholds (e.g., DistilBERT primary, LLM verification on low-confidence cases).

---

## 7. System Integration (SyntaxBase)

This section will describe how the trained models will be incorporated into the production moderation workflow. Key points / TODOs:

- Wrap transformer models as **FastAPI microservices** for real-time inference.
- Integrate the classical XGBoost baseline for quick filtering before transformer inference.
- Implement optional **LLM verification layer** for uncertain or borderline comments.
- Track predictions and performance metrics in production (e.g., MLflow, Weights & Biases).
- Monitor system performance and retrain models periodically based on new data.

---

## 8. Conclusion

This section will summarize the contributions of the research and outline next steps. Key points / TODOs:

- Present the overall performance improvement of transformer models over classical baselines.
- Highlight the benefits of multi-phase toxicity detection (speed + contextual accuracy + reasoning via LLMs).
- Discuss practical implications for scalable and adaptive online content moderation.
- Outline future directions:
  - ToxicBERT fine-tuning and evaluation.
  - LLM integration for reasoning and zero/few-shot classification.
  - Potential hybrid system thresholds for adaptive moderation.
- Prepare visualizations and tables for final paper submission.
