# SyntaxBase Moderation Microservice

An interactive comment moderation system leveraging machine learning to classify comments into multiple toxicity levels: **safe**, **mild**, **toxic**, and **severe**. Built with Python, XGBoost, TF-IDF features, and numeric text features.


## Project Structure

```
SyntaxBase-moderation-microservice/
│
├── data/
│ ├── raw/
│ └── processed/
│ └── test/
│ └── utils/
|
├── docs/
│ └── architecture.md
│ └── docs.md
│ └── index.md
│ └── research_paper.md
│ └── training_logs.md
│
├── models/
│ └── saved/
│   └── bert/
|   └── classical/
|   └── toxic_bert/
│
├── notebooks/
│ └── 01_baseline_experiments.ipynb
│ └── 02_distilbert_experiments.ipynb
│ └── 03_toxicbert_experiments.ipynb
│ └── 04_model_comparison.ipynb
│ └── 05_error_analysis.ipynb
|
├── results/
│ └── comparisons/
|   └── bert/
│ └── logs/
|   └── classical/
|   └── transformer/
|       └── distil_bert/
|       └── toxic_bert/
│ └── metrics/
│ └── visuals/
│
├── src/
│ ├── classical/
│ ├── llm/
│ ├── transformer/
│ ├── utils/
│ │ └── generate_comments/
│ │     └── all_round_comments.py # general comments from various sources
│ │     └── forum_based_comments.py # comments that will likely be in forum
│ │ └── evaluate_both_models.py
│ │ └── evaluate_checkpoints.py
│ │ └── predict_comments_bert.py
│ │ └── predict_comments_toxicbert.py
│ │ └── predict_comments.py
│ │ ├── preprocessing.py
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Features

- **Multilevel toxicity classification**: safe, mild, toxic, severe  
- **Classical ML baseline**: TF-IDF + numeric features (char count, word count, capitalization, punctuation, swear word detection)  
- **Transformer-based models**: DistilBERT, ToxicBERT  
- **Interactive CLI** for live comment moderation  
- **Experiment tracking** through Jupyter notebooks  

**Planned / Future Enhancements:**

- Hybrid framework combining **rule-based**, **transformer-based**, and **LLM reasoning** (Phase 3)  
- Open-source LLM support: **LLaMA3**, **Mistral**, **Phi-4** for context-aware moderation  
- REST API for web integration and production deployment  
- Advanced monitoring and retraining with MLflow or Weights & Biases  

---

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/SyntaxBase-moderation-microservice.git
cd SyntaxBase-moderation-microservice
```
Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
### Optional
```bash
brew install git-lfs         # macOS
sudo apt-get install git-lfs # Linux
git lfs install
git lfs track "*.csv"
```
## Usage
### 1. Preprocess dataset
```bash
python src/utils/preprocessing.py
```
This will:

- Clean and normalize comments
- Compute numeric features
- Assign score and label (safe/mild/toxic/severe)
- Save processed CSV to data/processed/

### 2. Train models
```bash
python src/train_models.py
```
- Trains XGBoost, Random Forest, or Logistic Regression models
- Combines TF-IDF features and numeric features
- Saves trained models, vectorizer, and label encoder in models/saved/

### 3. Predict comments interactively
```bash
python src/utils/predict_comments.py
```
Example:

```pgsql
Comment Moderation Interactive Tool
Type your comment and press Enter. Type 'exit' to quit.

Your comment: you are stupid
Predicted label: mild
```
### 4. Evaluate baseline (Jupyter Notebook)

Open notebooks/01_baseline_experiments.ipynb to see:
- Class distribution and dataset statistics
- TF-IDF feature inspection
- Model evaluation: precision, recall, F1-score, confusion matrices
- Visualizations per-class F1

## Notes

The score system weights different toxicity labels to create a multilevel classification.
Swear words are detected as a separate numeric feature, but final prediction relies on combined model features.

## Future Work

Integrate transformer embeddings and LLM reasoning (e.g., LLaMA3, open-source models) for context-aware moderation.
Improve recall for minority classes (mild, toxic, severe).
Add a REST API for web integration.
Track experiments using MLflow or Weights & Biases.
Develop a hybrid moderation framework combining rule-based, ML-based, and LLM reasoning for scalable and adaptive online toxicity detection.

### License
MIT License © 2025 Ivan Smiljanić