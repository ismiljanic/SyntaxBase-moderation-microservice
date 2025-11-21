# SyntaxBase Moderation Microservice

**SyntaxBase Moderation Microservice** is a scalable, interactive comment moderation system designed to automatically classify user comments into multiple toxicity levels: **safe**, **mild**, **toxic**, and **severe**.

The system uses a hybrid pipeline combining classical machine learning and modern NLP models:

1. **Classical ML** – XGBoost with TF-IDF and numeric text features for fast first-pass classification.
2. **BERT-based models** – Deep NLP models for semantic understanding and finer toxicity detection.
3. **ToxicBERT** – Specialized model for aggressive or identity-based content.
4. **LLM fallback** – Large language model handles uncertain cases and provides reasoning.

The microservice is fully **Dockerized** and orchestrated via **docker-compose**, with each component running as an independent service for flexible scaling and fault isolation.

## Key Features

- Multi-level toxicity detection: safe, mild, toxic, severe  
- Hybrid architecture combining classical ML, transformers, and LLMs  
- Dockerized services for reproducible deployment  
- Detailed reasoning for LLM-classified content  

## Tech Stack

Python, FastAPI, XGBoost, Transformers, Docker, docker-compose

## Project Structure

```
SyntaxBase-moderation-microservice/
│
├── data/
│ └── processed/
│ ├── raw/
│ └── test/
│ └── utils/
|
├── docker/
│ └── api/
    └── Dockerfile
│ ├── bert/
    └── Dockerfile
│ └── classical/
    └── Dockerfile
│ └── llm/
    └── Dockerfile
│ └── toxic_bert/
    └── Dockerfile
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
│ └── 06_metrics_analysis.ipynb
|
├── results/
│ └── comparisons/
|   └── bert/
|   └── llm/
│ └── model_comparison_summary.csv
│ └── runtime_memory_tradeoff.csv
|
│ └── logs/
|   └── classical/
|   └── llm/
|       └── meta-llama-3.1-8b-instruct/
|       └── phi-4-reasoning-plus/
|       └── qwen3-4b-thinking-2507/
|   └── transformer/
|       └── distil_bert/
|       └── toxic_bert/
│ └── metrics/
│ └── visuals/
│
├── src/
│ ├── classical/
│ ├── llm/
│   ├── prompts/
│   ├── evaluator.py/
│   ├── llm_moderation_client.py/
│   ├── llm_service.py/
│ ├── transformer/
│ ├── utils/
│ │ └── generate_comments/
│ │     └── all_round_comments.py # general comments from various sources
│ │     └── forum_based_comments.py # comments that will likely be in forum
│ │ └── aggregateLlmResults.py
│ │ └── compare_all_llms.py
│ │ └── evaluate_both_models.py
│ │ └── evaluate_checkpoints.py
│ │ └── model_comparison_summary.py
│ │ └── predict_comments_bert.py
│ │ └── predict_comments_toxicbert.py
│ │ └── predict_comments.py
│ │ ├── preprocessing.py
│ │ ├── runtime_memory_tradeoff_comparison.py
| │api_service.py   
| |config.py
| |inference.py   
| |model_loader.py   
| |utils.py   
|
├── .gitattributes
├── .gitignore
├── docker-compose.yml
├── LICENSE
├── README.md
└── requirements.txt
```

## Features

**Multilevel Toxicity Classification**  
- Classifies comments into **safe**, **mild**, **toxic**, and **severe** categories.

**Classical ML Baseline**  
- **TF-IDF features** for textual representation.  
- **Numeric text features** including character count, word count, capitalization ratio, punctuation count, and swear word detection.  
- Fast and interpretable baseline for moderation.

**Transformer-Based Models**  
- **DistilBERT** for semantic understanding and contextual analysis.  
- **ToxicBERT** for fine-grained toxic content detection.  
- Confidence scoring and threshold-based decision-making.

**Hybrid Moderation Pipeline**  
- Multi-step classification: **classical ML → BERT → ToxicBERT → LLM reasoning**.  
- LLM invoked only when previous models are uncertain or disagree.  
- Flexible orchestration ensures high accuracy while minimizing compute costs.

**LLM Integration**  
- Uses **Qwen3-4B** for context-aware moderation and reasoning.  
- Supports reasoning output along with label prediction for transparency.  
- Easily extendable to other open-source LLMs (e.g., LLaMA3, Mistral, Phi-4).

**Interactive CLI**  
- Live moderation interface for testing individual comments.  
- Provides labels, confidence scores, and LLM reasoning when applicable.

**REST API & Microservices**  
- Fully containerized services for **classical**, **BERT**, **ToxicBERT**, **LLM**, and API orchestration.  
- Docker Compose for multi-service deployment and health monitoring.  
- Endpoints for remote moderation requests.

**Experiment Tracking & Notebooks**  
- Jupyter notebooks for model evaluation, error analysis, and comparison across classical and transformer models.  
- Supports reproducible experiments and visualizations.

**Monitoring & Logging**  
- Logs and metrics stored for each service (classical, transformers, LLM).  
- Facilitates debugging and performance analysis.

**Planned / Future Enhancements**  
- Continuous retraining and monitoring with **MLflow** or **Weights & Biases**.  
- Hybrid model improvements and additional LLM integrations.  
- Advanced web integration with dashboards and user feedback loop.

# Installation & Setup

## 1. Clone the repository
```bash
git clone https://github.com/ismiljanic/SyntaxBase-moderation-microservice.git
cd SyntaxBase-moderation-microservice
```

## 2. Install Docker & Docker Compose
Make sure you have Docker and Docker Compose installed.

Check versions:
```bash
docker --version
docker-compose --version
```

## 3. Build and start services
All core services (Classical ML, BERT, ToxicBERT, LLM, API) are containerized.
```bash
docker-compose build
docker-compose up -dx
```

This will build and start all microservices.

Each service exposes a port for API requests:
- **classical**: 7001
- **bert**: 7002
- **toxicbert**: 7003
- **llm**: 7004
- **api**: 8000

## 4. Verify services
Check logs to ensure services are running:
```bash
docker-compose logs -f api
docker-compose ps
```

---

# Usage

## 1. Using the REST API
Send a POST request to the API endpoint:
```bash
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "Some test comment"}'
```

Response will include:
- `final_label`
- Intermediate labels & confidence scores from classical, BERT, and ToxicBERT
- LLM reasoning (if invoked)

> **Note on LLM verification**: To use LLM-based moderation, you need to have **LM Studio** running with a model loaded, or another local open-source LLM available via API. This codebase assumes **LM Studio is running** on the default endpoint (typically `http://localhost:1234/v1`). Without a running LLM server, the LLM service will not be able to provide reasoning or verification.

## 2. Interactive CLI (optional)
If you want to use the CLI for single-comment moderation:
### 1.Classical phase 1 moderation
```bash
api python src/utils/predict_comments.py
```
or
```bash
docker-compose exec api python src/utils/predict_comments.py
```
### 2.Distil BERT moderation (phase 2)
```bash
python src/utils/predict_comments_bert.py
```
or
```bash
docker-compose exec api python src/utils/predict_comments_bert.py
```
### 3.ToxicBERT moderation (phase 2)
```bash
 python src/utils/predict_comments.py
```
or
```bash
docker-compose exec api python src/utils/predict_comments_toxicbert.py
```

## 3. Monitoring
- Each service has logs accessible via `docker-compose logs <service_name>`
- Health checks are built-in for all microservices
- detailed training logs are available in **docs/training_log.md**
- detailed results and comparisons are available in **results/** folder

---

# Notes
- Models are preloaded in each service; no local setup of `.pt`/`.safetensors` files is needed for API use
- Future updates may include retraining scripts and experiment tracking with MLflow or W&B, which can also run inside containers


### License
MIT License © 2025 Ivan Smiljanić