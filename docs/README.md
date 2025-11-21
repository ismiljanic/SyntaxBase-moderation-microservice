# Multi-Phase Toxicity Detection System – Documentation

Welcome to the official documentation for the Multi-Phase Toxicity Detection System, developed as part of the SyntaxBase platform.

This project implements and compares classical, transformer-based, and LLM-based models for comment toxicity detection. The system is designed to integrate with real-time forum moderation via SyntaxBase.

---

## Documentation Overview

- [Architecture](architecture.md) – High-level system design, model pipeline, and microservice integration.
- [Training Logs](training_log.md) – Chronological records of all training runs, checkpoints, and metrics.
- [Research Paper](research_paper.md) – Full research-style write-up, including dataset, methodology, experiments, results and more.
- [Official documentation](docs.md) – Full official documentation for this project.
- [Notebooks](../notebooks/) – Jupyter notebooks for baseline experiments, transformer evaluation, model comparison, error and metrics analysis.
- [Results](../results/) – Saved metrics, model comparisons, logs and visual evaluation artifacts
- [Models](../models/saved/) – Stored classical and transformer model checkpoints, tokenizers, and label mappings.

---

## Quick Links

- Classical ML Baseline: `01_baseline_experiments.ipynb`
- DistilBERT Evaluation: `02_distilbert_experiments.ipynb`
- ToxicBERT Evaluation: `03_toxicbert_experiments.ipynb`
- Model Comparison & Error Analysis: `04_model_comparison.ipynb`, `05_error_analysis.ipynb`

---

> For any questions about setup, dataset preprocessing, or evaluation scripts, refer to `training_logs.md`, `research_paper.md` or the research paper draft with full documentation `docs.md`.
