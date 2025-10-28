## Phase 1 – Classical Baseline (TF-IDF + XGBoost)
**Date:** 2025-10-10  
**Hardware:** Mac M1 Max, CPU  
**Dataset:** jigsaw_multilevel_features.csv  
**Description:** TF-IDF (1,3) + numeric features → XGBoost classifier  
**Runtime:** 3m 24s  
**Results:**
- Accuracy: 0.8758  
- Macro F1: 0.6393  
- Weighted F1: 0.8938  

**Notes:**
- Struggles with minority classes (mild/severe).
- Feature scaling improves XGBoost stability.
- Saved artifacts: 
    - `models/saved/classical/xgboost.pkl`
    - `models/saved/classical/vectorizer.pkl`.

---