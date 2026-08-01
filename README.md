# Vehicle Insurance Claim Fraud Detection

A machine learning system for detecting fraudulent vehicle insurance claims, served as a production-ready REST API.

## Overview

Insurance claim fraud is a costly problem — this project builds an end-to-end pipeline covering data exploration, feature engineering, model training, and deployment. The model flags suspicious claims for manual review by an analyst, prioritising recall to minimise missed fraud cases.

**Dataset**: [Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (Kaggle)  
15,420 claims · 33 features · ~6% fraud rate (heavily imbalanced)

## Results

| Metric | Value |
|--------|-------|
| CV Average Precision | 0.26 |
| Recall (fraud class) @ threshold 0.091 | ~0.60 |
| Best hyperparameters | `n_estimators=200`, `max_depth=10`, `learning_rate=0.1` |

Decision threshold **0.091** was chosen as a deliberate compromise between two strategies:
- **~0.04** → 70% recall, 20% precision (aggressive flagging)
- **~0.33** → 16% recall, 50% precision (high-confidence auto-block)

At 0.091 the system maximises fraud capture while keeping analyst workload manageable.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data & EDA | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Modelling | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| Explainability | `shap`, `statsmodels` (logistic regression) |
| Serving | `FastAPI`, `uvicorn`, `pydantic` |
| Deployment | `Docker` |

## Project Structure

```
.
├── main.py                              # FastAPI application
├── requirements.txt                     # Production dependencies
├── vehicle insurance claim fraud/
│   ├── data.ipynb                       # EDA, feature engineering, training
│   ├── fraud_model_pipeline.pkl         # Serialised model pipeline
│   └── Dockerfile                       # Container definition
```

## ML Pipeline

```
Raw data (with NaN)
      │
      ▼
SimpleImputer (median)     ← handles missing Age / Deductible / DriverRating
      │
      ▼
SMOTE                      ← oversamples minority class (training only)
      │
      ▼
XGBClassifier              ← tuned via GridSearchCV (5-fold stratified CV)
      │
      ▼
predict_proba → threshold → fraud flag
```

The entire pipeline (imputer + model) is serialised to a single `.pkl` file, so no preprocessing is required at inference time.

## Key Engineering Decisions

- **Ordinal encoding** for naturally ordered categorical features (`VehiclePrice`, `AgeOfVehicle`, `Days_Policy_Accident`, etc.) instead of one-hot encoding, preserving ordinal information and reducing dimensionality.
- **SMOTE inside the pipeline** to prevent data leakage — resampling is applied only to training folds, never to validation data.
- **Imputer inside the pipeline** so the API can accept requests with missing values for nullable fields (`Age`, `Deductible`, `DriverRating`) without any client-side preprocessing.
- **Chi-square test** confirms `Fault` is statistically significant (χ² = 207.7, p ≈ 4.3e-47).
- **SHAP TreeExplainer** used for global and per-prediction interpretability.
- **Logistic regression** (statsmodels) alongside XGBoost for statistical inference and coefficient interpretation.

## API

### Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Returns fraud probability and binary flag |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Month": 1,
    "WeekOfMonth": 3,
    "MonthClaimed": 1,
    "WeekOfMonthClaimed": 4,
    "Age": 34,
    "VehiclePrice": 4,
    "RepNumber": 15,
    "Deductible": 400,
    "DriverRating": 4,
    "Days_Policy_Accident": 3,
    "Days_Policy_Claim": 2,
    "PastNumberOfClaims": 0,
    "AgeOfVehicle": 5,
    "AgeOfPolicyHolder": 4,
    "NumberOfSuppliments": 0,
    "NumberOfCars": 0,
    "Year": 1994,
    "claim_delay_approx": 1.0
  }'
```

### Example response

```json
{
  "fraud_probability": 0.043,
  "is_fraud": false
}
```

## Docker

Build and run from the **repository root**:

```bash
docker build -f "vehicle insurance claim fraud/Dockerfile" -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## Reproducing the Model

```bash
# Install Jupyter dependencies
pip install kagglehub jupyterlab xgboost imbalanced-learn shap statsmodels

# Open and run all cells
jupyter lab "vehicle insurance claim fraud/data.ipynb"
```

The notebook will download the dataset via `kagglehub`, run the full pipeline and overwrite `fraud_model_pipeline.pkl`.
