# Online Fraud Detection — End-to-End ML & MLOps Capstone

This project provides a reproducible baseline for transaction fraud classification, a FastAPI prediction service, automated tests, GitHub Actions CI, local MLflow tracking, and optional Azure ML scripts.

## Important limitations
- No dataset is bundled. Use a dataset you are permitted to use and verify its target column.
- Training uses the selected CSV target and excludes that target from the features.
- Dataset-specific feature engineering and business validation may be needed.
- Azure scripts are templates; they do not prove that a workspace, endpoint, or deployment exists. Run them only after configuring and verifying your Azure subscription, workspace, permissions, and quota.
- A local MLflow run is not the same as Azure ML tracking or a deployed Azure endpoint.

## 1. Setup (VS Code terminal)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put your dataset at `data/transactions.csv`. Do not commit sensitive data.

## 2. Configure the dataset

The training script expects `data/onlinefraud.csv` and uses `isFraud` as the target. If your CSV has a different path or target column, update `DATA_PATH` and `TARGET_COLUMN` in `src/train.py` first.

```python
import pandas as pd
df = pd.read_csv("data/onlinefraud.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df["YOUR_TARGET_COLUMN"].value_counts(dropna=False))
```

The target must contain two classes, with the positive/fraud class encoded as `1` and normal transactions as `0`. If the dataset uses another convention, map it explicitly before training.

## 3. Train, evaluate, and log

```bash
python -m src.train
```

Outputs:
- `models/fraud_pipeline.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/precision_recall_curve.png`
- MLflow run metadata in `mlflow.db` and logged model artifacts under `mlruns/`

The script compares Logistic Regression, Decision Tree, and Random Forest. It uses a stratified holdout split, class weights where supported, and selects by average precision (PR-AUC). It deliberately avoids SMOTE by default to keep the first-day pipeline simple and prevent resampling leakage.

View MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Then open http://127.0.0.1:5000.

## 4. Run API

Train the model first, then:
```bash
python -m uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs. `POST /predict` expects a JSON object with the feature columns used during training. The API rejects missing/extra columns. `GET /health` is a basic process/model health check, not a full readiness probe.

## 5. Tests

```bash
pytest -q
```
Tests use a temporary synthetic model so they do not require your real dataset. They verify health, valid prediction, missing fields, and extra fields.

## 6. GitHub Actions

Create a GitHub repository, commit source code (not dataset/model files), push to GitHub, and inspect the Actions tab. The CI workflow installs dependencies and runs tests. It does not train the model or deploy Azure.

## 7. Azure ML (optional, only after local demo works)

See `azure/README.md`. Azure workspace creation, compute provisioning, endpoint deployment, online inference, and cost cleanup are separate steps. Do not claim deployment unless the endpoint is actually created and tested.

