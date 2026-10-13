import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import api.main as api_main


def setup_module():
    # Deterministic temporary synthetic model makes API tests independent of the real dataset.
    X = pd.DataFrame({"amount": [10, 20, 30, 40, 50, 60, 70, 80], "hour": [1, 2, 3, 4, 5, 6, 7, 8]})
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pipe = Pipeline([
        ("preprocess", ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler())]), ["amount", "hour"])])),
        ("model", LogisticRegression()),
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, "tests/test_model.joblib")
    api_main.MODEL_PATH = __import__("pathlib").Path("tests/test_model.joblib")
    api_main.load_model()


def test_health():
    with TestClient(api_main.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_valid_prediction():
    with TestClient(api_main.app) as client:
        response = client.post("/predict", json={"amount": 75, "hour": 7})
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in [0, 1]
        assert 0 <= body["fraud_probability"] <= 1


def test_missing_field_rejected():
    with TestClient(api_main.app) as client:
        response = client.post("/predict", json={"amount": 75})
        assert response.status_code == 422


def test_extra_field_rejected():
    with TestClient(api_main.app) as client:
        response = client.post("/predict", json={"amount": 75, "hour": 7, "unexpected": 1})
        assert response.status_code == 422
