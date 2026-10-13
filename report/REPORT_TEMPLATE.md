# Online Fraud Transaction Detection using Machine Learning and MLOps

**Student:** [Your name]  
**Training / batch:** [Fill in]  
**Date:** [Fill in]

## Abstract
This project develops a baseline binary classifier to identify potentially fraudulent transactions and packages it behind a REST API. It also demonstrates automated testing, CI, experiment tracking, and an optional Azure ML deployment path. Replace this paragraph with actual results after running the project.

## 1. Problem statement
Fraud detection is an imbalanced binary classification problem. False negatives can permit fraudulent transactions, while false positives can inconvenience genuine customers. The goal is to compare candidate classifiers and evaluate them using precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy alone.

## 2. Dataset
- Source and licence: [Provide source]
- Number of rows/columns: [Actual values]
- Target column: [Actual column]
- Positive class count/rate: [Actual values]
- Sensitive fields and handling: [Explain]
- Train/test split: stratified 80/20, random seed 42 (unless changed)

## 3. Preprocessing and EDA
Describe missing values, duplicate handling, identifier removal, numerical imputation/scaling, categorical imputation/encoding, target imbalance, and plots. State all dataset-specific assumptions.

## 4. Model development
Models: Logistic Regression, Decision Tree, Random Forest. Selection metric: average precision (PR-AUC). Include the actual comparison table from `artifacts/model_comparison.csv` and explain trade-offs. Describe any limitations of the holdout validation.

## 5. Final model and serialization
- Selected model: [Actual result]
- Metrics: [Actual metrics]
- Pipeline path: `models/fraud_pipeline.joblib`
- Reload validation: [Pass/fail evidence]

## 6. FastAPI
Document `GET /health`, `POST /predict`, request schema, response schema, and actual Swagger UI test. Include screenshots.

## 7. Testing and CI
Describe pytest cases and actual GitHub Actions run URL/status. Do not mark CI as passed unless the workflow ran successfully on GitHub.

## 8. MLflow and Azure ML
Separate local MLflow tracking from Azure ML workspace tracking. Include experiment/run identifiers and registry version if present. For Azure, record workspace, endpoint name, deployment state, test result, and date only if actually completed. Otherwise state: "Azure managed endpoint deployment was not completed in this submission; the deployment approach and prerequisites are documented."

## 9. Monitoring and retraining
Propose monitoring input distributions, score distributions, precision/recall when labels arrive, latency, errors, and drift. Trigger investigation/retraining based on validated thresholds and human review. Describe safe staged rollout and rollback.

## 10. Ethics, privacy, and limitations
Discuss false positives/negatives, dataset representativeness, potential bias, privacy/access controls, data leakage risks, and the fact that model output is a risk signal rather than proof of fraud.

## 11. Conclusion and future work
Summarize measured results and realistic next steps, such as threshold calibration, time-based validation, stronger feature engineering, model explainability, monitoring, and a verified cloud deployment.

## References
- Dataset source and licence: [Add]
- Scikit-learn documentation: https://scikit-learn.org/
- FastAPI documentation: https://fastapi.tiangolo.com/
- MLflow documentation: https://mlflow.org/docs/latest/
- Azure Machine Learning documentation: https://learn.microsoft.com/azure/machine-learning/

## Appendix: evidence
Attach genuine screenshots of dataset preview, EDA, model metrics, confusion matrix, PR curve, saved model, Swagger API, pytest, GitHub Actions, MLflow UI, and Azure endpoint only if deployed and tested.
