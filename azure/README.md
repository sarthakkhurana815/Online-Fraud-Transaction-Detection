# Azure ML extension — configure only after local pipeline works

This directory contains a minimal Azure ML workspace verification / MLflow tracking example. It does not automatically create resources or deploy an endpoint.

## Prerequisites
1. An Azure subscription with permission to create/use Azure Machine Learning resources.
2. Azure CLI installed and logged in (`az login`).
3. Python dependencies: `pip install azure-ai-ml azure-identity`.
4. A workspace and compute/endpoint quota if you intend to deploy.

## Verify access
```bash
az account show
az ml workspace show --name YOUR_WORKSPACE --resource-group YOUR_RESOURCE_GROUP
```

If the workspace does not exist, create it through Azure Portal or follow your organization's approved provisioning process. Do not paste access keys or secrets into source code or a report.

## Track against Azure ML
Set environment variables in your terminal:
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_RESOURCE_GROUP`
- `AZURE_WORKSPACE_NAME`

Then run `python azure/verify_workspace.py`. It verifies authentication and prints the workspace details; it does not create an endpoint.

## Endpoint deployment
A real managed online endpoint requires a deployment YAML/score script, model artifact, environment image, compute capacity, permissions, and endpoint testing. Do not claim the endpoint task is complete until the endpoint has a successful deployment state and a real request returns a prediction. Keep an eye on charges and delete temporary endpoints/compute when no longer needed.
