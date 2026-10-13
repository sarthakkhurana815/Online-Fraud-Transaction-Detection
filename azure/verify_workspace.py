"""Verify that Azure ML workspace access is configured. Does not create or deploy resources."""
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

required = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]
missing = [key for key in required if not os.getenv(key)]
if missing:
    raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

client = MLClient(
    DefaultAzureCredential(),
    os.environ["AZURE_SUBSCRIPTION_ID"],
    os.environ["AZURE_RESOURCE_GROUP"],
    os.environ["AZURE_WORKSPACE_NAME"],
)
workspace = client.workspaces.get(os.environ["AZURE_WORKSPACE_NAME"])
print("Azure ML workspace access verified.")
print("Workspace:", workspace.name)
print("Location:", workspace.location)
print("Resource group:", workspace.resource_group)
print("This script did not create an endpoint or deploy a model.")
