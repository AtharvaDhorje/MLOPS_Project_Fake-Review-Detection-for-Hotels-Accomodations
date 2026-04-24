# Start MLflow tracking server using sqlite backend
# Usage: Run this script in PowerShell. It will start mlflow server on port 5000

param(
    [int]$Port = 5000,
    [string]$DbPath = "mlflow.db",
    [string]$ArtifactRoot = "./mlruns"
)

Write-Host "Starting MLflow server on http://127.0.0.1:$Port using SQLite backend ($DbPath)"
mlflow server --backend-store-uri "sqlite:///$DbPath" --default-artifact-root "$ArtifactRoot" --host 127.0.0.1 -p $Port
