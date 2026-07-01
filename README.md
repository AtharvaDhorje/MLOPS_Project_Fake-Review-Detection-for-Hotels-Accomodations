# MLOPS_Project_Fake-Review-Detection-for-Hotels & Accomodations

This project provides a baseline pipeline to detect fake (deceptive) reviews for the hotel & accommodation domain using the provided `deceptive-opinion.csv` dataset.

Getting started

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train a baseline model (TF-IDF + LogisticRegression):

```powershell
python src\train.py --data-path deceptive-opinion.csv --model-out models\tfidf_lr.joblib
```

3. Evaluate the saved model:

```powershell
python src\evaluate.py --data-path deceptive-opinion.csv --model-path models\tfidf_lr.joblib
```

4. Run the Streamlit demo:

```powershell
streamlit run app\streamlit_app.py
```

Project structure

- `src/` - data loading, training, evaluation scripts
- `app/` - Streamlit demo
- `models/` - trained model output (gitignored)

Notes

- The dataset column names are auto-detected; if your CSV uses custom names, open `src/data.py` to adapt the column selectors.
- This repo includes a lightweight baseline; consider fine-tuning transformer models for production-quality performance.

MLOps additions

- DVC: pipeline skeleton in `dvc.yaml` and parameters in `params.yaml`. To initialize DVC and add data:

```powershell
dvc init
dvc add deceptive-opinion.csv
git add deceptive-opinion.csv.dvc .gitignore
git commit -m "Add dataset to DVC"
```

- MLflow: training logs are recorded by `src/train_mlflow.py`. Run MLflow UI locally with:

```powershell
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

- CI/CD: GitHub Actions workflows under `.github/workflows` provide CI and Docker-based CD (requires DockerHub secrets).
- Monitoring: drift utilities in `src/monitor.py` include PSI and KS test helpers; integrate them into periodic checks.

Deployment

- A `Dockerfile` is provided to containerize the Streamlit demo. Use the GitHub Actions CD workflow to push Docker images.

Reproducibility

- Use `dvc repro` to run the pipeline once DVC is initialized and data is tracked.
- Use `mlflow` to compare runs and register models.
