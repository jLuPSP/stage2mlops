# Mock Level 2 MLOps Pipeline

This is a simplified demonstration of a Level 2 MLOps pipeline inspired by Google Cloud Platform's best practices.

## Components
- **train.py**: Trains a simple XGBoost model on the Iris dataset.
- **evaluate.py**: Evaluates the model using F1 score and blocks if below threshold.
- **register_model.py**: Registers the model and saves metadata.
- **app.py**: Serves the model via a FastAPI endpoint.
- **.github/workflows/mlops.yml**: GitHub Actions CI pipeline to automate training, evaluation, and registration.

## Triggering the Pipeline
You can run the pipeline automatically on push to `main`, or manually via the GitHub Actions interface using `workflow_dispatch`.

## Requirements
- Python 3.9
- scikit-learn
- xgboost
- joblib
- fastapi
- uvicorn

## Run the API locally
```bash
uvicorn app:app --reload
```

## Predict Example
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
"""