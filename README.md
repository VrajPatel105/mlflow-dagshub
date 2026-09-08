# mlops-mlflow using dagshub with model registry

Training and tracking an ML model on the Iris dataset using MLflow, with experiments logged to DagsHub and a model registry for versioning.

## What's here

- `train.py` : trains the model and logs the run to MLflow
- `iris_dt.py` : decision tree training on the Iris dataset
- `hyperparameter_tuning.py` : hyperparameter search, logged as separate MLflow runs
- `register_model.py` / `model_register101.py` : registers a trained model into the MLflow model registry
- `model_stage_transition.py` : moves a registered model between stages (e.g. staging to production)
- `inference.py` : loads a registered model and runs predictions
- `mlruns/` : local MLflow tracking data
- `mlflow.db` : MLflow backend store
