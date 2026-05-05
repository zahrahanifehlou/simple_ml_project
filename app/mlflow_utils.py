import mlflow
import mlflow.sklearn

def start_run(run_name="iris_experiment"):
    mlflow.set_experiment("iris_ml_project")
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_model(model, name="model"):
    mlflow.sklearn.log_model(model, name)