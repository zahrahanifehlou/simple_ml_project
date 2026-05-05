import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from pathlib import Path
import mlflow

def objective(trial):

    iris = load_iris()
    X, y = iris.data, iris.target

    C = trial.suggest_loguniform("C", 1e-3, 1e2)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    max_iter = 200

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({"C": C, "solver": solver, "max_iter": max_iter, "model": "LogisticRegression"})

        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        score = cross_val_score(model, X, y, cv=3).mean()

        mlflow.log_metric("cv_score", float(score))
        return score


def main():

    mlruns_dir = (Path(__file__).resolve().parents[1] / "mlruns")
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("simple-ml")

    with mlflow.start_run(run_name="optuna_tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_score", float(study.best_value))

        print("Best params:", study.best_params)
        print("Best score:", study.best_value)


if __name__ == "__main__":
    main()