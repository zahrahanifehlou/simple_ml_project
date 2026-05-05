import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from mlflow_utils import start_run, log_params, log_metrics

def objective(trial):

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    model = LogisticRegression(C=C, solver=solver, max_iter=300)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    log_params({"C": C, "solver": solver})
    log_metrics({"accuracy": acc})

    return acc


def main():

    with start_run("optuna_iris_tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        print("Best params:", study.best_params)
        print("Best accuracy:", study.best_value)


if __name__ == "__main__":
    main()