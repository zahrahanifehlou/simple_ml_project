import pandas as pd
from model import train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn

def main():

    iris = load_iris(as_frame=True)
    data = iris.frame
    data["Species"] = iris.target

    X = data.drop("Species", axis=1)
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run(run_name="baseline_model"):

        C = 1.0
        solver = "lbfgs"
        max_iter = 200

        mlflow.log_params({"C": C, "solver": solver, "max_iter": max_iter, "model": "LogisticRegression"})

        model = train_model(X_train, y_train, C=C, solver=solver, max_iter=max_iter)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)

        mlflow.log_metrics({"accuracy": acc})
        mlflow.sklearn.log_model(model, "model")

        save_model(model)

    print("Model trained + logged in MLflow!")

if __name__ == "__main__":
    main()