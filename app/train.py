import pandas as pd
from model import train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from mlflow_utils import start_run, log_params, log_metrics, log_model

def main():

    iris = load_iris(as_frame=True)
    data = iris.frame
    data["Species"] = iris.target

    X = data.drop("Species", axis=1)
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with start_run("baseline_model"):

        model = train_model(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)

        log_params({"model": "LogisticRegression"})
        log_metrics({"accuracy": acc})
        log_model(model)

        save_model(model)

    print("Model trained + logged in MLflow!")

if __name__ == "__main__":
    main()