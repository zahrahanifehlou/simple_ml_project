import pandas as pd
from model import train_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "iris.csv")

    data = pd.read_csv(data_path)

    X = data.drop("Species", axis=1)
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = train_model(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    save_model(model)
    print("Model trained on Iris dataset and saved!")

if __name__ == "__main__":
    main()