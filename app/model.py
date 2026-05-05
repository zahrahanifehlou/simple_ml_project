from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X, y, C=1.0):
    model = LogisticRegression(C=C, max_iter=200)
    model.fit(X, y)
    return model

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    return joblib.load(path)