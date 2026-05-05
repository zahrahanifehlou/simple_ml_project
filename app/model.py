from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X, y, C=1.0, solver="lbfgs", max_iter=200):
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
    model.fit(X, y)
    return model

def save_model(model, path="model.pkl"):
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    return joblib.load(path)