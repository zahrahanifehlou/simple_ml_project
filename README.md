# simple_ml_project

A minimal machine learning example using the Iris dataset and scikit-learn.

- Trains a `LogisticRegression` classifier on Iris.
- Saves the trained model to `model.pkl`.
- Tracks experiments with MLflow (metrics + model artifacts).
- Includes a small prediction script, an Optuna tuning script, and a CI workflow.

## Project structure

- `app/train.py`
  - Loads the Iris dataset
  - Splits into train/test
  - Trains the model
  - Prints accuracy
  - Logs accuracy to MLflow
  - Logs the trained model to MLflow as an artifact
  - Saves the model to `model.pkl`
- `app/predict.py`
  - Loads `model.pkl`
  - Runs a sample prediction
- `app/tune.py`
  - Hyperparameter tuning with Optuna (e.g. `LogisticRegression` `C`, `solver`)
- `app/model.py`
  - `train_model`, `save_model`, `load_model`
- `data/Iris.csv`
  - Included dataset file (training currently uses `sklearn.datasets.load_iris`)

## Requirements

- Python 3.10+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

From the repository root:

```bash
python app/train.py
```

This will:

- Print the model accuracy
- Log the accuracy metric to MLflow
- Log the trained model to MLflow as an artifact
- Write the trained model to `model.pkl`

## MLflow

MLflow is used for:

- **Tracking experiments** (e.g. accuracy)
- **Tracking parameters** (e.g. `C`, `solver`, `max_iter`)
- **Logging model artifacts** (the trained sklearn model)

Run the training script first (it creates an MLflow run):

```bash
python app/train.py
```

To view runs locally, start the MLflow UI from the repository root:

```bash
mlflow ui
```

Then open the shown URL (by default `http://127.0.0.1:5000`).

## Optuna hyperparameter tuning

Optuna is used for:

- **Hyperparameter tuning** (e.g., `LogisticRegression` `C`, `solver`, etc.)

The tuning script also logs each trial to MLflow (nested runs), including:

- **Trial params** (`C`, `solver`, `max_iter`)
- **Trial metric** (`cv_score`)
- **Best summary** (`best_*` params + `best_cv_score`)

Run:

```bash
python app/tune.py
```

This will print the best parameters and the best cross-validation score found.

## Run a prediction

After training (or if `model.pkl` already exists):

```bash
python app/predict.py
```

## Docker

Build:

```bash
docker build -t simple-ml-app .
```

Run (trains by default; see `Dockerfile` CMD):

```bash
docker run --rm simple-ml-app
```

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

On pushes to `main`, it:

- Installs dependencies
- Runs `python app/train.py`
- Runs `python app/predict.py`
- Builds the Docker image
