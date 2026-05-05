# simple_ml_project

A minimal machine learning example using the Iris dataset and scikit-learn.

- Trains a `LogisticRegression` classifier on Iris.
- Saves the trained model to `model.pkl`.
- Includes a small prediction script and a CI workflow.

## Project structure

- `app/train.py`
  - Loads the Iris dataset
  - Splits into train/test
  - Trains the model
  - Prints accuracy
  - Saves the model to `model.pkl`
- `app/predict.py`
  - Loads `model.pkl`
  - Runs a sample prediction
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
- Write the trained model to `model.pkl`

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
