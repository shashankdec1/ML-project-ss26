# Classic MLflow Lifecycle Project

This is a small, production-style machine learning project using MLflow.

The goal is simple: predict whether the home team wins a football match from match statistics.

## Why This Is A Good ML Case

The dataset has structured tabular data: possession, shots, fouls, cards, corners, formations, and final scores.

The target is:

```text
home_win = 1 when home_score > away_score, otherwise 0
```

This is a good classic ML problem because the input is mostly numeric/categorical tabular data, not images or text. A simple sklearn pipeline is enough and is easier to understand, test, and deploy.

## Project Structure

```text
data/matches.csv           # dataset
src/data.py                # loads data and creates the target
src/modeling.py            # preprocessing + models
src/evaluate.py            # metrics
src/registry.py            # MLflow model alias helpers
train.py                   # full ML lifecycle
app.py                     # inference API
scripts/sample_request.py  # creates an example API payload
tests/test_data.py         # small data contract test
```

## Backend Intuition

The backend has two parts:

1. Training backend: `train.py`
2. Inference backend: `app.py`

The training backend reads the CSV, creates the target, splits the data, trains models, logs metrics to MLflow, registers model versions, and points the `production` alias to the best accepted model.

The inference backend follows:

```text
football_home_win_classifier@production
```

That means the API does not need to know the exact model version. On each request, it checks which version the MLflow `production` alias points to. If the alias changed, the API reloads the new version and exposes it without code changes.

## Lifecycle

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a quick test

```bash
python -m pytest -q
```

### 3. Train, evaluate, register, and promote

```bash
python train.py
```

This command does the full lifecycle:

- trains a good compact decision tree model
- logs metrics and artifacts to MLflow
- registers the model in the MLflow Model Registry
- trains an improved logistic regression challenger
- compares the challenger against the old model using F1 score
- automatically moves the `production` alias if the challenger improves enough

### 4. View MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

```text
http://127.0.0.1:5000
```

### 5. Serve the production model

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 6. Create a sample request

```bash
python scripts/sample_request.py
```

Copy the JSON into the `/predict` endpoint in the FastAPI docs.

## Why The Model Is Considered Good

The model is considered good when it beats a naive decision rule using real evaluation metrics on a held-out test set.

This project uses F1 score as the main promotion metric because the goal is classification and the classes may not be perfectly balanced. Accuracy is also logged, but F1 is better when false positives and false negatives both matter.

The first accepted model must reach:

```text
F1 >= 0.55
```

This threshold is intentionally modest because the dataset is small. It prevents registering a clearly weak model while keeping the project realistic for a compact demo.

## Automatic Improvement Rule

The challenger model replaces the old production model only when:

```text
challenger_f1 >= current_f1 + 0.01
```

This avoids replacing production for tiny random changes. If the challenger is better, MLflow updates:

```text
alias production -> new model version
```

Because the API loads the model through this alias, the latest production model is exposed without changing API code.

## Notes

This dataset contains match statistics, so the API is best understood as an in-match or post-match prediction service. A true pre-match predictor would need only information available before kickoff, such as teams, venue, rankings, injuries, and historical form.
