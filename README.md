# Classic MLflow Lifecycle Project

This is a small, production-style machine learning project using MLflow.

The goal is to predict whether the home team wins a football match from match statistics.

## Why This Is A Good ML Case

The dataset is structured tabular data, which makes it a good fit for scikit-learn pipelines. It contains match context and match statistics such as possession, shots, fouls, cards, corners, formations, and the gameweek.

The target is:

```text
home_win = 1 when home_score > away_score, otherwise 0
```

This is a strong classic ML problem because the inputs are a mix of numeric and categorical features, not images or free text. That makes preprocessing, model comparison, and deployment straightforward.

## Feature Engineering

The feature engineering in this project is intentionally simple and practical:

- `gameweek`
  - Gives the model season context. Early and late matches can behave differently because team form and squad stability change over time.
- Match statistics such as possession, shots, saves, cards, fouls, corners, crosses, interceptions, and offsides
  - These are direct signals of match control, attacking pressure, defensive effort, and discipline.
  - They help the model learn which teams are creating chances and which teams are under pressure.
- `round` and `dayofweek`
  - These add schedule context. Match dynamics can vary across competition rounds and across the week.
- `home_formation` and `away_formation`
  - Formations give tactical context. They can help capture style differences that raw scorelines do not show.

The preprocessing pipeline also adds a few standard ML safeguards:

- numeric features are imputed with the median
  - This keeps the pipeline robust when a match statistic is missing.
- numeric features are standardized
  - This helps linear models train more consistently.
- categorical features are one-hot encoded
  - This lets the model use formations and round/day categories without treating them as ordered values.
- `handle_unknown="ignore"` is enabled
  - This protects inference when a new category appears later.

## Models Used And Why

Two models are trained in `train.py`:

- Decision Tree Classifier
  - Used as the baseline model.
  - It is easy to interpret and can learn non-linear rules from the tabular match data.
  - The tree is intentionally kept shallow with `max_depth=2` and `min_samples_leaf=5` so it stays compact and does not overfit too quickly.
- Logistic Regression
  - Used as the challenger model.
  - It is a strong, simple baseline for tabular classification.
  - It often generalizes well when the features are already well prepared.
  - It works nicely with the scaling and one-hot encoding in the preprocessing pipeline.

Both models use `class_weight="balanced"` because classification targets in sports data are often not perfectly balanced. That helps the model pay attention to the smaller class instead of predicting the majority outcome too often.

## Project Structure

```text
data/matches.csv        # dataset
src/data.py             # loads data and creates the target
src/modeling.py         # preprocessing + models
src/evaluate.py         # metrics
src/registry.py         # MLflow model alias helpers
train.py                # full ML lifecycle
app.py                  # inference API
scripts/sample_request.py  # creates an example API payload
tests/test_data.py      # small data contract test
```

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

- trains a compact decision tree baseline
- logs metrics and artifacts to MLflow
- registers the model in the MLflow Model Registry
- trains a logistic regression challenger
- compares the challenger against the baseline using F1 score
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
