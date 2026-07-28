# FIFA Match Prediction with MLflow

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
pip install -r requirements.txt
```

## Dependencies

The project uses:

- `numpy`
- `pandas`
- `scikit-learn`
- `mlflow`
- `joblib`

## End-to-End Workflow

Run the scripts in this order:

1. Prepare the data

```bash
python src/prepare_data.py
```

This reads `data/matches.csv`, creates rolling form features, and writes:

- `data/training_data.csv`
- `data/prediction_matches.csv`

2. Train and evaluate models

```bash
python src/train.py
```

This trains Logistic Regression, Random Forest, and Gradient Boosting models, logs them to MLflow, and saves the best model to `models/best_model.pkl`.

- trains a compact decision tree baseline
- logs metrics and artifacts to MLflow
- registers the model in the MLflow Model Registry
- trains a logistic regression challenger
- compares the challenger against the baseline using F1 score
- automatically moves the `production` alias if the challenger improves enough

### 4. View MLflow

```bash
python src/register_model.py
```

This finds the best completed MLflow run, registers the model as `FIFA_Match_Predictor`, and assigns the `champion` alias.

4. Generate final predictions

```bash
python src/prediction1.py
```

This loads the saved model, predicts the semi-finals, third-place match, and final, then writes the results to `data/final_predictions.csv`.

## Notes

- `prepare_data.py` reserves the final four matches in `matches.csv` for prediction.
- The model predicts three classes:
  - `0` = Away Win
  - `1` = Draw
  - `2` = Home Win
- For knockout matches, if the model predicts a draw, the code selects the team with the higher win probability.

## Generated Files

The following files are created by the scripts and do not need to be committed:

- `data/training_data.csv`
- `data/prediction_matches.csv`
- `data/final_predictions.csv`
- `models/best_model.pkl`
- `mlflow.db`
- `mlruns/`

## Example Output

After running the full pipeline, you will have:

- a trained local model in `models/best_model.pkl`
- an MLflow experiment with run metrics and artifacts
- tournament predictions in `data/final_predictions.csv`

# Fifa-predict-model---ML-project
