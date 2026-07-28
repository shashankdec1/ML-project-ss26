# FIFA Match Prediction with MLflow

This project builds a machine-learning pipeline to predict FIFA match outcomes from team form statistics. It prepares match data, trains and compares a few scikit-learn models, tracks results in MLflow, registers the best model, and generates knockout-stage predictions.

## Project Structure

- `src/prepare_data.py` - transforms raw match data into training and prediction datasets
- `src/train.py` - trains candidate models and logs metrics to MLflow
- `src/register_model.py` - registers the best MLflow run as the production model
- `src/prediction1.py` - loads the saved model and creates final tournament predictions
- `data/matches.csv` - raw match dataset
- `data/training_data.csv` - generated training set
- `data/prediction_matches.csv` - generated prediction set for the final four
- `data/final_predictions.csv` - generated prediction output
- `models/best_model.pkl` - saved best model package
- `mlflow.db` - local MLflow tracking database
- `mlruns/` - MLflow run artifacts

## Why The Feature Engineering Helps

The raw match table becomes much more useful after feature engineering because soccer outcomes depend on form, context, and tactical setup, not just the final score.

- Rolling form features
  - These summarize recent performance instead of relying only on one match.
  - They help the model capture momentum, consistency, and short-term trends.
  - This is useful because team strength changes over the tournament.
- Match-statistics features
  - Inputs such as shots, possession, fouls, and cards provide a stronger signal than score alone.
  - They help the model understand which team controlled play and which team was under pressure.
- Tournament-context features
  - Final-four matches are not the same as group-stage matches.
  - Context features help the model learn patterns that depend on stage, opponent balance, and match importance.
- Categorical match descriptors
  - Features like round and team identity are encoded so the model can learn team-specific and stage-specific effects.
  - One-hot style encoding is used because these values are not ordinal.

In short, the feature engineering turns raw match records into signals that better reflect form, control, and tournament pressure.

## Which Models Are Used And Why

This project compares three classic scikit-learn models:

- Logistic Regression
  - Used as a strong baseline.
  - It is fast, stable, and easy to interpret.
  - It works well when the engineered features already carry most of the signal.
- Random Forest
  - Used because it captures non-linear relationships and feature interactions well.
  - It is robust on tabular data and usually performs well without much tuning.
  - It is a good choice when the data has mixed feature types and noisy patterns.
- Gradient Boosting
  - Used because boosting often gives the best performance on structured tabular problems.
  - It builds a stronger predictor by correcting previous mistakes step by step.
  - It is useful when match outcomes depend on subtle combinations of form and context.

These models give a good balance between simplicity and predictive power. Logistic Regression gives a clean baseline, Random Forest adds robustness, and Gradient Boosting often provides the strongest final score.

## Requirements

- Python 3.10 or newer is recommended
- Install dependencies with:

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

3. Register the best model in MLflow

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
