# FIFA Match Prediction with MLflow

This project builds a  machine-learning pipeline to predict FIFA match outcomes from team form statistics. It prepares match data, trains and compares a few scikit-learn models, tracks results in MLflow, registers the best model, and generates knockout-stage predictions.

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

# Fifa-predict-model---ML-project
