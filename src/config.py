from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "matches.csv"
ARTIFACT_ROOT = ROOT_DIR / "mlartifacts"
MLFLOW_DB = ROOT_DIR / "mlflow.db"

TRACKING_URI = f"sqlite:///{MLFLOW_DB}"
EXPERIMENT_NAME = "football-home-win"
MODEL_NAME = "football_home_win_classifier"
PRODUCTION_ALIAS = "production"

RANDOM_STATE = 42
MIN_F1_TO_REGISTER = 0.55
MIN_IMPROVEMENT_TO_REPLACE = 0.01
