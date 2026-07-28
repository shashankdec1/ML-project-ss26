import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict

from src.config import MODEL_NAME, PRODUCTION_ALIAS, TRACKING_URI
from src.data import FEATURES, NUMERIC_FEATURES


mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
cached_model = None
cached_version = None

app = FastAPI(title="Football Home Win Inference API")


class MatchFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gameweek: float
    home_possession: int
    away_possession: int
    home_sot: int
    away_sot: int
    home_total_shots: int
    away_total_shots: int
    home_saves: int
    away_saves: int
    home_cards_yellow: int
    away_cards_yellow: int
    home_cards_red: int
    away_cards_red: int
    home_fouls: int
    away_fouls: int
    home_corners: int
    away_corners: int
    home_crosses: int
    away_crosses: int
    home_interceptions: int
    away_interceptions: int
    home_offsides: float
    away_offsides: float
    round: str
    dayofweek: str
    home_formation: str
    away_formation: str


def load_production_model():
    global cached_model, cached_version

    model_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    if cached_version != model_version.version:
        cached_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
        cached_version = model_version.version

    return cached_model, cached_version


@app.get("/health")
def health():
    _, version = load_production_model()
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
        "version": version,
    }


@app.post("/predict")
def predict(payload: MatchFeatures):
    model, version = load_production_model()
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    row[NUMERIC_FEATURES] = row[NUMERIC_FEATURES].astype(float)
    prediction = int(model.predict(row)[0])
    return {
        "home_win": bool(prediction),
        "class": prediction,
        "model_name": MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
        "version": version,
    }
