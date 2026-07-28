import pandas as pd

from src.config import DATA_PATH


TARGET = "home_win"

NUMERIC_FEATURES = [
    "gameweek",
    "home_possession",
    "away_possession",
    "home_sot",
    "away_sot",
    "home_total_shots",
    "away_total_shots",
    "home_saves",
    "away_saves",
    "home_cards_yellow",
    "away_cards_yellow",
    "home_cards_red",
    "away_cards_red",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_crosses",
    "away_crosses",
    "home_interceptions",
    "away_interceptions",
    "home_offsides",
    "away_offsides",
]

CATEGORICAL_FEATURES = [
    "round",
    "dayofweek",
    "home_formation",
    "away_formation",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["home_score", "away_score"])
    df[TARGET] = (df["home_score"] > df["away_score"]).astype(int)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    return df[FEATURES], df[TARGET]
