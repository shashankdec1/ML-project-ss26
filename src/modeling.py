from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def decision_tree_model():
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def improved_model():
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
