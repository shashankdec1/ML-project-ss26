import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    MIN_F1_TO_REGISTER,
    MIN_IMPROVEMENT_TO_REPLACE,
    MODEL_NAME,
    PRODUCTION_ALIAS,
    RANDOM_STATE,
    TRACKING_URI,
)
from src.data import load_training_data
from src.evaluate import classification_metrics
from src.modeling import decision_tree_model, improved_model
from src.registry import get_production_model, promote_to_production


def train_and_log(model, model_label, x_train, x_test, y_train, y_test):
    with mlflow.start_run(run_name=model_label) as run:
        model.fit(x_train, y_train)
        metrics = classification_metrics(model, x_test, y_test)

        mlflow.log_param("model_label", model_label)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1", metrics["f1"])
        if "roc_auc" in metrics:
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=x_test.head(2),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "version": model_info.registered_model_version,
            "metrics": metrics,
        }


def should_replace(current_f1, challenger_f1):
    if current_f1 is None:
        return challenger_f1 >= MIN_F1_TO_REGISTER
    return challenger_f1 >= current_f1 + MIN_IMPROVEMENT_TO_REPLACE


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=f"file://{ARTIFACT_ROOT}",
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    x, y = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    good = train_and_log(decision_tree_model(), "good-decision-tree", x_train, x_test, y_train, y_test)
    promote_to_production(client, good["version"])

    challenger = train_and_log(improved_model(), "challenger-logistic-regression", x_train, x_test, y_train, y_test)

    if should_replace(good["metrics"]["f1"], challenger["metrics"]["f1"]):
        promote_to_production(client, challenger["version"])
        selected = challenger
        decision = "challenger replaced the old production model"
    else:
        selected = good
        decision = "old production model stayed active"

    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Production alias: {PRODUCTION_ALIAS}")
    print(f"Good model F1: {good['metrics']['f1']:.3f}")
    print(f"Challenger model F1: {challenger['metrics']['f1']:.3f}")
    print(f"Decision: {decision}")
    print(f"Live production version: {selected['version']}")


if __name__ == "__main__":
    main()
