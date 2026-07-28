from mlflow.tracking import MlflowClient

from src.config import MODEL_NAME, PRODUCTION_ALIAS


def get_production_model(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        return None


def promote_to_production(client: MlflowClient, model_version: str):
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, model_version)
