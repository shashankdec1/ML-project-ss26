from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    if len(set(y_test)) == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return metrics
