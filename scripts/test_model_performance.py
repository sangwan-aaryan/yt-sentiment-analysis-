import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def test_pipeline_model_meets_holdout_thresholds():
    with open("lgbm_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    holdout_data = pd.read_csv("data/interim/test_processed.csv")
    if not {"clean_comment", "category"}.issubset(holdout_data.columns):
        raise ValueError(
            "Holdout data must contain 'clean_comment' and 'category' columns"
        )

    features = vectorizer.transform(
        holdout_data["clean_comment"].fillna("").astype(str)
    )
    predictions = model.predict(features)
    labels = holdout_data["category"]

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=1,
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=1,
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=1,
        ),
    }
    for name, score in metrics.items():
        assert score >= 0.40, f"{name} should be at least 0.40, got {score}"