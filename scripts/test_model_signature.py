import pickle
import uuid

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature


def test_model_signature_and_vectorizer_artifact_round_trip(tmp_path):
    with open("lgbm_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    test_data = pd.read_csv("data/interim/test_processed.csv")
    sample = vectorizer.transform(test_data["clean_comment"].iloc[:5])
    input_example = pd.DataFrame(
        sample.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    signature = infer_signature(input_example, model.predict(input_example))

    previous_tracking_uri = mlflow.get_tracking_uri()
    database_path = (tmp_path / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    try:
        experiment_id = mlflow.create_experiment(
            f"signature-test-{uuid.uuid4().hex}",
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_run() as run:
            model_info = mlflow.sklearn.log_model(
                model,
                name="lgbm_model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=[
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                ],
            )
            mlflow.log_artifact(
                "tfidf_vectorizer.pkl",
                artifact_path="preprocessing",
            )

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        loaded_signature = loaded_model.metadata.signature
        assert loaded_signature is not None
        assert loaded_signature.inputs.input_names() == input_example.columns.tolist()
        assert len(loaded_model.predict(input_example)) == len(input_example)

        artifacts = mlflow.tracking.MlflowClient().list_artifacts(
            run.info.run_id,
            "preprocessing",
        )
        assert any(
            artifact.path.endswith("tfidf_vectorizer.pkl")
            for artifact in artifacts
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)