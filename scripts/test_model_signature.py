import pickle
import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://16.178.121.158:5000")

@pytest.mark.parametrize("model_name, alias", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_model_signature_with_matching_vectorizer(model_name, alias):
    client = MlflowClient()

    version = client.get_model_version_by_alias(model_name, alias)
    vectorizer_path = client.download_artifacts(
        version.run_id,
        "preprocessing/tfidf_vectorizer.pkl",
    )
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    input_data = vectorizer.transform(["hi how are you"])
    input_df = pd.DataFrame(
        input_data.toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    signature = model.metadata.signature
    assert signature is not None
    assert signature.inputs.input_names() == input_df.columns.tolist()

    prediction = model.predict(input_df)

    assert len(prediction) == input_df.shape[0]