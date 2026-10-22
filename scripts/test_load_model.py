import pytest
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://16.178.121.158:5000")

@pytest.mark.parametrize("model_name, alias", [
    ("yt_chrome_plugin_model", "staging"),
])
def test_load_staging_model_alias(model_name, alias):
    client = MlflowClient()

    version = client.get_model_version_by_alias(model_name, alias)
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)

    assert version.version
    assert model is not None, "Model failed to load"
    print(
        f"Model '{model_name}' version {version.version} loaded successfully "
        f"from alias '{alias}'."
    )