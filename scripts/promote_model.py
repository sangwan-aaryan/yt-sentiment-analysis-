import mlflow
import os

def promote_model():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://16.178.121.158:5000",
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    model_name = "yt_chrome_plugin_model"
    staging_version = client.get_model_version_by_alias(model_name, "staging")
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version.version,
    )
    print(
        f"Model version {staging_version.version} for {model_name} "
        "is now available through the 'production' alias"
    )

if __name__ == "__main__":
    promote_model()