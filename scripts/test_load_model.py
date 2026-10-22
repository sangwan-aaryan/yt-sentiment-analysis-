import json
import os
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient


def test_staging_alias_points_to_pipeline_registered_model():
    registration_path = Path("registered_model_info.json")
    if not registration_path.is_file():
        pytest.skip("Run dvc repro before checking the registered alias")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://16.178.121.158:5000",
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    registration_info = json.loads(
        registration_path.read_text(encoding="utf-8")
    )

    assert registration_info["alias"] == "staging"
    version = client.get_model_version_by_alias(
        registration_info["name"],
        registration_info["alias"],
    )

    assert str(version.version) == str(registration_info["version"])
    assert version.run_id == registration_info["run_id"]