# register model

import json
import mlflow
import logging
import os

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict) -> dict:
    """Register the model to the MLflow Model Registry."""
    try:
        if not model_info.get("run_id") or not model_info.get("model_path"):
            raise ValueError("Model info must include run_id and model_path")

        model_uri = model_info.get("model_uri")
        if not model_uri:
            model_uri = (
                f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            )
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Use an alias instead of the deprecated registry stages API.
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version,
        )

        registration_info = {
            "name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "model_uri": model_uri,
            "alias": "staging",
        }
        logger.debug(
            'Model %s version %s registered with the "staging" alias.',
            model_name,
            model_version.version,
        )
        return registration_info
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://16.178.121.158:5000")
    )

    try:
        model_info_path = os.path.join(root_dir, "experiment_info.json")
        model_info = load_model_info(model_info_path)
        registration_info = register_model("yt_chrome_plugin_model", model_info)
        registration_info_path = os.path.join(
            root_dir, "registered_model_info.json"
        )
        with open(registration_info_path, "w") as file:
            json.dump(registration_info, file, indent=4)
    except Exception:
        logger.exception("Failed to complete the model registration process")
        raise

if __name__ == '__main__':
    main()