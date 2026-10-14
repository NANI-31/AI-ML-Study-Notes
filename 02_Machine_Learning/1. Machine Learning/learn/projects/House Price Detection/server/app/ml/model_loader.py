import json
import logging
from typing import Any, Dict, Optional
import joblib

logger = logging.getLogger("uvicorn.error")

# Cached model and metadata instances in memory
_model_instance: Optional[Any] = None
_metadata_instance: Optional[Dict[str, Any]] = None


def load_model(model_path: str) -> Any:
    """
    Loads the trained ML pipeline from disk once and keeps it in memory.
    Directly loads the model using the string path from .env.
    """
    global _model_instance

    logger.info("Initializing ML model loader...")
    logger.info(f"Target model artifact path: {model_path}")

    try:
        loaded_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        error_msg = f"Model artifact not found at: {model_path}"
        logger.critical(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        logger.critical(f"Failed to load ML pipeline from {model_path}: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failure: {e}") from e

    # Verify that the loaded object is a valid estimator with a predict method
    if not hasattr(loaded_pipeline, "predict"):
        error_msg = f"Loaded object from {model_path} does not implement the 'predict' method."
        logger.critical(error_msg)
        raise TypeError(error_msg)

    _model_instance = loaded_pipeline
    logger.info("ML model pipeline loaded successfully into memory.")
    return _model_instance


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Loads the model metadata JSON for versioning, feature lists, and health checks
    using the direct string path from .env.
    """
    global _metadata_instance

    logger.info(f"Target metadata path: {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _metadata_instance = json.load(f)
        logger.info("Model metadata loaded successfully.")
        return _metadata_instance
    except FileNotFoundError:
        logger.warning(f"Metadata file not found at: {metadata_path}. Using fallback defaults.")
        _metadata_instance = {
            "model_name": "Indian House Price Prediction Model",
            "version": "1.0.0",
            "target": "price_inr"
        }
        return _metadata_instance
    except Exception as e:
        logger.warning(f"Error reading metadata from {metadata_path}: {e}. Using fallback defaults.")
        _metadata_instance = {
            "model_name": "Indian House Price Prediction Model",
            "version": "1.0.0",
            "target": "price_inr"
        }
        return _metadata_instance


def get_model() -> Any:
    """
    Returns the in-memory loaded ML pipeline singleton.
    Raises RuntimeError if accessed before model initialization.
    """
    if _model_instance is None:
        raise RuntimeError("ML model has not been initialized. Ensure server startup lifespan has executed.")
    return _model_instance


def get_metadata() -> Dict[str, Any]:
    """
    Returns the in-memory model metadata dictionary.
    """
    global _metadata_instance
    if _metadata_instance is None:
        return {
            "model_name": "Indian House Price Prediction Model",
            "version": "1.0.0",
            "target": "price_inr"
        }
    return _metadata_instance
