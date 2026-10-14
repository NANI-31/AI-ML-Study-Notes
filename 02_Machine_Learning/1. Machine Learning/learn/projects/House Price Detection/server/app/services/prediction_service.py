import logging
import numpy as np
import pandas as pd
from app.ml.model_loader import get_model, get_metadata
from app.schemas.prediction import (
    HousePredictionRequest,
    HousePredictionResponse,
    PriceValuation,
    ModelInfo,
)

logger = logging.getLogger("uvicorn.error")

from app.ml.feature_contract import ALL_EXPECTED_FEATURES, validate_feature_contract


def predict_property_price(request_data: HousePredictionRequest) -> HousePredictionResponse:
    """
    Executes the trained ML model pipeline on the validated property input.
    Maps request fields to DataFrame, injects NaN for deferred coordinate fields,
    verifies feature alignment, and formats the output into Lakhs and Crores.
    """
    pipeline = get_model()
    metadata = get_metadata()

    # 1. Convert Pydantic request to dictionary
    raw_dict = request_data.model_dump()

    # 2. Handle deferred coordinate features
    # TODO: Add latitude and longitude when geographic prediction support is introduced.
    if "latitude" not in raw_dict or raw_dict["latitude"] is None:
        raw_dict["latitude"] = np.nan
    if "longitude" not in raw_dict or raw_dict["longitude"] is None:
        raw_dict["longitude"] = np.nan

    # 3. Construct 1-row DataFrame with expected columns in exact order
    missing_keys = [col for col in ALL_EXPECTED_FEATURES if col not in raw_dict]
    if missing_keys:
        err_msg = f"Model input feature mismatch. Expected: {ALL_EXPECTED_FEATURES}. Missing: {missing_keys}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Filter strictly to the model's expected schema (ignoring non-modeled fields like pincode/mandal)
    feature_row = {col: raw_dict[col] for col in ALL_EXPECTED_FEATURES}
    input_df = pd.DataFrame([feature_row], columns=ALL_EXPECTED_FEATURES)

    # 4. Invoke trained Scikit-Learn pipeline
    try:
        raw_prediction = pipeline.predict(input_df)
        predicted_inr = float(raw_prediction[0])
    except Exception as e:
        logger.error(f"Inference execution failed inside ML pipeline: {e}", exc_info=True)
        raise RuntimeError(f"Prediction computation failure: {e}") from e

    # 5. Currency conversions (1 Lakh = 100,000 INR; 1 Crore = 10,000,000 INR)
    price_lakh = round(predicted_inr / 100000.0, 3)
    price_crore = round(predicted_inr / 10000000.0, 4)

    logger.info(
        f"Prediction successful for {request_data.state} / {request_data.district} "
        f"({request_data.property_type}): ₹ {predicted_inr:,.0f} (₹ {price_lakh} L / ₹ {price_crore} Cr)"
    )

    return HousePredictionResponse(
        success=True,
        prediction=PriceValuation(
            price_inr=round(predicted_inr, 2),
            price_lakh=price_lakh,
            price_crore=price_crore,
        ),
        model=ModelInfo(
            name=metadata.get("model_name", "Indian House Price Prediction Model"),
            version=metadata.get("version", "1.0.0"),
        ),
    )
