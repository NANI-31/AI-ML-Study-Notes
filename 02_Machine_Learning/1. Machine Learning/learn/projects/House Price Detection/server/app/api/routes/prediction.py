import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import HousePredictionRequest, HousePredictionResponse
from app.services.prediction_service import predict_property_price

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=HousePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Property Price",
    description="Accepts residential property specifications and returns an estimated market valuation in INR, Lakhs, and Crores using the trained ML pipeline."
)
def predict_price(payload: HousePredictionRequest):
    try:
        response = predict_property_price(payload)
        return response
    except RuntimeError as re:
        if "not been initialized" in str(re):
            logger.error(f"Prediction requested while model is unavailable: {re}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model is currently unavailable. Please check server logs."
            )
        logger.error(f"Inference runtime error: {re}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute property price prediction."
        )
    except ValueError as ve:
        logger.warning(f"Feature mismatch or invalid value: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred."
        )
