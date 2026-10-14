from fastapi import APIRouter, status
import app.ml.model_loader as ml_loader

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns service health status and confirms whether the ML pipeline is loaded into memory."
)
def health_check():
    is_loaded = ml_loader._model_instance is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded
    }
