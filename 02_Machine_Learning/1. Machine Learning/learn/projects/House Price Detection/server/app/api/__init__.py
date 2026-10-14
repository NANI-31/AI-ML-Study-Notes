from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.locations import router as locations_router
from app.api.routes.map import router as map_router

# Standard v1 router (/api/v1/...)
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(prediction_router)
api_v1_router.include_router(locations_router)
api_v1_router.include_router(map_router)

# Direct API router (/api/...) for flexible client compatibility
api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(prediction_router)
api_router.include_router(locations_router)
api_router.include_router(map_router)
