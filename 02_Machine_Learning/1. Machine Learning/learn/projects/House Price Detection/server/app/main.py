import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.ml.model_loader import load_model, load_metadata
from app.api import api_v1_router, api_router

# Configure root logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the ML model and metadata once into memory during startup.
    Ensures the server fails immediately if model artifacts cannot be loaded.
    """
    logger.info("==========================================================")
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("==========================================================")

    # 1. Startup validation: Load Model
    try:
        load_model(settings.MODEL_PATH)
    except Exception as e:
        logger.critical(f"FATAL: Application startup aborted. Could not load model: {e}")
        raise e

    # 2. Load Model Metadata
    try:
        load_metadata(settings.MODEL_METADATA_PATH)
    except Exception as e:
        logger.warning(f"Could not load metadata, proceeding with defaults: {e}")

    logger.info("Startup complete. Application is ready to receive inference requests.")
    yield
    logger.info("Shutting down application...")


# FastAPI Application instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade FastAPI service delivering residential property valuations "
        "across Indian States, Districts, and Micro-markets using a Scikit-Learn Gradient Boosting pipeline."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include versioned and direct API routers
app.include_router(api_v1_router)
app.include_router(api_router)


@app.get(
    "/",
    tags=["Root"],
    summary="Root Service Endpoint",
    description="Returns welcome message and API version."
)
def root():
    return {
        "message": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
