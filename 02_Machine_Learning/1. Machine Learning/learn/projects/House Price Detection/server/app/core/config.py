from typing import List, Union, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    Paths are loaded directly as strings from .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core Application Settings
    APP_NAME: str = "Indian House Price Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    ALLOWED_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]

    # Machine Learning Model Artifact Paths (Direct strings from .env)
    MODEL_PATH: str = "app/model/house_price_model.joblib"
    MODEL_METADATA_PATH: str = "app/model/model_metadata.json"

    # Location and Dataset Paths (Direct strings from .env, zero os/pathlib usage)
    STATES_DATA_PATH: str = "app/data/locations/states.json"
    DISTRICTS_DATA_PATH: str = "app/data/locations/districts.json"
    SUBDISTRICTS_DATA_PATH: str = "app/data/locations/subdistricts.json"
    VILLAGES_DATA_PATH: str = "app/data/locations/villages.json"
    LOCATION_STATISTICS_PATH: str = "app/data/locations/location_statistics.json"
    DATASET_DIR: str = "app/dataset"

    # Optional External Maps Geocoding API Keys
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
