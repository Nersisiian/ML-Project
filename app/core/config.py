from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "real-estate-price-predictor"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    API_RATE_LIMIT: int = 100  # requests per minute
    API_RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Database
    REDIS_URL: str = "redis://localhost:6379/0"
    POSTGRES_URL: Optional[str] = None
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MODEL_NAME: str = "real_estate_predictor"
    MODEL_STAGE: str = "Production"
    MODEL_VERSION: str = "latest"
    
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-west-2"
    MODEL_BUCKET: str = "s3://real-estate-models"
    
    # Feature Store
    FEAST_SERVER_URL: str = "http://localhost:6566"
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()