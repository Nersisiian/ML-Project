"""Application constants"""

from enum import Enum
from typing import Dict, Any

# Model constants
MODEL_FEATURES = [
    'square_feet',
    'bedrooms', 
    'bathrooms',
    'year_built',
    'lot_size',
    'pool',
    'fireplace',
    'garage_spaces',
    'condition_score',
    'property_age',
    'property_age_squared',
    'sqft_per_bedroom',
    'bath_bedroom_ratio',
    'log_sqft',
    'is_new_construction',
    'is_luxury'
]

# API constants
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Rate limiting
DEFAULT_RATE_LIMIT = 100
DEFAULT_RATE_WINDOW = 60

# Cache constants
CACHE_TTL = {
    'prediction': 3600,  # 1 hour
    'feature': 86400,    # 24 hours
    'model': 300         # 5 minutes
}

# Model stages
class ModelStage(str, Enum):
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

# HTTP status codes
class HTTPStatus:
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    RATE_LIMIT = 429
    INTERNAL_ERROR = 500

# Error messages
ERROR_MESSAGES: Dict[int, str] = {
    HTTPStatus.BAD_REQUEST: "Invalid request parameters",
    HTTPStatus.UNAUTHORIZED: "Authentication required",
    HTTPStatus.FORBIDDEN: "Access forbidden",
    HTTPStatus.NOT_FOUND: "Resource not found",
    HTTPStatus.RATE_LIMIT: "Rate limit exceeded",
    HTTPStatus.INTERNAL_ERROR: "Internal server error"
}

# Feature groups
FEATURE_GROUPS = {
    'property': ['square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size'],
    'amenities': ['pool', 'fireplace', 'garage_spaces', 'condition_score'],
    'derived': ['property_age', 'property_age_squared', 'sqft_per_bedroom'],
    'temporal': ['month_sin', 'month_cos', 'dow_sin', 'dow_cos']
}

# Monitoring thresholds
MONITORING_THRESHOLDS = {
    'max_latency_ms': 500,
    'max_error_rate': 0.05,
    'min_accuracy': 0.85,
    'max_drift_score': 0.2,
    'min_cache_hit_ratio': 0.3
}

# Data validation rules
VALIDATION_RULES = {
    'required_columns': MODEL_FEATURES + ['price'],
    'unique_columns': ['property_id'],
    'null_threshold': 0.3,
    'outlier_multiplier': 1.5
}