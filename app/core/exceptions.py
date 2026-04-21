from fastapi import HTTPException, status

class AppException(HTTPException):
    """Base application exception"""
    def __init__(self, status_code: int, detail: str, headers: dict = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class ModelNotLoadedError(AppException):
    """Model not loaded error"""
    def __init__(self, detail: str = "Model not loaded"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

class ValidationError(AppException):
    """Input validation error"""
    def __init__(self, detail: str = "Validation failed"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

class RateLimitError(AppException):
    """Rate limit exceeded error"""
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

class NotFoundError(AppException):
    """Resource not found error"""
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)