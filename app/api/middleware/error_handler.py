from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging
from typing import Union

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware:
    """Global error handling middleware"""
    
    async def __call__(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return await self.handle_exception(request, exc)
    
    async def handle_exception(self, request: Request, exc: Exception):
        """Handle different exception types"""
        
        if isinstance(exc, RequestValidationError):
            return await self._handle_validation_error(request, exc)
        elif isinstance(exc, StarletteHTTPException):
            return await self._handle_http_error(request, exc)
        else:
            return await self._handle_generic_error(request, exc)
    
    async def _handle_validation_error(self, request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        logger.warning(f"Validation error: {errors}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Validation error",
                "errors": errors,
                "path": request.url.path,
                "method": request.method
            }
        )
    
    async def _handle_http_error(self, request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        
        logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method
            }
        )
    
    async def _handle_generic_error(self, request: Request, exc: Exception):
        """Handle unexpected errors"""
        
        error_id = generate_error_id()
        
        logger.error(
            f"Unhandled error {error_id}: {str(exc)}",
            exc_info=True,
            extra={
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error_id": error_id,
                "path": request.url.path,
                "method": request.method
            }
        )

def generate_error_id() -> str:
    """Generate unique error ID"""
    import uuid
    return str(uuid.uuid4())[:8]

def register_error_handlers(app):
    """Register error handlers with FastAPI app"""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return await ErrorHandlerMiddleware()._handle_validation_error(request, exc)
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return await ErrorHandlerMiddleware()._handle_http_error(request, exc)
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        return await ErrorHandlerMiddleware()._handle_generic_error(request, exc)