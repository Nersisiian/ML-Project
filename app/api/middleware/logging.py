from fastapi import Request
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LoggingMiddleware:
    """Request/response logging middleware"""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        await self.log_request(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        await self.log_response(request, response, duration)
        
        # Add headers
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", generate_request_id())
        
        return response
    
    async def log_request(self, request: Request):
        """Log incoming request"""
        
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_body = body[:1000].decode()  # Limit size
            except:
                pass
        
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "request_body": request_body,
                "request_id": request.headers.get("X-Request-ID")
            }
        )
    
    async def log_response(self, request: Request, response, duration: float):
        """Log outgoing response"""
        
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration * 1000,
            "client_ip": request.client.host if request.client else None,
            "request_id": request.headers.get("X-Request-ID")
        }
        
        if response.status_code >= 500:
            logger.error(f"Response error: {response.status_code}", extra=log_data)
        elif response.status_code >= 400:
            logger.warning(f"Response client error: {response.status_code}", extra=log_data)
        else:
            logger.info(f"Response: {response.status_code}", extra=log_data)

def generate_request_id() -> str:
    """Generate unique request ID"""
    import uuid
    return str(uuid.uuid4())