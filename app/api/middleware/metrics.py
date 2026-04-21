from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics for middleware
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently in progress',
    ['method']
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000]
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000]
)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request/response metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Track request size
        request_body_size = 0
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                request_body_size = len(body)
            except:
                pass
        
        # Increment in-progress counter
        http_requests_in_progress.labels(method=request.method).inc()
        
        # Start timing
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get endpoint path (without query params)
            endpoint = request.url.path
            
            # Record metrics
            http_requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            # Track response size
            response_body_size = 0
            if hasattr(response, 'body'):
                response_body_size = len(response.body) if response.body else 0
            elif hasattr(response, 'content'):
                response_body_size = len(response.content) if response.content else 0
            
            http_response_size_bytes.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(response_body_size)
            
            # Track request size
            if request_body_size > 0:
                http_request_size_bytes.labels(
                    method=request.method,
                    endpoint=endpoint
                ).observe(request_body_size)
            
            # Add metrics headers
            response.headers["X-Request-Duration-Ms"] = f"{duration * 1000:.2f}"
            response.headers["X-Request-Size-Bytes"] = str(request_body_size)
            response.headers["X-Response-Size-Bytes"] = str(response_body_size)
            
            # Log metrics for debugging (sampled)
            if duration > 1.0:  # Log slow requests
                logger.warning(
                    f"Slow request: {request.method} {endpoint} took {duration*1000:.2f}ms",
                    extra={
                        "method": request.method,
                        "endpoint": endpoint,
                        "duration_ms": duration * 1000,
                        "status": response.status_code,
                        "request_size": request_body_size,
                        "response_size": response_body_size
                    }
                )
            
            return response
            
        except Exception as e:
            # Record error metrics
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            logger.error(f"Request failed: {e}", exc_info=True)
            raise
            
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=request.method).dec()

class BusinessMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting business-specific metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Business metrics
        self.predictions_by_model = Counter(
            'predictions_by_model_total',
            'Total predictions by model version',
            ['model_version']
        )
        
        self.predictions_by_zipcode = Counter(
            'predictions_by_zipcode_total',
            'Total predictions by zipcode',
            ['zipcode']
        )
        
        self.price_distribution = Histogram(
            'predicted_price_distribution',
            'Distribution of predicted prices',
            buckets=[100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000, 5000000]
        )
        
        self.feature_usage = Counter(
            'feature_usage_total',
            'Feature usage count',
            ['feature']
        )
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Track business metrics for prediction endpoints
        if request.url.path == "/api/v1/predict" and request.method == "POST":
            # These metrics will be updated in the prediction handler
            # This middleware just captures the response
            pass
        
        return response

class PerformanceMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    async def dispatch(self, request: Request, call_next):
        # Track database query metrics if applicable
        start_db_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Here you could add database query tracking
        # db_duration = time.time() - start_db_time
        
        return response

def setup_metrics_middleware(app):
    """Setup all metrics middleware for FastAPI app"""
    
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(BusinessMetricsMiddleware)
    app.add_middleware(PerformanceMetricsMiddleware)
    
    logger.info("Metrics middleware configured")
    
    return app