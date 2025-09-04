# ABOUTME: Monitoring middleware for request tracking and metrics collection  
# ABOUTME: Automatically instruments all requests with timing and error metrics
import time
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import PrometheusMetrics

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic request monitoring and metrics collection.
    
    Tracks request duration, counts, and error rates automatically.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = PrometheusMetrics()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        # Skip metrics collection for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Extract route info
        method = request.method
        path = request.url.path
        
        # Get model variant from request if available
        model_variant = "unknown"
        try:
            # Try to extract from request body for generation endpoints
            if hasattr(request.state, "model_variant"):
                model_variant = request.state.model_variant
        except:
            pass
        
        # Start metrics timer
        timer = self.metrics.start_request_timer(method, path, model_variant)
        
        # Initialize response variables
        response = None
        status_code = 500  # Default to error in case of exception
        
        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            
            # Add timing header
            duration = time.time() - start_time
            response.headers["X-Process-Time"] = str(round(duration, 4))
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            # Record error metric
            self.metrics.record_error(type(e).__name__, f"{method} {path}")
            raise
            
        finally:
            # Always record metrics
            duration = time.time() - start_time
            
            # Record request completion
            self.metrics.record_request(method, path, status_code, model_variant)
            self.metrics.record_request_duration(method, path, status_code, duration, model_variant)
            
            # Log request (optional, can be controlled by config)
            logger.info(
                f"Request processed: {method} {path} - {status_code} - {duration:.4f}s",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration": duration,
                    "model_variant": model_variant
                }
            )


def add_monitoring_middleware(app: FastAPI) -> None:
    """
    Add monitoring middleware to FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(MonitoringMiddleware)
    logger.info("Monitoring middleware added to FastAPI app")


class WebSocketConnectionTracker:
    """
    Helper class for tracking WebSocket connections.
    
    Use this in WebSocket endpoints to automatically track connection metrics.
    """
    
    def __init__(self):
        self.metrics = PrometheusMetrics()
    
    async def __aenter__(self):
        """Connection opened"""
        self.metrics.increment_active_connections()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Connection closed"""
        self.metrics.decrement_active_connections()


async def track_websocket_connection():
    """
    Context manager for tracking WebSocket connections.
    
    Usage:
        async with track_websocket_connection():
            # WebSocket handling code
            pass
    """
    return WebSocketConnectionTracker()