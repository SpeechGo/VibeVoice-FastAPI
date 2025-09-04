# ABOUTME: Request timeout middleware for FastAPI with configurable timeout duration
# ABOUTME: Provides 5-minute hard timeout for generation requests while exempting health checks

import asyncio
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from api.models.responses import ErrorResponse

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeout limits.
    
    Provides configurable timeout for requests while exempting health check endpoints.
    Default timeout is 5 minutes (300 seconds) for generation requests.
    """
    
    def __init__(self, app, timeout_seconds: float = 300.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.exempt_paths = {"/healthz", "/readyz", "/health", "/ready"}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with timeout enforcement.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware/route handler
            
        Returns:
            Response with timeout enforcement applied
        """
        # Check if this path should be exempt from timeout
        if request.url.path in self.exempt_paths:
            logger.debug(f"Exempting {request.url.path} from timeout")
            return await call_next(request)
        
        # Get request ID for logging and echo; fall back to incoming header
        request_id = (
            getattr(request.state, "request_id", None)
            or request.headers.get("X-Request-ID")
            or "unknown"
        )
        
        try:
            # Apply timeout to the request processing
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
            
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout after {self.timeout_seconds}s",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "timeout_seconds": self.timeout_seconds
                }
            )
            
            # Return timeout error response
            error_response = ErrorResponse(
                code="REQUEST_TIMEOUT",
                message=f"Request timed out after {self.timeout_seconds} seconds"
            )
            
            return JSONResponse(
                status_code=408,
                content=error_response.model_dump(),
                headers={"X-Request-ID": request_id}
            )
        
        except Exception as e:
            logger.error(
                f"Unexpected error in timeout middleware: {e}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e)
                }
            )
            # Re-raise to let other error handlers deal with it
            raise
