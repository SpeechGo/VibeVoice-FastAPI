# ABOUTME: Rate limiting middleware for FastAPI with per-IP limits and configurable thresholds
# ABOUTME: Uses in-memory storage for rate limiting with bypass for health check endpoints

import time
import logging
from typing import Callable, Dict, Optional
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from api.models.responses import ErrorResponse

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Checks X-Forwarded-For header first, then falls back to client IP.
    
    Args:
        request: The HTTP request object
        
    Returns:
        Client IP address as string
    """
    # Check for forwarded IP (common in load balancer setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in case of multiple proxies
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header (another common proxy header)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client IP
    client_host = request.client.host if request.client else "unknown"
    return client_host


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window approach.
    
    For production use, this should be replaced with Redis-based storage
    to support distributed deployments.
    """
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, key: str, limit: int) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Rate limit key (typically IP address)
            limit: Number of requests allowed per window
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Clean old requests outside the window
        request_times = self.requests[key]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if we're under the limit
        if len(request_times) < limit:
            # Add current request to the window
            request_times.append(current_time)
            return True
        
        return False
    
    def get_reset_time(self, key: str) -> Optional[float]:
        """
        Get the time when the rate limit resets for the given key.
        
        Args:
            key: Rate limit key
            
        Returns:
            Unix timestamp when limit resets, or None if no requests recorded
        """
        request_times = self.requests.get(key)
        if not request_times:
            return None
        
        # Reset time is when the oldest request in the window expires
        return request_times[0] + self.window_seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with per-IP limits and configurable thresholds.
    
    Provides protection against abuse while exempting health check endpoints.
    Uses sliding window rate limiting for more accurate rate control.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        redis_url: Optional[str] = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.exempt_paths = {"/healthz", "/readyz", "/health", "/ready"}
        
        # For now, use in-memory rate limiter
        # In production, this should use Redis for distributed rate limiting
        if redis_url:
            logger.warning("Redis-based rate limiting not implemented yet, using in-memory")
        
        self.rate_limiter = InMemoryRateLimiter(window_seconds=60)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting enforcement.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware/route handler
            
        Returns:
            Response with rate limiting applied or rate limit error
        """
        # Check if this path should be exempt from rate limiting
        if request.url.path in self.exempt_paths:
            logger.debug(f"Exempting {request.url.path} from rate limiting")
            return await call_next(request)
        
        # Get client IP and request ID
        client_ip = get_client_ip(request)
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Check rate limit
        rate_limit_key = f"ip:{client_ip}"
        
        if not self.rate_limiter.is_allowed(rate_limit_key, self.requests_per_minute):
            # Rate limit exceeded
            reset_time = self.rate_limiter.get_reset_time(rate_limit_key)
            
            logger.warning(
                f"Rate limit exceeded for IP {client_ip}",
                extra={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "rate_limit": self.requests_per_minute,
                    "reset_time": reset_time
                }
            )
            
            # Create rate limit error response
            error_response = ErrorResponse(
                code="RATE_LIMIT_EXCEEDED",
                message=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                details={
                    "limit": self.requests_per_minute,
                    "window": "60 seconds",
                    "reset_time": reset_time
                }
            )
            
            # Compute Retry-After (clamp to at least 1 second)
            retry_after = 60
            if reset_time:
                delta = int(reset_time - time.time())
                retry_after = str(max(1, delta))

            headers = {
                "X-Request-ID": request_id,
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Window": "60",
                "Retry-After": str(retry_after)
            }
            
            return JSONResponse(
                status_code=429,
                content=error_response.model_dump(),
                headers=headers
            )
        
        # Rate limit passed, continue with request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Window"] = "60"
        
        return response
