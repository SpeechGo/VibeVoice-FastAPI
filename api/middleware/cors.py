# ABOUTME: Enhanced CORS middleware with environment-driven configuration and WebSocket support
# ABOUTME: Provides secure CORS handling with configurable origins and proper WebSocket headers

import os
import logging
from typing import List, Optional, Sequence
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class EnhancedCORSMiddleware:
    """
    Enhanced CORS middleware wrapper that provides environment-driven configuration.
    
    This is a wrapper around Starlette's CORSMiddleware that adds:
    - Environment variable-based origin configuration
    - WebSocket support headers
    - Enhanced logging
    - Security-focused defaults
    """
    
    def __init__(
        self,
        app: ASGIApp,
        allow_origins: Optional[Sequence[str]] = None,
        allow_methods: Optional[Sequence[str]] = None,
        allow_headers: Optional[Sequence[str]] = None,
        allow_credentials: bool = True,
        allow_origin_regex: Optional[str] = None,
        expose_headers: Optional[Sequence[str]] = None,
        max_age: int = 600,
    ):
        # Get origins from environment if not provided
        if allow_origins is None:
            origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
            if origins_env:
                allow_origins = [origin.strip() for origin in origins_env.split(",")]
                logger.info(f"CORS origins from environment: {allow_origins}")
            else:
                # Default to restrictive policy - no wildcard in production
                allow_origins = []
                logger.warning("No CORS origins configured - using restrictive policy")
        
        # Enhanced method list for API and WebSocket support
        if allow_methods is None:
            allow_methods = [
                "DELETE",
                "GET", 
                "OPTIONS",
                "PATCH",
                "POST",
                "PUT"
            ]
        
        # Enhanced headers for WebSocket and API support
        if allow_headers is None:
            allow_headers = [
                "Accept",
                "Accept-Language",
                "Authorization",
                "Content-Language", 
                "Content-Type",
                "X-Request-ID",
                # WebSocket specific headers
                "Connection",
                "Upgrade", 
                "Sec-WebSocket-Key",
                "Sec-WebSocket-Version",
                "Sec-WebSocket-Protocol",
                "Sec-WebSocket-Extensions"
            ]
        
        # Headers to expose to the client
        if expose_headers is None:
            expose_headers = [
                "X-Request-ID",
                "X-RateLimit-Limit",
                "X-RateLimit-Window", 
                "X-Generation-Time"
            ]
        
        # Apply CORS middleware with enhanced configuration
        self.cors_middleware = CORSMiddleware(
            app,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
            allow_origin_regex=allow_origin_regex,
            expose_headers=expose_headers,
            max_age=max_age,
        )
        
        # Log configuration for security audit
        logger.info(
            "CORS middleware configured",
            extra={
                "allow_origins": list(allow_origins) if allow_origins else [],
                "allow_credentials": allow_credentials,
                "allow_methods": list(allow_methods),
                "max_age": max_age,
                "websocket_support": True
            }
        )
        
        # Security warning for wildcard origins
        if allow_origins and "*" in allow_origins:
            logger.warning(
                "CORS configured with wildcard origin - this is insecure for production!"
            )
    
    async def __call__(self, scope, receive, send):
        """ASGI application interface."""
        return await self.cors_middleware(scope, receive, send)


def setup_cors_middleware(app: ASGIApp) -> ASGIApp:
    """
    Convenience function to set up enhanced CORS middleware.
    
    Args:
        app: The ASGI application to wrap
        
    Returns:
        App wrapped with enhanced CORS middleware
    """
    return EnhancedCORSMiddleware(app)


def get_cors_origins_from_env() -> List[str]:
    """
    Parse CORS origins from environment variable.
    
    Returns:
        List of allowed origins from CORS_ALLOW_ORIGINS environment variable
    """
    origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
    if not origins_env:
        return []
    
    # Split by comma and clean whitespace
    origins = [origin.strip() for origin in origins_env.split(",")]
    
    # Filter out empty strings
    origins = [origin for origin in origins if origin]
    
    logger.debug(f"Parsed CORS origins from environment: {origins}")
    return origins


def validate_cors_origins(origins: List[str]) -> List[str]:
    """
    Validate and sanitize CORS origins.
    
    Args:
        origins: List of origin URLs to validate
        
    Returns:
        List of validated origins
    """
    validated = []
    
    for origin in origins:
        # Basic URL validation
        if not origin:
            continue
        
        # Check for wildcard
        if origin == "*":
            logger.warning("Wildcard CORS origin detected - use with caution!")
            validated.append(origin)
            continue
        
        # Basic URL format check
        if not (origin.startswith("http://") or origin.startswith("https://")):
            logger.warning(f"Invalid CORS origin format: {origin}")
            continue
        
        validated.append(origin)
    
    return validated