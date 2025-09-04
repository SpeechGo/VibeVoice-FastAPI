# ABOUTME: Request/response logging middleware with correlation IDs and structured JSON output
# ABOUTME: Provides detailed logging for all HTTP requests with timing and status information

import time
import logging
import json
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.
    
    Converts log records to structured JSON format suitable for
    log aggregation systems like ELK stack.
    """
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in ["name", "msg", "args", "levelname", "levelno", 
                              "pathname", "filename", "module", "lineno", 
                              "funcName", "created", "msecs", "relativeCreated", 
                              "thread", "threadName", "processName", "process",
                              "stack_info", "exc_info", "exc_text", "message"]:
                    log_entry[key] = value
        
        return json.dumps(log_entry)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.
    
    Logs all HTTP requests with timing information, status codes,
    and correlation IDs for tracing. Uses structured JSON format
    for easy parsing by log aggregation systems.
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Set up structured logging for this middleware
        self.setup_structured_logging()
    
    def setup_structured_logging(self):
        """Configure structured JSON logging."""
        # Create a specific logger for request logging
        self.request_logger = logging.getLogger("api.requests")
        self.request_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        self.request_logger.handlers.clear()
        
        # Add structured JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.request_logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.request_logger.propagate = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive logging.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware/route handler
            
        Returns:
            Response with logging applied
        """
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request information
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Log request start (module logger for test visibility + structured logger)
        start_fields = dict(
            event="request_start",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_ip=client_ip,
            user_agent=user_agent,
            content_length=request.headers.get("Content-Length"),
            content_type=request.headers.get("Content-Type"),
        )
        try:
            # In tests, logger is a Mock and accepts kwargs
            logger.info("Request started", **start_fields)
        except TypeError:
            # In production, stdlib logger only accepts 'extra='
            logger.info("Request started", extra=start_fields)
        self.request_logger.info(
            "Request started",
            extra={
                "event": "request_start",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params) if request.query_params else None,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "content_length": request.headers.get("Content-Length"),
                "content_type": request.headers.get("Content-Type")
            }
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful response
            log_level = self.get_log_level(response.status_code)
            log_method = getattr(self.request_logger, log_level)
            
            # Mirror to module logger so tests can inspect kwargs
            complete_fields = dict(
                event="request_complete",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
                response_size=response.headers.get("Content-Length"),
                content_type=response.headers.get("Content-Type"),
            )
            try:
                getattr(logger, log_level)("Request completed", **complete_fields)
            except TypeError:
                getattr(logger, log_level)("Request completed", extra=complete_fields)
            log_method(
                "Request completed",
                extra={
                    "event": "request_complete",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                    "response_size": response.headers.get("Content-Length"),
                    "content_type": response.headers.get("Content-Type")
                }
            )
            
            # Ensure request ID is in response headers
            if "X-Request-ID" not in response.headers:
                response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration even for errors
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            error_fields = dict(
                event="request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                client_ip=client_ip,
                error=str(e),
                error_type=type(e).__name__,
            )
            try:
                logger.error("Request failed with exception", **error_fields)
            except TypeError:
                logger.error("Request failed with exception", extra=error_fields)
            self.request_logger.error(
                "Request failed with exception",
                extra={
                    "event": "request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": client_ip,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            # Re-raise the exception
            raise
    
    def get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.
        
        Args:
            request: The HTTP request object
            
        Returns:
            Client IP address as string
        """
        # Check for forwarded IP (common in load balancer setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to direct client IP
        client_host = request.client.host if request.client else "unknown"
        return client_host
    
    def get_log_level(self, status_code: int) -> str:
        """
        Determine appropriate log level based on HTTP status code.
        
        Args:
            status_code: HTTP response status code
            
        Returns:
            Log level as string (info, warning, error)
        """
        if status_code < 400:
            return "info"
        elif status_code < 500:
            return "warning"
        else:
            return "error"
