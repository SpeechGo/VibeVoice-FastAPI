# ABOUTME: Structured logging configuration for VibeVoice-FastAPI with request ID tracking  
# ABOUTME: Provides JSON logging, performance metrics, and context management per contract

import logging
import logging.handlers
import sys
from contextlib import contextmanager
from typing import Optional, Dict, Any
import json
import time

import structlog


# Global context storage for request IDs
_context_storage = {}


def configure_logging(
    log_level: str = "info",
    log_file: Optional[str] = None,
    enable_json: bool = True
) -> None:
    """
    Configure structured logging for the VibeVoice-FastAPI service.
    
    Args:
        log_level: Logging level (debug, info, warning, error, critical)
        log_file: Optional file path for log output (defaults to stdout)
        enable_json: Whether to use JSON formatting (default True)
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing configuration
    structlog.reset_defaults()
    logging.getLogger().handlers.clear()
    
    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    # Set up handler first
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)
    
    handler.setLevel(numeric_level)
    
    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)
    
    # Configure structlog processors  
    if enable_json:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                timestamper,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                timestamper,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


@contextmanager
def request_id_context(request_id: str):
    """
    Context manager for request ID tracking in logs.
    
    Args:
        request_id: Unique identifier for the request
    """
    # Store the request ID in context variables
    token = structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield
    finally:
        # Clear the context
        token.reset()


def log_request_metrics(
    method: str,
    path: str, 
    status_code: int,
    duration_ms: float,
    request_id: str,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log request performance metrics.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_id: Unique request identifier
        additional_data: Optional additional metrics data
    """
    logger = structlog.get_logger("api.metrics")
    
    metrics_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "request_id": request_id
    }
    
    if additional_data:
        metrics_data.update(additional_data)
    
    logger.info("request_completed", **metrics_data)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class StructuredLoggerAdapter:
    """
    Adapter to provide structured logging interface compatible with existing code.
    """
    
    def __init__(self, logger_name: str):
        self.logger = structlog.get_logger(logger_name)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log an exception with full traceback."""
        self.logger.error(message, exc_info=True, **kwargs)