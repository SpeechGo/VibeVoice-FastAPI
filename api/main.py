# ABOUTME: FastAPI application instance with middleware and exception handlers
# ABOUTME: Main entry point with CORS, lifespan management, monitoring, and error handling
from contextlib import asynccontextmanager
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.middleware import (
    TimeoutMiddleware,
    RateLimitMiddleware, 
    LoggingMiddleware,
    EnhancedCORSMiddleware
)
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import get_settings
from api.models.errors import (
    ServiceBusyError, 
    ModelNotReadyError, 
    GenerationTimeoutError, 
    InvalidVoiceError
)
from api.models.responses import ErrorResponse
from api.core.voice_service import VoiceService

# Import monitoring system
from api.monitoring.middleware import add_monitoring_middleware
from api.monitoring.background_tasks import start_monitoring_tasks, stop_monitoring_tasks
from api.monitoring.metrics import PrometheusMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting VibeVoice FastAPI service")
    
    monitoring_manager = None
    
    try:
        # Initialize monitoring system
        logger.info("Initializing monitoring system")
        metrics = PrometheusMetrics()
        logger.info("Prometheus metrics initialized")
        
        # Start monitoring background tasks
        monitoring_manager = await start_monitoring_tasks(gpu_monitoring_interval=10.0)
        logger.info("Monitoring background tasks started")
        
        # Initialize the voice service (model loads in constructor)
        voice_service = VoiceService.instance()
        if voice_service.ready():
            logger.info("Voice service initialized and ready")
        else:
            logger.warning("Voice service initialized but model not ready")
        
        logger.info("VibeVoice FastAPI service startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start service components: {e}")
        # Continue anyway - service will report not ready where appropriate
    
    yield
    
    # Shutdown
    logger.info("Shutting down VibeVoice FastAPI service")
    
    try:
        # Stop monitoring tasks
        if monitoring_manager:
            await stop_monitoring_tasks()
            logger.info("Monitoring tasks stopped")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("VibeVoice FastAPI service shutdown completed")

# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="VibeVoice API",
    description="Text-to-speech API using VibeVoice model",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware stack - Applied in reverse order of desired execution
# Order: CORS -> Monitoring -> Logging -> Rate Limiting -> Timeout -> Request Processing

# Enhanced CORS middleware (outermost - first to see request, last to see response)  
app.add_middleware(EnhancedCORSMiddleware)

# Add monitoring middleware
add_monitoring_middleware(app)

# Logging middleware (logs all requests/responses with correlation IDs)
app.add_middleware(LoggingMiddleware)

# Rate limiting middleware (applied after logging so rate limits are logged)
app.add_middleware(
    RateLimitMiddleware, 
    requests_per_minute=settings.rate_limit_requests_per_minute
)

# Timeout middleware (innermost - closest to request processing)
app.add_middleware(
    TimeoutMiddleware,
    timeout_seconds=float(settings.timeout_sec)
)

# Note: Request ID generation is now handled by LoggingMiddleware

# Exception handlers
@app.exception_handler(ServiceBusyError)
async def service_busy_handler(request: Request, exc: ServiceBusyError):
    """Handle service busy errors (429)"""
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            code="SERVICE_BUSY",
            message="Service is currently at capacity. Please try again later.",
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    """Handle model not ready errors (503)"""
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            code="MODEL_NOT_READY",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

@app.exception_handler(GenerationTimeoutError)
async def generation_timeout_handler(request: Request, exc: GenerationTimeoutError):
    """Handle generation timeout errors (408)"""
    return JSONResponse(
        status_code=408,
        content=ErrorResponse(
            code="GENERATION_TIMEOUT",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

@app.exception_handler(InvalidVoiceError)
async def invalid_voice_handler(request: Request, exc: InvalidVoiceError):
    """Handle invalid voice errors (422)"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="INVALID_VOICE",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle general HTTP exceptions"""
    # Map specific status codes to appropriate error codes
    if exc.status_code == 406:
        error_code = "NOT_ACCEPTABLE"
    else:
        error_code = "HTTP_ERROR"
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=error_code,
            message=str(exc.detail),
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors as 406 Not Acceptable"""
    return JSONResponse(
        status_code=406,
        content=ErrorResponse(
            code="NOT_ACCEPTABLE",
            message=str(exc),
        ).model_dump(),
        headers={"X-Request-ID": getattr(request.state, "request_id", "")}
    )

# Include routers
from api.routes import voice, health, metrics
from api.routes.websocket import websocket_generate_endpoint

app.include_router(voice.router, prefix="/api", tags=["voice"])
app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, tags=["monitoring"])

# Add WebSocket endpoint
app.websocket("/ws/generate")(websocket_generate_endpoint)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve the main UI at root
@app.get("/")
async def serve_index():
    """Serve the main web UI"""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return {"message": "VibeVoice API", "status": "Web UI not available"}