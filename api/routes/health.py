# ABOUTME: Health check API routes
# ABOUTME: Implements /healthz and /readyz endpoints for service monitoring
from fastapi import APIRouter, Depends
import torch

from api.dependencies import get_voice_service
from api.core.voice_service import VoiceService
from api.models.responses import HealthStatus, ReadinessStatus
from api.config import get_settings

router = APIRouter()

@router.get("/healthz", response_model=HealthStatus)
async def health_check():
    """
    Basic health check - service is running
    """
    return HealthStatus(status="ok")

@router.get("/readyz", response_model=ReadinessStatus)
async def readiness_check(
    voice_service: VoiceService = Depends(get_voice_service)
):
    """
    Readiness check - service is ready to handle requests
    
    Returns detailed status including model loading and device info
    """
    settings = get_settings()
    
    return ReadinessStatus(
        ready=voice_service.ready(),
        device=settings.device,
        dtype=settings.dtype,
        model_loaded=voice_service.ready(),
        max_concurrency=settings.max_concurrency,
        current_concurrency=getattr(voice_service, '_current_concurrency', 0)
    )