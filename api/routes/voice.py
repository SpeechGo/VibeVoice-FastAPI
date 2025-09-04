# ABOUTME: Voice generation API routes
# ABOUTME: Implements /api/generate POST and /api/voices GET endpoints with content negotiation
import time
from typing import Union
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import Response

from api.dependencies import get_voice_service
from api.core.voice_service import VoiceService
from api.models.requests import VoiceGenerationRequest
from api.models.responses import VoiceListResponse, GenerateJsonResponse, VoiceInfo
from api.utils.streaming import negotiate_accept
from api.utils.audio import wav_to_base64

router = APIRouter()

@router.post("/generate", response_model=None)
async def generate_voice(
    request: Request,
    voice_request: VoiceGenerationRequest,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """
    Generate text-to-speech audio
    
    Content negotiation based on Accept header:
    - audio/wav: Returns WAV binary data
    - application/json: Returns JSON with base64-encoded audio
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Content negotiation
    try:
        # Default to JSON when Accept is */*
        content_type = negotiate_accept(request, ["application/json", "audio/wav"])
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e))
    
    # Generate audio
    result = await voice_service.generate_blocking(voice_request)
    generation_time = time.time() - start_time
    
    if content_type == "audio/wav":
        # Return WAV bytes directly
        return Response(
            content=result.wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Request-ID": request_id,
                "X-Generation-Time": str(generation_time),
            }
        )
    else:
        # Return JSON with base64 audio
        audio_base64 = wav_to_base64(result.wav_bytes)
        
        return GenerateJsonResponse(
            audio_base64=audio_base64,
            duration_sec=result.duration_sec,
            sample_rate=result.sample_rate,
            num_channels=1,
            format="wav",
            request_id=request_id,
            generation_time_sec=generation_time
        )

@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(
    include_hidden: bool = False,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """
    List available voice presets
    
    Args:
        include_hidden: Whether to include hidden voices (not used in v1)
    """
    voices = voice_service.list_voices(include_hidden=include_hidden)
    return VoiceListResponse(voices=voices)
