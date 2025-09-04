# ABOUTME: This file defines Pydantic models for API response payloads.
# ABOUTME: These models ensure consistent response structure per the Public Interface contract.

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel


class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: Optional[str] = None
    language: Optional[str] = None


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]


class GenerateJsonResponse(BaseModel):
    audio_base64: str
    duration_sec: float
    sample_rate: int
    num_channels: int = 1
    format: Literal['wav'] = 'wav'
    request_id: str
    generation_time_sec: float


class HealthStatus(BaseModel):
    status: Literal['ok','error']


class ReadinessStatus(BaseModel):
    ready: bool
    device: str
    dtype: str
    model_loaded: bool
    max_concurrency: int
    current_concurrency: int


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None