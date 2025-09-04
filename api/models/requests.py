# ABOUTME: This file defines Pydantic models for API request payloads.
# ABOUTME: These models enforce validation and structure per the Public Interface contract.

from typing import List, Optional, Literal, Annotated
from pydantic import BaseModel, Field, StringConstraints

Script = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class VoiceGenerationRequest(BaseModel):
    script: Script
    speakers: Annotated[List[str], Field(min_length=1, max_length=4)]
    cfg_scale: float = Field(1.3, ge=0.1, le=5.0)
    inference_steps: int = Field(5, ge=1, le=64)
    sample_rate: int = Field(24000, description="Target sample rate in Hz")
    format: Literal['wav'] = 'wav'  # v1 scope
    seed: Optional[int] = Field(None, ge=0)


class VoiceListRequest(BaseModel):
    include_hidden: bool = False