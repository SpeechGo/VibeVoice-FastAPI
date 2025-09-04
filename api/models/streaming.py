# ABOUTME: This file defines Pydantic models for WebSocket streaming protocol messages.
# ABOUTME: These models ensure consistent WebSocket message structure per the Public Interface contract.

from typing import Literal, Optional
from pydantic import BaseModel


class StreamingInit(BaseModel):
    """Initial message sent when WebSocket streaming starts."""
    type: Literal['init'] = 'init'
    request_id: str
    sample_rate: int
    num_channels: int = 1
    format: Literal['pcm16'] = 'pcm16'


class StreamingProgress(BaseModel):
    """Progress update message sent during WebSocket streaming."""
    type: Literal['progress'] = 'progress'
    request_id: str
    chunk_index: int
    ms_emitted: int


class StreamingFinal(BaseModel):
    """Final message sent when WebSocket streaming completes successfully."""
    type: Literal['final'] = 'final'
    request_id: str
    total_ms: int


class StreamingError(BaseModel):
    """Error message sent when WebSocket streaming encounters an error."""
    type: Literal['error'] = 'error'
    request_id: str
    code: str
    message: str