# ABOUTME: This file implements the WebSocket streaming endpoint for real-time audio generation.
# ABOUTME: Handles WebSocket protocol messages, binary PCM16 frames, error handling, and proper close codes.

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect, Depends
from pydantic import ValidationError

from api.core.connection_manager import connection_manager
from api.core.voice_service import VoiceService
from api.dependencies import get_voice_service
from api.models.requests import VoiceGenerationRequest
from api.models.streaming import StreamingInit, StreamingProgress, StreamingFinal, StreamingError
from api.models.errors import (
    ServiceBusyError,
    ModelNotReadyError,
    GenerationTimeoutError,
    InvalidVoiceError
)

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Handles WebSocket streaming audio generation protocol."""
    
    def __init__(self, voice_service: VoiceService):
        self.voice_service = voice_service
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle a complete WebSocket connection lifecycle.
        
        Args:
            websocket: The WebSocket connection to handle
        """
        connection_id = await connection_manager.connect(websocket)
        
        try:
            await self._handle_generation_request(websocket, connection_id)
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket handler {connection_id}: {e}")
            await self._send_error(
                connection_id,
                "internal_error",
                f"Internal server error: {str(e)}"
            )
        finally:
            await connection_manager.disconnect(connection_id)
    
    async def _handle_generation_request(self, websocket: WebSocket, connection_id: str):
        """Handle a single generation request over WebSocket.
        
        Args:
            websocket: The WebSocket connection
            connection_id: The connection ID for this WebSocket
        """
        # Receive the request JSON
        try:
            request_data = await websocket.receive_json()
        except Exception as e:
            await self._send_error(
                connection_id,
                "invalid_request",
                f"Failed to parse JSON request: {str(e)}"
            )
            return
        
        # Add request_id if not provided
        if "request_id" not in request_data:
            request_data["request_id"] = str(uuid.uuid4())
        
        request_id = request_data["request_id"]
        
        # Validate request
        try:
            # Remove request_id for validation since it's not part of VoiceGenerationRequest
            validation_data = request_data.copy()
            validation_data.pop("request_id", None)
            request = VoiceGenerationRequest(**validation_data)
        except ValidationError as e:
            await self._send_error(
                connection_id,
                "validation_error",
                f"Invalid request: {str(e)}",
                request_id
            )
            return
        
        # Check service readiness
        if not self.voice_service.ready():
            await self._send_error(
                connection_id,
                "service_not_ready",
                "Voice service is not ready",
                request_id
            )
            return
        
        # Start streaming generation
        await self._stream_generation(connection_id, request, request_id)
    
    async def _stream_generation(self, connection_id: str, request: VoiceGenerationRequest, request_id: str):
        """Stream audio generation for a request.
        
        Args:
            connection_id: The WebSocket connection ID
            request: The validated generation request
            request_id: The request ID for tracking
        """
        # Send initial message
        init_message = StreamingInit(
            request_id=request_id,
            sample_rate=request.sample_rate,
            num_channels=1,
            format="pcm16"
        )
        
        await connection_manager.send_text(connection_id, init_message.model_dump_json())
        
        # Set up cancellation tracking
        cancelled = False
        
        def cancel_check() -> bool:
            return cancelled or not connection_manager.is_connected(connection_id)
        
        def cancel_callback():
            nonlocal cancelled
            cancelled = True
        
        # Track generation start
        connection_manager.start_generation(connection_id, cancel_callback)
        
        try:
            # Stream PCM16 chunks
            chunk_index = 0
            total_samples = 0
            start_time = asyncio.get_event_loop().time()
            
            async for pcm_chunk in self.voice_service.stream_pcm16(
                request, 
                cancel_check=cancel_check,
                timeout_sec=300
            ):
                if cancel_check():
                    logger.info(f"Generation cancelled for request {request_id}")
                    break
                
                # Send binary PCM16 frame
                await connection_manager.send_bytes(connection_id, pcm_chunk)
                
                # Calculate progress info
                chunk_samples = len(pcm_chunk) // 2  # PCM16 = 2 bytes per sample
                total_samples += chunk_samples
                ms_emitted = int((total_samples / request.sample_rate) * 1000)
                
                # Send progress update every few chunks
                if chunk_index % 5 == 0:  # Every 5th chunk
                    progress_message = StreamingProgress(
                        request_id=request_id,
                        chunk_index=chunk_index,
                        ms_emitted=ms_emitted
                    )
                    await connection_manager.send_text(connection_id, progress_message.model_dump_json())
                
                chunk_index += 1
            
            # Send final message if not cancelled
            if not cancel_check():
                total_ms = int((total_samples / request.sample_rate) * 1000)
                final_message = StreamingFinal(
                    request_id=request_id,
                    total_ms=total_ms
                )
                
                await connection_manager.send_text(connection_id, final_message.model_dump_json())
                await connection_manager.disconnect(connection_id, code=1000)  # Normal close
            else:
                # Cancelled - close with going away code
                await connection_manager.disconnect(connection_id, code=1001)  # Going away
        
        except ServiceBusyError as e:
            await self._send_error(
                connection_id,
                "service_busy",
                str(e),
                request_id
            )
            await connection_manager.disconnect(connection_id, code=1013)  # Try again later
        
        except GenerationTimeoutError as e:
            await self._send_error(
                connection_id,
                "generation_timeout", 
                str(e),
                request_id
            )
            await connection_manager.disconnect(connection_id, code=1011)  # Internal error
        
        except InvalidVoiceError as e:
            await self._send_error(
                connection_id,
                "invalid_voice",
                str(e),
                request_id
            )
            await connection_manager.disconnect(connection_id, code=1011)  # Internal error
        
        except ModelNotReadyError as e:
            await self._send_error(
                connection_id,
                "model_not_ready",
                str(e), 
                request_id
            )
            await connection_manager.disconnect(connection_id, code=1011)  # Internal error
        
        except Exception as e:
            logger.error(f"Unexpected error during generation for {request_id}: {e}")
            await self._send_error(
                connection_id,
                "internal_error",
                f"Internal server error: {str(e)}",
                request_id
            )
            await connection_manager.disconnect(connection_id, code=1011)  # Internal error
        
        finally:
            connection_manager.finish_generation(connection_id)
    
    async def _send_error(self, connection_id: str, code: str, message: str, request_id: Optional[str] = None):
        """Send an error message over WebSocket.
        
        Args:
            connection_id: The WebSocket connection ID
            code: The error code
            message: The error message
            request_id: Optional request ID for tracking
        """
        error_message = StreamingError(
            request_id=request_id or "unknown",
            code=code,
            message=message
        )
        
        await connection_manager.send_text(connection_id, error_message.model_dump_json())


# WebSocket endpoint function
async def websocket_generate_endpoint(
    websocket: WebSocket,
    voice_service: VoiceService = Depends(get_voice_service)
):
    """WebSocket endpoint for streaming audio generation.
    
    Protocol:
    1. Client sends JSON VoiceGenerationRequest (with optional request_id)
    2. Server sends StreamingInit JSON message
    3. Server sends binary PCM16 frames
    4. Server optionally sends StreamingProgress JSON messages
    5. Server sends StreamingFinal JSON message and closes (1000)
    6. On error: Server sends StreamingError JSON message and closes (1011/1013)
    
    Args:
        websocket: The WebSocket connection
        voice_service: The voice generation service
    """
    handler = WebSocketHandler(voice_service)
    await handler.handle_connection(websocket)