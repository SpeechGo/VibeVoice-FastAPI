# ABOUTME: This file provides streaming utilities for HTTP content negotiation and audio streaming.
# ABOUTME: Includes AudioStreamer class for real-time audio generation and content type handling.

import asyncio
import threading
from typing import AsyncGenerator, Optional, Callable, List
from starlette.requests import Request
import logging

logger = logging.getLogger(__name__)


def negotiate_accept(request: Request, allowed: list[str]) -> str:
    """Negotiate the best Accept header match from allowed content types.
    
    Args:
        request: The Starlette/FastAPI request object
        allowed: List of content types the server can provide
        
    Returns:
        The best matching content type from allowed list
        
    Raises:
        ValueError: If no acceptable content type is found (should map to 406)
    """
    accept_header = request.headers.get('accept', '*/*')
    
    # Handle wildcard case
    if accept_header == '*/*':
        # Default to first allowed type
        return allowed[0] if allowed else 'application/json'
    
    # Parse Accept header (simplified - doesn't handle full HTTP spec)
    accepted_types = []
    for media_type in accept_header.split(','):
        media_type = media_type.strip().split(';')[0].strip()
        accepted_types.append(media_type)
    
    # Find the first match
    for accepted_type in accepted_types:
        if accepted_type in allowed:
            return accepted_type
        
        # Handle wildcard sub-types like "audio/*"
        if '/*' in accepted_type:
            main_type = accepted_type.split('/')[0]
            for allowed_type in allowed:
                if allowed_type.startswith(main_type + '/'):
                    return allowed_type
    
    # No match found
    raise ValueError(f"No acceptable content type found. Accept: {accept_header}, Allowed: {allowed}")


class AudioStreamer:
    """Manages real-time audio streaming with cancellation support.
    
    This class provides an interface similar to the demo's AudioStreamer but
    is self-contained and doesn't import from the parent repo.
    """
    
    def __init__(self, sample_rate: int = 24000, chunk_duration_sec: float = 0.5):
        """Initialize the audio streamer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_sec: Target duration for each PCM16 chunk
        """
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.samples_per_chunk = int(sample_rate * chunk_duration_sec)
        self.bytes_per_chunk = self.samples_per_chunk * 2  # PCM16 = 2 bytes per sample
        
        self._is_cancelled = False
        self._lock = threading.Lock()
        
        logger.debug(f"AudioStreamer initialized: {sample_rate}Hz, {chunk_duration_sec}s chunks")
    
    def is_cancelled(self) -> bool:
        """Check if streaming has been cancelled."""
        with self._lock:
            return self._is_cancelled
    
    def cancel(self):
        """Cancel the streaming operation."""
        with self._lock:
            self._is_cancelled = True
            logger.debug("AudioStreamer cancelled")
    
    def reset(self):
        """Reset the streamer for reuse."""
        with self._lock:
            self._is_cancelled = False
            logger.debug("AudioStreamer reset")
    
    async def stream_from_generator(self, 
                                   audio_generator: AsyncGenerator[bytes, None],
                                   cancel_check: Optional[Callable[[], bool]] = None) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from an async generator with cancellation support.
        
        Args:
            audio_generator: Async generator yielding raw audio bytes
            cancel_check: Optional function that returns True if streaming should be cancelled
            
        Yields:
            PCM16 audio chunks
        """
        try:
            buffer = bytearray()
            
            async for raw_chunk in audio_generator:
                # Check for cancellation
                if self.is_cancelled() or (cancel_check and cancel_check()):
                    logger.debug("Streaming cancelled")
                    break
                
                if not raw_chunk:
                    continue
                
                buffer.extend(raw_chunk)
                
                # Yield complete chunks
                while len(buffer) >= self.bytes_per_chunk:
                    chunk = bytes(buffer[:self.bytes_per_chunk])
                    buffer = buffer[self.bytes_per_chunk:]
                    
                    yield chunk
                    
                    # Allow other coroutines to run
                    await asyncio.sleep(0)
            
            # Yield any remaining data as final chunk
            if buffer and not self.is_cancelled():
                yield bytes(buffer)
                
        except Exception as e:
            logger.error(f"Error in audio streaming: {e}")
            raise
        finally:
            logger.debug("Audio streaming completed")
    
    async def stream_with_timeout(self,
                                 audio_generator: AsyncGenerator[bytes, None],
                                 timeout_sec: Optional[float] = None,
                                 cancel_check: Optional[Callable[[], bool]] = None) -> AsyncGenerator[bytes, None]:
        """Stream audio with timeout support.
        
        Args:
            audio_generator: Async generator yielding raw audio bytes
            timeout_sec: Optional timeout in seconds
            cancel_check: Optional cancellation check function
            
        Yields:
            PCM16 audio chunks
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        if timeout_sec is None:
            async for chunk in self.stream_from_generator(audio_generator, cancel_check):
                yield chunk
        else:
            try:
                async with asyncio.timeout(timeout_sec):
                    async for chunk in self.stream_from_generator(audio_generator, cancel_check):
                        yield chunk
            except asyncio.TimeoutError:
                self.cancel()
                logger.warning(f"Audio streaming timed out after {timeout_sec}s")
                raise


class ChunkBuffer:
    """Buffer for managing audio chunks with size and timing constraints."""
    
    def __init__(self, max_chunks: int = 10):
        """Initialize the chunk buffer.
        
        Args:
            max_chunks: Maximum number of chunks to buffer
        """
        self.max_chunks = max_chunks
        self._buffer: List[bytes] = []
        self._lock = threading.Lock()
    
    def add_chunk(self, chunk: bytes) -> bool:
        """Add a chunk to the buffer.
        
        Args:
            chunk: Audio chunk to add
            
        Returns:
            True if chunk was added, False if buffer is full
        """
        with self._lock:
            if len(self._buffer) >= self.max_chunks:
                return False
            
            self._buffer.append(chunk)
            return True
    
    def get_chunk(self) -> Optional[bytes]:
        """Get the next chunk from the buffer.
        
        Returns:
            Next audio chunk or None if buffer is empty
        """
        with self._lock:
            if not self._buffer:
                return None
            
            return self._buffer.pop(0)
    
    def get_all_chunks(self) -> List[bytes]:
        """Get all chunks from the buffer and clear it.
        
        Returns:
            All buffered audio chunks
        """
        with self._lock:
            chunks = self._buffer.copy()
            self._buffer.clear()
            return chunks
    
    def size(self) -> int:
        """Get the current buffer size."""
        with self._lock:
            return len(self._buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.max_chunks
    
    def clear(self):
        """Clear all chunks from the buffer."""
        with self._lock:
            self._buffer.clear()