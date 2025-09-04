# ABOUTME: This file provides audio processing utilities for format conversion and manipulation.
# ABOUTME: Functions handle PCM16 conversion, WAV file creation, and audio concatenation operations.

from typing import Iterable
import numpy as np
import struct
import io
import base64


def float32_to_pcm16(x: np.ndarray) -> bytes:
    """Convert float32 audio array to PCM16 bytes.
    
    Args:
        x: Float32 numpy array with audio samples in range [-1, 1]
        
    Returns:
        Raw PCM16 bytes (little-endian, signed 16-bit integers)
    """
    # Ensure input is numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Ensure float32 dtype
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    
    # Clip to valid range [-1, 1]
    x = np.clip(x, -1.0, 1.0)
    
    # Convert to 16-bit integers
    # Scale by 32767 (max value for signed 16-bit) and convert to int16
    x_int16 = (x * 32767).astype(np.int16)
    
    # Convert to bytes (little-endian)
    return x_int16.tobytes()


def pcm16_to_wav(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Convert PCM16 bytes to WAV file format.
    
    Args:
        pcm: Raw PCM16 bytes (little-endian)
        sample_rate: Sample rate in Hz (e.g., 24000)
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        
    Returns:
        Complete WAV file as bytes
    """
    # Calculate WAV header parameters
    byte_rate = sample_rate * num_channels * 2  # 2 bytes per sample for PCM16
    block_align = num_channels * 2
    data_size = len(pcm)
    file_size = 36 + data_size  # WAV header is 44 bytes total, minus 8 for RIFF header
    
    # Create WAV header
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (PCM format)
        1,                 # AudioFormat (PCM)
        num_channels,      # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        16,                # BitsPerSample
        b'data',           # Subchunk2ID
        data_size          # Subchunk2Size
    )
    
    # Combine header and data
    return wav_header + pcm


def concat_pcm16(chunks: Iterable[bytes]) -> bytes:
    """Concatenate multiple PCM16 byte chunks into a single audio stream.
    
    Args:
        chunks: Iterable of PCM16 byte chunks to concatenate
        
    Returns:
        Concatenated PCM16 bytes
    """
    # Use BytesIO for efficient concatenation
    output = io.BytesIO()
    
    for chunk in chunks:
        if chunk:  # Skip empty chunks
            output.write(chunk)
    
    return output.getvalue()


def validate_pcm16_chunk_size(chunk: bytes, sample_rate: int, target_duration_sec: float = 1.0) -> bool:
    """Validate that a PCM16 chunk is approximately the expected duration.
    
    Args:
        chunk: PCM16 bytes to validate
        sample_rate: Sample rate in Hz
        target_duration_sec: Expected duration in seconds
        
    Returns:
        True if chunk size is reasonable for target duration
    """
    if not chunk:
        return False
    
    # Calculate actual duration
    # PCM16 = 2 bytes per sample, mono
    num_samples = len(chunk) // 2
    actual_duration = num_samples / sample_rate
    
    # Allow 20% tolerance
    tolerance = 0.2
    min_duration = target_duration_sec * (1 - tolerance)
    max_duration = target_duration_sec * (1 + tolerance)
    
    return min_duration <= actual_duration <= max_duration


def normalize_audio_level(pcm: bytes, target_peak: float = 0.9) -> bytes:
    """Normalize audio level to target peak amplitude.
    
    Args:
        pcm: Input PCM16 bytes
        target_peak: Target peak amplitude (0.0 to 1.0)
        
    Returns:
        Normalized PCM16 bytes
    """
    if not pcm:
        return pcm
    
    # Convert to numpy array
    audio_data = np.frombuffer(pcm, dtype=np.int16)
    
    if len(audio_data) == 0:
        return pcm
    
    # Convert to float32 for processing
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # Find current peak
    current_peak = np.abs(audio_float).max()
    
    if current_peak > 0:
        # Calculate scaling factor
        scale_factor = target_peak / current_peak
        
        # Apply scaling
        audio_float *= scale_factor
        
        # Convert back to PCM16
        return float32_to_pcm16(audio_float)
    
    return pcm


def wav_to_base64(wav_bytes: bytes) -> str:
    """Convert WAV bytes to base64 string.
    
    Args:
        wav_bytes: WAV file as bytes
        
    Returns:
        Base64 encoded string of WAV data
    """
    return base64.b64encode(wav_bytes).decode('utf-8')