#!/usr/bin/env python3
"""Test the VoiceService to see if our compatibility fixes work."""

import sys
import os
sys.path.insert(0, '/home/bumpyclock/Projects/VibeVoice')
sys.path.insert(0, '/home/bumpyclock/Projects/VibeVoice-FastAPI')

from api.core.voice_service import VoiceService
from api.models.requests import VoiceGenerationRequest
import asyncio

async def test_voice_service():
    """Test the VoiceService."""
    print("Testing VoiceService...")
    
    # Get the singleton instance
    service = VoiceService.instance()
    
    # Check if model is loaded
    is_ready = service.ready()  # It's a method, not a property
    print(f"Service ready: {is_ready}")
    
    if not is_ready:
        print("Service not ready, waiting for model to load...")
        # Wait a bit for model to load
        for i in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            is_ready = service.ready()
            if is_ready:
                print(f"Service ready after {i+1} seconds")
                break
    
    if not is_ready:
        print("Service failed to become ready")
        return
    
    # Try a simple generation
    print("\nTrying audio generation...")
    request = VoiceGenerationRequest(
        script="Speaker 0: Hello, this is a test.",
        speakers=["en-Alice_woman"],  # Use an actual voice from the voices directory
        cfg_scale=1.3
    )
    
    try:
        result = await service.generate_blocking(request)
        print(f"✓ Generation successful!")
        print(f"  WAV bytes size: {len(result.wav_bytes)} bytes")
        print(f"  Sample rate: {result.sample_rate} Hz")
        print(f"  Duration: {result.duration_sec:.2f} seconds")
        
        # Save the audio to a file for verification
        output_file = "/tmp/test_output.wav"
        with open(output_file, "wb") as f:
            f.write(result.wav_bytes)
        print(f"  Audio saved to: {output_file}")
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_voice_service())