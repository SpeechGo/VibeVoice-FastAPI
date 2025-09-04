#!/usr/bin/env python
"""Simple test to verify API works with correct transformers version."""

import httpx
import time
import sys

def test_api():
    """Test the API endpoint."""
    print("Testing API with transformers 4.51.3...")
    
    # Test data - corrected format with proper voice ID
    request_data = {
        "script": "Alice: Hello, this is a test.",
        "speakers": ["en-Alice_woman"],
        "cfg_scale": 1.3,
        "inference_steps": 5
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            print("Sending request to API...")
            response = client.post(
                "http://localhost:8000/api/generate",
                json=request_data,
                headers={"Accept": "audio/wav"}
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Check if we got WAV data
                content_type = response.headers.get("content-type", "")
                print(f"Content-Type: {content_type}")
                
                if "audio/wav" in content_type:
                    wav_size = len(response.content)
                    print(f"✓ Success! Generated WAV file: {wav_size} bytes")
                    
                    # Save for inspection
                    with open("/tmp/test_api_output.wav", "wb") as f:
                        f.write(response.content)
                    print("Audio saved to /tmp/test_api_output.wav")
                else:
                    print(f"Unexpected content type: {content_type}")
                    print(f"Response: {response.text[:500]}")
            else:
                print(f"Error: {response.text}")
                
    except httpx.ConnectError:
        print("ERROR: Could not connect to API. Is the server running?")
        print("Start it with: uv run uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)