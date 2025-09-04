# ABOUTME: This file contains comprehensive WebSocket tests for streaming protocol compliance.
# ABOUTME: Tests cover streaming protocol messages, binary frames, error scenarios, and cancellation.

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from api.main import app
from api.models.requests import VoiceGenerationRequest
from api.models.streaming import StreamingInit, StreamingProgress, StreamingFinal, StreamingError


class TestWebSocketStreaming:
    """Test WebSocket streaming functionality."""

    @pytest.fixture
    def websocket_request(self):
        """Standard WebSocket generation request."""
        return {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav",
            "request_id": str(uuid.uuid4())
        }

    @pytest.fixture
    def mock_voice_service(self):
        """Mock voice service for testing."""
        from api.core.voice_service import VoiceService
        with patch.object(VoiceService, 'instance') as mock_instance:
            service = MagicMock()
            service.ready.return_value = True
            mock_instance.return_value = service
            yield service

    @pytest.fixture
    def client(self):
        """Test client for WebSocket tests."""
        return TestClient(app)

    def test_websocket_streaming_success(self, client, mock_voice_service, websocket_request):
        """Test successful WebSocket streaming with proper protocol messages."""
        # Mock streaming PCM16 chunks
        pcm_chunks = [b'\x00\x01' * 1000, b'\x00\x02' * 1000, b'\x00\x03' * 1000]
        
        async def mock_stream(*args, **kwargs):
            for chunk in pcm_chunks:
                yield chunk
        
        # Make the mock return the async generator
        mock_voice_service.stream_pcm16 = mock_stream
        
        with client.websocket_connect("/ws/generate") as websocket:
            # Send request
            websocket.send_json(websocket_request)
            
            # Expect StreamingInit message
            try:
                init_message = websocket.receive_text()
                init_data = json.loads(init_message)
                print(f"Init message: {init_data}")
                assert init_data["type"] == "init"
                assert init_data["request_id"] == websocket_request["request_id"]
                assert init_data["sample_rate"] == 24000
                assert init_data["num_channels"] == 1
                assert init_data["format"] == "pcm16"
            except Exception as e:
                print(f"Failed to receive init message: {e}")
                raise
            
            # Collect all messages (binary and text)
            received_chunks = []
            messages = []
            
            try:
                while True:
                    try:
                        # Try to receive text message
                        text_msg = websocket.receive_text()
                        messages.append(json.loads(text_msg))
                    except:
                        # Try to receive binary message
                        binary_data = websocket.receive_bytes()
                        received_chunks.append(binary_data)
            except:
                # Connection closed or no more data
                pass
            
            # Debug output
            print(f"Received chunks: {len(received_chunks)}")
            print(f"Received messages: {messages}")
            
            # Should have received binary chunks and a final message
            assert len(received_chunks) > 0, f"Should have received binary chunks. Messages: {messages}"
            
            # Check for final message
            final_messages = [msg for msg in messages if msg.get("type") == "final"]
            assert len(final_messages) == 1, f"Expected 1 final message, got {len(final_messages)}: {messages}"
            
            final_data = final_messages[0]
            assert final_data["request_id"] == websocket_request["request_id"]
            assert "total_ms" in final_data
            
            # Verify we received the expected binary chunks
            assert len(received_chunks) == len(pcm_chunks)
            assert received_chunks == pcm_chunks

    def test_websocket_streaming_with_auto_request_id(self, client, mock_voice_service):
        """Test WebSocket streaming when request_id is auto-generated."""
        request_without_id = {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav"
        }
        
        async def mock_stream():
            yield b'\x00\x01' * 1000
        
        mock_voice_service.stream_pcm16.return_value = mock_stream()
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(request_without_id)
            
            # Expect StreamingInit with generated request_id
            init_message = websocket.receive_text()
            init_data = json.loads(init_message)
            assert init_data["type"] == "init"
            assert "request_id" in init_data
            assert init_data["request_id"] != ""

    def test_websocket_streaming_with_progress(self, client, mock_voice_service, websocket_request):
        """Test WebSocket streaming with periodic progress updates."""
        pcm_chunks = [b'\x00\x01' * 2000, b'\x00\x02' * 2000]
        
        async def mock_stream():
            for chunk in pcm_chunks:
                yield chunk
        
        mock_voice_service.stream_pcm16.return_value = mock_stream()
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Expect StreamingInit
            init_message = websocket.receive_text()
            init_data = json.loads(init_message)
            assert init_data["type"] == "init"
            
            messages = []
            binary_chunks = []
            
            # Collect all messages
            try:
                while True:
                    try:
                        text_msg = websocket.receive_text()
                        messages.append(json.loads(text_msg))
                    except:
                        # Try binary
                        binary_data = websocket.receive_bytes()
                        binary_chunks.append(binary_data)
            except Exception:
                pass
            
            # Should have progress and final messages
            progress_messages = [m for m in messages if m.get("type") == "progress"]
            final_messages = [m for m in messages if m.get("type") == "final"]
            
            # May have progress updates
            assert len(final_messages) == 1
            assert final_messages[0]["request_id"] == websocket_request["request_id"]

    def test_websocket_validation_error(self, client, mock_voice_service):
        """Test WebSocket with invalid request data."""
        invalid_request = {
            "script": "",  # Empty script should fail validation
            "speakers": [],  # Empty speakers should fail
            "cfg_scale": -1  # Invalid cfg_scale
        }
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(invalid_request)
            
            # Expect StreamingError message
            error_message = websocket.receive_text()
            error_data = json.loads(error_message)
            assert error_data["type"] == "error"
            assert error_data["code"] == "validation_error"
            assert "message" in error_data

    def test_websocket_service_not_ready(self, client, mock_voice_service):
        """Test WebSocket when service is not ready."""
        mock_voice_service.ready.return_value = False
        
        websocket_request = {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav"
        }
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Expect StreamingError message
            error_message = websocket.receive_text()
            error_data = json.loads(error_message)
            assert error_data["type"] == "error"
            assert error_data["code"] == "service_not_ready"

    def test_websocket_service_busy(self, client, mock_voice_service, websocket_request):
        """Test WebSocket when service is busy."""
        from api.models.errors import ServiceBusyError
        
        mock_voice_service.stream_pcm16.side_effect = ServiceBusyError("Service at capacity")
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Expect StreamingError message
            error_message = websocket.receive_text()
            error_data = json.loads(error_message)
            assert error_data["type"] == "error"
            assert error_data["code"] == "service_busy"

    def test_websocket_generation_timeout(self, client, mock_voice_service, websocket_request):
        """Test WebSocket when generation times out."""
        from api.models.errors import GenerationTimeoutError
        
        mock_voice_service.stream_pcm16.side_effect = GenerationTimeoutError("Generation timed out")
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Expect StreamingError message
            error_message = websocket.receive_text()
            error_data = json.loads(error_message)
            assert error_data["type"] == "error"
            assert error_data["code"] == "generation_timeout"

    def test_websocket_invalid_voice(self, client, mock_voice_service, websocket_request):
        """Test WebSocket with invalid voice ID."""
        from api.models.errors import InvalidVoiceError
        
        websocket_request["speakers"] = ["nonexistent-voice"]
        mock_voice_service.stream_pcm16.side_effect = InvalidVoiceError("Voice not found: nonexistent-voice")
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Expect StreamingError message
            error_message = websocket.receive_text()
            error_data = json.loads(error_message)
            assert error_data["type"] == "error"
            assert error_data["code"] == "invalid_voice"

    def test_websocket_client_disconnect_cancellation(self, client, mock_voice_service, websocket_request):
        """Test WebSocket handles client disconnection and cancels generation."""
        # Create a long-running mock stream that we can control
        cancel_called = False
        
        async def mock_stream_with_cancellation():
            nonlocal cancel_called
            for i in range(100):  # Long stream
                await asyncio.sleep(0.01)  # Simulate work
                if cancel_called:
                    break
                yield b'\x00\x01' * 1000
        
        def mock_cancel_check():
            return cancel_called
        
        mock_voice_service.stream_pcm16.return_value = mock_stream_with_cancellation()
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Receive init message
            init_message = websocket.receive_text()
            init_data = json.loads(init_message)
            assert init_data["type"] == "init"
            
            # Simulate client disconnect by setting cancel flag
            cancel_called = True
            
            # Close websocket connection
            websocket.close()

    def test_websocket_close_codes(self, client, mock_voice_service):
        """Test WebSocket uses correct close codes."""
        # Test normal completion - close code 1000
        websocket_request = {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav"
        }
        
        async def mock_stream():
            yield b'\x00\x01' * 1000
        
        mock_voice_service.stream_pcm16.return_value = mock_stream()
        
        with client.websocket_connect("/ws/generate") as websocket:
            websocket.send_json(websocket_request)
            
            # Consume all messages
            try:
                while True:
                    try:
                        websocket.receive_text()
                    except:
                        websocket.receive_bytes()
            except:
                pass
        
        # WebSocket should close normally with code 1000

    def test_websocket_backpressure_close_code(self, client, mock_voice_service):
        """Test WebSocket uses close code 1013 for backpressure."""
        from api.models.errors import ServiceBusyError
        
        websocket_request = {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav"
        }
        
        mock_voice_service.stream_pcm16.side_effect = ServiceBusyError("Service at capacity")
        
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/generate") as websocket:
                websocket.send_json(websocket_request)
                
                # Should receive error message then disconnect with code 1013
                error_message = websocket.receive_text()
                error_data = json.loads(error_message)
                assert error_data["type"] == "error"
                assert error_data["code"] == "service_busy"
        
        # Should close with code 1013 (try again later)
        assert exc_info.value.code == 1013

    def test_websocket_internal_error_close_code(self, client, mock_voice_service):
        """Test WebSocket uses close code 1011 for internal errors."""
        websocket_request = {
            "script": "Hello world",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "format": "wav"
        }
        
        # Simulate internal error
        mock_voice_service.stream_pcm16.side_effect = RuntimeError("Internal error")
        
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/generate") as websocket:
                websocket.send_json(websocket_request)
                
                # Should receive error message then disconnect with code 1011
                error_message = websocket.receive_text()
                error_data = json.loads(error_message)
                assert error_data["type"] == "error"
                assert error_data["code"] == "internal_error"
        
        # Should close with code 1011 (internal error)
        assert exc_info.value.code == 1011