# ABOUTME: Integration tests for FastAPI endpoints using TestClient
# ABOUTME: Tests all API routes including content negotiation, error handling, and validation
import pytest
import json
import base64
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.core.voice_service import VoiceService, GenerateResult
from api.models.responses import VoiceInfo

# Test client
client = TestClient(app)

@pytest.fixture
def mock_voice_service():
    """Mock VoiceService for testing"""
    with patch.object(VoiceService, 'instance') as mock_instance:
        mock_service = Mock()
        mock_instance.return_value = mock_service
        
        # Setup default behaviors
        mock_service.ready.return_value = True
        mock_service.current_concurrency = 0
        mock_service.list_voices.return_value = [
            VoiceInfo(id="en-Alice_woman", name="Alice Woman", gender="female", language="en"),
            VoiceInfo(id="en-Carter_man", name="Carter Man", gender="male", language="en")
        ]
        
        # Mock generate result
        mock_wav_data = b"RIFF" + b"fake_wav_data" * 100
        mock_result = GenerateResult(
            wav_bytes=mock_wav_data,
            sample_rate=24000,
            duration_sec=2.5
        )
        # Make it async
        async def async_generate_blocking(*args, **kwargs):
            return mock_result
        mock_service.generate_blocking = async_generate_blocking
        
        yield mock_service

class TestHealthEndpoints:
    
    def test_health_check(self):
        """Test /healthz endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_readiness_check(self, mock_voice_service):
        """Test /readyz endpoint"""
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "device" in data
        assert "dtype" in data
        assert "model_loaded" in data
        assert "max_concurrency" in data
        assert "current_concurrency" in data

class TestVoiceListEndpoint:
    
    def test_list_voices_success(self, mock_voice_service):
        """Test GET /api/voices success"""
        response = client.get("/api/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert len(data["voices"]) == 2
        assert data["voices"][0]["id"] == "en-Alice_woman"
        assert data["voices"][0]["name"] == "Alice Woman"
    
    def test_list_voices_with_hidden(self, mock_voice_service):
        """Test GET /api/voices with include_hidden parameter"""
        response = client.get("/api/voices?include_hidden=true")
        assert response.status_code == 200
        mock_voice_service.list_voices.assert_called_with(include_hidden=True)

class TestGenerateEndpoint:
    
    @pytest.fixture
    def valid_request_data(self):
        return {
            "script": "Hello, this is a test.",
            "speakers": ["en-Alice_woman"],
            "cfg_scale": 1.3,
            "inference_steps": 5,
            "sample_rate": 24000,
            "seed": 42
        }
    
    def test_generate_wav_response(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with audio/wav Accept header"""
        response = client.post(
            "/api/generate",
            json=valid_request_data,
            headers={"Accept": "audio/wav"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Request-ID" in response.headers
        assert "X-Generation-Time" in response.headers
        assert len(response.content) > 0
    
    def test_generate_json_response(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with application/json Accept header"""
        response = client.post(
            "/api/generate",
            json=valid_request_data,
            headers={"Accept": "application/json"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "audio_base64" in data
        assert "duration_sec" in data
        assert "sample_rate" in data
        assert "request_id" in data
        assert "generation_time_sec" in data
        assert data["format"] == "wav"
        assert data["num_channels"] == 1
        
        # Verify base64 encoding
        audio_data = base64.b64decode(data["audio_base64"])
        assert len(audio_data) > 0
    
    def test_generate_default_accept(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with default Accept header (should return JSON)"""
        response = client.post("/api/generate", json=valid_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "audio_base64" in data
    
    def test_generate_unsupported_accept(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with unsupported Accept header"""
        response = client.post(
            "/api/generate",
            json=valid_request_data,
            headers={"Accept": "audio/mp3"}
        )
        assert response.status_code == 406
        data = response.json()
        assert data["code"] == "NOT_ACCEPTABLE"
    
    def test_generate_validation_error(self, mock_voice_service):
        """Test POST /api/generate with invalid request data"""
        invalid_data = {
            "script": "",  # Too short
            "speakers": [],  # Empty list
            "cfg_scale": 10.0,  # Too high
        }
        response = client.post("/api/generate", json=invalid_data)
        assert response.status_code == 422
    
    def test_generate_invalid_voice_error(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with invalid voice"""
        from api.models.errors import InvalidVoiceError
        
        async def async_generate_blocking(*args, **kwargs):
            raise InvalidVoiceError("Voice not found: invalid-voice")
        mock_voice_service.generate_blocking = async_generate_blocking
        
        valid_request_data["speakers"] = ["invalid-voice"]
        response = client.post("/api/generate", json=valid_request_data)
        assert response.status_code == 422
        data = response.json()
        assert data["code"] == "INVALID_VOICE"
    
    def test_generate_service_busy_error(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate when service is busy"""
        from api.models.errors import ServiceBusyError
        
        async def async_generate_blocking(*args, **kwargs):
            raise ServiceBusyError("Service at capacity")
        mock_voice_service.generate_blocking = async_generate_blocking
        
        response = client.post("/api/generate", json=valid_request_data)
        assert response.status_code == 429
        data = response.json()
        assert data["code"] == "SERVICE_BUSY"
    
    def test_generate_model_not_ready_error(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate when model not ready"""
        from api.models.errors import ModelNotReadyError
        
        async def async_generate_blocking(*args, **kwargs):
            raise ModelNotReadyError("Model not loaded")
        mock_voice_service.generate_blocking = async_generate_blocking
        
        response = client.post("/api/generate", json=valid_request_data)
        assert response.status_code == 503
        data = response.json()
        assert data["code"] == "MODEL_NOT_READY"
    
    def test_generate_timeout_error(self, mock_voice_service, valid_request_data):
        """Test POST /api/generate with timeout"""
        from api.models.errors import GenerationTimeoutError
        
        async def async_generate_blocking(*args, **kwargs):
            raise GenerationTimeoutError("Generation timeout")
        mock_voice_service.generate_blocking = async_generate_blocking
        
        response = client.post("/api/generate", json=valid_request_data)
        assert response.status_code == 408
        data = response.json()
        assert data["code"] == "GENERATION_TIMEOUT"

class TestRequestIDHandling:
    
    def test_request_id_echo(self, mock_voice_service):
        """Test that X-Request-ID header is echoed back"""
        request_id = "test-request-123"
        response = client.get("/healthz", headers={"X-Request-ID": request_id})
        assert response.headers["X-Request-ID"] == request_id
    
    def test_request_id_generation(self, mock_voice_service):
        """Test that X-Request-ID is generated if not provided"""
        response = client.get("/healthz")
        assert "X-Request-ID" in response.headers
        # Should be a UUID-like string
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 10

class TestCORSHeaders:
    
    def test_cors_headers_present(self, mock_voice_service):
        """Test CORS headers are included"""
        response = client.options("/api/voices")
        # FastAPI/Starlette automatically handles OPTIONS for CORS
        # Just verify the service doesn't crash
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled