# ABOUTME: Comprehensive test suite for all Pydantic models in the VibeVoice FastAPI service
# ABOUTME: Tests validate model schemas, field constraints, validation rules, and error handling per Public Interfaces specification

import pytest
from typing import List, Dict, Any
from pydantic import ValidationError
import json

# These imports will initially fail - that's expected for TDD
from api.models.requests import VoiceGenerationRequest, VoiceListRequest
from api.models.responses import (
    VoiceInfo, VoiceListResponse, GenerateJsonResponse, 
    HealthStatus, ReadinessStatus, ErrorResponse
)
from api.models.streaming import (
    StreamingInit, StreamingProgress, StreamingFinal, StreamingError
)
from api.models.errors import (
    ServiceBusyError, ModelNotReadyError, GenerationTimeoutError, InvalidVoiceError
)


class TestVoiceGenerationRequest:
    """Test VoiceGenerationRequest model validation and constraints."""
    
    def test_valid_request_creation(self):
        """Test creating a valid VoiceGenerationRequest."""
        request = VoiceGenerationRequest(
            script="Speaker 1: Hello world! Speaker 2: How are you?",
            speakers=["alice", "bob"],
            cfg_scale=1.5,
            inference_steps=10,
            sample_rate=24000,
            format="wav",
            seed=42
        )
        
        assert request.script == "Speaker 1: Hello world! Speaker 2: How are you?"
        assert request.speakers == ["alice", "bob"]
        assert request.cfg_scale == 1.5
        assert request.inference_steps == 10
        assert request.sample_rate == 24000
        assert request.format == "wav"
        assert request.seed == 42
    
    def test_default_values(self):
        """Test default values are applied correctly."""
        request = VoiceGenerationRequest(
            script="Hello world!",
            speakers=["alice"]
        )
        
        assert request.cfg_scale == 1.3
        assert request.inference_steps == 5
        assert request.sample_rate == 24000
        assert request.format == "wav"
        assert request.seed is None
    
    def test_script_validation(self):
        """Test script field validation constraints."""
        # Test minimum length
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="", speakers=["alice"])
        assert "at least 1 character" in str(exc_info.value) or "string_too_short" in str(exc_info.value)
        
        # Test maximum length
        long_script = "x" * 5001
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script=long_script, speakers=["alice"])
        assert "at most 5000 characters" in str(exc_info.value) or "string_too_long" in str(exc_info.value)
        
        # Test whitespace stripping
        request = VoiceGenerationRequest(
            script="  Hello world!  ",
            speakers=["alice"]
        )
        assert request.script == "Hello world!"
    
    def test_speakers_validation(self):
        """Test speakers list validation constraints."""
        # Test minimum speakers
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=[])
        assert "min_items" in str(exc_info.value) or "at least 1" in str(exc_info.value)
        
        # Test maximum speakers
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["a", "b", "c", "d", "e"])
        assert "max_items" in str(exc_info.value) or "at most 4" in str(exc_info.value)
        
        # Test valid range
        request = VoiceGenerationRequest(script="Hello", speakers=["a", "b", "c", "d"])
        assert len(request.speakers) == 4
    
    def test_cfg_scale_validation(self):
        """Test cfg_scale field validation constraints."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], cfg_scale=0.05)
        assert "greater than or equal to 0.1" in str(exc_info.value) or "greater_than_equal" in str(exc_info.value)
        
        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], cfg_scale=5.1)
        assert "less than or equal to 5" in str(exc_info.value) or "less_than_equal" in str(exc_info.value)
        
        # Test valid range
        request = VoiceGenerationRequest(script="Hello", speakers=["alice"], cfg_scale=2.5)
        assert request.cfg_scale == 2.5
    
    def test_inference_steps_validation(self):
        """Test inference_steps field validation constraints."""
        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], inference_steps=0)
        assert "greater than or equal to 1" in str(exc_info.value)
        
        # Test maximum value
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], inference_steps=65)
        assert "less than or equal to 64" in str(exc_info.value)
        
        # Test valid range
        request = VoiceGenerationRequest(script="Hello", speakers=["alice"], inference_steps=32)
        assert request.inference_steps == 32
    
    def test_seed_validation(self):
        """Test seed field validation constraints."""
        # Test negative value
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], seed=-1)
        assert "greater than or equal to 0" in str(exc_info.value)
        
        # Test valid values
        request1 = VoiceGenerationRequest(script="Hello", speakers=["alice"], seed=0)
        assert request1.seed == 0
        
        request2 = VoiceGenerationRequest(script="Hello", speakers=["alice"], seed=123456)
        assert request2.seed == 123456
    
    def test_format_literal_validation(self):
        """Test format field only accepts 'wav'."""
        # Invalid format should fail
        with pytest.raises(ValidationError) as exc_info:
            VoiceGenerationRequest(script="Hello", speakers=["alice"], format="mp3")
        assert "wav" in str(exc_info.value) or "literal" in str(exc_info.value).lower()


class TestVoiceListRequest:
    """Test VoiceListRequest model validation."""
    
    def test_valid_request_creation(self):
        """Test creating a valid VoiceListRequest."""
        request = VoiceListRequest(include_hidden=True)
        assert request.include_hidden is True
    
    def test_default_values(self):
        """Test default values are applied correctly."""
        request = VoiceListRequest()
        assert request.include_hidden is False


class TestVoiceInfo:
    """Test VoiceInfo model validation."""
    
    def test_valid_voice_info_creation(self):
        """Test creating a valid VoiceInfo."""
        voice = VoiceInfo(
            id="alice",
            name="Alice",
            gender="female",
            language="en"
        )
        
        assert voice.id == "alice"
        assert voice.name == "Alice"
        assert voice.gender == "female"
        assert voice.language == "en"
    
    def test_minimal_voice_info(self):
        """Test creating VoiceInfo with only required fields."""
        voice = VoiceInfo(id="alice", name="Alice")
        
        assert voice.id == "alice"
        assert voice.name == "Alice"
        assert voice.gender is None
        assert voice.language is None


class TestVoiceListResponse:
    """Test VoiceListResponse model validation."""
    
    def test_valid_response_creation(self):
        """Test creating a valid VoiceListResponse."""
        voices = [
            VoiceInfo(id="alice", name="Alice", gender="female"),
            VoiceInfo(id="bob", name="Bob", gender="male")
        ]
        response = VoiceListResponse(voices=voices)
        
        assert len(response.voices) == 2
        assert response.voices[0].id == "alice"
        assert response.voices[1].id == "bob"
    
    def test_empty_voices_list(self):
        """Test creating VoiceListResponse with empty voices list."""
        response = VoiceListResponse(voices=[])
        assert response.voices == []


class TestGenerateJsonResponse:
    """Test GenerateJsonResponse model validation."""
    
    def test_valid_response_creation(self):
        """Test creating a valid GenerateJsonResponse."""
        response = GenerateJsonResponse(
            audio_base64="SGVsbG8gV29ybGQ=",
            duration_sec=2.5,
            sample_rate=24000,
            num_channels=1,
            format="wav",
            request_id="req-123",
            generation_time_sec=1.2
        )
        
        assert response.audio_base64 == "SGVsbG8gV29ybGQ="
        assert response.duration_sec == 2.5
        assert response.sample_rate == 24000
        assert response.num_channels == 1
        assert response.format == "wav"
        assert response.request_id == "req-123"
        assert response.generation_time_sec == 1.2
    
    def test_default_values(self):
        """Test default values are applied correctly."""
        response = GenerateJsonResponse(
            audio_base64="SGVsbG8gV29ybGQ=",
            duration_sec=2.5,
            sample_rate=24000,
            request_id="req-123",
            generation_time_sec=1.2
        )
        
        assert response.num_channels == 1
        assert response.format == "wav"


class TestHealthStatus:
    """Test HealthStatus model validation."""
    
    def test_valid_health_status(self):
        """Test creating valid HealthStatus."""
        status_ok = HealthStatus(status="ok")
        assert status_ok.status == "ok"
        
        status_error = HealthStatus(status="error")
        assert status_error.status == "error"
    
    def test_invalid_status_value(self):
        """Test invalid status value fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            HealthStatus(status="unknown")
        assert "ok" in str(exc_info.value) or "error" in str(exc_info.value)


class TestReadinessStatus:
    """Test ReadinessStatus model validation."""
    
    def test_valid_readiness_status(self):
        """Test creating a valid ReadinessStatus."""
        status = ReadinessStatus(
            ready=True,
            device="cuda:0",
            dtype="bfloat16",
            model_loaded=True,
            max_concurrency=2,
            current_concurrency=1
        )
        
        assert status.ready is True
        assert status.device == "cuda:0"
        assert status.dtype == "bfloat16"
        assert status.model_loaded is True
        assert status.max_concurrency == 2
        assert status.current_concurrency == 1


class TestErrorResponse:
    """Test ErrorResponse model validation."""
    
    def test_valid_error_response(self):
        """Test creating a valid ErrorResponse."""
        error = ErrorResponse(
            code="MODEL_NOT_READY",
            message="Model is still loading",
            details={"retry_after": 30}
        )
        
        assert error.code == "MODEL_NOT_READY"
        assert error.message == "Model is still loading"
        assert error.details == {"retry_after": 30}
    
    def test_error_response_without_details(self):
        """Test creating ErrorResponse without details."""
        error = ErrorResponse(
            code="INVALID_INPUT",
            message="Invalid script format"
        )
        
        assert error.code == "INVALID_INPUT"
        assert error.message == "Invalid script format"
        assert error.details is None


class TestStreamingModels:
    """Test streaming protocol models validation."""
    
    def test_streaming_init(self):
        """Test StreamingInit model validation."""
        init = StreamingInit(
            request_id="req-123",
            sample_rate=24000,
            num_channels=1,
            format="pcm16"
        )
        
        assert init.type == "init"
        assert init.request_id == "req-123"
        assert init.sample_rate == 24000
        assert init.num_channels == 1
        assert init.format == "pcm16"
    
    def test_streaming_init_defaults(self):
        """Test StreamingInit default values."""
        init = StreamingInit(
            request_id="req-123",
            sample_rate=24000
        )
        
        assert init.type == "init"
        assert init.num_channels == 1
        assert init.format == "pcm16"
    
    def test_streaming_progress(self):
        """Test StreamingProgress model validation."""
        progress = StreamingProgress(
            request_id="req-123",
            chunk_index=5,
            ms_emitted=2500
        )
        
        assert progress.type == "progress"
        assert progress.request_id == "req-123"
        assert progress.chunk_index == 5
        assert progress.ms_emitted == 2500
    
    def test_streaming_final(self):
        """Test StreamingFinal model validation."""
        final = StreamingFinal(
            request_id="req-123",
            total_ms=5000
        )
        
        assert final.type == "final"
        assert final.request_id == "req-123"
        assert final.total_ms == 5000
    
    def test_streaming_error(self):
        """Test StreamingError model validation."""
        error = StreamingError(
            request_id="req-123",
            code="TIMEOUT",
            message="Generation timed out after 5 minutes"
        )
        
        assert error.type == "error"
        assert error.request_id == "req-123"
        assert error.code == "TIMEOUT"
        assert error.message == "Generation timed out after 5 minutes"


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_service_busy_error(self):
        """Test ServiceBusyError exception."""
        error = ServiceBusyError("Service is at capacity")
        assert str(error) == "Service is at capacity"
        assert isinstance(error, Exception)
    
    def test_model_not_ready_error(self):
        """Test ModelNotReadyError exception."""
        error = ModelNotReadyError("Model is still loading")
        assert str(error) == "Model is still loading"
        assert isinstance(error, Exception)
    
    def test_generation_timeout_error(self):
        """Test GenerationTimeoutError exception."""
        error = GenerationTimeoutError("Generation exceeded timeout")
        assert str(error) == "Generation exceeded timeout"
        assert isinstance(error, Exception)
    
    def test_invalid_voice_error(self):
        """Test InvalidVoiceError exception."""
        error = InvalidVoiceError("Voice 'unknown' not found")
        assert str(error) == "Voice 'unknown' not found"
        assert isinstance(error, Exception)


class TestModelSerialization:
    """Test JSON serialization/deserialization of models."""
    
    def test_voice_generation_request_serialization(self):
        """Test VoiceGenerationRequest JSON round-trip."""
        request = VoiceGenerationRequest(
            script="Hello world!",
            speakers=["alice"],
            cfg_scale=1.5,
            seed=42
        )
        
        # Test serialization
        json_data = request.model_dump()
        assert json_data["script"] == "Hello world!"
        assert json_data["speakers"] == ["alice"]
        assert json_data["cfg_scale"] == 1.5
        assert json_data["seed"] == 42
        
        # Test deserialization
        request_from_json = VoiceGenerationRequest(**json_data)
        assert request_from_json.script == request.script
        assert request_from_json.speakers == request.speakers
        assert request_from_json.cfg_scale == request.cfg_scale
        assert request_from_json.seed == request.seed
    
    def test_generate_json_response_serialization(self):
        """Test GenerateJsonResponse JSON round-trip."""
        response = GenerateJsonResponse(
            audio_base64="SGVsbG8gV29ybGQ=",
            duration_sec=2.5,
            sample_rate=24000,
            request_id="req-123",
            generation_time_sec=1.2
        )
        
        # Test serialization
        json_data = response.model_dump()
        assert json_data["audio_base64"] == "SGVsbG8gV29ybGQ="
        assert json_data["duration_sec"] == 2.5
        assert json_data["sample_rate"] == 24000
        assert json_data["request_id"] == "req-123"
        assert json_data["generation_time_sec"] == 1.2
        
        # Test deserialization
        response_from_json = GenerateJsonResponse(**json_data)
        assert response_from_json.audio_base64 == response.audio_base64
        assert response_from_json.duration_sec == response.duration_sec
        assert response_from_json.sample_rate == response.sample_rate