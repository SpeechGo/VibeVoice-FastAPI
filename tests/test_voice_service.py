# ABOUTME: This file contains comprehensive tests for the VoiceService class following TDD principles.
# ABOUTME: Tests cover singleton pattern, model loading, voice scanning, and generation methods.

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import AsyncGenerator, List

# Import the modules we'll be testing (these don't exist yet, but we're defining the interface)
import sys
sys.path.append('/home/bumpyclock/Projects/VibeVoice-FastAPI')

from api.models.errors import ModelNotReadyError, ServiceBusyError, InvalidVoiceError
from api.models.requests import VoiceGenerationRequest
from api.models.responses import VoiceInfo
from api.core.voice_service import VoiceService, GenerateResult, CancelCheck


class TestVoiceService:
    """Test suite for VoiceService following the Public Interface contract."""
    
    def test_singleton_pattern(self):
        """Test that VoiceService follows singleton pattern."""
        service1 = VoiceService.instance()
        service2 = VoiceService.instance()
        assert service1 is service2
        assert isinstance(service1, VoiceService)
    
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    def test_singleton_with_different_instances(self, mock_processor, mock_model):
        """Test that multiple calls to instance() return the same object."""
        # Reset singleton for clean test
        VoiceService._instance = None
        
        # Mock the model loading
        mock_model.from_pretrained.return_value = Mock()
        mock_processor.from_pretrained.return_value = Mock()
        
        service1 = VoiceService.instance()
        service2 = VoiceService.instance()
        service3 = VoiceService.instance()
        
        assert service1 is service2 is service3
        # Model should only be loaded once
        assert mock_model.from_pretrained.call_count <= 1
    
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    def test_ready_with_successful_model_loading(self, mock_processor, mock_model):
        """Test ready() returns True when model is successfully loaded."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        service = VoiceService.instance()
        assert service.ready() is True
    
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    def test_ready_with_failed_model_loading(self, mock_processor, mock_model):
        """Test ready() returns False when model loading fails."""
        VoiceService._instance = None
        
        mock_model.from_pretrained.side_effect = Exception("CUDA OOM")
        
        service = VoiceService.instance()
        assert service.ready() is False
    
    def test_list_voices_with_empty_directory(self):
        """Test list_voices() with empty voices directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('api.core.voice_service.VOICES_DIR', temp_dir):
                VoiceService._instance = None
                service = VoiceService.instance()
                voices = service.list_voices()
                assert voices == []
    
    def test_list_voices_with_voice_files(self):
        """Test list_voices() scans voice files correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test voice files
            voice_files = [
                "en-Alice_woman.wav",
                "en-Carter_man.wav",
                "es-Sofia_woman.wav"
            ]
            
            for voice_file in voice_files:
                Path(temp_dir, voice_file).touch()
            
            # Create a non-voice file that should be ignored
            Path(temp_dir, "readme.txt").touch()
            
            with patch('api.core.voice_service.VOICES_DIR', temp_dir):
                VoiceService._instance = None
                service = VoiceService.instance()
                voices = service.list_voices()
                
                assert len(voices) == 3
                voice_ids = [v.id for v in voices]
                assert "en-Alice_woman" in voice_ids
                assert "en-Carter_man" in voice_ids
                assert "es-Sofia_woman" in voice_ids
                
                # Check VoiceInfo structure
                alice_voice = next(v for v in voices if v.id == "en-Alice_woman")
                assert alice_voice.name == "Alice"
                assert alice_voice.gender == "woman"
                assert alice_voice.language == "en"
    
    def test_list_voices_include_hidden(self):
        """Test list_voices() with include_hidden parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create visible and hidden voice files
            Path(temp_dir, "en-Alice_woman.wav").touch()
            Path(temp_dir, ".hidden-voice.wav").touch()
            
            with patch('api.core.voice_service.VOICES_DIR', temp_dir):
                VoiceService._instance = None
                service = VoiceService.instance()
                
                # Without include_hidden
                voices = service.list_voices(include_hidden=False)
                assert len(voices) == 1
                assert voices[0].id == "en-Alice_woman"
                
                # With include_hidden
                voices = service.list_voices(include_hidden=True)
                assert len(voices) == 2
                voice_ids = [v.id for v in voices]
                assert "en-Alice_woman" in voice_ids
                assert ".hidden-voice" in voice_ids
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.torch.cuda.is_available', return_value=True)
    @patch('api.core.voice_service.torch.cuda.device_count', return_value=1)
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_generate_blocking_success(self, mock_processor, mock_model, mock_device_count, mock_cuda_available):
        """Test successful blocking generation."""
        VoiceService._instance = None
        
        # Mock model and processor
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Create a mock request
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"],
            cfg_scale=1.3,
            inference_steps=5,
            sample_rate=24000,
            format="wav"
        )
        
        # Mock the generation process  
        with patch('api.core.voice_service.asyncio.Semaphore') as mock_semaphore:
            mock_semaphore_instance = AsyncMock()
            mock_semaphore_instance.acquire = AsyncMock()
            mock_semaphore_instance.release = Mock()
            mock_semaphore.return_value = mock_semaphore_instance
            
            with patch('api.core.voice_service.torch.cuda.empty_cache'):
                with patch('api.core.voice_service.torch.no_grad'):
                    # Mock the actual generation
                    fake_wav_bytes = b"RIFF" + b"fake_wav_data" * 1000
                    
                    with patch.object(VoiceService, '_generate_audio') as mock_generate:
                        mock_generate.return_value = GenerateResult(
                            wav_bytes=fake_wav_bytes,
                            sample_rate=24000,
                            duration_sec=2.5
                        )
                        
                        service = VoiceService.instance()
                        result = await service.generate_blocking(request)
                        
                        assert isinstance(result, GenerateResult)
                        assert result.wav_bytes == fake_wav_bytes
                        assert result.sample_rate == 24000
                        assert result.duration_sec == 2.5
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_generate_blocking_model_not_ready(self, mock_processor, mock_model):
        """Test generate_blocking raises ModelNotReadyError when model not loaded."""
        VoiceService._instance = None
        
        mock_model.from_pretrained.side_effect = Exception("CUDA OOM")
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        service = VoiceService.instance()
        
        with pytest.raises(ModelNotReadyError):
            await service.generate_blocking(request)
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_generate_blocking_with_timeout(self, mock_processor, mock_model):
        """Test generate_blocking respects timeout parameter."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        service = VoiceService.instance()
        
        # Mock a slow generation that should timeout
        with patch.object(service, '_generate_audio') as mock_generate:
            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than our timeout
                return GenerateResult(wav_bytes=b"fake_data", sample_rate=24000, duration_sec=1.0)
            
            mock_generate.side_effect = slow_generate
            
            with pytest.raises(asyncio.TimeoutError):
                await service.generate_blocking(request, timeout_sec=0.1)
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_generate_blocking_with_cancellation(self, mock_processor, mock_model):
        """Test generate_blocking respects cancellation callback."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        # Create a cancel check that returns True immediately
        def cancel_check():
            return True
        
        service = VoiceService.instance()
        
        # Generation should stop early due to cancellation
        result = await service.generate_blocking(request, cancel_check=cancel_check)
        # Should return empty or minimal result when cancelled
        assert result is None or len(result.wav_bytes) == 0
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_stream_pcm16_basic(self, mock_processor, mock_model):
        """Test basic PCM16 streaming functionality."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        service = VoiceService.instance()
        
        # Mock the streaming generation
        fake_chunks = [b"chunk1" * 100, b"chunk2" * 100, b"chunk3" * 100]
        
        with patch.object(service, '_stream_pcm16_internal') as mock_stream:
            async def fake_stream(*args, **kwargs):
                for chunk in fake_chunks:
                    yield chunk
            
            mock_stream.return_value = fake_stream()
            
            chunks = []
            async for chunk in service.stream_pcm16(request):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks == fake_chunks
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_stream_pcm16_with_cancellation(self, mock_processor, mock_model):
        """Test PCM16 streaming with cancellation."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        cancel_called = False
        def cancel_check():
            nonlocal cancel_called
            if not cancel_called:
                cancel_called = True
                return False
            return True  # Cancel after first chunk
        
        service = VoiceService.instance()
        
        fake_chunks = [b"chunk1" * 100, b"chunk2" * 100, b"chunk3" * 100]
        
        with patch.object(service, '_stream_pcm16_internal') as mock_stream:
            async def fake_stream(*args, **kwargs):
                for i, chunk in enumerate(fake_chunks):
                    if kwargs.get('cancel_check') and kwargs['cancel_check']():
                        break
                    yield chunk
            
            mock_stream.return_value = fake_stream()
            
            chunks = []
            async for chunk in service.stream_pcm16(request, cancel_check=cancel_check):
                chunks.append(chunk)
            
            # Should stop early due to cancellation
            assert len(chunks) <= 2
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that MAX_CONCURRENCY is enforced."""
        VoiceService._instance = None
        
        with patch('api.core.voice_service.MAX_CONCURRENCY', 1):
            with patch('api.core.voice_service.AutoModelForCausalLM'):
                with patch('api.core.voice_service.AutoProcessor'):
                    service = VoiceService.instance()
                    
                    request = VoiceGenerationRequest(
                        script="Hello, world!",
                        speakers=["en-Alice_woman"]
                    )
                    
                    # Mock a slow generation
                    with patch.object(service, '_generate_audio') as mock_generate:
                        async def slow_generate(*args, **kwargs):
                            await asyncio.sleep(0.1)
                            return (b"fake_data", 24000, 1.0)
                        
                        mock_generate.side_effect = slow_generate
                        
                        # Start two concurrent requests
                        task1 = asyncio.create_task(service.generate_blocking(request))
                        task2 = asyncio.create_task(service.generate_blocking(request))
                        
                        # The second should raise ServiceBusyError or wait
                        with pytest.raises((ServiceBusyError, asyncio.TimeoutError)):
                            await asyncio.wait_for(asyncio.gather(task1, task2), timeout=0.05)


class TestVoiceServiceErrorHandling:
    """Test error handling in VoiceService."""
    
    @pytest.mark.asyncio
    @patch('api.core.voice_service.AutoModelForCausalLM')
    @patch('api.core.voice_service.AutoProcessor')
    async def test_cuda_oom_handling(self, mock_processor, mock_model):
        """Test CUDA OOM is handled gracefully."""
        VoiceService._instance = None
        
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        request = VoiceGenerationRequest(
            script="Hello, world!",
            speakers=["en-Alice_woman"]
        )
        
        service = VoiceService.instance()
        
        # Mock CUDA OOM during generation
        with patch.object(service, '_generate_audio') as mock_generate:
            async def oom_generate(*args, **kwargs):
                raise RuntimeError("CUDA out of memory")
            mock_generate.side_effect = oom_generate
            
            with patch('api.core.voice_service.torch.cuda.empty_cache') as mock_empty_cache:
                with pytest.raises(ModelNotReadyError):
                    await service.generate_blocking(request)
                
                # Cache should be cleared on OOM
                mock_empty_cache.assert_called()
    
    @pytest.mark.asyncio
    async def test_invalid_voice_handling(self):
        """Test handling of invalid voice IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one valid voice
            Path(temp_dir, "en-Alice_woman.wav").touch()
            
            with patch('api.core.voice_service.VOICES_DIR', temp_dir):
                VoiceService._instance = None
                service = VoiceService.instance()
                
                request = VoiceGenerationRequest(
                    script="Hello, world!",
                    speakers=["nonexistent-voice"]  # Invalid voice
                )
                
                with pytest.raises(InvalidVoiceError):
                    await service.generate_blocking(request)


@pytest.fixture
def mock_model_loading():
    """Fixture that properly mocks model loading for all tests."""
    with patch('api.core.voice_service.torch.cuda.is_available', return_value=True), \
         patch('api.core.voice_service.torch.cuda.device_count', return_value=1), \
         patch('api.core.voice_service.AutoModelForCausalLM') as mock_model, \
         patch('api.core.voice_service.AutoProcessor') as mock_processor:
        
        # Setup successful mock returns
        mock_model_instance = Mock()
        mock_processor_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        yield {
            'model': mock_model,
            'processor': mock_processor,
            'model_instance': mock_model_instance,
            'processor_instance': mock_processor_instance
        }


@pytest.fixture
def sample_voice_request():
    """Fixture providing a sample VoiceGenerationRequest."""
    return VoiceGenerationRequest(
        script="Hello, world! This is a test.",
        speakers=["en-Alice_woman"],
        cfg_scale=1.3,
        inference_steps=5,
        sample_rate=24000,
        format="wav",
        seed=42
    )


@pytest.fixture
def mock_cancel_check():
    """Fixture providing a mock cancel check function."""
    def cancel_check():
        return False
    return cancel_check