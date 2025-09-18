# ABOUTME: This file implements the core VoiceService class with singleton pattern and model integration.
# ABOUTME: Provides blocking generation, streaming, voice scanning, and CUDA-optimized model loading from HuggingFace Hub.

from __future__ import annotations
import os
import asyncio
import threading
import time
import logging
import re
from pathlib import Path
from typing import AsyncIterator, Optional, Callable, Protocol, List
from dataclasses import dataclass

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
import sys
import os

# Optional VibeVoice streaming support
AsyncAudioStreamer = None
VibeVoiceProcessor = None
VibeVoiceForConditionalGenerationInference = None

# Try to import VibeVoice model if available
try:
    # Add parent directory to path temporarily to import VibeVoice
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vibevoice_dir = os.path.join(os.path.dirname(parent_dir), 'VibeVoice')
    if os.path.exists(vibevoice_dir):
        sys.path.insert(0, vibevoice_dir)
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.modular.streamer import AsyncAudioStreamer
        VIBEVOICE_AVAILABLE = True
        sys.path.pop(0)  # Remove from path after import
    else:
        VIBEVOICE_AVAILABLE = False
except ImportError:
    VIBEVOICE_AVAILABLE = False
    AsyncAudioStreamer = None
    VibeVoiceProcessor = None
    VibeVoiceForConditionalGenerationInference = None
    
if not VIBEVOICE_AVAILABLE:
    # Fallback to AutoModelForCausalLM if VibeVoice not available
    from transformers import AutoModelForCausalLM

from api.models.errors import ModelNotReadyError, ServiceBusyError, InvalidVoiceError, GenerationTimeoutError
from api.models.requests import VoiceGenerationRequest
from api.models.responses import VoiceInfo
from api.utils.audio import float32_to_pcm16, pcm16_to_wav, concat_pcm16
from api.utils.streaming import AudioStreamer

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Configuration - these would normally come from config.py
MODEL_PATH = os.getenv('MODEL_PATH', 'microsoft/VibeVoice-1.5B')
VOICES_DIR = os.getenv('VOICES_DIR', 'voices')
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '1'))
TIMEOUT_SEC = int(os.getenv('TIMEOUT_SEC', '300'))

logger = logging.getLogger(__name__)


class CancelCheck(Protocol):
    """Protocol for cancellation check functions."""
    def __call__(self) -> bool: 
        """Return True if operation should be cancelled."""
        ...


@dataclass
class GenerateResult:
    """Result from blocking generation."""
    wav_bytes: bytes
    sample_rate: int
    duration_sec: float


class VoiceService:
    """Core voice generation service using VibeVoice models from HuggingFace Hub.
    
    Implements singleton pattern and provides both blocking and streaming generation
    with CUDA optimization and concurrency control.
    """
    
    _instance: Optional['VoiceService'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Private constructor - use instance() class method instead."""
        if VoiceService._instance is not None:
            raise RuntimeError("Use VoiceService.instance() instead of direct instantiation")
        
        self._model = None
        self._processor = None
        self._device = 'cuda:0'
        # Select dtype adaptively; can be overridden by env TORCH_DTYPE
        self._dtype = self._select_dtype()
        self._model_loaded = False
        self._model_error = None
        self._voices_cache: Optional[List[VoiceInfo]] = None
        self._voices_cache_time = 0
        self._cache_ttl = 60  # Cache voices for 60 seconds
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._current_concurrency = 0
        self._concurrency_lock = threading.Lock()
        
        # Initialize model loading (skip in test mode)
        import os
        if not os.getenv('TESTING', False):
            self._load_model()

    def _select_dtype(self):
        """Select optimal torch dtype with environment override.

        Priority:
        - TORCH_DTYPE=bf16/fp16/f16/float16/float32
        - If GPU supports bf16, use bfloat16; else use float16.
        Also enables TF32 where beneficial.
        """
        # Env override
        dtype_env = os.getenv('TORCH_DTYPE', '').lower()
        str_to_dtype = {
            'bf16': torch.bfloat16,
            'bfloat16': torch.bfloat16,
            'fp16': torch.float16,
            'f16': torch.float16,
            'float16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if dtype_env in str_to_dtype:
            selected = str_to_dtype[dtype_env]
        else:
            # Auto-detect
            selected = torch.float16
            try:
                is_bf16_supported = getattr(torch.cuda, 'is_bf16_supported', None)
                if callable(is_bf16_supported) and is_bf16_supported():
                    selected = torch.bfloat16
                else:
                    # Heuristic using device capability for Ampere/Ada/Hopper
                    if torch.cuda.is_available():
                        major, minor = torch.cuda.get_device_capability(0)
                        # Ada (8.9) and Hopper (9.x) have solid bf16 support
                        if (major, minor) >= (8, 9):
                            selected = torch.bfloat16
            except Exception:
                pass

        # Enable TF32 when available to improve throughput on Ampere+ for matmul/convs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        logging.getLogger(__name__).info(f"Selected torch dtype: {selected}")
        return selected
    
    @classmethod
    def instance(cls) -> 'VoiceService':
        """Get or create the singleton VoiceService instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _fix_prepare_cache_compatibility(self):
        """Fix compatibility issues with VibeVoice model."""
        if VIBEVOICE_AVAILABLE and self._model is not None:
            # Fix 1: Add missing num_hidden_layers to config
            if hasattr(self._model, 'config') and not hasattr(self._model.config, 'num_hidden_layers'):
                # Get it from the language model config if available
                if hasattr(self._model, 'model') and hasattr(self._model.model, 'language_model'):
                    if hasattr(self._model.model.language_model, 'config'):
                        lm_config = self._model.model.language_model.config
                        if hasattr(lm_config, 'num_hidden_layers'):
                            self._model.config.num_hidden_layers = lm_config.num_hidden_layers
                            logger.info(f"Added missing num_hidden_layers={lm_config.num_hidden_layers} to model config")
            
            # Fix 2: Handle _prepare_cache_for_generation argument mismatch
            # The issue is VibeVoice calls it with 6 args but base class expects 5
            from transformers.generation.utils import GenerationMixin
            
            # Get the original method from GenerationMixin base class
            base_prepare_cache = GenerationMixin._prepare_cache_for_generation
            
            # Create a wrapper that adapts the call
            def wrapped_prepare_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device=None):
                # In transformers 4.51.3, the method expects device as 6th arg
                # Pass all arguments including device
                return base_prepare_cache(self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)
            
            # Replace the method on the model instance
            import types
            self._model._prepare_cache_for_generation = types.MethodType(wrapped_prepare_cache, self._model)
            logger.info("Applied _prepare_cache_for_generation compatibility fix")
    
    def _load_model(self):
        """Load the VibeVoice model from HuggingFace Hub."""
        try:
            logger.info(f"Loading model: {MODEL_PATH} on {self._device}")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, but service is configured for cuda:0")
            
            if torch.cuda.device_count() == 0:
                raise RuntimeError("No CUDA devices available")
            
            # Check if flash_attn is available (can be disabled via env)
            disable_fa = os.getenv('DISABLE_FLASH_ATTN', '0') in ('1', 'true', 'True')
            if not disable_fa:
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = 'flash_attention_2'
                    logger.info("Flash Attention 2 is available, using it for better performance")
                except ImportError:
                    attn_impl = 'sdpa'
                    logger.info("Flash Attention 2 not available, using sdpa attention")
            else:
                attn_impl = 'sdpa'
                logger.info("Flash Attention explicitly disabled via env; using sdpa attention")
            
            # Load model with specific configuration for VibeVoice
            if VIBEVOICE_AVAILABLE:
                logger.info("Using local VibeVoice implementation")
                self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=self._dtype,
                    device_map={'': self._device},
                    attn_implementation=attn_impl
                )
                self._processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
            else:
                logger.info("Using transformers AutoModel with trust_remote_code")
                self._model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    torch_dtype=self._dtype,
                    device_map={'': self._device},
                    attn_implementation=attn_impl
                )
                self._processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
            logger.info("Processor loaded successfully")
            
            # Set model to evaluation mode
            self._model.eval()
            logger.info("Model set to evaluation mode")
            
            # Setup noise scheduler exactly like the demo
            if hasattr(self._model, 'model') and hasattr(self._model.model, 'noise_scheduler'):
                self._model.model.noise_scheduler = self._model.model.noise_scheduler.from_config(
                    self._model.model.noise_scheduler.config,
                    algorithm_type='sde-dpmsolver++',
                    beta_schedule='squaredcos_cap_v2'
                )
                # Set default inference steps
                if hasattr(self._model, 'set_ddpm_inference_steps'):
                    self._model.set_ddpm_inference_steps(num_steps=5)
                logger.info("Noise scheduler configured for SDE-DPMSolver++")
            
            # Apply compatibility fix
            self._fix_prepare_cache_compatibility()
            
            self._model_loaded = True
            self._model_error = None
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
            # Try fallback attention implementation
            if "flash_attention_2" in str(e) or "flash_attn" in str(e) or "FlashAttention" in str(e):
                try:
                    logger.info("Falling back to sdpa attention implementation")
                    if VIBEVOICE_AVAILABLE:
                        self._model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            MODEL_PATH,
                            torch_dtype=self._dtype,
                            device_map={'': self._device},
                            attn_implementation='sdpa'
                        )
                        self._processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
                    else:
                        self._model = AutoModelForCausalLM.from_pretrained(
                            MODEL_PATH,
                            trust_remote_code=True,
                            torch_dtype=self._dtype,
                            device_map={'': self._device},
                            attn_implementation='sdpa'
                        )
                        self._processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
                    
                    # Set model to evaluation mode
                    self._model.eval()
                    
                    # Setup noise scheduler exactly like the demo
                    if hasattr(self._model, 'model') and hasattr(self._model.model, 'noise_scheduler'):
                        self._model.model.noise_scheduler = self._model.model.noise_scheduler.from_config(
                            self._model.model.noise_scheduler.config,
                            algorithm_type='sde-dpmsolver++',
                            beta_schedule='squaredcos_cap_v2'
                        )
                        # Set default inference steps
                        if hasattr(self._model, 'set_ddpm_inference_steps'):
                            self._model.set_ddpm_inference_steps(num_steps=5)
                        logger.info("Noise scheduler configured for SDE-DPMSolver++ (fallback)")
                    
                    self._model_loaded = True
                    self._model_error = None
                    logger.info("Model loaded successfully with sdpa attention")
                    return
                except Exception as fallback_e:
                    logger.error(f"Fallback attention also failed: {fallback_e}")
                    e = fallback_e
            
            self._model_loaded = False
            self._model_error = str(e)
            self._model = None
            self._processor = None
    
    def ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model_loaded and self._model is not None and self._processor is not None
    
    def list_voices(self, include_hidden: bool = False) -> List[VoiceInfo]:
        """List available voice presets by scanning the voices directory.
        
        Args:
            include_hidden: If True, include hidden voices (starting with '.')
            
        Returns:
            List of VoiceInfo objects for available voices
        """
        # Check cache first
        current_time = time.time()
        if (self._voices_cache is not None and 
            current_time - self._voices_cache_time < self._cache_ttl):
            voices = self._voices_cache
        else:
            # Scan voices directory
            voices = self._scan_voices_directory()
            self._voices_cache = voices
            self._voices_cache_time = current_time
        
        # Filter hidden voices if requested
        if not include_hidden:
            voices = [v for v in voices if not v.id.startswith('.')]
        
        return voices
    
    def _scan_voices_directory(self) -> List[VoiceInfo]:
        """Scan the voices directory for available voice files.
        
        Returns:
            List of VoiceInfo objects
        """
        voices = []
        voices_path = Path(VOICES_DIR)
        
        if not voices_path.exists():
            logger.warning(f"Voices directory does not exist: {voices_path}")
            return voices
        
        # Scan for WAV files (reference audio) and JSON/YAML files (metadata)
        for voice_file in voices_path.iterdir():
            if voice_file.is_file() and voice_file.suffix.lower() in ['.wav', '.json', '.yaml']:
                voice_info = self._parse_voice_file(voice_file)
                if voice_info:
                    voices.append(voice_info)
        
        logger.info(f"Found {len(voices)} voices in {voices_path}")
        return voices
    
    def _parse_voice_file(self, voice_file: Path) -> Optional[VoiceInfo]:
        """Parse a voice file to extract VoiceInfo.
        
        Args:
            voice_file: Path to the voice file
            
        Returns:
            VoiceInfo object or None if parsing fails
        """
        try:
            # Use filename stem as voice ID
            voice_id = voice_file.stem
            
            # Try to parse metadata from filename (e.g., "en-Alice_woman.wav")
            # Pattern: {lang}-{name}_{gender}.{ext}
            match = re.match(r'^([a-z]{2})-([A-Za-z]+)_([a-z]+)', voice_id)
            
            if match:
                language, name, gender = match.groups()
                return VoiceInfo(
                    id=voice_id,
                    name=name,
                    language=language,
                    gender=gender
                )
            else:
                # Fallback: use filename as name
                return VoiceInfo(
                    id=voice_id,
                    name=voice_id.replace('_', ' ').replace('-', ' ').title(),
                    language=None,
                    gender=None
                )
                
        except Exception as e:
            logger.warning(f"Failed to parse voice file {voice_file}: {e}")
            return None
    
    async def generate_blocking(self,
                               req: VoiceGenerationRequest,
                               cancel_check: Optional[CancelCheck] = None,
                               timeout_sec: Optional[float] = None) -> GenerateResult:
        """Generate audio synchronously and return complete WAV file.
        
        Args:
            req: Voice generation request
            cancel_check: Optional cancellation check function
            timeout_sec: Optional timeout in seconds (default from config)
            
        Returns:
            GenerateResult with WAV bytes and metadata
            
        Raises:
            ModelNotReadyError: If model is not loaded
            ServiceBusyError: If at maximum concurrency
            InvalidVoiceError: If requested voice is not found
            GenerationTimeoutError: If generation times out
        """
        if not self.ready():
            error_msg = self._model_error or "Model not loaded"
            raise ModelNotReadyError(f"Model not ready: {error_msg}")
        
        # Check voice validity
        self._validate_voices(req.speakers)
        
        # Use configured timeout if not specified
        if timeout_sec is None:
            timeout_sec = TIMEOUT_SEC
        
        # Acquire semaphore with timeout
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=1.0)
        except asyncio.TimeoutError:
            raise ServiceBusyError("Service is at maximum capacity")
        
        try:
            with self._concurrency_lock:
                self._current_concurrency += 1
            
            # Run generation with timeout
            try:
                result = await asyncio.wait_for(
                    self._generate_audio(req, cancel_check),
                    timeout=timeout_sec
                )
                return result
            except asyncio.TimeoutError:
                raise GenerationTimeoutError(f"Generation timed out after {timeout_sec}s")
                
        finally:
            with self._concurrency_lock:
                self._current_concurrency -= 1
            self._semaphore.release()
    
    async def stream_pcm16(self,
                          req: VoiceGenerationRequest,
                          cancel_check: Optional[CancelCheck] = None,
                          timeout_sec: Optional[float] = None) -> AsyncIterator[bytes]:
        """Stream audio as PCM16 chunks.
        
        Args:
            req: Voice generation request
            cancel_check: Optional cancellation check function
            timeout_sec: Optional timeout in seconds
            
        Yields:
            Raw PCM16 audio chunks (0.5-2.0s each)
            
        Raises:
            ModelNotReadyError: If model is not loaded
            ServiceBusyError: If at maximum concurrency
            InvalidVoiceError: If requested voice is not found
        """
        if not self.ready():
            error_msg = self._model_error or "Model not loaded"
            raise ModelNotReadyError(f"Model not ready: {error_msg}")
        
        # Check voice validity
        self._validate_voices(req.speakers)
        
        # Use configured timeout if not specified
        if timeout_sec is None:
            timeout_sec = TIMEOUT_SEC
        
        # Acquire semaphore
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=1.0)
        except asyncio.TimeoutError:
            raise ServiceBusyError("Service is at maximum capacity")
        
        try:
            with self._concurrency_lock:
                self._current_concurrency += 1
            
            # Stream generation
            async for chunk in self._stream_pcm16_internal(req, cancel_check, timeout_sec):
                if cancel_check and cancel_check():
                    break
                yield chunk
                
        finally:
            with self._concurrency_lock:
                self._current_concurrency -= 1
            self._semaphore.release()
    
    def _validate_voices(self, speaker_ids: List[str]):
        """Validate that all requested voices exist.
        
        Args:
            speaker_ids: List of voice IDs to validate
            
        Raises:
            InvalidVoiceError: If any voice is not found
        """
        if not speaker_ids:
            raise InvalidVoiceError("At least one speaker must be provided")

        available_voices = {v.id for v in self.list_voices(include_hidden=True)}
        
        for speaker_id in speaker_ids:
            if speaker_id not in available_voices:
                raise InvalidVoiceError(f"Voice not found: {speaker_id}")

    def _format_script_text(self, req: VoiceGenerationRequest) -> str:
        """Format script text into speaker-tagged lines.

        Args:
            req: Voice generation request

        Returns:
            Formatted script text matching model expectations
        """
        lines = req.script.strip().split('\n')
        formatted_lines: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Speaker ') and ':' in line:
                formatted_lines.append(line)
            else:
                speaker_index = len(formatted_lines) % len(req.speakers)
                formatted_lines.append(f"Speaker {speaker_index}: {line}")

        return '\n'.join(formatted_lines)

    def _load_voice_samples(self, speaker_ids: List[str]) -> List[np.ndarray]:
        """Load and preprocess reference voice samples.

        Args:
            speaker_ids: Speaker identifiers to load

        Returns:
            List of numpy arrays containing voice audio
        """
        voice_samples: List[np.ndarray] = []

        for speaker_id in speaker_ids:
            voice_path = os.path.join(VOICES_DIR, f"{speaker_id}.wav")
            if not os.path.exists(voice_path):
                raise InvalidVoiceError(f"Voice file not found: {speaker_id}")

            import soundfile as sf

            audio, sample_rate = sf.read(voice_path)
            if sample_rate != 24000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=24000)

            voice_samples.append(audio)

        return voice_samples

    def _prepare_model_inputs(self, req: VoiceGenerationRequest):
        """Prepare model inputs for generation based on request parameters."""
        script_text = self._format_script_text(req)
        voice_samples = self._load_voice_samples(req.speakers)

        inputs = self._processor(
            text=[script_text],
            voice_samples=[voice_samples],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self._device)

        return inputs

    async def _generate_audio(self,
                             req: VoiceGenerationRequest,
                             cancel_check: Optional[CancelCheck] = None) -> GenerateResult:
        """Internal method to generate audio using the loaded model.
        
        Args:
            req: Voice generation request
            cancel_check: Optional cancellation check function
            
        Returns:
            GenerateResult with generated audio
        """
        try:
            start_time = time.time()
            
            # Run generation in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_model_inference,
                req
            )
            
            if cancel_check and cancel_check():
                return GenerateResult(wav_bytes=b'', sample_rate=req.sample_rate, duration_sec=0.0)
            
            audio_array, sample_rate = result
            
            # Convert to PCM16 and then to WAV
            pcm16_bytes = float32_to_pcm16(audio_array)
            wav_bytes = pcm16_to_wav(pcm16_bytes, sample_rate)
            
            duration_sec = len(audio_array) / sample_rate
            generation_time = time.time() - start_time
            
            # Clean up to prevent memory leaks
            del audio_array
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Generated {duration_sec:.2f}s audio in {generation_time:.2f}s")
            
            return GenerateResult(
                wav_bytes=wav_bytes,
                sample_rate=sample_rate,
                duration_sec=duration_sec
            )
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                # Clear CUDA cache and raise ModelNotReadyError
                torch.cuda.empty_cache()
                logger.error(f"CUDA OOM during generation: {e}")
                raise ModelNotReadyError(f"CUDA out of memory: {e}")
            else:
                logger.error(f"Runtime error during generation: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            raise
    
    def _run_model_inference(self, req: VoiceGenerationRequest) -> tuple[np.ndarray, int]:
        """Run the actual model inference in a thread-safe manner.

        Args:
            req: Generation request parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Set random seed if provided
            if req.seed is not None:
                torch.manual_seed(req.seed)
                np.random.seed(req.seed)

            # Process with the model
            with torch.no_grad():
                # Prepare inputs
                inputs = self._prepare_model_inputs(req)
                
                # Set inference steps if configurable
                if hasattr(self._model, 'set_ddpm_inference_steps'):
                    self._model.set_ddpm_inference_steps(num_steps=req.inference_steps)
                
                # Generate audio exactly like the demo
                if hasattr(self._model, 'generate'):
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=req.cfg_scale,
                        tokenizer=self._processor.tokenizer,
                        generation_config={
                            'do_sample': False,
                        },
                        verbose=False,
                        refresh_negative=True  # Like the demo
                    )
                    
                    # Extract audio from outputs
                    # The output is a VibeVoiceGenerationOutput with 'speech_outputs' field
                    if hasattr(outputs, 'speech_outputs') and outputs.speech_outputs:
                        # speech_outputs is a list of torch tensors
                        audio_tensor = outputs.speech_outputs[0]
                        if torch.is_tensor(audio_tensor):
                            # Convert bfloat16 to float32 before numpy conversion
                            if audio_tensor.dtype == torch.bfloat16:
                                audio_tensor = audio_tensor.float()
                            audio_array = audio_tensor.cpu().numpy()
                        else:
                            audio_array = audio_tensor
                    elif hasattr(outputs, 'audios'):
                        audio_tensor = outputs.audios[0]
                        if audio_tensor.dtype == torch.bfloat16:
                            audio_tensor = audio_tensor.float()
                        audio_array = audio_tensor.cpu().numpy()
                    elif torch.is_tensor(outputs):
                        if outputs.dtype == torch.bfloat16:
                            outputs = outputs.float()
                        audio_array = outputs[0].cpu().numpy()
                    else:
                        # The outputs object itself might be the VibeVoiceGenerationOutput
                        # Log for debugging
                        logger.warning(f"Unexpected outputs type: {type(outputs)}, attributes: {dir(outputs)}")
                        audio_array = outputs
                else:
                    # Fallback if generate method not available
                    outputs = self._model(**inputs)
                    if hasattr(outputs, 'audio'):
                        audio_array = outputs.audio[0].cpu().numpy()
                    elif hasattr(outputs, 'logits'):
                        # This is likely text output, not audio
                        raise RuntimeError("Model returned text instead of audio")
                    else:
                        audio_array = outputs[0].cpu().numpy()
                
                # Ensure audio is float32 and normalized
                if hasattr(audio_array, 'dtype'):
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)
                else:
                    # If it's not a numpy array yet, convert it
                    logger.warning(f"Audio array is not numpy, type: {type(audio_array)}")
                    audio_array = np.array(audio_array, dtype=np.float32)
                
                # Normalize if needed
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                
                # Clean up tensors to prevent memory leak
                if 'outputs' in locals():
                    del outputs
                if 'inputs' in locals():
                    del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear CUDA cache
                
                return audio_array, req.sample_rate
                
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    async def _stream_pcm16_internal(self,
                                    req: VoiceGenerationRequest,
                                    cancel_check: Optional[CancelCheck] = None,
                                    timeout_sec: Optional[float] = None) -> AsyncIterator[bytes]:
        """Internal method for streaming PCM16 audio chunks.

        Args:
            req: Voice generation request
            cancel_check: Optional cancellation check function
            timeout_sec: Optional timeout in seconds

        Yields:
            PCM16 audio chunks
        """
        try:
            if (
                VIBEVOICE_AVAILABLE
                and AsyncAudioStreamer is not None
                and VibeVoiceForConditionalGenerationInference is not None
                and isinstance(self._model, VibeVoiceForConditionalGenerationInference)
            ):
                async for chunk in self._stream_pcm16_vibevoice(req, cancel_check, timeout_sec):
                    yield chunk
                return

            # Fallback: generate entire audio then chunk
            streamer = AudioStreamer(sample_rate=req.sample_rate, chunk_duration_sec=1.5)

            result = await self._generate_audio(req, cancel_check)

            if cancel_check and cancel_check():
                return

            wav_bytes = result.wav_bytes
            pcm16_bytes = wav_bytes[44:] if len(wav_bytes) > 44 else b''

            chunk_size = streamer.bytes_per_chunk
            for i in range(0, len(pcm16_bytes), chunk_size):
                if cancel_check and cancel_check():
                    break

                chunk = pcm16_bytes[i:i + chunk_size]
                if chunk:
                    yield chunk

                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    async def _stream_pcm16_vibevoice(self,
                                      req: VoiceGenerationRequest,
                                      cancel_check: Optional[CancelCheck],
                                      timeout_sec: Optional[float]) -> AsyncIterator[bytes]:
        """Stream audio chunks using the VibeVoice async audio streamer."""

        if AsyncAudioStreamer is None:
            return

        loop = asyncio.get_event_loop()
        start_time = loop.time()

        streamer = AsyncAudioStreamer(batch_size=1)
        generation_exception: list[Exception] = []

        def run_generation():
            try:
                if req.seed is not None:
                    torch.manual_seed(req.seed)
                    np.random.seed(req.seed)

                inputs = self._prepare_model_inputs(req)

                if hasattr(self._model, 'set_ddpm_inference_steps'):
                    self._model.set_ddpm_inference_steps(num_steps=req.inference_steps)

                self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=req.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={'do_sample': False},
                    audio_streamer=streamer,
                    verbose=False,
                    refresh_negative=True,
                )
            except Exception as exc:  # noqa: BLE001
                generation_exception.append(exc)
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

        stream_iter = streamer.get_stream(0).__aiter__()

        try:
            while True:
                if cancel_check and cancel_check():
                    streamer.end()
                    break

                timeout_remaining = None
                if timeout_sec is not None:
                    elapsed = loop.time() - start_time
                    timeout_remaining = timeout_sec - elapsed
                    if timeout_remaining <= 0:
                        streamer.end()
                        raise GenerationTimeoutError(f"Generation timed out after {timeout_sec}s")

                try:
                    if timeout_remaining is not None:
                        audio_chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=timeout_remaining)
                    else:
                        audio_chunk = await stream_iter.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    streamer.end()
                    raise GenerationTimeoutError(f"Generation timed out after {timeout_sec}s") from exc

                if audio_chunk is None:
                    continue

                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.to(torch.float32)
                    np_chunk = audio_chunk.detach().cpu().numpy()
                else:
                    np_chunk = np.asarray(audio_chunk)

                if np_chunk.size == 0:
                    continue

                np_chunk = np_chunk.astype(np.float32).reshape(-1)
                max_val = np.max(np.abs(np_chunk))
                if max_val > 1.0 and max_val != 0:
                    np_chunk = np_chunk / max_val

                pcm_chunk = float32_to_pcm16(np_chunk)
                if pcm_chunk:
                    yield pcm_chunk

        finally:
            streamer.end()
            generation_thread.join(timeout=1.0)
            if generation_thread.is_alive():
                logger.warning("Generation thread did not terminate promptly")

        if generation_exception:
            raise generation_exception[0]

    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': MODEL_PATH,
            'device': self._device,
            'dtype': str(self._dtype),
            'model_loaded': self._model_loaded,
            'model_error': self._model_error,
            'max_concurrency': MAX_CONCURRENCY,
            'current_concurrency': self._current_concurrency
        }
    
    def clear_cache(self):
        """Clear the voices cache to force refresh."""
        self._voices_cache = None
        self._voices_cache_time = 0
        logger.info("Voices cache cleared")
