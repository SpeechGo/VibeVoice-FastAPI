# ABOUTME: Configuration system for VibeVoice-FastAPI service with environment variable handling
# ABOUTME: Provides Settings singleton with validation per contract specifications

import os
from typing import List, Literal

# Module-level constants for easy import
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/VibeVoice-1.5B")
VOICES_DIR = os.getenv("VOICES_DIR", "voices/")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", "300"))
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")


class Settings:
    """
    Configuration settings for VibeVoice-FastAPI service.
    Singleton class that loads configuration from environment variables
    with validation and defaults per contract specifications.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Always re-initialize to pick up environment changes
        # This is important for testing
            
        # Load and validate all settings
        self._model_path = self._get_model_path()
        self._voices_dir = self._get_voices_dir()
        self._max_concurrency = self._get_max_concurrency()
        self._timeout_sec = self._get_timeout_sec()
        self._cors_allow_origins = self._get_cors_allow_origins()
        self._log_level = self._get_log_level()
        self._rate_limit_requests_per_minute = self._get_rate_limit_requests_per_minute()
        
        # v1 contract: device pinned to cuda:0
        self._device = "cuda:0"
        self._dtype = "bfloat16"
    
    @classmethod
    def _reset_instance(cls):
        """Reset singleton instance for testing purposes only."""
        cls._instance = None
    
    def _get_model_path(self) -> str:
        """Get and validate MODEL_PATH environment variable."""
        model_path = os.getenv("MODEL_PATH", "microsoft/VibeVoice-1.5B")
        
        supported_models = [
            "microsoft/VibeVoice-1.5B",
            "microsoft/VibeVoice-Large"
        ]
        
        if model_path not in supported_models:
            raise ValueError(
                f"Unsupported model: {model_path}. "
                f"Supported models: {', '.join(supported_models)}"
            )
            
        return model_path
    
    def _get_voices_dir(self) -> str:
        """Get VOICES_DIR environment variable."""
        return os.getenv("VOICES_DIR", "voices/")
    
    def _get_max_concurrency(self) -> int:
        """Get and validate MAX_CONCURRENCY environment variable."""
        max_concurrency = int(os.getenv("MAX_CONCURRENCY", "1"))
        
        if max_concurrency <= 0:
            raise ValueError("MAX_CONCURRENCY must be positive")
            
        return max_concurrency
    
    def _get_timeout_sec(self) -> int:
        """Get and validate TIMEOUT_SEC environment variable."""
        timeout_sec = int(os.getenv("TIMEOUT_SEC", "300"))
        
        if timeout_sec <= 0:
            raise ValueError("TIMEOUT_SEC must be positive")
            
        return timeout_sec
    
    def _get_cors_allow_origins(self) -> List[str]:
        """Parse comma-separated CORS_ALLOW_ORIGINS environment variable."""
        origins_str = os.getenv("CORS_ALLOW_ORIGINS", "")
        
        if not origins_str.strip():
            return []
            
        return [origin.strip() for origin in origins_str.split(",")]
    
    def _get_log_level(self) -> Literal["debug", "info", "warning", "error", "critical"]:
        """Get and validate LOG_LEVEL environment variable."""
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        
        if log_level not in valid_levels:
            raise ValueError(
                f"Invalid log level: {log_level}. "
                f"Valid levels: {', '.join(valid_levels)}"
            )
            
        return log_level
    
    def _get_rate_limit_requests_per_minute(self) -> int:
        """Get and validate RATE_LIMIT_REQUESTS_PER_MINUTE environment variable."""
        requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
        
        if requests_per_minute <= 0:
            raise ValueError("RATE_LIMIT_REQUESTS_PER_MINUTE must be positive")
            
        return requests_per_minute
    
    # Properties to provide immutable access
    @property
    def model_path(self) -> str:
        return self._model_path
    
    @property 
    def voices_dir(self) -> str:
        return self._voices_dir
    
    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency
    
    @property
    def timeout_sec(self) -> int:
        return self._timeout_sec
    
    @property
    def cors_allow_origins(self) -> List[str]:
        return self._cors_allow_origins.copy()  # Return copy to prevent mutation
    
    @property
    def log_level(self) -> str:
        return self._log_level
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def dtype(self) -> str:
        return self._dtype
    
    @property
    def rate_limit_requests_per_minute(self) -> int:
        return self._rate_limit_requests_per_minute


# Global function to get settings instance
def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()