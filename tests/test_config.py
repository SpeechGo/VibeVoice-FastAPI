# ABOUTME: Test cases for configuration system validating environment variable handling
# ABOUTME: Ensures Settings class properly loads, validates, and provides defaults per contract

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSettings:
    """Test configuration system per contract specifications."""

    def setup_method(self):
        """Clear environment variables and reset singleton before each test."""
        env_vars = [
            "MODEL_PATH",
            "VOICES_DIR", 
            "MAX_CONCURRENCY",
            "TIMEOUT_SEC",
            "CORS_ALLOW_ORIGINS",
            "LOG_LEVEL"
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
                
        # Reset singleton for testing
        from api.config import Settings
        Settings._reset_instance()

    def test_default_values_per_contract(self):
        """Test that default values match contract specifications."""
        from api.config import Settings
        
        settings = Settings()
        
        # Contract specifications
        assert settings.model_path == "microsoft/VibeVoice-1.5B"
        assert settings.voices_dir == "voices/"
        assert settings.max_concurrency == 1
        assert settings.timeout_sec == 300
        assert settings.cors_allow_origins == []
        assert settings.log_level == "info"

    def test_environment_variable_override(self):
        """Test that environment variables properly override defaults."""
        from api.config import Settings
        
        with patch.dict(os.environ, {
            "MODEL_PATH": "microsoft/VibeVoice-Large",
            "VOICES_DIR": "/custom/voices/",
            "MAX_CONCURRENCY": "4",
            "TIMEOUT_SEC": "600",
            "CORS_ALLOW_ORIGINS": "http://localhost:3000,https://example.com",
            "LOG_LEVEL": "debug"
        }):
            settings = Settings()
            
            assert settings.model_path == "microsoft/VibeVoice-Large"
            assert settings.voices_dir == "/custom/voices/"
            assert settings.max_concurrency == 4
            assert settings.timeout_sec == 600
            assert settings.cors_allow_origins == ["http://localhost:3000", "https://example.com"]
            assert settings.log_level == "debug"

    def test_cors_origins_parsing(self):
        """Test comma-separated CORS origins parsing."""
        from api.config import Settings
        
        test_cases = [
            ("", []),
            ("http://localhost:3000", ["http://localhost:3000"]),
            ("http://localhost:3000,https://example.com", ["http://localhost:3000", "https://example.com"]),
            ("http://localhost:3000, https://example.com", ["http://localhost:3000", "https://example.com"]),
        ]
        
        for origins_str, expected in test_cases:
            with patch.dict(os.environ, {"CORS_ALLOW_ORIGINS": origins_str}):
                settings = Settings()
                assert settings.cors_allow_origins == expected

    def test_validation_positive_integers(self):
        """Test validation of positive integer fields."""
        from api.config import Settings
        
        with patch.dict(os.environ, {"MAX_CONCURRENCY": "0"}):
            with pytest.raises(ValueError, match="must be positive"):
                Settings()
        
        with patch.dict(os.environ, {"TIMEOUT_SEC": "-1"}):
            with pytest.raises(ValueError, match="must be positive"):
                Settings()

    def test_validation_log_level(self):
        """Test validation of log level values."""
        from api.config import Settings
        
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        
        for level in valid_levels:
            with patch.dict(os.environ, {"LOG_LEVEL": level}):
                settings = Settings()
                assert settings.log_level == level
                
        with patch.dict(os.environ, {"LOG_LEVEL": "invalid"}):
            with pytest.raises(ValueError, match="Invalid log level"):
                Settings()

    def test_model_path_validation(self):
        """Test validation of supported model paths."""
        from api.config import Settings
        
        valid_models = [
            "microsoft/VibeVoice-1.5B",
            "microsoft/VibeVoice-Large"
        ]
        
        for model in valid_models:
            with patch.dict(os.environ, {"MODEL_PATH": model}):
                settings = Settings()
                assert settings.model_path == model
                
        with patch.dict(os.environ, {"MODEL_PATH": "invalid/model"}):
            with pytest.raises(ValueError, match="Unsupported model"):
                Settings()

    def test_voices_dir_path_handling(self):
        """Test voices directory path handling and validation."""
        from api.config import Settings
        
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_path = Path(temp_dir) / "voices"
            voices_path.mkdir()
            
            with patch.dict(os.environ, {"VOICES_DIR": str(voices_path)}):
                settings = Settings()
                assert settings.voices_dir == str(voices_path)

    def test_singleton_behavior(self):
        """Test that Settings behaves as singleton within same environment."""
        from api.config import Settings
        
        # Don't reset singleton for this test
        settings1 = Settings()
        settings2 = Settings()
        
        # Should be the same instance
        assert settings1 is settings2
        
        # But values should be the same even after re-initialization
        assert settings1.model_path == settings2.model_path

    def test_device_pinning_contract(self):
        """Test that device is pinned to cuda:0 per v1 contract."""
        from api.config import Settings
        
        settings = Settings()
        
        # v1 contract specifies cuda:0 pinning
        assert settings.device == "cuda:0"
        assert settings.dtype == "bfloat16"

    def test_immutability(self):
        """Test that settings are immutable after creation."""
        from api.config import Settings
        
        settings = Settings()
        
        # Should not be able to modify settings
        with pytest.raises(AttributeError):
            settings.model_path = "different/model"