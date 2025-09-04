# ABOUTME: Test cases for structured logging infrastructure with request ID tracking
# ABOUTME: Validates logging configuration, structured output, and performance metrics integration

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest
import structlog


class TestLoggingConfiguration:
    """Test logging system configuration and functionality."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Clear any existing loggers
        structlog.reset_defaults()
        logging.getLogger().handlers.clear()

    def test_configure_logging_with_defaults(self):
        """Test logging configuration with default settings."""
        from api.logging_config import configure_logging
        
        configure_logging()
        
        # Get the configured logger
        logger = structlog.get_logger("test")
        
        # Should be able to log without errors
        logger.info("test message", key="value")

    def test_configure_logging_with_custom_level(self):
        """Test logging configuration with custom log level."""
        from api.logging_config import configure_logging
        
        configure_logging(log_level="debug")
        
        # Verify debug level is set
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_json_output_format(self):
        """Test that logs are output in JSON format."""
        from api.logging_config import configure_logging
        
        # Capture log output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            configure_logging(log_file=f.name)
            
            logger = structlog.get_logger("test")
            logger.info("test message", key="value", number=42)
            
            # Force flush
            for handler in logging.getLogger().handlers:
                handler.flush()
            
        # Read the log file and verify JSON format
        with open(f.name, 'r') as file:
            log_content = file.read()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        
        # Should have at least one log entry
        assert len(log_lines) >= 1
        
        # Parse the last log entry
        log_entry = json.loads(log_lines[-1])
        
        assert log_entry["event"] == "test message"
        assert log_entry["key"] == "value"
        assert log_entry["number"] == 42
        assert "timestamp" in log_entry
        assert "level" in log_entry

    def test_request_id_context_manager(self):
        """Test request ID context manager for tracing."""
        from api.logging_config import configure_logging, request_id_context
        
        configure_logging()
        
        test_request_id = str(uuid4())
        
        with request_id_context(test_request_id):
            logger = structlog.get_logger("test")
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
                # Reconfigure to capture this specific log
                handler = logging.FileHandler(f.name)
                handler.setFormatter(
                    structlog.stdlib.ProcessorFormatter(
                        processor=structlog.dev.ConsoleRenderer()
                    )
                )
                root_logger = logging.getLogger()
                root_logger.handlers = [handler]
                
                logger.info("test with request id")
                handler.flush()
                
                # Read and check for request ID
                log_content = Path(f.name).read_text()
                assert test_request_id in log_content

    def test_performance_logging_helpers(self):
        """Test performance logging helper functions."""
        from api.logging_config import configure_logging, log_request_metrics
        
        configure_logging()
        
        # Test log_request_metrics function
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            configure_logging(log_file=f.name)
            
            log_request_metrics(
                method="POST",
                path="/api/generate",
                status_code=200,
                duration_ms=1500.5,
                request_id="test-123"
            )
            
            logging.shutdown()
            
        # Verify metrics were logged
        log_content = Path(f.name).read_text()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        log_entry = json.loads(log_lines[-1])
        
        assert log_entry["event"] == "request_completed"
        assert log_entry["method"] == "POST"
        assert log_entry["path"] == "/api/generate"
        assert log_entry["status_code"] == 200
        assert log_entry["duration_ms"] == 1500.5
        assert log_entry["request_id"] == "test-123"

    def test_error_logging_with_exception(self):
        """Test error logging with exception details."""
        from api.logging_config import configure_logging
        
        configure_logging()
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            configure_logging(log_file=f.name)
            
            logger = structlog.get_logger("test")
            
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                logger.error("Test error occurred", exc_info=e)
            
            logging.shutdown()
            
        # Verify exception details are captured
        log_content = Path(f.name).read_text()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        log_entry = json.loads(log_lines[-1])
        
        assert log_entry["event"] == "Test error occurred"
        assert log_entry["level"] == "error"
        assert "exception" in log_entry or "exc_info" in log_entry

    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        from api.logging_config import configure_logging
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            # Configure with WARNING level
            configure_logging(log_level="warning", log_file=f.name)
            
            logger = structlog.get_logger("test")
            logger.debug("debug message")  # Should not appear
            logger.info("info message")    # Should not appear  
            logger.warning("warning message")  # Should appear
            logger.error("error message")      # Should appear
            
            logging.shutdown()
            
        # Check that only warning and error messages appear
        log_content = Path(f.name).read_text()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        
        # Should have exactly 2 log entries (warning and error)
        assert len(log_lines) == 2
        
        log1 = json.loads(log_lines[0])
        log2 = json.loads(log_lines[1])
        
        assert log1["event"] == "warning message"
        assert log2["event"] == "error message"

    def test_context_preservation(self):
        """Test that logging context is preserved across calls."""
        from api.logging_config import configure_logging, request_id_context
        
        configure_logging()
        
        test_request_id = "context-test-123"
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            configure_logging(log_file=f.name)
            
            with request_id_context(test_request_id):
                logger = structlog.get_logger("test")
                logger.info("first message")
                logger.info("second message", extra_key="extra_value")
            
            logging.shutdown()
            
        # Both messages should have the request ID
        log_content = Path(f.name).read_text()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        
        assert len(log_lines) >= 2
        
        for line in log_lines:
            log_entry = json.loads(line)
            assert log_entry.get("request_id") == test_request_id

    def test_multiple_request_contexts(self):
        """Test handling of nested request contexts."""
        from api.logging_config import configure_logging, request_id_context
        
        configure_logging()
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            configure_logging(log_file=f.name)
            
            with request_id_context("outer-123"):
                logger = structlog.get_logger("test")
                logger.info("outer message")
                
                with request_id_context("inner-456"):
                    logger.info("inner message")
                
                logger.info("outer again")
            
            logging.shutdown()
            
        # Verify request IDs are correct
        log_content = Path(f.name).read_text()
        log_lines = [line for line in log_content.strip().split('\n') if line]
        
        entries = [json.loads(line) for line in log_lines if line]
        
        # Find our test messages
        outer1 = next(e for e in entries if e["event"] == "outer message")
        inner = next(e for e in entries if e["event"] == "inner message")
        outer2 = next(e for e in entries if e["event"] == "outer again")
        
        assert outer1.get("request_id") == "outer-123"
        assert inner.get("request_id") == "inner-456"
        assert outer2.get("request_id") == "outer-123"