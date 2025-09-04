# ABOUTME: Pytest configuration and shared fixtures
# ABOUTME: Sets up test environment and common fixtures for API tests
import pytest
import asyncio
from unittest.mock import patch

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_torch_cuda():
    """Mock torch.cuda.is_available to return True for all tests"""
    with patch('torch.cuda.is_available', return_value=True):
        yield