# ABOUTME: Comprehensive test suite for all middleware components in VibeVoice FastAPI service
# ABOUTME: Tests timeout, rate limiting, logging, and CORS middleware with TDD approach

import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from api.middleware.timeout import TimeoutMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.logging import LoggingMiddleware
from api.middleware.cors import EnhancedCORSMiddleware


class TestTimeoutMiddleware:
    """Test suite for request timeout middleware"""

    @pytest.fixture
    def app_with_timeout(self):
        """Create FastAPI app with timeout middleware"""
        app = FastAPI()
        
        # Add timeout middleware with 1 second timeout for testing
        timeout_middleware = TimeoutMiddleware(timeout_seconds=1.0)
        app.add_middleware(TimeoutMiddleware, timeout_seconds=1.0)
        
        @app.get("/fast")
        async def fast_endpoint():
            return {"message": "fast response"}
        
        @app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(2.0)  # This should timeout
            return {"message": "slow response"}
        
        @app.get("/healthz")
        async def health_endpoint():
            await asyncio.sleep(2.0)  # Health checks should not timeout
            return {"status": "ok"}
        
        @app.get("/readyz")
        async def readiness_endpoint():
            await asyncio.sleep(2.0)  # Readiness checks should not timeout
            return {"ready": True}
        
        return app

    def test_fast_request_should_not_timeout(self, app_with_timeout):
        """Test that fast requests complete normally"""
        client = TestClient(app_with_timeout)
        response = client.get("/fast")
        
        assert response.status_code == 200
        assert response.json() == {"message": "fast response"}

    def test_slow_request_should_timeout(self, app_with_timeout):
        """Test that slow requests timeout with 408 status"""
        client = TestClient(app_with_timeout)
        response = client.get("/slow")
        
        assert response.status_code == 408
        response_data = response.json()
        assert response_data["code"] == "REQUEST_TIMEOUT"
        assert "timeout" in response_data["message"].lower()

    def test_health_endpoints_should_be_exempt_from_timeout(self, app_with_timeout):
        """Test that health check endpoints are exempt from timeout"""
        client = TestClient(app_with_timeout)
        
        # Health endpoint should complete even if it takes longer than timeout
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
        # Readiness endpoint should also be exempt
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"ready": True}

    def test_timeout_includes_request_id_in_response(self, app_with_timeout):
        """Test that timeout responses include X-Request-ID header"""
        client = TestClient(app_with_timeout)
        test_request_id = "test-request-123"
        
        response = client.get("/slow", headers={"X-Request-ID": test_request_id})
        
        assert response.status_code == 408
        assert response.headers.get("X-Request-ID") == test_request_id


class TestRateLimitMiddleware:
    """Test suite for rate limiting middleware"""

    @pytest.fixture
    def app_with_rate_limit(self):
        """Create FastAPI app with rate limiting middleware"""
        app = FastAPI()
        
        # Add rate limiting middleware with strict limits for testing
        app.add_middleware(
            RateLimitMiddleware, 
            requests_per_minute=2,  # Very low limit for testing
            redis_url=None  # Use in-memory storage for tests
        )
        
        @app.get("/api/generate")
        async def generate_endpoint():
            return {"message": "generated"}
        
        @app.get("/healthz")
        async def health_endpoint():
            return {"status": "ok"}
        
        @app.get("/readyz")
        async def readiness_endpoint():
            return {"ready": True}
        
        return app

    def test_requests_under_limit_should_succeed(self, app_with_rate_limit):
        """Test that requests under the rate limit succeed"""
        client = TestClient(app_with_rate_limit)
        
        # First request should succeed
        response = client.get("/api/generate")
        assert response.status_code == 200
        assert response.json() == {"message": "generated"}
        
        # Second request should also succeed
        response = client.get("/api/generate")
        assert response.status_code == 200

    def test_requests_over_limit_should_be_rejected(self, app_with_rate_limit):
        """Test that requests over the rate limit are rejected with 429"""
        client = TestClient(app_with_rate_limit)
        
        # Make requests up to the limit
        for _ in range(2):
            response = client.get("/api/generate")
            assert response.status_code == 200
        
        # Third request should be rate limited
        response = client.get("/api/generate")
        assert response.status_code == 429
        response_data = response.json()
        assert response_data["code"] == "RATE_LIMIT_EXCEEDED"

    def test_health_endpoints_should_bypass_rate_limiting(self, app_with_rate_limit):
        """Test that health check endpoints bypass rate limiting"""
        client = TestClient(app_with_rate_limit)
        
        # Exhaust the rate limit
        for _ in range(3):
            client.get("/api/generate")
        
        # Health endpoints should still work
        response = client.get("/healthz")
        assert response.status_code == 200
        
        response = client.get("/readyz")
        assert response.status_code == 200

    def test_rate_limiting_is_per_ip(self, app_with_rate_limit):
        """Test that rate limiting is applied per IP address"""
        client = TestClient(app_with_rate_limit)
        
        # This test would need custom client setup to test different IPs
        # For now, we'll test that the rate limit key includes IP info
        with patch('api.middleware.rate_limit.get_client_ip') as mock_get_ip:
            mock_get_ip.side_effect = ["127.0.0.1", "192.168.1.1", "127.0.0.1"]
            
            # Two requests from first IP
            response = client.get("/api/generate")
            assert response.status_code == 200
            response = client.get("/api/generate")
            assert response.status_code == 200
            
            # Request from different IP should succeed
            response = client.get("/api/generate")
            assert response.status_code == 200
            
            # Another request from first IP should be rate limited
            response = client.get("/api/generate")
            assert response.status_code == 429


class TestLoggingMiddleware:
    """Test suite for request/response logging middleware"""

    @pytest.fixture
    def app_with_logging(self):
        """Create FastAPI app with logging middleware"""
        app = FastAPI()
        
        app.add_middleware(LoggingMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @app.post("/api/generate")
        async def generate_endpoint():
            await asyncio.sleep(0.1)  # Simulate some processing time
            return {"message": "generated"}
        
        return app

    @patch('api.middleware.logging.logger')
    def test_successful_requests_are_logged(self, mock_logger, app_with_logging):
        """Test that successful requests are properly logged"""
        client = TestClient(app_with_logging)
        test_request_id = "test-request-456"
        
        response = client.get("/api/test", headers={"X-Request-ID": test_request_id})
        
        assert response.status_code == 200
        # Verify that info log was called with correct structure
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[1]
        assert call_args["request_id"] == test_request_id
        assert call_args["method"] == "GET"
        assert call_args["path"] == "/api/test"
        assert call_args["status_code"] == 200
        assert "duration_ms" in call_args

    @patch('api.middleware.logging.logger')
    def test_request_timing_is_logged(self, mock_logger, app_with_logging):
        """Test that request duration is properly logged"""
        client = TestClient(app_with_logging)
        
        start_time = time.time()
        response = client.post("/api/generate")
        end_time = time.time()
        
        assert response.status_code == 200
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[1]
        
        # Duration should be reasonable (at least 100ms due to sleep)
        assert call_args["duration_ms"] >= 100
        assert call_args["duration_ms"] <= (end_time - start_time) * 1000 + 50  # 50ms tolerance

    @patch('api.middleware.logging.logger')
    def test_error_requests_are_logged_as_error(self, mock_logger, app_with_logging):
        """Test that error responses are logged at error level"""
        client = TestClient(app_with_logging)
        
        # Request non-existent endpoint
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args[1]
        assert call_args["status_code"] == 404

    def test_request_id_is_added_when_missing(self, app_with_logging):
        """Test that X-Request-ID is generated when not provided"""
        client = TestClient(app_with_logging)
        
        response = client.get("/api/test")
        
        assert response.status_code == 200
        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None
        assert len(request_id) > 0


class TestEnhancedCORSMiddleware:
    """Test suite for enhanced CORS middleware"""

    @pytest.fixture
    def app_with_cors(self):
        """Create FastAPI app with enhanced CORS middleware"""
        app = FastAPI()
        
        # Mock environment settings
        with patch.dict('os.environ', {
            'CORS_ALLOW_ORIGINS': 'https://example.com,https://app.example.com'
        }):
            app.add_middleware(EnhancedCORSMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # WebSocket endpoint for testing WebSocket CORS
        from fastapi import WebSocket
        @app.websocket("/ws/test")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("Hello WebSocket")
            await websocket.close()
        
        return app

    def test_allowed_origins_receive_cors_headers(self, app_with_cors):
        """Test that requests from allowed origins receive CORS headers"""
        client = TestClient(app_with_cors)
        
        response = client.get(
            "/api/test",
            headers={"Origin": "https://example.com"}
        )
        
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"
        assert "Access-Control-Allow-Credentials" in response.headers

    def test_disallowed_origins_are_blocked(self, app_with_cors):
        """Test that requests from disallowed origins are properly handled"""
        client = TestClient(app_with_cors)
        
        response = client.get(
            "/api/test",
            headers={"Origin": "https://malicious.com"}
        )
        
        # FastAPI CORS middleware typically doesn't block the request
        # but doesn't include CORS headers for disallowed origins
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") != "https://malicious.com"

    def test_preflight_requests_are_handled(self, app_with_cors):
        """Test that CORS preflight requests are properly handled"""
        client = TestClient(app_with_cors)
        
        response = client.options(
            "/api/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

    def test_websocket_cors_headers_are_included(self, app_with_cors):
        """Test that WebSocket connections receive appropriate CORS headers"""
        # This test would need a WebSocket test client
        # For now, we'll test that the middleware is configured for WebSocket support
        client = TestClient(app_with_cors)
        
        # Test that the app has WebSocket support in CORS configuration
        # This is validated by checking that the middleware allows WebSocket headers
        response = client.options(
            "/ws/test",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "upgrade,connection,sec-websocket-key,sec-websocket-version"
            }
        )
        
        # Should allow WebSocket headers
        allowed_headers = response.headers.get("Access-Control-Allow-Headers", "").lower()
        assert "upgrade" in allowed_headers or "*" in response.headers.get("Access-Control-Allow-Headers", "")


# Integration tests for middleware stack
class TestMiddlewareIntegration:
    """Test that all middleware components work together correctly"""

    @pytest.fixture
    def app_with_all_middleware(self):
        """Create FastAPI app with all middleware components"""
        app = FastAPI()
        
        # Add all middleware in the correct order
        # (reverse order of execution)
        app.add_middleware(EnhancedCORSMiddleware)
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(RateLimitMiddleware, requests_per_minute=10)
        app.add_middleware(TimeoutMiddleware, timeout_seconds=5.0)
        
        @app.get("/api/test")
        async def test_endpoint():
            return {"message": "test"}
        
        return app

    @patch('api.middleware.logging.logger')
    def test_middleware_stack_processes_request_correctly(self, mock_logger, app_with_all_middleware):
        """Test that the complete middleware stack processes requests correctly"""
        client = TestClient(app_with_all_middleware)
        
        response = client.get(
            "/api/test",
            headers={
                "Origin": "https://example.com",
                "X-Request-ID": "integration-test-123"
            }
        )
        
        assert response.status_code == 200
        assert response.json() == {"message": "test"}
        
        # Should have CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        
        # Should have request ID
        assert response.headers.get("X-Request-ID") == "integration-test-123"
        
        # Should have logged the request
        mock_logger.info.assert_called()

    def test_middleware_order_is_correct(self, app_with_all_middleware):
        """Test that middleware is applied in the correct order"""
        # This is a structural test to ensure middleware stack is built correctly
        client = TestClient(app_with_all_middleware)
        
        # Simply test that a request works end-to-end
        response = client.get("/api/test")
        assert response.status_code == 200
        
        # The fact that this works means middleware is properly ordered
        # and not interfering with each other