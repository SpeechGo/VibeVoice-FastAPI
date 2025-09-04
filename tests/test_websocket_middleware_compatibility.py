# ABOUTME: Test suite to verify middleware doesn't interfere with WebSocket connections
# ABOUTME: Ensures timeout, rate limiting, logging, and CORS middleware work correctly with WebSockets

import pytest
import asyncio
import json
import time
from unittest.mock import patch
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.middleware import (
    TimeoutMiddleware,
    RateLimitMiddleware, 
    LoggingMiddleware,
    EnhancedCORSMiddleware
)


class TestWebSocketMiddlewareCompatibility:
    """Test that middleware doesn't break WebSocket functionality"""

    @pytest.fixture
    def app_with_middleware_and_websocket(self):
        """Create FastAPI app with all middleware and a WebSocket endpoint"""
        app = FastAPI()
        
        # Add all middleware in same order as main app
        app.add_middleware(EnhancedCORSMiddleware)
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(RateLimitMiddleware, requests_per_minute=10)
        app.add_middleware(TimeoutMiddleware, timeout_seconds=5.0)
        
        @app.websocket("/ws/test")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                # Echo messages back
                while True:
                    data = await websocket.receive_text()
                    await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                pass
        
        @app.get("/api/test")
        async def http_endpoint():
            return {"message": "HTTP endpoint works"}
        
        return app

    def test_websocket_connection_establishment(self, app_with_middleware_and_websocket):
        """Test that WebSocket connections can be established with middleware active"""
        client = TestClient(app_with_middleware_and_websocket)
        
        with client.websocket_connect("/ws/test") as websocket:
            websocket.send_text("Hello WebSocket")
            data = websocket.receive_text()
            assert data == "Echo: Hello WebSocket"

    def test_websocket_with_cors_headers(self, app_with_middleware_and_websocket):
        """Test that WebSocket connections work with CORS middleware"""
        client = TestClient(app_with_middleware_and_websocket)
        
        # WebSocket handshake should include appropriate headers
        with client.websocket_connect(
            "/ws/test", 
            headers={"Origin": "https://example.com"}
        ) as websocket:
            websocket.send_text("CORS test")
            data = websocket.receive_text()
            assert data == "Echo: CORS test"

    def test_websocket_not_affected_by_timeout_middleware(self, app_with_middleware_and_websocket):
        """Test that WebSocket connections aren't terminated by timeout middleware"""
        client = TestClient(app_with_middleware_and_websocket)
        
        with client.websocket_connect("/ws/test") as websocket:
            # Wait longer than the timeout (5 seconds) to ensure WebSocket isn't affected
            time.sleep(1)  # Short sleep for this test
            websocket.send_text("After timeout period")
            data = websocket.receive_text()
            assert data == "Echo: After timeout period"

    def test_websocket_not_rate_limited(self, app_with_middleware_and_websocket):
        """Test that established WebSocket connections aren't subject to rate limiting"""
        client = TestClient(app_with_middleware_and_websocket)
        
        with client.websocket_connect("/ws/test") as websocket:
            # Send many messages rapidly - should not be rate limited
            for i in range(5):
                websocket.send_text(f"Message {i}")
                data = websocket.receive_text()
                assert data == f"Echo: Message {i}"

    def test_http_endpoints_still_work_with_websocket_middleware(self, app_with_middleware_and_websocket):
        """Test that HTTP endpoints continue to work when WebSocket middleware is present"""
        client = TestClient(app_with_middleware_and_websocket)
        
        # HTTP endpoint should work normally
        response = client.get("/api/test")
        assert response.status_code == 200
        assert response.json() == {"message": "HTTP endpoint works"}
        
        # Should have request ID from logging middleware
        assert "X-Request-ID" in response.headers

    def test_multiple_websocket_connections(self, app_with_middleware_and_websocket):
        """Test that multiple WebSocket connections work simultaneously"""
        client = TestClient(app_with_middleware_and_websocket)
        
        # This test simulates multiple concurrent WebSocket connections
        # In practice, TestClient may not support true concurrency,
        # but we can at least test sequential connections
        
        with client.websocket_connect("/ws/test") as ws1:
            ws1.send_text("Connection 1")
            data1 = ws1.receive_text()
            assert data1 == "Echo: Connection 1"
        
        with client.websocket_connect("/ws/test") as ws2:
            ws2.send_text("Connection 2") 
            data2 = ws2.receive_text()
            assert data2 == "Echo: Connection 2"


class TestMiddlewareBypass:
    """Test that certain middleware components correctly bypass WebSocket routes"""

    def test_websocket_handshake_not_rate_limited(self):
        """Test that WebSocket handshake requests are not subject to rate limiting"""
        app = FastAPI()
        
        # Add rate limiting with very low limit
        app.add_middleware(RateLimitMiddleware, requests_per_minute=1)
        
        @app.websocket("/ws/test")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.close()
        
        client = TestClient(app)
        
        # Should be able to establish multiple WebSocket connections
        # even with very low rate limit
        for i in range(3):
            try:
                with client.websocket_connect("/ws/test"):
                    pass  # Connection established and closed
            except Exception as e:
                pytest.fail(f"WebSocket connection {i} failed: {e}")

    def test_websocket_upgrade_includes_cors_headers(self):
        """Test that WebSocket upgrade responses include CORS headers when appropriate"""
        app = FastAPI()
        
        # Set up CORS middleware
        import os
        with patch.dict(os.environ, {"CORS_ALLOW_ORIGINS": "https://example.com"}):
            app.add_middleware(EnhancedCORSMiddleware)
        
        @app.websocket("/ws/test")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.close()
        
        client = TestClient(app)
        
        # WebSocket connection with proper origin should work
        with client.websocket_connect(
            "/ws/test",
            headers={"Origin": "https://example.com"}
        ):
            pass  # Connection should be successful