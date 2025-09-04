# ABOUTME: Tests for monitoring and metrics system
# ABOUTME: Covers Prometheus metrics, GPU monitoring, and metrics endpoint functionality
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from prometheus_client import REGISTRY, CollectorRegistry
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class TestPrometheusMetrics:
    """Test prometheus metrics collection and registration"""
    
    @pytest.fixture(autouse=True)
    def setup_metrics(self):
        """Setup clean metrics registry for each test"""
        # Create a new registry for each test to avoid conflicts
        self.test_registry = CollectorRegistry()
        
        # Patch the metrics module to use our test registry
        with patch('api.monitoring.metrics.REGISTRY', self.test_registry):
            yield
    
    def test_metrics_initialization(self):
        """Test that all required metrics are properly initialized"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Check that all required metrics exist
        assert hasattr(metrics, 'requests_total')
        assert hasattr(metrics, 'request_duration_seconds') 
        assert hasattr(metrics, 'gpu_utilization_ratio')
        assert hasattr(metrics, 'active_connections')
        assert hasattr(metrics, 'model_inference_time_seconds')
    
    def test_request_metrics_recording(self):
        """Test recording request metrics with proper labels"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Test recording request start
        with patch.object(metrics.request_duration_seconds, 'time') as mock_timer:
            mock_timer.return_value = Mock()
            timer = metrics.start_request_timer('POST', '/api/generate')
            assert timer is not None
        
        # Test recording request completion
        metrics.record_request('POST', '/api/generate', 200, 'microsoft/VibeVoice-1.5B')
        
        # Verify counter was incremented
        # Note: In real implementation, we'd check the metric values
        # but for this test, we're verifying the method calls don't error
    
    def test_gpu_metrics_recording(self):
        """Test recording GPU utilization metrics"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Test setting GPU utilization
        metrics.set_gpu_utilization(0.85)
        metrics.set_gpu_memory_usage(8192, 16384)  # 8GB used, 16GB total
        
        # Test GPU temperature recording
        metrics.set_gpu_temperature(72.5)
    
    def test_websocket_connection_tracking(self):
        """Test WebSocket connection counting"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Test connection increment/decrement
        initial_count = 0
        metrics.increment_active_connections()
        metrics.increment_active_connections()
        metrics.decrement_active_connections()
        
        # Should have 1 active connection
    
    def test_model_inference_timing(self):
        """Test model inference time recording"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Test inference timing with context manager
        with patch.object(metrics.model_inference_time_seconds, 'time') as mock_timer:
            mock_timer.return_value = Mock()
            timer = metrics.start_inference_timer('microsoft/VibeVoice-1.5B', 'generation')
            assert timer is not None
    
    def test_metrics_singleton_pattern(self):
        """Test that PrometheusMetrics follows singleton pattern"""
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics1 = PrometheusMetrics()
        metrics2 = PrometheusMetrics()
        
        assert metrics1 is metrics2


class TestGPUMonitor:
    """Test GPU monitoring functionality"""
    
    @pytest.fixture
    def mock_nvml(self):
        """Mock pynvml library"""
        with patch('api.monitoring.gpu_monitor.pynvml') as mock_nvml:
            mock_nvml.nvmlInit.return_value = None
            mock_nvml.nvmlDeviceGetCount.return_value = 1
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
            mock_nvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=85, memory=70)
            mock_nvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
                used=8 * 1024**3,  # 8GB
                total=16 * 1024**3  # 16GB
            )
            mock_nvml.nvmlDeviceGetTemperature.return_value = 72
            mock_nvml.NVML_TEMPERATURE_GPU = 0
            yield mock_nvml
    
    def test_gpu_monitor_initialization(self, mock_nvml):
        """Test GPU monitor initializes correctly"""
        from api.monitoring.gpu_monitor import GPUMonitor
        
        monitor = GPUMonitor()
        assert monitor.device_count == 1
        assert monitor.handles is not None
    
    def test_gpu_stats_collection(self, mock_nvml):
        """Test GPU statistics collection"""
        from api.monitoring.gpu_monitor import GPUMonitor
        
        monitor = GPUMonitor()
        stats = monitor.get_gpu_stats()
        
        assert 'devices' in stats
        assert len(stats['devices']) == 1
        
        device_stats = stats['devices'][0]
        assert 'utilization_gpu' in device_stats
        assert 'utilization_memory' in device_stats
        assert 'memory_used_mb' in device_stats
        assert 'memory_total_mb' in device_stats
        assert 'temperature' in device_stats
        
        # Check values match our mock
        assert device_stats['utilization_gpu'] == 85
        assert device_stats['utilization_memory'] == 70
        assert device_stats['memory_used_mb'] == 8192  # 8GB in MB
        assert device_stats['temperature'] == 72
    
    def test_gpu_monitor_no_gpu_fallback(self):
        """Test GPU monitor handles case when no GPU available"""
        with patch('api.monitoring.gpu_monitor.pynvml') as mock_nvml:
            mock_nvml.nvmlInit.side_effect = Exception("No GPU")
            
            from api.monitoring.gpu_monitor import GPUMonitor
            
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()
            
            # Should return empty stats when no GPU
            assert stats['devices'] == []
            assert stats['available'] is False
    
    @pytest.mark.asyncio
    async def test_gpu_monitoring_task(self, mock_nvml):
        """Test background GPU monitoring task"""
        from api.monitoring.gpu_monitor import GPUMonitor
        from api.monitoring.metrics import PrometheusMetrics
        
        monitor = GPUMonitor()
        metrics = PrometheusMetrics()
        
        # Mock the metrics setter
        with patch.object(metrics, 'set_gpu_utilization') as mock_set_util, \
             patch.object(metrics, 'set_gpu_memory_usage') as mock_set_memory, \
             patch.object(metrics, 'set_gpu_temperature') as mock_set_temp:
            
            # Run monitoring task for a short time
            task = asyncio.create_task(monitor.start_monitoring(metrics, interval=0.1))
            await asyncio.sleep(0.2)  # Let it run for 200ms
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Verify metrics were called
            assert mock_set_util.called
            assert mock_set_memory.called
            assert mock_set_temp.called


class TestMetricsEndpoint:
    """Test metrics endpoint for Prometheus scraping"""
    
    @pytest.fixture
    def test_client(self):
        """Create FastAPI test client"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from api.routes.metrics import router
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    def test_metrics_endpoint_exists(self, test_client):
        """Test that metrics endpoint is accessible"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_endpoint_content_type(self, test_client):
        """Test that metrics endpoint returns proper content type"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_metrics_endpoint_prometheus_format(self, test_client):
        """Test that metrics endpoint returns Prometheus format"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for presence of expected metric names
        assert "vibe_requests_total" in content
        assert "vibe_request_duration_seconds" in content
        assert "vibe_gpu_utilization_ratio" in content
        assert "vibe_active_connections" in content
        assert "vibe_model_inference_time_seconds" in content
    
    def test_metrics_endpoint_performance(self, test_client):
        """Test that metrics endpoint responds quickly"""
        start_time = time.time()
        response = test_client.get("/metrics")
        end_time = time.time()
        
        assert response.status_code == 200
        # Metrics endpoint should respond in under 100ms
        assert (end_time - start_time) < 0.1


class TestMonitoringIntegration:
    """Test integration of monitoring system with FastAPI app"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application"""
        from fastapi import FastAPI
        return FastAPI()
    
    def test_monitoring_middleware_integration(self, mock_app):
        """Test that monitoring middleware can be integrated"""
        from api.monitoring.middleware import MonitoringMiddleware
        
        # Should not raise exception when adding middleware
        mock_app.add_middleware(MonitoringMiddleware)
    
    @pytest.mark.asyncio
    async def test_request_timing_middleware(self):
        """Test request timing middleware functionality"""
        from api.monitoring.middleware import add_monitoring_middleware
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            await asyncio.sleep(0.01)  # Simulate some work
            return {"status": "ok"}
        
        # Add monitoring middleware
        add_monitoring_middleware(app)
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        # Middleware should add timing headers
        assert "X-Process-Time" in response.headers
    
    def test_background_tasks_startup(self):
        """Test that background monitoring tasks start correctly"""
        from api.monitoring.background_tasks import start_monitoring_tasks
        from api.monitoring.metrics import PrometheusMetrics
        
        metrics = PrometheusMetrics()
        
        # Should not raise exception
        tasks = start_monitoring_tasks(metrics)
        assert isinstance(tasks, list)
        
        # Clean up tasks
        for task in tasks:
            if not task.done():
                task.cancel()


class TestMonitoringErrors:
    """Test error handling in monitoring system"""
    
    def test_metrics_with_nvml_unavailable(self):
        """Test metrics system when NVML is unavailable"""
        with patch('api.monitoring.gpu_monitor.pynvml', side_effect=ImportError):
            from api.monitoring.gpu_monitor import GPUMonitor
            
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()
            
            # Should gracefully handle missing NVML
            assert stats['available'] is False
            assert stats['devices'] == []
    
    def test_metrics_registry_errors(self):
        """Test handling of Prometheus registry errors"""
        from api.monitoring.metrics import PrometheusMetrics
        
        with patch('prometheus_client.Counter', side_effect=ValueError("Registry error")):
            # Should handle registry errors gracefully
            try:
                metrics = PrometheusMetrics()
            except Exception as e:
                pytest.fail(f"PrometheusMetrics should handle registry errors: {e}")
    
    def test_gpu_monitoring_exception_handling(self):
        """Test GPU monitoring handles exceptions gracefully"""
        from api.monitoring.gpu_monitor import GPUMonitor
        
        with patch('api.monitoring.gpu_monitor.pynvml') as mock_nvml:
            mock_nvml.nvmlInit.return_value = None
            mock_nvml.nvmlDeviceGetCount.return_value = 1
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
            # Make GPU stats calls raise exceptions
            mock_nvml.nvmlDeviceGetUtilizationRates.side_effect = Exception("GPU error")
            
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()
            
            # Should return empty stats on error
            assert len(stats['devices']) == 0 or stats['devices'][0].get('error') is not None