# ABOUTME: Prometheus metrics collection for VibeVoice FastAPI service
# ABOUTME: Defines counters, histograms, and gauges for monitoring API performance and GPU utilization
import time
import logging
from typing import Optional, Dict, Any
from threading import Lock

from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge,
    REGISTRY
)

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Singleton class for managing Prometheus metrics for VibeVoice API.
    
    Provides thread-safe access to all metrics and ensures consistent labeling
    across the application.
    """
    
    _instance: Optional['PrometheusMetrics'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'PrometheusMetrics':
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Prometheus metrics"""
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
            
        try:
            # Request metrics
            self.requests_total = Counter(
                'vibe_requests_total',
                'Total number of requests processed by VibeVoice API',
                ['route', 'status', 'model_variant'],
                registry=REGISTRY
            )
            
            self.request_duration_seconds = Histogram(
                'vibe_request_duration_seconds',
                'Request duration in seconds',
                ['route', 'status', 'model_variant'],
                buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, float('inf')),
                registry=REGISTRY
            )
            
            # GPU metrics
            self.gpu_utilization_ratio = Gauge(
                'vibe_gpu_utilization_ratio',
                'GPU utilization ratio (0-1)',
                ['device_id'],
                registry=REGISTRY
            )
            
            self.gpu_memory_usage_ratio = Gauge(
                'vibe_gpu_memory_usage_ratio', 
                'GPU memory usage ratio (0-1)',
                ['device_id'],
                registry=REGISTRY
            )
            
            self.gpu_memory_used_bytes = Gauge(
                'vibe_gpu_memory_used_bytes',
                'GPU memory used in bytes',
                ['device_id'],
                registry=REGISTRY
            )
            
            self.gpu_memory_total_bytes = Gauge(
                'vibe_gpu_memory_total_bytes',
                'GPU memory total in bytes',
                ['device_id'],
                registry=REGISTRY
            )
            
            self.gpu_temperature_celsius = Gauge(
                'vibe_gpu_temperature_celsius',
                'GPU temperature in Celsius',
                ['device_id'],
                registry=REGISTRY
            )
            
            # WebSocket connection metrics
            self.active_connections = Gauge(
                'vibe_active_connections',
                'Number of active WebSocket connections',
                registry=REGISTRY
            )
            
            # Model inference metrics
            self.model_inference_time_seconds = Histogram(
                'vibe_model_inference_time_seconds',
                'Model inference time in seconds',
                ['model_variant', 'operation_type'],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, float('inf')),
                registry=REGISTRY
            )
            
            self.model_queue_size = Gauge(
                'vibe_model_queue_size',
                'Number of requests waiting in model queue',
                registry=REGISTRY
            )
            
            # Error metrics
            self.errors_total = Counter(
                'vibe_errors_total',
                'Total number of errors by type',
                ['error_type', 'route'],
                registry=REGISTRY
            )
            
            self._initialized = True
            logger.info("Prometheus metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
            raise
    
    def record_request(self, method: str, path: str, status_code: int, model_variant: str = "unknown") -> None:
        """Record a completed request"""
        try:
            route = f"{method} {path}"
            status = str(status_code)
            
            self.requests_total.labels(
                route=route,
                status=status,
                model_variant=model_variant
            ).inc()
            
        except Exception as e:
            logger.error(f"Error recording request metric: {e}")
    
    def start_request_timer(self, method: str, path: str, model_variant: str = "unknown"):
        """Start timing a request - returns timer object"""
        try:
            route = f"{method} {path}"
            return self.request_duration_seconds.labels(
                route=route,
                status="pending", 
                model_variant=model_variant
            ).time()
        except Exception as e:
            logger.error(f"Error starting request timer: {e}")
            return None
    
    def record_request_duration(self, method: str, path: str, status_code: int, 
                              duration: float, model_variant: str = "unknown") -> None:
        """Record request duration manually"""
        try:
            route = f"{method} {path}"
            status = str(status_code)
            
            self.request_duration_seconds.labels(
                route=route,
                status=status,
                model_variant=model_variant
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Error recording request duration: {e}")
    
    def set_gpu_utilization(self, utilization: float, device_id: str = "0") -> None:
        """Set GPU utilization ratio (0-1)"""
        try:
            # Convert percentage to ratio if needed
            if utilization > 1.0:
                utilization = utilization / 100.0
                
            self.gpu_utilization_ratio.labels(device_id=device_id).set(utilization)
        except Exception as e:
            logger.error(f"Error setting GPU utilization metric: {e}")
    
    def set_gpu_memory_usage(self, used_bytes: int, total_bytes: int, device_id: str = "0") -> None:
        """Set GPU memory usage metrics"""
        try:
            self.gpu_memory_used_bytes.labels(device_id=device_id).set(used_bytes)
            self.gpu_memory_total_bytes.labels(device_id=device_id).set(total_bytes)
            
            if total_bytes > 0:
                ratio = used_bytes / total_bytes
                self.gpu_memory_usage_ratio.labels(device_id=device_id).set(ratio)
            
        except Exception as e:
            logger.error(f"Error setting GPU memory metrics: {e}")
    
    def set_gpu_temperature(self, temperature: float, device_id: str = "0") -> None:
        """Set GPU temperature in Celsius"""
        try:
            self.gpu_temperature_celsius.labels(device_id=device_id).set(temperature)
        except Exception as e:
            logger.error(f"Error setting GPU temperature metric: {e}")
    
    def increment_active_connections(self) -> None:
        """Increment active WebSocket connections count"""
        try:
            self.active_connections.inc()
        except Exception as e:
            logger.error(f"Error incrementing active connections: {e}")
    
    def decrement_active_connections(self) -> None:
        """Decrement active WebSocket connections count"""
        try:
            self.active_connections.dec()
        except Exception as e:
            logger.error(f"Error decrementing active connections: {e}")
    
    def set_active_connections(self, count: int) -> None:
        """Set active WebSocket connections count"""
        try:
            self.active_connections.set(count)
        except Exception as e:
            logger.error(f"Error setting active connections: {e}")
    
    def start_inference_timer(self, model_variant: str, operation_type: str = "generation"):
        """Start timing model inference"""
        try:
            return self.model_inference_time_seconds.labels(
                model_variant=model_variant,
                operation_type=operation_type
            ).time()
        except Exception as e:
            logger.error(f"Error starting inference timer: {e}")
            return None
    
    def record_inference_time(self, duration: float, model_variant: str, 
                            operation_type: str = "generation") -> None:
        """Record model inference time manually"""
        try:
            self.model_inference_time_seconds.labels(
                model_variant=model_variant,
                operation_type=operation_type
            ).observe(duration)
        except Exception as e:
            logger.error(f"Error recording inference time: {e}")
    
    def set_model_queue_size(self, size: int) -> None:
        """Set current model queue size"""
        try:
            self.model_queue_size.set(size)
        except Exception as e:
            logger.error(f"Error setting model queue size: {e}")
    
    def record_error(self, error_type: str, route: str = "unknown") -> None:
        """Record an error occurrence"""
        try:
            self.errors_total.labels(
                error_type=error_type,
                route=route
            ).inc()
        except Exception as e:
            logger.error(f"Error recording error metric: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metric values for debugging"""
        try:
            return {
                "active_connections": self.active_connections._value._value,
                "model_queue_size": self.model_queue_size._value._value,
                "total_requests": sum(
                    sample.value for sample in self.requests_total.collect()[0].samples
                ),
                "gpu_metrics_count": len([
                    sample for sample in self.gpu_utilization_ratio.collect()[0].samples
                    if sample.value > 0
                ])
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"error": str(e)}