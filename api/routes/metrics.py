# ABOUTME: Metrics endpoint for Prometheus scraping in proper text format
# ABOUTME: Provides /metrics endpoint that returns all metrics in Prometheus exposition format
import logging
from fastapi import APIRouter, Response
from prometheus_client import REGISTRY, generate_latest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics", response_class=Response)
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns all registered metrics in Prometheus exposition format.
    This endpoint is designed to be scraped by Prometheus server.
    
    Returns:
        Response: Plain text response with metrics in Prometheus format
    """
    try:
        # Generate metrics in Prometheus format
        metrics_data = generate_latest(REGISTRY)
        
        # Return as plain text with proper content type
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        
        # Return empty metrics response on error
        return Response(
            content="# Error generating metrics\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=500
        )


@router.get("/metrics/health")
async def get_metrics_health():
    """
    Health check endpoint for metrics system.
    
    Returns basic health information about the metrics collection system.
    """
    try:
        from api.monitoring.metrics import PrometheusMetrics
        from api.monitoring.gpu_monitor import GPUMonitor
        
        # Get metrics instance
        metrics = PrometheusMetrics()
        
        # Get GPU monitor status
        gpu_monitor = GPUMonitor()
        gpu_stats = gpu_monitor.get_gpu_stats()
        
        # Get metrics summary
        metrics_summary = metrics.get_metrics_summary()
        
        return {
            "status": "ok",
            "metrics_initialized": hasattr(metrics, '_initialized'),
            "gpu_monitoring_available": gpu_stats["available"],
            "gpu_device_count": gpu_stats.get("device_count", 0),
            "metrics_summary": metrics_summary
        }
        
    except Exception as e:
        logger.error(f"Error in metrics health check: {e}")
        return {
            "status": "error",
            "error": str(e)
        }