# ABOUTME: Background tasks for monitoring system including GPU monitoring and metrics collection
# ABOUTME: Manages lifecycle of monitoring tasks and ensures they run continuously without blocking the main app
import asyncio
import logging
from typing import List, Optional

from .metrics import PrometheusMetrics
from .gpu_monitor import GPUMonitorManager

logger = logging.getLogger(__name__)


class MonitoringTaskManager:
    """
    Manages all background monitoring tasks.
    
    Provides centralized control over GPU monitoring and other periodic tasks.
    """
    
    def __init__(self):
        self.gpu_monitor_manager = GPUMonitorManager()
        self.tasks: List[asyncio.Task] = []
        self.metrics = PrometheusMetrics()
        self._shutdown_requested = False
    
    async def start_all_tasks(self, gpu_monitoring_interval: float = 10.0):
        """
        Start all monitoring background tasks.
        
        Args:
            gpu_monitoring_interval: Interval in seconds for GPU monitoring
        """
        if self.tasks:
            logger.warning("Monitoring tasks already started")
            return
        
        logger.info("Starting monitoring background tasks")
        
        try:
            # Start GPU monitoring
            await self.gpu_monitor_manager.start(self.metrics, gpu_monitoring_interval)
            
            # Start metrics maintenance task
            metrics_task = asyncio.create_task(self._metrics_maintenance_task())
            self.tasks.append(metrics_task)
            
            logger.info(f"Started {len(self.tasks)} monitoring background tasks")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring tasks: {e}")
            await self.stop_all_tasks()
            raise
    
    async def stop_all_tasks(self):
        """Stop all monitoring tasks gracefully"""
        logger.info("Stopping monitoring background tasks")
        self._shutdown_requested = True
        
        try:
            # Stop GPU monitoring
            await self.gpu_monitor_manager.stop()
            
            # Cancel other tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            self.tasks.clear()
            logger.info("All monitoring tasks stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring tasks: {e}")
    
    async def _metrics_maintenance_task(self):
        """
        Background task for metrics maintenance.
        
        Performs periodic cleanup and health checks on metrics.
        """
        logger.info("Starting metrics maintenance task")
        
        while not self._shutdown_requested:
            try:
                # Update queue size metrics if we have access to queue info
                # This would be integrated with the voice service
                # For now, we'll just perform basic maintenance
                
                # Log metrics summary periodically (every 5 minutes)
                await asyncio.sleep(300)  # 5 minutes
                
                if not self._shutdown_requested:
                    summary = self.metrics.get_metrics_summary()
                    logger.debug(f"Metrics summary: {summary}")
                
            except asyncio.CancelledError:
                logger.info("Metrics maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics maintenance task: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def get_status(self) -> dict:
        """Get status of all monitoring tasks"""
        return {
            "gpu_monitoring_running": self.gpu_monitor_manager.is_running(),
            "background_tasks_count": len([t for t in self.tasks if not t.done()]),
            "shutdown_requested": self._shutdown_requested
        }


# Global instance
_monitoring_manager: Optional[MonitoringTaskManager] = None


def get_monitoring_manager() -> MonitoringTaskManager:
    """Get the global monitoring manager instance"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringTaskManager()
    return _monitoring_manager


async def start_monitoring_tasks(gpu_monitoring_interval: float = 10.0) -> MonitoringTaskManager:
    """
    Start all monitoring background tasks.
    
    Args:
        gpu_monitoring_interval: Interval in seconds for GPU monitoring
        
    Returns:
        MonitoringTaskManager instance
    """
    manager = get_monitoring_manager()
    await manager.start_all_tasks(gpu_monitoring_interval)
    return manager


async def stop_monitoring_tasks():
    """Stop all monitoring background tasks"""
    global _monitoring_manager
    if _monitoring_manager is not None:
        await _monitoring_manager.stop_all_tasks()


def start_monitoring_tasks_sync(metrics_instance: Optional[PrometheusMetrics] = None) -> List[asyncio.Task]:
    """
    Synchronous wrapper to start monitoring tasks.
    
    For compatibility with older code patterns.
    
    Returns:
        List of background tasks
    """
    async def _start():
        manager = await start_monitoring_tasks()
        return manager.tasks
    
    # This is a simplified version - in practice you'd want to handle this more carefully
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_start())
    except RuntimeError:
        # No event loop running
        logger.warning("No event loop running - background tasks not started")
        return []