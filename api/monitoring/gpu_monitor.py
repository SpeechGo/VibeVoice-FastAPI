# ABOUTME: GPU monitoring using pynvml to collect GPU utilization, memory, and temperature stats
# ABOUTME: Provides background task for continuous GPU monitoring and Prometheus metrics updates
import asyncio
import logging
from typing import Dict, List, Any, Optional
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    GPU monitoring class using NVIDIA Management Library (NVML).
    
    Collects GPU utilization, memory usage, and temperature data.
    Gracefully handles systems without NVIDIA GPUs or NVML.
    """
    
    def __init__(self):
        """Initialize GPU monitoring"""
        self.available = False
        self.device_count = 0
        self.handles = []
        
        if not NVML_AVAILABLE:
            logger.warning("pynvml not available - GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            # Get handles for all devices
            self.handles = []
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
            
            self.available = True
            logger.info(f"GPU monitoring initialized - {self.device_count} device(s) detected")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self.available = False
            self.device_count = 0
            self.handles = []
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get current GPU statistics for all devices.
        
        Returns:
            Dict containing GPU stats or empty dict if unavailable
        """
        if not self.available or not NVML_AVAILABLE:
            return {
                "available": False,
                "devices": []
            }
        
        devices_stats = []
        
        for i, handle in enumerate(self.handles):
            try:
                device_stats = self._get_device_stats(handle, i)
                devices_stats.append(device_stats)
                
            except Exception as e:
                logger.error(f"Error getting stats for GPU {i}: {e}")
                devices_stats.append({
                    "device_id": str(i),
                    "error": str(e)
                })
        
        return {
            "available": True,
            "device_count": self.device_count,
            "devices": devices_stats
        }
    
    def _get_device_stats(self, handle, device_id: int) -> Dict[str, Any]:
        """Get stats for a single GPU device"""
        try:
            # Get utilization rates
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory information
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get device name
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
            except:
                name = f"GPU-{device_id}"
            
            # Get power usage if available
            power_watts = None
            try:
                power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            except:
                pass  # Not all GPUs support power monitoring
            
            # Get clock speeds if available
            graphics_clock = None
            memory_clock = None
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                pass  # Not all GPUs support clock monitoring
            
            stats = {
                "device_id": str(device_id),
                "name": name,
                "utilization_gpu": utilization.gpu,
                "utilization_memory": utilization.memory,
                "memory_used_bytes": memory_info.used,
                "memory_free_bytes": memory_info.free,
                "memory_total_bytes": memory_info.total,
                "memory_used_mb": memory_info.used // (1024 * 1024),
                "memory_free_mb": memory_info.free // (1024 * 1024),
                "memory_total_mb": memory_info.total // (1024 * 1024),
                "memory_usage_percent": (memory_info.used / memory_info.total) * 100,
                "temperature": temperature,
            }
            
            # Add optional metrics if available
            if power_watts is not None:
                stats["power_watts"] = power_watts
            
            if graphics_clock is not None:
                stats["graphics_clock_mhz"] = graphics_clock
            
            if memory_clock is not None:
                stats["memory_clock_mhz"] = memory_clock
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting GPU stats for device {device_id}: {e}")
            raise
    
    def get_primary_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Get stats for the primary GPU (device 0) only"""
        if not self.available or self.device_count == 0:
            return None
        
        stats = self.get_gpu_stats()
        if stats["devices"]:
            return stats["devices"][0]
        
        return None
    
    async def start_monitoring(self, metrics_instance, interval: float = 10.0):
        """
        Start background GPU monitoring task.
        
        Args:
            metrics_instance: PrometheusMetrics instance to update
            interval: Monitoring interval in seconds
        """
        if not self.available:
            logger.info("GPU monitoring not available - task not started")
            return
        
        logger.info(f"Starting GPU monitoring task with {interval}s interval")
        
        while True:
            try:
                stats = self.get_gpu_stats()
                
                if stats["available"] and stats["devices"]:
                    for device_stats in stats["devices"]:
                        if "error" in device_stats:
                            continue
                        
                        device_id = device_stats["device_id"]
                        
                        # Update Prometheus metrics
                        metrics_instance.set_gpu_utilization(
                            device_stats["utilization_gpu"] / 100.0,  # Convert to ratio
                            device_id
                        )
                        
                        metrics_instance.set_gpu_memory_usage(
                            device_stats["memory_used_bytes"],
                            device_stats["memory_total_bytes"],
                            device_id
                        )
                        
                        metrics_instance.set_gpu_temperature(
                            device_stats["temperature"],
                            device_id
                        )
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("GPU monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in GPU monitoring task: {e}")
                await asyncio.sleep(interval)  # Continue monitoring despite errors
    
    def cleanup(self):
        """Cleanup NVML resources"""
        if self.available and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                logger.info("GPU monitoring cleanup completed")
            except Exception as e:
                logger.warning(f"Error during GPU monitoring cleanup: {e}")


class GPUMonitorManager:
    """
    Manager class for GPU monitoring tasks.
    
    Handles lifecycle of GPU monitoring background tasks.
    """
    
    def __init__(self):
        self.monitor = GPUMonitor()
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self, metrics_instance, interval: float = 10.0):
        """Start GPU monitoring"""
        if self.monitoring_task is not None and not self.monitoring_task.done():
            logger.warning("GPU monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(
            self.monitor.start_monitoring(metrics_instance, interval)
        )
        
        logger.info("GPU monitoring manager started")
    
    async def stop(self):
        """Stop GPU monitoring"""
        if self.monitoring_task is not None and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            
            self.monitoring_task = None
        
        self.monitor.cleanup()
        logger.info("GPU monitoring manager stopped")
    
    def is_running(self) -> bool:
        """Check if monitoring is running"""
        return (
            self.monitoring_task is not None 
            and not self.monitoring_task.done()
        )
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current GPU stats synchronously"""
        return self.monitor.get_gpu_stats()