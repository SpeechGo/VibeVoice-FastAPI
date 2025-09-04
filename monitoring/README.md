# VibeVoice FastAPI Monitoring System

This directory contains a comprehensive monitoring solution for the VibeVoice FastAPI service, including Prometheus metrics collection, Grafana dashboards, and alerting rules.

## Components

### 1. Prometheus Metrics (`/metrics` endpoint)
- **Request metrics**: Request count, duration, error rates
- **GPU metrics**: Utilization, memory usage, temperature  
- **WebSocket metrics**: Active connections
- **Model metrics**: Inference time, queue size

### 2. Grafana Dashboard
- Real-time visualization of all metrics
- Pre-configured panels for key performance indicators
- GPU monitoring visualizations
- Request latency and throughput monitoring

### 3. Alert Rules
- Configurable thresholds for critical metrics
- GPU overheating and memory alerts
- High latency and error rate alerts
- Service availability monitoring

## Quick Start

### Using Docker Compose (Recommended)

1. **Start the monitoring stack:**
   ```bash
   cd monitoring
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access services:**
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Alertmanager: http://localhost:9093

3. **Start your VibeVoice FastAPI service:**
   ```bash
   cd ..
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

4. **Import dashboard:**
   - The dashboard should auto-import via provisioning
   - If not, manually import `grafana-dashboard.json` in Grafana

### Manual Setup

1. **Install Prometheus:**
   ```bash
   # Use the prometheus.yml configuration file
   prometheus --config.file=prometheus.yml
   ```

2. **Install Grafana:**
   ```bash
   # Import grafana-dashboard.json
   # Configure Prometheus datasource: http://localhost:9090
   ```

3. **Configure Alertmanager (optional):**
   ```bash
   alertmanager --config.file=alertmanager.yml
   ```

## Configuration Files

### `prometheus.yml`
Main Prometheus configuration with:
- Scrape targets (VibeVoice FastAPI service)
- Alert rules and recording rules
- Global settings and timeouts

### `alert_rules.yml`  
Alert definitions for:
- High error rates (>5%)
- High latency (>30s)
- GPU issues (utilization >95%, temperature >85°C)
- Service availability

### `recording_rules.yml`
Pre-computed aggregations for:
- Request rates and percentiles
- GPU utilization averages
- Error rate calculations

### `grafana-dashboard.json`
Grafana dashboard with panels for:
- Request rate and latency
- GPU utilization and memory
- WebSocket connections
- Model inference performance
- Error tracking

### `alertmanager.yml`
Alertmanager routing and notification configuration:
- Slack/email notifications
- PagerDuty integration (critical alerts)
- Alert grouping and inhibition rules

## Metrics Reference

### Request Metrics
```
vibe_requests_total{route, status, model_variant}
vibe_request_duration_seconds{route, status, model_variant}
```

### GPU Metrics
```
vibe_gpu_utilization_ratio{device_id}
vibe_gpu_memory_usage_ratio{device_id}
vibe_gpu_memory_used_bytes{device_id}
vibe_gpu_memory_total_bytes{device_id}
vibe_gpu_temperature_celsius{device_id}
```

### Connection Metrics
```
vibe_active_connections
```

### Model Metrics
```
vibe_model_inference_time_seconds{model_variant, operation_type}
vibe_model_queue_size
```

### Error Metrics
```
vibe_errors_total{error_type, route}
```

## Alert Thresholds

| Alert | Threshold | Duration | Severity |
|-------|-----------|----------|----------|
| High Error Rate | >5% | 2min | Warning |
| High Latency | >30s | 5min | Warning |
| Critical Latency | >60s | 2min | Critical |
| GPU High Utilization | >95% | 10min | Warning |
| GPU High Memory | >90% | 5min | Critical |
| GPU High Temperature | >85°C | 5min | Critical |
| Service Down | N/A | 1min | Critical |
| Queue Backup | >10 requests | 2min | Warning |
| Too Many Connections | >100 | 5min | Warning |

## Customization

### Adding New Metrics
1. Add metrics to `api/monitoring/metrics.py`
2. Update dashboard panels in `grafana-dashboard.json`
3. Add alert rules if needed in `alert_rules.yml`

### Modifying Alert Thresholds
Edit the `expr` values in `alert_rules.yml`:
```yaml
- alert: CustomAlert
  expr: your_metric > threshold
  for: duration
```

### Dashboard Customization
1. Edit `grafana-dashboard.json` directly, or
2. Use Grafana UI and export updated JSON

### Notification Channels
Configure `alertmanager.yml` with your:
- Slack webhook URLs
- Email SMTP settings  
- PagerDuty service keys

## Troubleshooting

### Metrics Not Appearing
1. Check if FastAPI service is running: `curl http://localhost:8000/metrics`
2. Verify Prometheus can scrape: Check Prometheus targets page
3. Check logs for GPU monitoring issues (NVML not available)

### Grafana Dashboard Issues
1. Verify Prometheus datasource is configured correctly
2. Check dashboard queries match your metric names
3. Import fresh dashboard JSON if panels are broken

### Alert Issues
1. Check Prometheus rules evaluation: Go to Rules page
2. Verify Alertmanager is receiving alerts
3. Check notification channel configurations

### GPU Monitoring Not Working
1. Ensure NVIDIA drivers are installed
2. Verify `pynvml` package is available
3. Check service logs for GPU initialization errors

## Performance Impact

The monitoring system is designed for minimal performance impact:
- Metrics collection: <1ms per request overhead
- GPU monitoring: 10-second intervals by default
- Background tasks use minimal CPU/memory
- Prometheus scraping: 10-15 second intervals

## Security Considerations

- Metrics endpoint (`/metrics`) exposes system information
- Consider restricting access in production environments
- Alert notifications may contain sensitive data
- Use HTTPS for Grafana in production

## Support

For issues or questions:
1. Check service logs: `docker-compose logs vibevoice-fastapi`
2. Review Prometheus targets: http://localhost:9090/targets
3. Verify dashboard queries in Grafana explore mode
4. Check alert evaluation in Prometheus rules page