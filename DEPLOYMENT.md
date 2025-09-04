# VibeVoice-FastAPI Docker Deployment Guide

## Overview

This document provides comprehensive instructions for deploying VibeVoice-FastAPI using Docker with CUDA support, reverse proxy configuration, and production monitoring.

## Quick Start

### Development Deployment

1. **Build the Docker image:**
   ```bash
   ./scripts/docker_build.sh
   ```

2. **Run the service:**
   ```bash
   ./scripts/docker_run.sh
   ```

3. **Or use Docker Compose:**
   ```bash
   docker-compose up -d
   ```

### Production Deployment

1. **Install as systemd service:**
   ```bash
   cd deployment/
   sudo ./install.sh
   ```

2. **Configure environment:**
   ```bash
   sudo nano /opt/vibevoice-fastapi/.env.production
   ```

3. **Start the service:**
   ```bash
   sudo systemctl start vibevoice
   sudo systemctl status vibevoice
   ```

## Architecture

### Multi-Stage Docker Build

The Dockerfile uses a 3-stage build process:

1. **Base Stage**: CUDA 12.1 with system dependencies
2. **Dependencies Stage**: Model caching and Python dependencies
3. **Runtime Stage**: Minimal runtime image with pre-cached models

### Key Features

- **CUDA Support**: GPU acceleration for TTS inference
- **Model Caching**: Pre-downloads models during build
- **Health Checks**: Automated container health monitoring
- **Security**: Non-root user execution
- **Resource Limits**: Memory and CPU constraints

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `microsoft/VibeVoice-1.5B` | HuggingFace model path |
| `VOICES_DIR` | `/app/voices` | Voice files directory |
| `MAX_CONCURRENCY` | `1` | Maximum concurrent requests |
| `TIMEOUT_SEC` | `300` | Request timeout in seconds |
| `CORS_ALLOW_ORIGINS` | `""` | Comma-separated CORS origins |
| `LOG_LEVEL` | `info` | Logging level |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |

### Volume Mounts

- `./voices:/app/voices` - Voice files (persistent)
- `vibevoice_models:/app/model_cache` - Model cache (persistent)
- `./logs:/app/logs` - Application logs
- `./config:/app/config` - Configuration overrides (optional)

## Scripts

### docker_build.sh

Builds the Docker image with various options:

```bash
# Basic build
./scripts/docker_build.sh

# With specific CUDA version
./scripts/docker_build.sh --cuda-version 11.8

# Build and push to registry
./scripts/docker_build.sh --tag v1.0.0 --push

# Build without cache
./scripts/docker_build.sh --no-cache
```

### docker_run.sh

Runs the Docker container with proper GPU support:

```bash
# Basic run
./scripts/docker_run.sh

# Custom port and configuration
./scripts/docker_run.sh --port 8080 --env-file .env.production

# Interactive mode for debugging
./scripts/docker_run.sh --interactive --rm

# Run in foreground with logs
./scripts/docker_run.sh --foreground --verbose
```

## Reverse Proxy (Nginx)

The deployment includes a comprehensive Nginx configuration:

### Features

- **SSL Termination**: HTTPS with configurable certificates
- **Rate Limiting**: API protection against abuse
- **WebSocket Support**: Real-time communication proxying
- **Load Balancing**: Multiple backend support
- **Health Checks**: Automatic upstream monitoring

### Configuration Files

- `deployment/nginx.conf` - Main Nginx configuration
- `deployment/proxy_params.conf` - Proxy headers and timeouts
- `deployment/websocket_params.conf` - WebSocket-specific settings
- `deployment/rate_limit.conf` - Rate limiting rules

### SSL Setup

1. Place SSL certificates in `deployment/ssl/`:
   ```
   deployment/ssl/
   ├── cert.pem
   └── key.pem
   ```

2. Uncomment SSL configuration in `nginx.conf`

3. Update server_name with your domain

## Production Deployment

### System Service (systemd)

The `vibevoice.service` file provides:

- **Auto-restart**: Service restarts on failure
- **Resource Limits**: Memory and process constraints
- **Security**: Sandboxing and privilege restrictions
- **Logging**: Structured journald integration

### Installation Steps

1. **Run the installer:**
   ```bash
   cd deployment/
   sudo ./install.sh
   ```

2. **Edit production environment:**
   ```bash
   sudo nano /opt/vibevoice-fastapi/.env.production
   ```

3. **Start and enable service:**
   ```bash
   sudo systemctl start vibevoice
   sudo systemctl enable vibevoice
   ```

4. **Monitor service:**
   ```bash
   sudo systemctl status vibevoice
   sudo journalctl -u vibevoice -f
   ```

## Monitoring

### Prometheus Metrics

The service exposes metrics at `/metrics` endpoint:

- Request counters and histograms
- GPU utilization metrics
- Model loading performance
- Error rates and types

### Grafana Dashboards

Production deployment includes Grafana with pre-configured dashboards:

- **API Performance**: Request rates, latencies, errors
- **System Resources**: CPU, memory, GPU usage
- **TTS Metrics**: Generation times, model performance

Access Grafana at: `http://localhost:3000`
Default credentials: `admin` / `vibevoice_admin_2024`

### Log Aggregation

Optional Loki integration for centralized logging:

- Structured JSON logs from FastAPI
- Nginx access and error logs
- System and container logs

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check NVIDIA Docker support
   docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
   
   # Install nvidia-docker2 if needed
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Model Download Failures**
   ```bash
   # Check HuggingFace Hub connectivity
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/VibeVoice-1.5B')"
   
   # Use HF_TOKEN for private models
   export HF_TOKEN=your_token_here
   ```

3. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./voices ./logs
   ```

4. **Port Conflicts**
   ```bash
   # Check port usage
   sudo netstat -tlnp | grep :8000
   
   # Use different port
   ./scripts/docker_run.sh --port 8001
   ```

### Performance Optimization

1. **GPU Memory**: Adjust model size based on VRAM
2. **Concurrency**: Increase MAX_CONCURRENCY for more throughput
3. **Caching**: Use persistent model cache volume
4. **Resources**: Monitor and adjust container limits

### Health Monitoring

The service provides health endpoints:

- `/health` - Basic health check
- `/metrics` - Prometheus metrics
- Container health checks every 30 seconds

## Security Considerations

- **Non-root execution**: Container runs as unprivileged user
- **Network isolation**: Service runs in isolated Docker network
- **Resource limits**: Memory and CPU constraints prevent abuse
- **Rate limiting**: Nginx protects against DDoS
- **CORS configuration**: Restrict origins in production

## Backup and Recovery

### Model Cache

The model cache should be backed up for faster deployments:

```bash
# Backup model cache
docker run --rm -v vibevoice_models:/data -v $(pwd):/backup ubuntu tar czf /backup/models-backup.tar.gz -C /data .

# Restore model cache
docker run --rm -v vibevoice_models:/data -v $(pwd):/backup ubuntu tar xzf /backup/models-backup.tar.gz -C /data
```

### Configuration

Backup the following directories:
- `/opt/vibevoice-fastapi/.env.production`
- `/opt/vibevoice-fastapi/voices/`
- `/opt/vibevoice-fastapi/deployment/ssl/`

## Scaling

### Horizontal Scaling

1. **Load Balancer**: Configure Nginx upstream with multiple backends
2. **Container Orchestration**: Use Docker Swarm or Kubernetes
3. **Model Sharding**: Distribute different models across instances

### Vertical Scaling

1. **GPU Memory**: Use larger GPU or multiple GPUs
2. **CPU Cores**: Increase worker processes
3. **Memory**: Adjust container memory limits

## Updates and Maintenance

### Rolling Updates

1. **Build new image:**
   ```bash
   ./scripts/docker_build.sh --tag v1.1.0
   ```

2. **Update compose file:**
   ```bash
   sed -i 's/:latest/:v1.1.0/' docker-compose.yml
   ```

3. **Rolling restart:**
   ```bash
   docker-compose up -d --no-deps vibevoice-api
   ```

### Log Rotation

Logs are automatically rotated using logrotate configuration installed by the setup script.

## Support

For deployment issues:

1. Check service logs: `sudo journalctl -u vibevoice -f`
2. Check container logs: `docker logs vibevoice-api`
3. Verify GPU access: `docker exec vibevoice-api nvidia-smi`
4. Test endpoints: `curl http://localhost:8000/health`