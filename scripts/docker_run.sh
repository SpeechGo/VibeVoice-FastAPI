#!/bin/bash
# ABOUTME: Docker run script with nvidia-docker runtime and proper configuration
# ABOUTME: Handles GPU mapping, volume mounts, environment variables, and networking

set -euo pipefail

# Default values
IMAGE_NAME="vibevoice-fastapi"
TAG="latest"
CONTAINER_NAME="vibevoice-api"
HOST_PORT=8000
CONTAINER_PORT=8000
GPU_DEVICE="all"
DETACH=true
REMOVE=false
INTERACTIVE=false
VERBOSE=false
ENV_FILE=""
VOICES_DIR="./voices"
LOGS_DIR="./logs"
CONFIG_DIR="./config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

show_help() {
    cat << EOF
VibeVoice-FastAPI Docker Run Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -i, --image IMAGE           Docker image (default: vibevoice-fastapi:latest)
    -n, --name NAME             Container name (default: vibevoice-api)
    -p, --port PORT             Host port mapping (default: 8000)
    --container-port PORT       Container port (default: 8000)
    -g, --gpu DEVICE            GPU device(s) to use (default: all)
    -e, --env-file FILE         Environment file to load
    -v, --voices-dir DIR        Voices directory (default: ./voices)
    -l, --logs-dir DIR          Logs directory (default: ./logs)
    -c, --config-dir DIR        Config directory (default: ./config)
    --fg, --foreground          Run in foreground (not detached)
    --rm                        Remove container when it exits
    --interactive               Run in interactive mode
    --verbose                   Verbose output
    -h, --help                  Show this help message

ENVIRONMENT VARIABLES:
    MODEL_PATH                  HuggingFace model path (default: microsoft/VibeVoice-1.5B)
    MAX_CONCURRENCY            Max concurrent requests (default: 1)
    TIMEOUT_SEC                 Request timeout (default: 300)
    CORS_ALLOW_ORIGINS          Allowed CORS origins
    LOG_LEVEL                   Log level (default: info)
    CUDA_VISIBLE_DEVICES        CUDA device visibility

EXAMPLES:
    # Basic run
    $0

    # Run with custom port and model
    $0 --port 8080 --env MODEL_PATH=microsoft/VibeVoice-Large

    # Run in foreground with logs
    $0 --foreground --verbose

    # Run with custom directories
    $0 --voices-dir /path/to/voices --logs-dir /path/to/logs

    # Run with environment file
    $0 --env-file .env.production

    # Run interactively for debugging
    $0 --interactive --rm

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -p|--port)
            HOST_PORT="$2"
            shift 2
            ;;
        --container-port)
            CONTAINER_PORT="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -v|--voices-dir)
            VOICES_DIR="$2"
            shift 2
            ;;
        -l|--logs-dir)
            LOGS_DIR="$2"
            shift 2
            ;;
        -c|--config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --fg|--foreground)
            DETACH=false
            shift
            ;;
        --rm)
            REMOVE=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$HOST_PORT" =~ ^[0-9]+$ ]] || [ "$HOST_PORT" -lt 1 ] || [ "$HOST_PORT" -gt 65535 ]; then
    log_error "Invalid host port: $HOST_PORT"
    exit 1
fi

if [[ ! "$CONTAINER_PORT" =~ ^[0-9]+$ ]] || [ "$CONTAINER_PORT" -lt 1 ] || [ "$CONTAINER_PORT" -gt 65535 ]; then
    log_error "Invalid container port: $CONTAINER_PORT"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker is available
if ! docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    log_warn "NVIDIA Docker support not detected. Running without GPU support."
    GPU_DEVICE=""
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    log_error "Docker image not found: $IMAGE_NAME"
    log_info "Please build the image first using: ./scripts/docker_build.sh"
    exit 1
fi

# Create directories if they don't exist
for dir in "$VOICES_DIR" "$LOGS_DIR" "$CONFIG_DIR"; do
    if [[ ! -d "$dir" ]]; then
        log_info "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Check if container is already running
if docker ps -q --filter "name=$CONTAINER_NAME" | grep -q .; then
    log_warn "Container $CONTAINER_NAME is already running"
    log_info "Stopping existing container..."
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
fi

# Build docker run command
RUN_CMD="docker run"

# Add GPU support if available
if [[ -n "$GPU_DEVICE" ]]; then
    RUN_CMD="$RUN_CMD --gpus $GPU_DEVICE"
fi

# Add runtime options
if [[ $DETACH == true && $INTERACTIVE == false ]]; then
    RUN_CMD="$RUN_CMD -d"
fi

if [[ $REMOVE == true ]]; then
    RUN_CMD="$RUN_CMD --rm"
fi

if [[ $INTERACTIVE == true ]]; then
    RUN_CMD="$RUN_CMD -it"
fi

# Add name and port mapping
RUN_CMD="$RUN_CMD --name $CONTAINER_NAME"
RUN_CMD="$RUN_CMD -p $HOST_PORT:$CONTAINER_PORT"

# Add volume mounts
RUN_CMD="$RUN_CMD -v $(realpath "$VOICES_DIR"):/app/voices:rw"
RUN_CMD="$RUN_CMD -v $(realpath "$LOGS_DIR"):/app/logs:rw"

if [[ -d "$CONFIG_DIR" ]]; then
    RUN_CMD="$RUN_CMD -v $(realpath "$CONFIG_DIR"):/app/config:ro"
fi

# Add environment variables
RUN_CMD="$RUN_CMD -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
RUN_CMD="$RUN_CMD -e NVIDIA_VISIBLE_DEVICES=all"
RUN_CMD="$RUN_CMD -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"

# Load environment file if specified
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    log_info "Loading environment from: $ENV_FILE"
    RUN_CMD="$RUN_CMD --env-file $ENV_FILE"
fi

# Add default environment variables if not in env file
if [[ -z "$ENV_FILE" ]]; then
    RUN_CMD="$RUN_CMD -e MODEL_PATH=${MODEL_PATH:-microsoft/VibeVoice-1.5B}"
    RUN_CMD="$RUN_CMD -e MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}"
    RUN_CMD="$RUN_CMD -e TIMEOUT_SEC=${TIMEOUT_SEC:-300}"
    RUN_CMD="$RUN_CMD -e LOG_LEVEL=${LOG_LEVEL:-info}"
fi

# Add health check options
RUN_CMD="$RUN_CMD --health-cmd 'curl -f http://localhost:8000/health || exit 1'"
RUN_CMD="$RUN_CMD --health-interval 30s"
RUN_CMD="$RUN_CMD --health-timeout 30s"
RUN_CMD="$RUN_CMD --health-start-period 60s"
RUN_CMD="$RUN_CMD --health-retries 3"

# Add resource limits
RUN_CMD="$RUN_CMD --memory 8g"
RUN_CMD="$RUN_CMD --shm-size 2g"

# Add the image name
RUN_CMD="$RUN_CMD $IMAGE_NAME"

# Add command override for interactive mode
if [[ $INTERACTIVE == true ]]; then
    RUN_CMD="$RUN_CMD /bin/bash"
fi

# Show run command if verbose
if [[ $VERBOSE == true ]]; then
    log_info "Docker run command:"
    echo "$RUN_CMD"
    echo
fi

# Execute the run command
log_info "Starting VibeVoice container..."
log_info "Container: $CONTAINER_NAME"
log_info "Image: $IMAGE_NAME"
log_info "Port: $HOST_PORT -> $CONTAINER_PORT"
log_info "GPU: ${GPU_DEVICE:-disabled}"

if eval "$RUN_CMD"; then
    if [[ $DETACH == true && $INTERACTIVE == false ]]; then
        log_success "Container started successfully"
        log_info "Container ID: $(docker ps -q --filter "name=$CONTAINER_NAME")"
        log_info "Access the API at: http://localhost:$HOST_PORT"
        log_info "Health check: http://localhost:$HOST_PORT/health"
        log_info "API docs: http://localhost:$HOST_PORT/docs"
        echo
        log_info "To view logs: docker logs -f $CONTAINER_NAME"
        log_info "To stop container: docker stop $CONTAINER_NAME"
    else
        log_success "Container session completed"
    fi
else
    log_error "Failed to start container"
    exit 1
fi