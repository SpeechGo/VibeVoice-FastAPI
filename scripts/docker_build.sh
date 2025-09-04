#!/bin/bash
# ABOUTME: Docker build script with CUDA version support and optimizations
# ABOUTME: Supports different build configurations and proper tagging

set -euo pipefail

# Default values
CUDA_VERSION="12.1"
IMAGE_NAME="vibevoice-fastapi"
TAG="latest"
DOCKERFILE="Dockerfile"
BUILD_ARGS=""
PUSH=false
NO_CACHE=false
VERBOSE=false

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
VibeVoice-FastAPI Docker Build Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -c, --cuda-version VERSION  CUDA version to use (default: 12.1)
    -t, --tag TAG              Image tag (default: latest)  
    -n, --name NAME            Image name (default: vibevoice-fastapi)
    -f, --dockerfile FILE      Dockerfile path (default: Dockerfile)
    -p, --push                 Push image to registry after build
    --no-cache                 Build without using cache
    --build-arg KEY=VALUE      Pass build argument
    -v, --verbose              Verbose output
    -h, --help                 Show this help message

EXAMPLES:
    # Basic build
    $0

    # Build with specific CUDA version
    $0 --cuda-version 11.8

    # Build and push to registry
    $0 --tag v1.0.0 --push

    # Build without cache
    $0 --no-cache

    # Build with custom build args
    $0 --build-arg MODEL_VERSION=1.5B --build-arg OPTIMIZE=true

SUPPORTED CUDA VERSIONS:
    11.8, 12.0, 12.1 (default), 12.2

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        -v|--verbose)
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

# Validate CUDA version
case $CUDA_VERSION in
    11.8|12.0|12.1|12.2)
        ;;
    *)
        log_error "Unsupported CUDA version: $CUDA_VERSION"
        log_info "Supported versions: 11.8, 12.0, 12.1, 12.2"
        exit 1
        ;;
esac

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    log_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_warn "NVIDIA Docker support not detected. GPU functionality may not work."
fi

# Build the image
log_info "Starting Docker build..."
log_info "Image: ${IMAGE_NAME}:${TAG}"
log_info "CUDA Version: ${CUDA_VERSION}"
log_info "Dockerfile: ${DOCKERFILE}"

# Construct the build command
BUILD_CMD="docker build"

if [[ $NO_CACHE == true ]]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [[ $VERBOSE == true ]]; then
    BUILD_CMD="$BUILD_CMD --progress=plain"
fi

# Add build arguments
BUILD_CMD="$BUILD_CMD --build-arg CUDA_VERSION=${CUDA_VERSION}"
BUILD_CMD="$BUILD_CMD $BUILD_ARGS"

# Add tags
BUILD_CMD="$BUILD_CMD -t ${IMAGE_NAME}:${TAG}"
BUILD_CMD="$BUILD_CMD -t ${IMAGE_NAME}:cuda-${CUDA_VERSION}"

# Add context and dockerfile
BUILD_CMD="$BUILD_CMD -f ${DOCKERFILE} ."

# Show build command if verbose
if [[ $VERBOSE == true ]]; then
    log_info "Build command: $BUILD_CMD"
fi

# Execute build
if eval "$BUILD_CMD"; then
    log_success "Docker build completed successfully"
else
    log_error "Docker build failed"
    exit 1
fi

# Show image info
log_info "Built image details:"
docker images --filter "reference=${IMAGE_NAME}:${TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Push if requested
if [[ $PUSH == true ]]; then
    log_info "Pushing image to registry..."
    
    if docker push "${IMAGE_NAME}:${TAG}"; then
        log_success "Successfully pushed ${IMAGE_NAME}:${TAG}"
    else
        log_error "Failed to push image"
        exit 1
    fi
    
    if docker push "${IMAGE_NAME}:cuda-${CUDA_VERSION}"; then
        log_success "Successfully pushed ${IMAGE_NAME}:cuda-${CUDA_VERSION}"
    else
        log_warn "Failed to push CUDA-specific tag"
    fi
fi

# Test the built image
log_info "Testing the built image..."
if docker run --rm --gpus all "${IMAGE_NAME}:${TAG}" python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; then
    log_success "Image test passed"
else
    log_warn "Image test failed - this may be normal if GPU is not available"
fi

log_success "Build process completed!"
log_info "To run the image: docker run --rm --gpus all -p 8000:8000 ${IMAGE_NAME}:${TAG}"