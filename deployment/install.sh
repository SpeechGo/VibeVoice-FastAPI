#!/bin/bash
# ABOUTME: Installation script for VibeVoice-FastAPI systemd service
# ABOUTME: Sets up user, directories, permissions, and service registration

set -euo pipefail

# Configuration
SERVICE_NAME="vibevoice"
SERVICE_USER="vibevoice"
SERVICE_GROUP="vibevoice"
INSTALL_DIR="/opt/vibevoice-fastapi"
SERVICE_FILE="vibevoice.service"
LOG_DIR="/var/log/vibevoice"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root (use sudo)"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

log_info "Installing VibeVoice-FastAPI systemd service..."

# Create service user and group
if ! getent group "$SERVICE_GROUP" > /dev/null 2>&1; then
    log_info "Creating group: $SERVICE_GROUP"
    groupadd --system "$SERVICE_GROUP"
else
    log_info "Group $SERVICE_GROUP already exists"
fi

if ! getent passwd "$SERVICE_USER" > /dev/null 2>&1; then
    log_info "Creating user: $SERVICE_USER"
    useradd --system --gid "$SERVICE_GROUP" --home-dir "$INSTALL_DIR" \
            --shell /bin/false --comment "VibeVoice Service User" "$SERVICE_USER"
else
    log_info "User $SERVICE_USER already exists"
fi

# Add service user to docker group
if getent group docker > /dev/null 2>&1; then
    log_info "Adding $SERVICE_USER to docker group"
    usermod -aG docker "$SERVICE_USER"
else
    log_warn "Docker group does not exist. The service user may not be able to access Docker."
fi

# Create installation directory
log_info "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/volumes/models"
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/config"

# Create log directory
log_info "Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"

# Copy application files
log_info "Copying application files..."
cp -r ../api "$INSTALL_DIR/"
cp -r ../voices "$INSTALL_DIR/"
cp ../docker-compose.yml "$INSTALL_DIR/"
cp ../Dockerfile "$INSTALL_DIR/"
cp ../.dockerignore "$INSTALL_DIR/"
cp ../pyproject.toml "$INSTALL_DIR/"
cp ../uv.lock "$INSTALL_DIR/" 2>/dev/null || true

# Copy deployment files
cp -r . "$INSTALL_DIR/deployment/"

# Set ownership and permissions
log_info "Setting ownership and permissions..."
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$LOG_DIR"

# Make scripts executable
chmod +x "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true

# Set proper permissions
chmod 755 "$INSTALL_DIR"
chmod 755 "$LOG_DIR"
chmod -R 644 "$INSTALL_DIR/api"
chmod -R 644 "$INSTALL_DIR/voices"
chmod 644 "$INSTALL_DIR/docker-compose.yml"
chmod 644 "$INSTALL_DIR/Dockerfile"

# Install systemd service
log_info "Installing systemd service..."
cp "$SERVICE_FILE" "/etc/systemd/system/"
systemctl daemon-reload

# Create environment file template
log_info "Creating environment file template..."
cat > "$INSTALL_DIR/.env.production" << 'EOF'
# VibeVoice-FastAPI Production Environment Variables

# Model Configuration
MODEL_PATH=microsoft/VibeVoice-1.5B
VOICES_DIR=/opt/vibevoice-fastapi/voices

# Performance
MAX_CONCURRENCY=2
TIMEOUT_SEC=300

# CORS (customize for your domain)
CORS_ALLOW_ORIGINS=https://your-domain.com

# Logging
LOG_LEVEL=info

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0

# Docker Configuration
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1
EOF

chown "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR/.env.production"
chmod 600 "$INSTALL_DIR/.env.production"

# Create logrotate configuration
log_info "Setting up log rotation..."
cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
    su $SERVICE_USER $SERVICE_GROUP
}
EOF

# Enable but don't start the service
log_info "Enabling systemd service..."
systemctl enable "$SERVICE_NAME.service"

log_success "Installation completed successfully!"
echo
log_info "Next steps:"
echo "  1. Edit the environment file: $INSTALL_DIR/.env.production"
echo "  2. Customize CORS origins and other settings as needed"
echo "  3. Start the service: sudo systemctl start $SERVICE_NAME"
echo "  4. Check status: sudo systemctl status $SERVICE_NAME"
echo "  5. View logs: sudo journalctl -u $SERVICE_NAME -f"
echo
log_info "Service will be available at http://localhost:8000 after starting"
log_info "API documentation will be at http://localhost:8000/docs"