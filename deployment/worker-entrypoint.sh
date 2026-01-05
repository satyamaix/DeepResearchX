#!/bin/bash
# =============================================================================
# DRX Deep Research - Celery Worker Entrypoint Script
# Handles graceful startup, shutdown, and signal handling
# =============================================================================

set -e

# Colors for logging (optional, disabled in production)
if [ "${APP_ENV}" != "production" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, initiating graceful shutdown..."

    # Send SIGTERM to Celery worker for warm shutdown
    if [ -n "$CELERY_PID" ]; then
        log_info "Sending SIGTERM to Celery worker (PID: $CELERY_PID)"
        kill -TERM "$CELERY_PID" 2>/dev/null || true

        # Wait for graceful shutdown (up to 30 seconds)
        WAIT_TIME=0
        while [ $WAIT_TIME -lt 30 ] && kill -0 "$CELERY_PID" 2>/dev/null; do
            sleep 1
            WAIT_TIME=$((WAIT_TIME + 1))
        done

        # Force kill if still running
        if kill -0 "$CELERY_PID" 2>/dev/null; then
            log_warn "Celery worker did not stop gracefully, sending SIGKILL"
            kill -9 "$CELERY_PID" 2>/dev/null || true
        fi
    fi

    log_info "Shutdown complete"
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT SIGQUIT

# =============================================================================
# Pre-flight checks
# =============================================================================

log_info "Starting DRX Celery Worker..."
log_info "Environment: ${APP_ENV:-development}"
log_info "Hostname: $(hostname)"

# Check required environment variables
REQUIRED_VARS="CELERY_BROKER_URL DATABASE_URL"
for var in $REQUIRED_VARS; do
    if [ -z "${!var}" ]; then
        log_error "Required environment variable $var is not set"
        exit 1
    fi
done

# Wait for Redis to be available
log_info "Checking Redis connectivity..."
REDIS_HOST=$(echo "${CELERY_BROKER_URL}" | sed -e 's|redis://||' -e 's|/.*||' -e 's|:.*||')
REDIS_PORT=$(echo "${CELERY_BROKER_URL}" | sed -e 's|redis://[^:]*:||' -e 's|/.*||')
REDIS_PORT=${REDIS_PORT:-6379}

RETRIES=30
until python -c "import redis; r = redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}); r.ping()" 2>/dev/null; do
    RETRIES=$((RETRIES - 1))
    if [ $RETRIES -eq 0 ]; then
        log_error "Failed to connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
        exit 1
    fi
    log_warn "Redis not available yet, retrying... ($RETRIES attempts left)"
    sleep 2
done
log_info "Redis connection successful"

# Wait for PostgreSQL to be available
log_info "Checking PostgreSQL connectivity..."
PG_HOST=$(echo "${DATABASE_URL}" | sed -e 's|postgresql://[^@]*@||' -e 's|/.*||' -e 's|:.*||')
RETRIES=30
until python -c "import psycopg; psycopg.connect('${DATABASE_URL}')" 2>/dev/null; do
    RETRIES=$((RETRIES - 1))
    if [ $RETRIES -eq 0 ]; then
        log_error "Failed to connect to PostgreSQL at ${PG_HOST}"
        exit 1
    fi
    log_warn "PostgreSQL not available yet, retrying... ($RETRIES attempts left)"
    sleep 2
done
log_info "PostgreSQL connection successful"

# =============================================================================
# Configure Celery worker settings
# =============================================================================

# Set defaults for Celery configuration
CELERY_CONCURRENCY=${CELERY_WORKER_CONCURRENCY:-4}
CELERY_PREFETCH=${CELERY_WORKER_PREFETCH_MULTIPLIER:-1}
CELERY_MAX_TASKS=${CELERY_WORKER_MAX_TASKS_PER_CHILD:-1000}
CELERY_MAX_MEMORY=${CELERY_WORKER_MAX_MEMORY_PER_CHILD:-2000000}
LOG_LEVEL=${LOG_LEVEL:-INFO}

log_info "Celery Configuration:"
log_info "  - Concurrency: ${CELERY_CONCURRENCY}"
log_info "  - Prefetch Multiplier: ${CELERY_PREFETCH}"
log_info "  - Max Tasks Per Child: ${CELERY_MAX_TASKS}"
log_info "  - Max Memory Per Child: ${CELERY_MAX_MEMORY} KB"
log_info "  - Log Level: ${LOG_LEVEL}"

# =============================================================================
# Start Celery worker
# =============================================================================

log_info "Starting Celery worker..."

# Build Celery command
CELERY_CMD="celery -A src.worker worker"
CELERY_CMD="$CELERY_CMD --loglevel=${LOG_LEVEL}"
CELERY_CMD="$CELERY_CMD --concurrency=${CELERY_CONCURRENCY}"
CELERY_CMD="$CELERY_CMD --prefetch-multiplier=${CELERY_PREFETCH}"
CELERY_CMD="$CELERY_CMD --max-tasks-per-child=${CELERY_MAX_TASKS}"
CELERY_CMD="$CELERY_CMD --max-memory-per-child=${CELERY_MAX_MEMORY}"
CELERY_CMD="$CELERY_CMD --pool=prefork"
CELERY_CMD="$CELERY_CMD --optimization=fair"
CELERY_CMD="$CELERY_CMD --hostname=celery@%h"

# Add production-specific flags
if [ "${APP_ENV}" = "production" ]; then
    CELERY_CMD="$CELERY_CMD --without-heartbeat"
    CELERY_CMD="$CELERY_CMD --without-gossip"
    CELERY_CMD="$CELERY_CMD --without-mingle"
fi

# Execute Celery worker
log_info "Executing: $CELERY_CMD"
exec $CELERY_CMD &
CELERY_PID=$!

log_info "Celery worker started with PID: $CELERY_PID"

# Wait for Celery process
wait $CELERY_PID
EXIT_CODE=$?

log_info "Celery worker exited with code: $EXIT_CODE"
exit $EXIT_CODE
