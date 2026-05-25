#!/bin/bash
set -e

# Project root directory
NEXUS_DIR="${NEXUS_DIR:-$HOME/dev/nexus}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_and_echo() {
    echo "\$ $*"  # Echoes the command with a '$' prefix
    "$@"          # Executes the command
}

echo -e "${RED}NUCLEAR RESET: Clearing ALL data...${NC}"

# Stop API and Worker processes
echo -e "${YELLOW}1. Stopping processes...${NC}"
run_and_echo pkill -f "nexus.api" 2>/dev/null || true
run_and_echo pkill -f "nexus.worker" 2>/dev/null || true
echo ""

# Stop Docker
echo -e "${YELLOW}2. Stopping Docker containers...${NC}"
run_and_echo docker-compose down
echo ""

# Delete Docker volumes
echo -e "${YELLOW}3. Deleting Docker volumes...${NC}"
run_and_echo docker volume rm nexus_postgres_data 2>/dev/null || true
run_and_echo docker volume rm nexus_prometheus_data 2>/dev/null || true
run_and_echo docker volume rm nexus_grafana_data 2>/dev/null || true
echo ""

# Clear metric files
echo -e "${YELLOW}4. Clearing metric files...${NC}"
run_and_echo cd "$NEXUS_DIR"
run_and_echo mkdir -p /tmp/prometheus_multiproc
run_and_echo rm -rf /tmp/prometheus_multiproc/*
run_and_echo ls -la /tmp/prometheus_multiproc/
echo ""

# Start Docker
echo -e "${YELLOW}5. Starting Docker services...${NC}"
run_and_echo docker-compose up -d
echo ""

echo -e "${YELLOW}6. Waiting for PostgreSQL to be healthy...${NC}"
run_and_echo sleep 15
echo ""

# Activate virtual environment and run migrations
echo -e "${YELLOW}7. Running database migrations...${NC}"
run_and_echo cd "$NEXUS_DIR"
run_and_echo source .venv/bin/activate
run_and_echo alembic upgrade head
echo ""

echo ""
echo -e "${GREEN}RESET COMPLETE!${NC}"
echo ""
echo "All data cleared and database tables created."
echo ""
echo "Next steps:"
echo "1. Terminal 1: export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc && python -m nexus.api"
echo "2. Terminal 2: export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc && python -m nexus.worker 3"
echo "3. Open Grafana: http://localhost:3000 (admin/admin)"
echo "4. Open Prometheus: http://localhost:9090"