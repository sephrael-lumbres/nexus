#!/bin/bash
# Load test runner script for Nexus

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
HOST="${HOST:-http://localhost:8000}"
USERS="${USERS:-50}"
SPAWN_RATE="${SPAWN_RATE:-10}"
DURATION="${DURATION:-60s}"
WORKERS="${WORKERS:-3}"

# Print banner
echo "========================================"
echo "  Nexus Load Test Runner"
echo "========================================"
echo ""

# Function to check if service is healthy
check_health() {
    echo -e "${YELLOW}Checking API health...${NC}"
    
    if curl -s "${HOST}/health" | grep -q "healthy"; then
        echo -e "${GREEN}✅ API is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ API is not healthy${NC}"
        return 1
    fi
}

# Function to start workers if not running
start_workers() {
    echo -e "${YELLOW}Starting ${WORKERS} workers...${NC}"
    
    # Start workers in background
    python -m nexus.worker ${WORKERS} &
    WORKER_PID=$!
    
    # Wait for workers to start
    sleep 2
    
    echo -e "${GREEN}✅ Workers started (PID: ${WORKER_PID})${NC}"
}

# Function to stop workers
stop_workers() {
    if [ ! -z "$WORKER_PID" ]; then
        echo -e "${YELLOW}Stopping workers...${NC}"
        kill $WORKER_PID 2>/dev/null || true
        wait $WORKER_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Workers stopped${NC}"
    fi
}

# Function to run locust test
run_locust() {
    local test_type="${1:-mixed}"
    
    echo ""
    echo -e "${YELLOW}Running Locust load test...${NC}"
    echo "  Host: ${HOST}"
    echo "  Users: ${USERS}"
    echo "  Spawn Rate: ${SPAWN_RATE}/s"
    echo "  Duration: ${DURATION}"
    echo "  Test Type: ${test_type}"
    echo ""
    
    case $test_type in
        throughput)
            TAGS="--tags throughput"
            ;;
        stress)
            TAGS="--tags stress"
            ;;
        endurance)
            TAGS="--tags endurance"
            ;;
        *)
            TAGS=""
            ;;
    esac
    
    locust -f loadtest/locustfile.py \
        --host="${HOST}" \
        --headless \
        -u "${USERS}" \
        -r "${SPAWN_RATE}" \
        -t "${DURATION}" \
        ${TAGS} \
        --html="loadtest/results/report_${test_type}_$(date +%Y-%m-%dT%H:%M:%S).html"
}

# Function to run benchmark
run_benchmark() {
    local mode="${1:-standard}"
    
    echo ""
    echo -e "${YELLOW}Running benchmark suite...${NC}"
    echo "  Mode: ${mode}"
    echo ""
    
    case $mode in
        quick)
            python -m loadtest.benchmark --quick --url "${HOST}"
            ;;
        full)
            python -m loadtest.benchmark --full --url "${HOST}" --output "loadtest/results/benchmark_$(date +%Y-%m-%dT%H:%M:%S).json"
            ;;
        *)
            python -m loadtest.benchmark --url "${HOST}" --output "loadtest/results/benchmark_$(date +%Y-%m-%dT%H:%M:%S).json"
            ;;
    esac
}

# Parse arguments
COMMAND="${1:-help}"
TEST_TYPE="${2:-mixed}"

# Trap to cleanup on exit
trap stop_workers EXIT

case $COMMAND in
    locust)
        check_health
        run_locust "${TEST_TYPE}"
        ;;
    
    benchmark)
        check_health
        run_benchmark "${TEST_TYPE}"
        ;;
    
    full)
        # Full test: start workers, run benchmark, run locust
        echo -e "${YELLOW}Running full test suite...${NC}"
        check_health || exit 1
        run_benchmark "standard"
        run_locust "mixed"
        ;;
    
    ci)
        # CI mode: quick tests for CI/CD pipelines
        echo -e "${YELLOW}Running CI test suite...${NC}"
        check_health || exit 1
        run_benchmark "quick"
        ;;
    
    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  locust [type]     Run Locust load test"
        echo "                    Types: throughput, stress, endurance, mixed (default)"
        echo ""
        echo "  benchmark [mode]  Run benchmark suite"
        echo "                    Modes: quick, standard (default), full"
        echo ""
        echo "  full              Run full test suite (benchmark + locust)"
        echo ""
        echo "  ci                Run quick tests for CI/CD"
        echo ""
        echo "Environment Variables:"
        echo "  HOST              API URL (default: http://localhost:8000)"
        echo "  USERS             Number of concurrent users (default: 50)"
        echo "  SPAWN_RATE        Users spawned per second (default: 10)"
        echo "  DURATION          Test duration (default: 60s)"
        echo "  WORKERS           Number of workers to start (default: 3)"
        echo ""
        echo "Examples:"
        echo "  $0 benchmark quick"
        echo "  $0 locust throughput"
        echo "  USERS=100 DURATION=120s $0 locust stress"
        echo "  $0 full"
        ;;
esac