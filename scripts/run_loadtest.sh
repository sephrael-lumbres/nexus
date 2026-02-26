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

# Global variables
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
LOCUST_RESULTS_DIR="loadtest/results/locust"
BENCHMARK_RESULTS_DIR="loadtest/results/benchmark"

# Ensure both results directories exist
mkdir -p "${LOCUST_RESULTS_DIR}"
mkdir -p "${BENCHMARK_RESULTS_DIR}"

echo "========================================"
echo "  Nexus Load Test Runner"
echo "========================================"

# Check if API is running and healthy
check_health() {
    echo -e "${YELLOW}Checking API health...${NC}"
    
    if curl -s "${HOST}/health" | grep -q "healthy"; then
        echo -e "${GREEN}  API is healthy${NC}"
        return 0
    else
        echo -e "${RED}  API is not healthy${NC}"
        echo -e "${RED}    Start API: python -m nexus.api${NC}"
        return 1
    fi
}

# Check if workers are running and export WORKER_COUNT so load tests can read it
check_workers() {
    echo -e "${YELLOW}Checking for workers...${NC}"
    
    # Verify that there are active workers and grab the number of active workers
    workers=$(curl -s http://localhost:8000/metrics | grep "nexus_workers_active" | grep -v "^#" | grep -v "0.0$" | awk '{print int($2)}')
    if [ "$workers" != "0" ] && [ -n "$workers" ]; then
        echo -e "${GREEN}  $workers Workers appear to be running${NC}"
        export WORKER_COUNT=$workers
        return 0
    else
        echo -e "${RED}  No active workers${NC}"
        echo -e "${YELLOW}    Start workers in separate terminal: WORKER_COUNT=3 python -m nexus.worker${NC}"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
        return 0
    fi
}

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

    # Export TEST_TYPE so locustfile.py can read it
    export TEST_TYPE="${test_type}"
    
    locust -f loadtest/locustfile.py \
        --host="${HOST}" \
        --headless \
        -u "${USERS}" \
        -r "${SPAWN_RATE}" \
        -t "${DURATION}" \
        ${TAGS} \
        --html="${LOCUST_RESULTS_DIR}/locust_${test_type}_${TIMESTAMP}.html"

    echo ""
    echo -e "${GREEN}Load test complete${NC}"
    
    # Check if jobs are still pending
    pending=$(curl -s "${HOST}/queue/stats" 2>/dev/null | jq -r '.pending // 0')
    if [ "$pending" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Note: ${pending} jobs still pending. Workers are processing...${NC}"
        echo -e "${YELLOW}      Check stats again in a few minutes for final results.${NC}"
    fi

    echo ""
    echo -e "${YELLOW}Results saved:${NC}"
    echo "  JSON: ${LOCUST_RESULTS_DIR}/locust_${test_type}_${TIMESTAMP}.json"
    echo "  HTML: ${LOCUST_RESULTS_DIR}/locust_${test_type}_${TIMESTAMP}.html"
}

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
            python -m loadtest.benchmark --full --url "${HOST}" --output "${BENCHMARK_RESULTS_DIR}/benchmark_${TIMESTAMP}.json"
            ;;
        *)
            python -m loadtest.benchmark --url "${HOST}" --output "${BENCHMARK_RESULTS_DIR}/benchmark_${TIMESTAMP}.json"
            ;;
    esac
}

# Parse arguments
COMMAND="${1:-help}"
TEST_TYPE="${2:-mixed}"

case $COMMAND in
    locust)
        check_health || exit 1
        check_workers || exit 1
        run_locust "${TEST_TYPE}"
        ;;
    
    benchmark)
        check_health || exit 1
        check_workers || exit 1
        run_benchmark "${TEST_TYPE}"
        ;;
    
    full)
        # Full test: run benchmark, then locust
        echo -e "${YELLOW}Running full test suite...${NC}"
        check_health || exit 1
        check_workers || exit 1
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
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo "  Terminal 1: python -m nexus.api"
        echo "  Terminal 2: python -m nexus.worker 3"
        echo "  Terminal 3: $0 <command>"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
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
        echo -e "${YELLOW}Environment Variables:${NC}"
        echo "  HOST              API URL (default: http://localhost:8000)"
        echo "  USERS             Number of concurrent users (default: 50)"
        echo "  SPAWN_RATE        Users spawned per second (default: 10)"
        echo "  DURATION          Test duration (default: 60s)"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  $0 benchmark quick"
        echo "  $0 locust throughput"
        echo "  USERS=100 DURATION=120s $0 locust stress"
        echo "  $0 full"
        ;;
esac