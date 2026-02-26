"""Locust load testing for Nexus job queue.

This file defines user behaviors for load testing:
- JobSubmitter: Submits jobs at high rate
- JobMonitor: Polls job status
- StatsViewer: Checks statistics endpoint
- MixedUser: Realistic mix of all behaviors

Usage:
    # Start Locust web UI
    locust -f loadtest/locustfile.py --host=http://localhost:8000

    # Headless mode (CI/CD)
    locust -f loadtest/locustfile.py --host=http://localhost:8000 \
        --headless -u 100 -r 10 -t 60s

Run scenarios:
    # High throughput test
    locust -f loadtest/locustfile.py --host=http://localhost:8000 \
        --headless -u 50 -r 5 -t 120s --tags throughput

    # Stress test
    locust -f loadtest/locustfile.py --host=http://localhost:8000 \
        --headless -u 200 -r 20 -t 60s --tags stress
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from locust import HttpUser, between, events, tag, task
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from nexus.models import JobRecord

# =============================================================================
# Test Data
# =============================================================================
PROMPTS = [
    "Explain microservices architecture in simple terms.",
    "What are the benefits of using Redis as a cache?",
    "How does PostgreSQL handle concurrent transactions?",
    "Describe the CAP theorem and its implications.",
    "What is the difference between SQL and NoSQL databases?",
    "Explain Docker containerization.",
    "What is Kubernetes and why is it useful?",
    "How do message queues improve system reliability?",
    "What are the SOLID principles in software design?",
    "Explain the concept of eventual consistency.",
    "What is a distributed system?",
    "How does load balancing work?",
    "What is the purpose of an API gateway?",
    "Explain the circuit breaker pattern.",
    "What are the benefits of infrastructure as code?",
]

MODELS = ["gpt-4o-mini", "gpt-4o"]

BATCH_SIZES = [2, 3, 5, 10]


def random_completion_payload() -> dict[str, Any]:
    """Generate a random completion job payload."""
    return {
        "job_type": "llm.completion",
        "input_data": {
            "prompt": random.choice(PROMPTS),
            "model": random.choice(MODELS),
            "max_tokens": random.randint(50, 200),
            "temperature": round(random.uniform(0.5, 1.0), 1),
        },
    }


def random_batch_payload() -> dict[str, Any]:
    """Generate a random batch job payload."""
    batch_size = random.choice(BATCH_SIZES)
    items = [
        {"id": f"q{i}", "prompt": random.choice(PROMPTS)}
        for i in range(batch_size)
    ]
    return {
        "job_type": "llm.batch",
        "input_data": {
            "items": items,
            "model": random.choice(MODELS),
            "max_tokens": random.randint(50, 100),
            "temperature": 0.7,
        },
    }


# =============================================================================
# Metrics Tracking
# =============================================================================
class MetricsCollector:
    """Collect custom metrics during load test."""

    def __init__(self):
        self.jobs_submitted = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.completion_times: list[float] = []

    def record_submission(self):
        self.jobs_submitted += 1

    def record_completion(self, job_data: dict):
        self.jobs_completed += 1
        if job_data.get("total_tokens"):
            self.total_tokens += job_data["total_tokens"]
        if job_data.get("cost_usd"):
            self.total_cost += job_data["cost_usd"]
        if job_data.get("duration_ms"):
            self.completion_times.append(job_data["duration_ms"])

    def record_failure(self):
        self.jobs_failed += 1

    def get_summary(self) -> dict:
        avg_completion_time = (
            sum(self.completion_times) / len(self.completion_times)
            if self.completion_times else 0
        )
        return {
            "worker_count": int(os.environ.get("WORKER_COUNT", 3)),
            "jobs_submitted": self.jobs_submitted,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "success_rate": (
                self.jobs_completed / self.jobs_submitted * 100
                if self.jobs_submitted > 0 else 0
            ),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_completion_time_ms": round(avg_completion_time, 2),
        }


# =============================================================================
# Global Variables
# =============================================================================
metrics = MetricsCollector()
test_start_time = None
test_end_time = None


# =============================================================================
# Database Stats Query Helper
# =============================================================================
async def get_db_stats(start_time: datetime, end_time: datetime) -> dict:
    """Query job statistics from database for jobs created in time window.

    Args:
        start_time: Start of test run
        end_time: End of test run

    Returns:
        Dictionary with job counts by status
    """
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://nexus:nexus@localhost:5432/nexus")

    engine = create_async_engine(database_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            # Query for jobs created during test run
            query = select(
                JobRecord.status,
                func.count(JobRecord.id).label('count')
            ).where(
                JobRecord.created_at >= start_time,
                JobRecord.created_at <= end_time
            ).group_by(JobRecord.status)

            result = await session.execute(query)
            status_counts = {row.status: row.count for row in result}

            return {
                'completed': status_counts.get('completed', 0),
                'failed': status_counts.get('failed', 0),
                'pending': status_counts.get('pending', 0),
                'processing': status_counts.get('running', 0),
                'total': sum(status_counts.values()),
            }
    finally:
        await engine.dispose()


# =============================================================================
# Event Listeners
# =============================================================================
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Record test start time."""
    global test_start_time
    test_start_time = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"Load test started: {test_start_time.isoformat()}")
    print(f"{'=' * 60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary and save to JSON when test stops."""
    global test_end_time
    test_end_time = datetime.now()
    summary = metrics.get_summary()
    duration = None

    # Extract test configuration from environment
    test_config = {
        "host": environment.parsed_options.host,
        "users": environment.parsed_options.num_users,
        "spawn_rate": environment.parsed_options.spawn_rate,
        "run_time": environment.parsed_options.run_time,
        "test_type": os.getenv("TEST_TYPE", "unknown"),
    }

    # Query database for stats
    try:
        server_stats = asyncio.run(get_db_stats(test_start_time, test_end_time))
    except Exception as e:
        print(f"Failed to query database: {e}")
        server_stats = None

    # Print to console
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    # Print test configuration
    print("TEST CONFIGURATION:")
    print(f"  Host:        {test_config['host']}")
    print(f"  Users:       {test_config['users']}")
    print(f"  Spawn Rate:  {test_config['spawn_rate']}/s")
    print(f"  Run Time:    {test_config['run_time']}")
    print(f"  Test Type:   {test_config['test_type']}")

    if test_start_time and test_end_time:
        duration = (test_end_time - test_start_time).total_seconds()
        print("\nTEST EXECUTION:")
        print(f"  Started:  {test_start_time.isoformat()}")
        print(f"  Ended:    {test_end_time.isoformat()}")
        print(f"  Duration: {duration:.2f}s")

    print("\nCLIENT STATS:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if server_stats:
        print("\nSERVER STATS (from database):")
        print(f"  completed:  {server_stats.get('completed', 'N/A')}")
        print(f"  failed:     {server_stats.get('failed', 'N/A')}")
        print(f"  pending:    {server_stats.get('pending', 'N/A')}")
        print(f"  processing: {server_stats.get('processing', 'N/A')}")
        print(f"  total:      {server_stats.get('total', 'N/A')}")

    print("=" * 60 + "\n")

    # Save to JSON
    output_dir = Path("loadtest/results/locust")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = test_start_time.strftime("%Y-%m-%dT%H:%M:%S")
    output_file = output_dir / f"locust_{test_config['test_type']}_{timestamp}.json"

    result = {
        "test_config": test_config,
        "test_start": test_start_time.isoformat() if test_start_time else None,
        "test_end": test_end_time.isoformat() if test_end_time else None,
        "duration_seconds": duration,
        "client_stats": summary,
        "server_stats": server_stats or {},
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {output_file}\n")


# =============================================================================
# User Behaviors
# =============================================================================
class JobSubmitter(HttpUser):
    """User that submits jobs at high rate.

    Used for throughput testing - minimal wait between requests.
    """

    wait_time = between(0.1, 0.5)  # Fast submission
    weight = 3  # Higher weight = more instances

    @tag("throughput", "submit")
    @task(10)
    def submit_completion_job(self):
        """Submit a completion job."""
        payload = random_completion_payload()

        with self.client.post(
            "/jobs",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                metrics.record_submission()
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected under load
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    @tag("throughput", "submit", "batch")
    @task(3)
    def submit_batch_job(self):
        """Submit a batch job."""
        payload = random_batch_payload()

        with self.client.post(
            "/jobs",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                metrics.record_submission()
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")


class JobMonitor(HttpUser):
    """User that monitors job status.

    Simulates clients polling for job completion.
    """

    wait_time = between(1, 3)
    weight = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_ids: list[str] = []

    @tag("monitor")
    @task(5)
    def submit_and_poll(self):
        """Submit a job and poll for completion."""
        # Submit job
        payload = random_completion_payload()

        with self.client.post(
            "/jobs",
            json=payload,
            name="/jobs [submit]",
            catch_response=True
        ) as response:
            if response.status_code != 201:
                return

            job_id = response.json()["id"]
            metrics.record_submission()

        # Poll for completion (up to 10 times)
        for _ in range(10):
            time.sleep(0.5)

            with self.client.get(
                f"/jobs/{job_id}",
                name="/jobs/{id} [poll]",
                catch_response=True,
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    status = data["status"]

                    if status == "completed":
                        metrics.record_completion(data)
                        response.success()
                        return
                    elif status == "failed" or status == "dead":
                        metrics.record_failure()
                        response.failure(f"Job {status}")
                        return
                    else:
                        # Still processing
                        response.success()
                else:
                    response.failure(f"Status {response.status_code}")
                    return

    @tag("monitor")
    @task(2)
    def list_jobs(self):
        """List recent jobs."""
        self.client.get("/jobs?limit=20", name="/jobs [list]")

    @tag("monitor")
    @task(1)
    def check_job_status(self):
        """Check status of a random job from list."""
        # Get a job from the list
        response = self.client.get("/jobs?limit=10", name="/jobs [list for check]")

        if response.status_code == 200:
            jobs = response.json().get("jobs", [])
            if jobs:
                job_id = random.choice(jobs)["id"]
                self.client.get(f"/jobs/{job_id}", name="/jobs/{id} [check]")


class StatsViewer(HttpUser):
    """User that views statistics and health.

    Simulates monitoring dashboards.
    """

    wait_time = between(2, 5)
    weight = 1

    @tag("stats")
    @task(5)
    def check_stats(self):
        """Check queue statistics."""
        self.client.get("/stats", name="/stats")

    @tag("stats")
    @task(3)
    def check_health(self):
        """Check health endpoint."""
        self.client.get("/health", name="/health")

    @tag("stats")
    @task(2)
    def check_queue_stats(self):
        """Check queue statistics."""
        self.client.get("/queue/stats", name="/queue/stats")

    @tag("stats")
    @task(1)
    def peek_pending(self):
        """Peek at pending queue."""
        self.client.get("/queue/pending?count=5", name="/queue/pending")


class MixedUser(HttpUser):
    """Realistic user with mixed behavior.

    Simulates real-world usage patterns.
    """

    wait_time = between(1, 5)
    weight = 5  # Most common user type

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted_jobs: list[str] = []

    @tag("mixed", "submit")
    @task(10)
    def submit_job(self):
        """Submit a job."""
        # 80% completion, 20% batch
        if random.random() < 0.8:
            payload = random_completion_payload()
        else:
            payload = random_batch_payload()

        with self.client.post("/jobs", json=payload, catch_response=True) as response:
            if response.status_code == 201:
                job_id = response.json()["id"]
                self.submitted_jobs.append(job_id)
                # Keep only last 20 jobs
                self.submitted_jobs = self.submitted_jobs[-20:]
                metrics.record_submission()
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    @tag("mixed", "monitor")
    @task(5)
    def check_my_job(self):
        """Check status of a submitted job."""
        if not self.submitted_jobs:
            return

        job_id = random.choice(self.submitted_jobs)

        with self.client.get(
            f"/jobs/{job_id}",
            name="/jobs/{id}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "completed":
                    metrics.record_completion(data)
                    # Remove from tracking
                    if job_id in self.submitted_jobs:
                        self.submitted_jobs.remove(job_id)
                response.success()
            elif response.status_code == 404:
                # Job may have been cleaned up
                if job_id in self.submitted_jobs:
                    self.submitted_jobs.remove(job_id)
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @tag("mixed", "stats")
    @task(2)
    def check_stats(self):
        """Check statistics."""
        self.client.get("/stats", name="/stats")

    @tag("mixed")
    @task(1)
    def check_health(self):
        """Check health."""
        self.client.get("/health", name="/health")


# =============================================================================
# Stress Test User
# =============================================================================
class StressUser(HttpUser):
    """Aggressive user for stress testing.

    Submits jobs as fast as possible.
    """

    wait_time = between(0.01, 0.1)  # Very fast
    weight = 1

    @tag("stress")
    @task
    def rapid_submit(self):
        """Submit jobs rapidly."""
        payload = random_completion_payload()

        with self.client.post(
            "/jobs",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 201]:
                metrics.record_submission()
                response.success()
            elif response.status_code == 429:
                # Expected under stress
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# =============================================================================
# Endurance Test User
# =============================================================================
class EnduranceUser(HttpUser):
    """User for long-running endurance tests.

    Steady, consistent load over time.
    """

    wait_time = between(2, 5)
    weight = 1

    @tag("endurance")
    @task(5)
    def steady_submit(self):
        """Submit jobs at steady rate."""
        payload = random_completion_payload()

        with self.client.post("/jobs", json=payload, catch_response=True) as response:
            if response.status_code == 201:
                metrics.record_submission()
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @tag("endurance")
    @task(2)
    def check_stats(self):
        """Monitor system health."""
        self.client.get("/stats")
        self.client.get("/health")
