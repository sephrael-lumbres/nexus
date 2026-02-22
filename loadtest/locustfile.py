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

import random
import time
from typing import Any

from locust import HttpUser, between, events, tag, task

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


# Global metrics collector
metrics = MetricsCollector()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary when test stops."""
    summary = metrics.get_summary()
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


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

        with self.client.post("/jobs", json=payload, name="/jobs [submit]") as response:
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
