"""Automated benchmark suite for Nexus job queue.

This script runs a series of benchmarks to measure:
- Throughput (jobs/sec and prompts/sec)
- Latency percentiles
- Queue processing time
- System limits
- Batch vs sequential speedup (Nx)

Usage:
    python -m loadtest.benchmark
    python -m loadtest.benchmark --quick
    python -m loadtest.benchmark --full
    python -m loadtest.benchmark --compare
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    base_url: str = "http://localhost:8000"
    quick_jobs: int = 50
    standard_jobs: int = 200
    full_jobs: int = 500
    ci_jobs: int = 500
    concurrent_submitters: int = 10
    poll_interval: float = 0.1
    poll_timeout: float = 120.0
    # Throughput comparison (batch vs sequential) settings.
    # prompts_per_batch must not exceed BatchHandler.MAX_CONCURRENCY (5).
    comparison_total_prompts: int = 300        # same prompt count for both conditions
    comparison_prompts_per_batch: int = 5      # saturates BatchHandler.MAX_CONCURRENCY
    comparison_trials: int = 5                 # independent repetitions per condition
    comparison_max_tokens: int = 100           # identical for both job types


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    jobs_submitted: int
    jobs_completed: int
    jobs_failed: int
    total_duration_seconds: float
    throughput_jobs_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    worker_count: int = 0
    job_type: str = "llm.completion"
    prompts_per_job: int = 1
    throughput_prompts_per_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "worker_count": self.worker_count,
            "jobs_submitted": self.jobs_submitted,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "success_rate": round(self.jobs_completed / self.jobs_submitted * 100, 2) if self.jobs_submitted > 0 else 0,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "throughput_jobs_per_sec": round(self.throughput_jobs_per_sec, 2),
            "prompts_per_job": self.prompts_per_job,
            "throughput_prompts_per_sec": round(self.throughput_prompts_per_sec, 2),
            "job_type": self.job_type,
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 2),
                "p95_ms": round(self.latency_p95_ms, 2),
                "p99_ms": round(self.latency_p99_ms, 2),
                "min_ms": round(self.latency_min_ms, 2),
                "max_ms": round(self.latency_max_ms, 2),
            },
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "errors": self.errors[:10],  # First 10 errors
        }


@dataclass
class ThroughputComparisonResult:
    """Per-trial results and aggregated statistics for a batch vs sequential comparison."""
    sequential_trials: list[BenchmarkResult]
    batch_trials: list[BenchmarkResult]
    total_prompts: int
    prompts_per_batch: int
    sequential_mean_prompts_per_sec: float
    sequential_stdev_prompts_per_sec: float
    sequential_cv_pct: float
    batch_mean_prompts_per_sec: float
    batch_stdev_prompts_per_sec: float
    batch_cv_pct: float
    mean_speedup_nx: float
    stdev_speedup_nx: float
    speedup_cv_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "prompts_per_batch": self.prompts_per_batch,
            "sequential_mean_prompts_per_sec": round(self.sequential_mean_prompts_per_sec, 2),
            "sequential_stdev_prompts_per_sec": round(self.sequential_stdev_prompts_per_sec, 2),
            "sequential_cv_pct": round(self.sequential_cv_pct, 2),
            "batch_mean_prompts_per_sec": round(self.batch_mean_prompts_per_sec, 2),
            "batch_stdev_prompts_per_sec": round(self.batch_stdev_prompts_per_sec, 2),
            "batch_cv_pct": round(self.batch_cv_pct, 2),
            "mean_speedup_nx": round(self.mean_speedup_nx, 2),
            "stdev_speedup_nx": round(self.stdev_speedup_nx, 2),
            "speedup_cv_pct": round(self.speedup_cv_pct, 2),
            "sequential_trials": [r.to_dict() for r in self.sequential_trials],
            "batch_trials": [r.to_dict() for r in self.batch_trials],
        }


# =============================================================================
# Benchmark Runner
# =============================================================================
class BenchmarkRunner:
    """Runs benchmarks against the Nexus API.

    Measures throughput, latency, and reliability.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client: httpx.AsyncClient | None = None

    async def setup(self) -> bool:
        """Setup benchmark environment."""
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=30.0,
        )

        # Check API health
        try:
            response = await self.client.get("/health")
            if response.status_code != 200:
                print(f"FAIL API health check failed: {response.status_code}")
                return False

            health = response.json()
            if health["status"] != "healthy":
                print(f"FAIL API unhealthy: {health}")
                return False

            print(f"PASS API healthy: {health}")
            return True

        except Exception as e:
            print(f"FAIL Cannot connect to API: {e}")
            return False

    async def teardown(self):
        """Cleanup after benchmarks."""
        if self.client:
            await self.client.aclose()

    def _get_worker_count(self) -> int:
        """Read Nexus worker count from environment.

        Reads WORKER_COUNT, matching the env var used to
        start the worker pool (e.g. WORKER_COUNT=5 python
        -m nexus.worker). Falls back to the Settings
        default of 3 if not set.
        """
        return int(os.environ.get("WORKER_COUNT", 3))

    def _warmup_job_count(self, cycles_per_worker: int = 2) -> int:
        """Compute warmup job count autoscaling to worker count, with a minimum of 10 warmup jobs."

        Args:
            cycles_per_worker: Multiplier to establish all DB and Redis connections.

        Returns:
            The number of warmup jobs needed to cycle each worker.
        """
        return max(10, self._get_worker_count() * cycles_per_worker)


    async def run_benchmark(
        self,
        name: str,
        job_count: int,
        job_type: str = "llm.completion",
        concurrent: int = 10,
        prompts_per_batch: int = 3,
        max_tokens: int = 100,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Benchmark name
            job_count: Number of jobs to submit
            job_type: Type of job (llm.completion or llm.batch)
            concurrent: Number of concurrent submissions
            prompts_per_batch: Number of prompts per llm.batch job payload
            max_tokens: Override payload default of 100

        Returns:
            BenchmarkResult with metrics
        """
        print(f"{'='*60}")
        print(f"Running: {name}")
        print(f"  Jobs: {job_count}, Concurrent: {concurrent}, Type: {job_type}")
        print(f"{'='*60}")

        # Track metrics
        job_ids: list[str] = []
        completion_times: list[float] = []
        errors: list[str] = []
        total_tokens = 0
        total_cost = 0.0

        # Server timestamps for throughput and latency calculation:
        #   created_at   — when the API inserted the job (pipeline start)
        #   completed_at — when the worker wrote the final result (pipeline end)
        # Throughput window: max(completed_at) - min(created_at), free of polling lag.
        server_created_times: list[datetime] = []
        server_completed_times: list[datetime] = []

        start_time = time.time()

        semaphore = asyncio.Semaphore(concurrent)

        async def submit_job(index: int) -> str | None:
            async with semaphore:
                payload = self._create_payload(job_type, index, prompts_per_batch, max_tokens)

                # Attempt to submit job up to 5 times total
                for attempt in range(5):
                    try:
                        response = await self.client.post("/jobs", json=payload)

                        if response.status_code == 201:
                            job_id = response.json()["id"]
                            return job_id

                        elif response.status_code == 429:
                            # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                            wait = 0.5 * (2 ** attempt)
                            print(f"\n  Rate limited, retrying in {wait:.1f}s "
                                f"(attempt {attempt + 1}/5)...")
                            await asyncio.sleep(wait)
                            continue  # retry the loop

                        else:
                            errors.append(f"Submit failed: {response.status_code}")
                            return None

                    except Exception as e:
                        errors.append(f"Submit error: {str(e)}")
                        return None

                # All 5 attempts exhausted
                errors.append("Submit failed: rate limit retries exhausted")
                return None

        # Submit all jobs
        print("  Submitting jobs...", end="", flush=True)
        tasks = [submit_job(i) for i in range(job_count)]
        results = await asyncio.gather(*tasks)
        job_ids = [jid for jid in results if jid is not None]
        print(f" {len(job_ids)}/{job_count} submitted")

        # Poll for completions
        print("  Waiting for completions...", end="", flush=True)
        completed = 0
        failed = 0
        pending_ids = set(job_ids)

        poll_start = time.time()
        while pending_ids and (time.time() - poll_start) < self.config.poll_timeout:
            # Check batch of jobs
            check_ids = list(pending_ids)[:50]  # Check 50 at a time

            for job_id in check_ids:
                try:
                    response = await self.client.get(f"/jobs/{job_id}")

                    if response.status_code == 200:
                        data = response.json()
                        status = data["status"]

                        if status == "completed":
                            # Collect timestamps for throughput and latency
                            raw_created = data.get("created_at")
                            raw_completed = data.get("completed_at")

                            if raw_created and raw_completed:
                                created_at = datetime.fromisoformat(raw_created)
                                completed_at = datetime.fromisoformat(raw_completed)

                                # End-to-end latency: queue wait + worker execution
                                completion_times.append((completed_at - created_at).total_seconds() * 1000)

                                # Throughput window
                                server_created_times.append(created_at)
                                server_completed_times.append(completed_at)

                            total_tokens += data.get("total_tokens", 0)
                            total_cost += data.get("cost_usd", 0)
                            pending_ids.discard(job_id)
                            completed += 1
                        elif status in ["failed", "dead"]:
                            pending_ids.discard(job_id)
                            failed += 1
                            errors.append(f"Job {job_id}: {status}")

                except Exception as e:
                    errors.append(f"Poll error: {str(e)}")

            # Break immediately when all jobs resolve to avoid an extra sleep
            if not pending_ids:
                break

            # Progress indicator
            print(".", end="", flush=True)
            await asyncio.sleep(self.config.poll_interval)

        end_time = time.time()
        total_duration = end_time - start_time

        print(f" {completed}/{len(job_ids)} completed")

        # Calculate latency percentiles
        # quantiles(n=100) returns 99 interpolated cut points.
        # index 49 = P50, 94 = P95, 98 = P99.
        if len(completion_times) >= 2:
            quantiles = statistics.quantiles(completion_times, n=100)
            p50 = quantiles[49]
            p95 = quantiles[94]
            p99 = quantiles[98]
            min_time = min(completion_times)
            max_time = max(completion_times)
        elif len(completion_times) == 1:
            p50 = p95 = p99 = min_time = max_time = completion_times[0]
        else:
            p50 = p95 = p99 = min_time = max_time = 0

        # Calculate throughput
        # Window: max(completed_at) - min(created_at).
        # Falls back to client wall-clock if server timestamps are unavailable.
        if server_created_times and server_completed_times:
            throughput_window = (
                max(server_completed_times) - min(server_created_times)
            ).total_seconds()
            print("  Calculating throughput via server-side timestamps")
            throughput = completed / throughput_window if throughput_window > 0 else 0
        else:
            print("  Server timestamps unavailable, calculating throughput via client wall-clock")
            throughput = completed / total_duration if total_duration > 0 else 0

        # Used to calculate prompts per second throughput
        prompts_per_job = prompts_per_batch if job_type == "llm.batch" else 1

        result = BenchmarkResult(
            name=name,
            jobs_submitted=len(job_ids),
            jobs_completed=completed,
            jobs_failed=failed,
            total_duration_seconds=total_duration,
            throughput_jobs_per_sec=throughput,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_min_ms=min_time,
            latency_max_ms=max_time,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            errors=errors,
            worker_count=self._get_worker_count(),
            job_type=job_type,
            prompts_per_job=prompts_per_job,
            throughput_prompts_per_sec=throughput * prompts_per_job,
        )

        self._print_result(result)
        return result

    def _create_payload(
        self,
        job_type: str,
        index: int,
        prompts_per_batch: int = 3,
        max_tokens: int = 100,
    ) -> dict[str, Any]:
        """Create job payload.

        Args:
            job_type: llm.completion or llm.batch
            index: Job index used to differentiate prompts
            prompts_per_batch: Number of prompts in an llm.batch payload
            max_tokens: Override payload default of 100
        """
        if job_type == "llm.batch":
            return {
                "job_type": "llm.batch",
                "input_data": {
                    "items": [
                        {
                            "id": f"q{i}",
                            "prompt": f"Benchmark question {index * prompts_per_batch + i}: Explain a technical concept.",
                        }
                        for i in range(prompts_per_batch)
                    ],
                    "model": "gpt-4o-mini",
                    "max_tokens": max_tokens,
                },
            }
        else:
            return {
                "job_type": "llm.completion",
                "input_data": {
                    "prompt": f"Benchmark question {index}: Explain a technical concept.",
                    "model": "gpt-4o-mini",
                    "max_tokens": max_tokens,
                },
            }

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print("\n  Results:")
        print(f"    Submitted:  {result.jobs_submitted}")
        print(f"    Completed:  {result.jobs_completed}")
        print(f"    Failed:     {result.jobs_failed}")
        print(f"    Duration:   {result.total_duration_seconds:.2f}s")
        if result.job_type == "llm.batch":
            print(f"    Prompts/Job: {result.prompts_per_job}")
            print(f"    Prompt Throughput: {result.throughput_prompts_per_sec:.2f} prompts/sec")
        print(f"    Job Throughput: {result.throughput_jobs_per_sec:.2f} jobs/sec")
        print(f"    Latency P50: {result.latency_p50_ms:.2f}ms")
        print(f"    Latency P95: {result.latency_p95_ms:.2f}ms")
        print(f"    Latency P99: {result.latency_p99_ms:.2f}ms")
        print()
        if result.errors:
            print(f"    Errors: {len(result.errors)}")


# =============================================================================
# Benchmark Suites
# =============================================================================
async def run_quick_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run quick benchmark suite."""
    runner = BenchmarkRunner(config)
    results = []

    if not await runner.setup():
        return results

    try:
        # Warmup: 1 cycle per worker, matches measurement concurrency.
        print("\nWarming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=runner._warmup_job_count(cycles_per_worker=1),
            concurrent=config.concurrent_submitters,
        )

        # Quick throughput test
        result = await runner.run_benchmark(
            "Quick Throughput",
            job_count=config.quick_jobs,
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

    finally:
        await runner.teardown()

    return results


async def run_standard_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run standard benchmark suite."""
    runner = BenchmarkRunner(config)
    results = []

    if not await runner.setup():
        return results

    try:
        # Warmup: 2 cycles per worker to match this suite's peak concurrency (High concurrency)
        print("\nWarming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=runner._warmup_job_count(cycles_per_worker=2),
            concurrent=config.concurrent_submitters * 2,
        )

        # Completion jobs
        result = await runner.run_benchmark(
            "Completion Jobs",
            job_count=config.standard_jobs,
            job_type="llm.completion",
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

        # Batch jobs
        result = await runner.run_benchmark(
            "Batch Jobs",
            job_count=config.standard_jobs // 2,
            job_type="llm.batch",
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

        # High concurrency
        result = await runner.run_benchmark(
            "High Concurrency",
            job_count=config.standard_jobs,
            concurrent=config.concurrent_submitters * 2,
        )
        results.append(result)

    finally:
        await runner.teardown()

    return results


async def run_full_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    runner = BenchmarkRunner(config)
    results = []

    if not await runner.setup():
        return results

    try:
        # Warmup: 3 cycles per worker to match this suite's peak concurrency (Stress Test)
        print("\nWarming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=runner._warmup_job_count(cycles_per_worker=3),
            concurrent=config.concurrent_submitters * 3,
        )

        # Baseline
        result = await runner.run_benchmark(
            "Baseline (Low Concurrency)",
            job_count=config.quick_jobs,
            concurrent=5,
        )
        results.append(result)

        # Standard throughput
        result = await runner.run_benchmark(
            "Standard Throughput",
            job_count=config.standard_jobs,
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

        # High throughput
        result = await runner.run_benchmark(
            "High Throughput",
            job_count=config.full_jobs,
            concurrent=config.concurrent_submitters * 2,
        )
        results.append(result)

        # Batch processing
        result = await runner.run_benchmark(
            "Batch Processing",
            job_count=config.standard_jobs // 2,
            job_type="llm.batch",
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

        # Stress test
        result = await runner.run_benchmark(
            "Stress Test",
            job_count=config.full_jobs,
            concurrent=config.concurrent_submitters * 3,
        )
        results.append(result)

    finally:
        await runner.teardown()

    return results


async def run_ci_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run CI regression-gate benchmark."""
    runner = BenchmarkRunner(config)
    results = []

    if not await runner.setup():
        return results

    try:
        print("\nWarming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=runner._warmup_job_count(cycles_per_worker=2),
            concurrent=config.concurrent_submitters,
        )

        result = await runner.run_benchmark(
            "CI Throughput",
            job_count=config.ci_jobs,
            concurrent=config.concurrent_submitters,
        )
        results.append(result)

    finally:
        await runner.teardown()

    return results


async def run_throughput_comparison(config: BenchmarkConfig) -> ThroughputComparisonResult | None:
    """Run batch vs sequential throughput comparison.

    Measures prompts/sec for both job types over multiple independent trials
    using a fixed, normalized prompt count. Returns a ThroughputComparisonResult
    with per-trial data and aggregated statistics.

    Controlled experiment design:
    - Same total prompts processed per trial in both conditions
    - Same prompt template, model, max_tokens, and submitter concurrency
    - prompts_per_batch set to BatchHandler.MAX_CONCURRENCY to ensure complete saturation
    - Per-trial warmups keep connection pools hot
    """
    runner = BenchmarkRunner(config)
    total_prompts      = max(config.comparison_total_prompts, runner._get_worker_count() * 100)
    prompts_per_batch  = config.comparison_prompts_per_batch
    trials             = config.comparison_trials
    max_tokens         = config.comparison_max_tokens

    seq_job_count   = total_prompts                       # e.g. 300 jobs × 1 prompt
    batch_job_count = total_prompts // prompts_per_batch  # e.g.  60 jobs × 5 prompts

    sequential_trials: list[BenchmarkResult] = []
    batch_trials: list[BenchmarkResult] = []

    if not await runner.setup():
        return None

    try:
        # Initial warmup: 1 cycle per worker, per job type saturates DB and Redis pools.
        await runner.run_benchmark(
            "Initial Warmup (sequential)",
            job_count=runner._warmup_job_count(cycles_per_worker=1),
            job_type="llm.completion",
            concurrent=config.concurrent_submitters,
        )

        await runner.run_benchmark(
            "Initial Warmup (batch)",
            job_count=runner._warmup_job_count(cycles_per_worker=1),
            job_type="llm.batch",
            prompts_per_batch=prompts_per_batch,
            max_tokens=max_tokens,
            concurrent=config.concurrent_submitters,
        )

        for trial in range(trials):
            trial_number = trial + 1
            print(f"\n{'='*60}")
            print(f"TRIAL {trial_number} of {trials}")
            print(f"{'='*60}")

            # Per-trial sequential warmup: 1 cycle per worker re-activates connections.
            await runner.run_benchmark(
                f"Warmup Sequential (trial {trial_number})",
                job_count=runner._warmup_job_count(cycles_per_worker=1),
                job_type="llm.completion",
                max_tokens=max_tokens,
                concurrent=config.concurrent_submitters,
            )

            seq_result = await runner.run_benchmark(
                f"Sequential (trial {trial_number})",
                job_count=seq_job_count,
                job_type="llm.completion",
                max_tokens=max_tokens,
                concurrent=config.concurrent_submitters,
            )
            sequential_trials.append(seq_result)

            # Per-trial batch warmup: 1 cycle per worker using the batch handler
            # path so the asyncio task scheduler and semaphore are pre-exercised.
            await runner.run_benchmark(
                f"Warmup Batch (trial {trial_number})",
                job_count=runner._warmup_job_count(cycles_per_worker=1),
                job_type="llm.batch",
                prompts_per_batch=prompts_per_batch,
                max_tokens=max_tokens,
                concurrent=config.concurrent_submitters,
            )

            batch_result = await runner.run_benchmark(
                f"Batch (trial {trial_number})",
                job_count=batch_job_count,
                job_type="llm.batch",
                prompts_per_batch=prompts_per_batch,
                max_tokens=max_tokens,
                concurrent=config.concurrent_submitters,
            )
            batch_trials.append(batch_result)

    finally:
        await runner.teardown()

    seq_pps   = [r.throughput_prompts_per_sec for r in sequential_trials]
    batch_pps = [r.throughput_prompts_per_sec for r in batch_trials]

    # Per-trial speedup ratios (one per trial, not mean-of-means)
    nx_values = [
        b / s for s, b in zip(seq_pps, batch_pps) if s > 0
    ]

    # stdev requires at least 2 samples; falls back to 0.0 for a single trial.
    mean_seq    = statistics.mean(seq_pps)
    mean_batch  = statistics.mean(batch_pps)
    stdev_seq   = statistics.stdev(seq_pps)   if len(seq_pps)   >= 2 else 0.0
    stdev_batch = statistics.stdev(batch_pps) if len(batch_pps) >= 2 else 0.0

    # CV = (stdev / mean) × 100% — dimensionless, comparable across configurations
    cv_seq   = (stdev_seq   / mean_seq)   * 100 if mean_seq   > 0 else 0.0
    cv_batch = (stdev_batch / mean_batch) * 100 if mean_batch > 0 else 0.0

    # Speedup ratio statistics (variance OF the ratio, not just mean ratio)
    mean_nx  = statistics.mean(nx_values) if nx_values else 0.0
    stdev_nx = statistics.stdev(nx_values) if len(nx_values) >= 2 else 0.0
    cv_nx    = (stdev_nx / mean_nx) * 100 if mean_nx > 0 else 0.0

    return ThroughputComparisonResult(
        sequential_trials=sequential_trials,
        batch_trials=batch_trials,
        total_prompts=total_prompts,
        prompts_per_batch=prompts_per_batch,
        sequential_mean_prompts_per_sec=mean_seq,
        sequential_stdev_prompts_per_sec=stdev_seq,
        sequential_cv_pct=cv_seq,
        batch_mean_prompts_per_sec=mean_batch,
        batch_stdev_prompts_per_sec=stdev_batch,
        batch_cv_pct=cv_batch,
        mean_speedup_nx=mean_nx,
        stdev_speedup_nx=stdev_nx,
        speedup_cv_pct=cv_nx,
    )


def generate_comparison_report(comparison: ThroughputComparisonResult, output_path: Path | None = None):
    """Generate throughput comparison report (batch vs sequential)."""
    worker_count = (
        comparison.sequential_trials[0].worker_count
        if comparison.sequential_trials
        else int(os.environ.get("WORKER_COUNT", 3))
    )
    generated_at = datetime.now(UTC).isoformat()

    print("\n" + "=" * 60)
    print("THROUGHPUT COMPARISON REPORT (Batch vs Sequential)")
    print("=" * 60)
    print(f"Generated:      {generated_at}")
    print(f"Nexus Workers:  {worker_count}")
    print(f"Trials:         {len(comparison.sequential_trials)}")
    print(f"Total Prompts:  {comparison.total_prompts} per trial")
    print(f"Prompts/Batch:  {comparison.prompts_per_batch}")
    print()

    # Per-trial table
    print(f"{'Trial':<8} {'Sequential (prompts/s)':>24} {'Batch (prompts/s)':>20} {'Nx':>9}")
    print("-" * 64)

    for i, (s, b) in enumerate(
        zip(comparison.sequential_trials, comparison.batch_trials), 1
    ):
        nx = (
            b.throughput_prompts_per_sec / s.throughput_prompts_per_sec
            if s.throughput_prompts_per_sec > 0 else 0
        )
        print(
            f"{i:<8} "
            f"{s.throughput_prompts_per_sec:>24.2f} "
            f"{b.throughput_prompts_per_sec:>20.2f} "
            f"{nx:>8.2f}x"
        )

    print("-" * 64)

    print(f"\nSummary (mean ± stdev over {len(comparison.sequential_trials)} trials):")
    print(f"  Sequential: {comparison.sequential_mean_prompts_per_sec:.2f} ± "
        f"{comparison.sequential_stdev_prompts_per_sec:.2f} prompts/sec "
        f"(CV: {comparison.sequential_cv_pct:.2f}%)")

    print(f"  Batch:      {comparison.batch_mean_prompts_per_sec:.2f} ± "
        f"{comparison.batch_stdev_prompts_per_sec:.2f} prompts/sec "
        f"(CV: {comparison.batch_cv_pct:.2f}%)")

    print(f"  Speedup Nx: {comparison.mean_speedup_nx:.2f}x ± "
        f"{comparison.stdev_speedup_nx:.2f}x "
        f"(CV: {comparison.speedup_cv_pct:.2f}%)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "generated_at": generated_at,
            "worker_count": worker_count,
            "mode": "throughput_comparison",
            **comparison.to_dict(),
        }, indent=2))
        print(f"\nReport saved to: {output_path}")

    print("=" * 60)


# =============================================================================
# Report Generation
# =============================================================================
def generate_report(results: list[BenchmarkResult], output_path: Path | None = None):
    """Generate benchmark report."""
    worker_count = results[0].worker_count if results else int(os.environ.get("WORKER_COUNT", 3))
    generated_at = datetime.now(UTC).isoformat()
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Generated: {generated_at}")
    print(f"Nexus Workers: {worker_count}")
    print()

    # Summary table
    print(f"{'Benchmark':<30} {'Jobs':>8} {'Throughput':>12} {'P50':>10} {'P95':>10} {'P99':>10} {'Success':>9}")
    print("-" * 100)

    for result in results:
        success_rate = result.jobs_completed / result.jobs_submitted * 100 if result.jobs_submitted > 0 else 0
        print(
            f"{result.name:<30} "
            f"{result.jobs_submitted:>8} "
            f"{result.throughput_jobs_per_sec:>10.2f}/s "
            f"{result.latency_p50_ms:>8.1f}ms "
            f"{result.latency_p95_ms:>8.1f}ms "
            f"{result.latency_p99_ms:>8.1f}ms "
            f"{success_rate:>8.1f}%"
        )

    print("-" * 100)

    # Aggregate stats
    if results:
        total_jobs = sum(r.jobs_submitted for r in results)
        total_completed = sum(r.jobs_completed for r in results)
        avg_throughput = statistics.mean(r.throughput_jobs_per_sec for r in results)
        avg_p95 = statistics.mean(r.latency_p95_ms for r in results)

        print("\nAggregate Statistics:")
        print(f"  Total Jobs Submitted: {total_jobs}")
        print(f"  Total Jobs Completed: {total_completed}")
        print(f"  Average Throughput:   {avg_throughput:.2f} jobs/sec")
        print(f"  Average P95 Latency:  {avg_p95:.2f}ms")

        # Check if we meet targets
        print("\nTarget Validation:")
        throughput_target = 50  # jobs/sec
        success_target = 99.0  # percent

        best_throughput = max(r.throughput_jobs_per_sec for r in results)
        overall_success = total_completed / total_jobs * 100 if total_jobs > 0 else 0

        if best_throughput >= throughput_target:
            print(f"  PASS Throughput: {best_throughput:.2f}/sec >= {throughput_target}/sec target")
        else:
            print(f"  FAIL Throughput: {best_throughput:.2f}/sec < {throughput_target}/sec target")

        if overall_success >= success_target:
            print(f"  PASS Success Rate: {overall_success:.2f}% >= {success_target}% target")
        else:
            print(f"  FAIL Success Rate: {overall_success:.2f}% < {success_target}% target")

    # Save to file
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "generated_at": generated_at,
            "worker_count": worker_count,
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(report_data, indent=2))
        print(f"\nReport saved to: {output_path}")

    print("=" * 60)


# =============================================================================
# Main
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Nexus Benchmark Suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer jobs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run CI regression-gate benchmark",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run batch vs sequential throughput comparison",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--submitters",
        type=int,
        default=10,
        help="Number of concurrent job submitters",
    )

    args = parser.parse_args()
    worker_count = int(os.environ.get("WORKER_COUNT", 3))

    config = BenchmarkConfig(
        base_url=args.url,
        concurrent_submitters=args.submitters,
    )

    print("Nexus Benchmark Suite")
    print(f"   URL: {config.base_url}")
    print(f"   Submitters: {config.concurrent_submitters}")
    print(f"   Workers: {worker_count}")

    results: list[BenchmarkResult] = []
    comparison: ThroughputComparisonResult | None = None

    if args.quick:
        print("   Mode: Quick")
        results = await run_quick_benchmark(config)
    elif args.full:
        print("   Mode: Full")
        results = await run_full_benchmark(config)
    elif args.ci:
        print("   Mode: CI")
        results = await run_ci_benchmark(config)
    elif args.compare:
        total_prompts = max(config.comparison_total_prompts, worker_count * 100)
        print("   Mode:          Throughput Comparison (Batch vs Sequential)")
        print(f"   Trials:        {config.comparison_trials}")
        print(f"   Total Prompts: {total_prompts} per trial")
        print(f"   Prompts/Batch: {config.comparison_prompts_per_batch}")
        comparison = await run_throughput_comparison(config)
    else:
        print("   Mode: Standard")
        results = await run_standard_benchmark(config)

    if comparison:
        generate_comparison_report(comparison, args.output)

        if comparison.mean_speedup_nx < 1.0:
            print("\nFAIL Batch is slower than sequential")
            sys.exit(1)
    elif results:
        generate_report(results, args.output)
        total_jobs = sum(r.jobs_submitted for r in results)
        total_completed = sum(r.jobs_completed for r in results)
        best_throughput = max(r.throughput_jobs_per_sec for r in results)
        overall_success = total_completed / total_jobs * 100 if total_jobs > 0 else 0

        # Exit with error if targets not met
        if best_throughput < 50 or overall_success < 99.0:
            sys.exit(1)
    else:
        print("No benchmark results (API may be unavailable)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
