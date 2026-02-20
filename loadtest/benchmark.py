"""Automated benchmark suite for Nexus job queue.

This script runs a series of benchmarks to measure:
- Throughput (jobs/sec)
- Latency percentiles
- Queue processing time
- System limits

Usage:
    python -m loadtest.benchmark
    python -m loadtest.benchmark --quick
    python -m loadtest.benchmark --full
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
    warmup_jobs: int = 10
    quick_jobs: int = 50
    standard_jobs: int = 200
    full_jobs: int = 500
    concurrent_workers: int = 10
    poll_interval: float = 0.1
    poll_timeout: float = 60.0


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "jobs_submitted": self.jobs_submitted,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "success_rate": round(self.jobs_completed / self.jobs_submitted * 100, 2) if self.jobs_submitted > 0 else 0,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "throughput_jobs_per_sec": round(self.throughput_jobs_per_sec, 2),
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
                print(f"❌ API health check failed: {response.status_code}")
                return False

            health = response.json()
            if health["status"] != "healthy":
                print(f"❌ API unhealthy: {health}")
                return False

            print(f"✅ API healthy: {health}")
            return True

        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False

    async def teardown(self):
        """Cleanup after benchmarks."""
        if self.client:
            await self.client.aclose()

    async def run_benchmark(
        self,
        name: str,
        job_count: int,
        job_type: str = "llm.completion",
        concurrent: int = 10,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Benchmark name
            job_count: Number of jobs to submit
            job_type: Type of job (llm.completion or llm.batch)
            concurrent: Number of concurrent submissions

        Returns:
            BenchmarkResult with metrics
        """
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  Jobs: {job_count}, Concurrent: {concurrent}, Type: {job_type}")
        print(f"{'='*60}")

        # Track metrics
        job_ids: list[str] = []
        submit_times: dict[str, float] = {}
        completion_times: list[float] = []
        errors: list[str] = []
        total_tokens = 0
        total_cost = 0.0

        # Submit jobs concurrently
        start_time = time.time()

        semaphore = asyncio.Semaphore(concurrent)

        async def submit_job(index: int) -> str | None:
            async with semaphore:
                payload = self._create_payload(job_type, index)

                try:
                    submit_start = time.time()
                    response = await self.client.post("/jobs", json=payload)

                    if response.status_code == 201:
                        job_id = response.json()["id"]
                        submit_times[job_id] = submit_start
                        return job_id
                    elif response.status_code == 429:
                        # Rate limited, retry after delay
                        await asyncio.sleep(0.5)
                        response = await self.client.post("/jobs", json=payload)
                        if response.status_code == 201:
                            job_id = response.json()["id"]
                            submit_times[job_id] = submit_start
                            return job_id

                    errors.append(f"Submit failed: {response.status_code}")
                    return None

                except Exception as e:
                    errors.append(f"Submit error: {str(e)}")
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
                            completion_time = (time.time() - submit_times[job_id]) * 1000
                            completion_times.append(completion_time)
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

            # Progress indicator
            print(".", end="", flush=True)
            await asyncio.sleep(self.config.poll_interval)

        end_time = time.time()
        total_duration = end_time - start_time

        print(f" {completed}/{len(job_ids)} completed")

        # Calculate latency percentiles
        if completion_times:
            sorted_times = sorted(completion_times)
            p50 = sorted_times[int(len(sorted_times) * 0.50)]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
            min_time = min(sorted_times)
            max_time = max(sorted_times)
        else:
            p50 = p95 = p99 = min_time = max_time = 0

        # Calculate throughput
        throughput = completed / total_duration if total_duration > 0 else 0

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
        )

        self._print_result(result)
        return result

    def _create_payload(self, job_type: str, index: int) -> dict[str, Any]:
        """Create job payload."""
        if job_type == "llm.batch":
            return {
                "job_type": "llm.batch",
                "input_data": {
                    "items": [
                        {"id": f"q{i}", "prompt": f"Question {index}-{i}"}
                        for i in range(3)
                    ],
                    "model": "gpt-4o-mini",
                    "max_tokens": 50,
                },
            }
        else:
            return {
                "job_type": "llm.completion",
                "input_data": {
                    "prompt": f"Benchmark question {index}: Explain a technical concept.",
                    "model": "gpt-4o-mini",
                    "max_tokens": 100,
                },
            }

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print("\n  Results:")
        print(f"    Submitted:  {result.jobs_submitted}")
        print(f"    Completed:  {result.jobs_completed}")
        print(f"    Failed:     {result.jobs_failed}")
        print(f"    Duration:   {result.total_duration_seconds:.2f}s")
        print(f"    Throughput: {result.throughput_jobs_per_sec:.2f} jobs/sec")
        print(f"    Latency P50: {result.latency_p50_ms:.2f}ms")
        print(f"    Latency P95: {result.latency_p95_ms:.2f}ms")
        print(f"    Latency P99: {result.latency_p99_ms:.2f}ms")
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
        # Warmup
        print("\n🔥 Warming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=config.warmup_jobs,
            concurrent=5,
        )

        # Quick throughput test
        result = await runner.run_benchmark(
            "Quick Throughput",
            job_count=config.quick_jobs,
            concurrent=config.concurrent_workers,
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
        # Warmup
        print("\n🔥 Warming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=config.warmup_jobs,
            concurrent=5,
        )

        # Completion jobs
        result = await runner.run_benchmark(
            "Completion Jobs",
            job_count=config.standard_jobs,
            job_type="llm.completion",
            concurrent=config.concurrent_workers,
        )
        results.append(result)

        # Batch jobs
        result = await runner.run_benchmark(
            "Batch Jobs",
            job_count=config.standard_jobs // 2,
            job_type="llm.batch",
            concurrent=config.concurrent_workers,
        )
        results.append(result)

        # High concurrency
        result = await runner.run_benchmark(
            "High Concurrency",
            job_count=config.standard_jobs,
            concurrent=config.concurrent_workers * 2,
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
        # Warmup
        print("\n🔥 Warming up...")
        await runner.run_benchmark(
            "Warmup",
            job_count=config.warmup_jobs,
            concurrent=5,
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
            concurrent=config.concurrent_workers,
        )
        results.append(result)

        # High throughput
        result = await runner.run_benchmark(
            "High Throughput",
            job_count=config.full_jobs,
            concurrent=config.concurrent_workers * 2,
        )
        results.append(result)

        # Batch processing
        result = await runner.run_benchmark(
            "Batch Processing",
            job_count=config.standard_jobs // 2,
            job_type="llm.batch",
            concurrent=config.concurrent_workers,
        )
        results.append(result)

        # Stress test
        result = await runner.run_benchmark(
            "Stress Test",
            job_count=config.full_jobs,
            concurrent=config.concurrent_workers * 3,
        )
        results.append(result)

    finally:
        await runner.teardown()

    return results


# =============================================================================
# Report Generation
# =============================================================================
def generate_report(results: list[BenchmarkResult], output_path: Path | None = None):
    """Generate benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().isoformat()}")
    print()

    # Summary table
    print(f"{'Benchmark':<30} {'Jobs':>8} {'Throughput':>12} {'P50':>10} {'P95':>10} {'Success':>10}")
    print("-" * 90)

    for result in results:
        success_rate = result.jobs_completed / result.jobs_submitted * 100 if result.jobs_submitted > 0 else 0
        print(
            f"{result.name:<30} "
            f"{result.jobs_submitted:>8} "
            f"{result.throughput_jobs_per_sec:>10.2f}/s "
            f"{result.latency_p50_ms:>8.1f}ms "
            f"{result.latency_p95_ms:>8.1f}ms "
            f"{success_rate:>8.1f}%"
        )

    print("-" * 90)

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
        print("\n🎯 Target Validation:")
        throughput_target = 50  # jobs/sec
        success_target = 99.0  # percent

        best_throughput = max(r.throughput_jobs_per_sec for r in results)
        overall_success = total_completed / total_jobs * 100 if total_jobs > 0 else 0

        if best_throughput >= throughput_target:
            print(f"  ✅ Throughput: {best_throughput:.2f}/sec >= {throughput_target}/sec target")
        else:
            print(f"  ❌ Throughput: {best_throughput:.2f}/sec < {throughput_target}/sec target")

        if overall_success >= success_target:
            print(f"  ✅ Success Rate: {overall_success:.2f}% >= {success_target}% target")
        else:
            print(f"  ❌ Success Rate: {overall_success:.2f}% < {success_target}% target")

    # Save to file
    if output_path:
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(json.dumps(report_data, indent=2))
        print(f"\n📄 Report saved to: {output_path}")

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
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        base_url=args.url,
        concurrent_workers=args.workers,
    )

    print("🚀 Nexus Benchmark Suite")
    print(f"   URL: {config.base_url}")
    print(f"   Workers: {config.concurrent_workers}")

    if args.quick:
        print("   Mode: Quick")
        results = await run_quick_benchmark(config)
    elif args.full:
        print("   Mode: Full")
        results = await run_full_benchmark(config)
    else:
        print("   Mode: Standard")
        results = await run_standard_benchmark(config)

    if results:
        generate_report(results, args.output)

        # Exit with error if targets not met
        best_throughput = max(r.throughput_jobs_per_sec for r in results)
        if best_throughput < 50:
            sys.exit(1)
    else:
        print("❌ No benchmark results (API may be unavailable)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
