# -*- coding: utf-8 -*-
"""Benchmark: Scheduling Policy Comparison

Compare different scheduling policies (LPM, DFS-Weight, FCFS, LOF, Random)
on simulated workloads to measure cache hit rate, scheduling latency, and
queue ordering behavior.

Usage:
    python benchmark/sglang_optimization/bench_scheduling.py
    python benchmark/sglang_optimization/bench_scheduling.py --policies lpm fcfs --num-requests 500

This benchmark does NOT require a running SGLang server. It directly
instantiates RadixCache and SchedulePolicy objects in simulation mode
and replays synthetic request traces.

Workloads:
    1. shared_prefix  - Many requests share a long System Prompt.
    2. no_sharing     - All requests have unique prompts.
    3. mixed          - 50% shared + 50% unique.
    4. long_queue     - 500+ requests to test LPM->FCFS degradation.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockAllocator:
    """A trivial allocator that tracks available / used slots."""

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self.available = total_slots
        self.device = torch.device("cpu")

    def alloc(self, n: int) -> torch.Tensor:
        if n > self.available:
            raise RuntimeError("OOM in mock allocator")
        indices = torch.arange(
            self.total_slots - self.available,
            self.total_slots - self.available + n,
            dtype=torch.int64,
        )
        self.available -= n
        return indices

    def free(self, indices: torch.Tensor):
        self.available += len(indices)

    def available_size(self) -> int:
        return self.available


class MockReq:
    """Lightweight request mock that carries fields needed by SchedulePolicy.

    We only need:
      - rid, origin_input_ids, output_ids, extra_key
      - sampling_params.max_new_tokens
      - prefix_indices, last_node, last_host_node, host_hit_length
      - time_stats.wait_queue_entry_time
      - routing_key, priority
    """

    _counter = 0

    def __init__(
        self,
        token_ids: List[int],
        max_new_tokens: int = 128,
        arrival_order: int = 0,
    ):
        MockReq._counter += 1
        self.rid = f"req-{MockReq._counter}"
        self.origin_input_ids = token_ids
        self.output_ids: List[int] = []
        self.fill_ids = list(token_ids)
        self.extra_key: Optional[str] = None
        self.lora_id: Optional[str] = None
        self.routing_key: Optional[str] = None
        self.priority: Optional[int] = 0

        # Prefix match results (populated by _compute_prefix_matches or init)
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.last_node = None
        self.last_host_node = None
        self.host_hit_length = 0
        self.mamba_branching_seqlen = None
        self.extend_input_len = len(token_ids)
        self.cache_protected_len = 0

        # Mock sampling params
        self.sampling_params = _MockSamplingParams(max_new_tokens)

        # Mock time stats
        self.time_stats = _MockTimeStats(arrival_order)


class _MockSamplingParams:
    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = False
        self.custom_params = None


class _MockTimeStats:
    def __init__(self, arrival_order: int):
        self.wait_queue_entry_time = float(arrival_order)


def make_cache(total_slots: int = 8192) -> Tuple[RadixCache, MockAllocator]:
    """Create a RadixCache in simulation mode."""
    allocator = MockAllocator(total_slots)
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
    )
    cache = RadixCache(params)
    return cache, allocator


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TOKENS = list(range(1, 401))  # 400-token shared prefix


def gen_shared_prefix_requests(
    n: int, user_len_range: Tuple[int, int] = (20, 80), seed: int = 42
) -> List[List[int]]:
    """Generate *n* requests that share SYSTEM_PROMPT_TOKENS."""
    rng = random.Random(seed)
    requests = []
    for _ in range(n):
        user_len = rng.randint(*user_len_range)
        user_tokens = [rng.randint(1000, 9999) for _ in range(user_len)]
        requests.append(SYSTEM_PROMPT_TOKENS + user_tokens)
    return requests


def gen_unique_requests(
    n: int, prompt_len_range: Tuple[int, int] = (100, 300), seed: int = 42
) -> List[List[int]]:
    """Generate *n* requests with entirely unique prompts."""
    rng = random.Random(seed)
    requests = []
    for _ in range(n):
        length = rng.randint(*prompt_len_range)
        requests.append([rng.randint(1000, 9999) for _ in range(length)])
    return requests


def gen_mixed_requests(
    n: int, shared_ratio: float = 0.5, seed: int = 42
) -> List[List[int]]:
    """Generate a mix of shared-prefix and unique requests."""
    n_shared = int(n * shared_ratio)
    n_unique = n - n_shared
    shared = gen_shared_prefix_requests(n_shared, seed=seed)
    unique = gen_unique_requests(n_unique, seed=seed + 1)
    combined = shared + unique
    rng = random.Random(seed + 2)
    rng.shuffle(combined)
    return combined


def gen_long_queue_requests(
    n: int, seed: int = 42
) -> List[List[int]]:
    """Generate a large batch of mixed requests for queue pressure test."""
    return gen_mixed_requests(max(n, 500), shared_ratio=0.3, seed=seed)


# ---------------------------------------------------------------------------
# Scheduling simulation
# ---------------------------------------------------------------------------


def simulate_scheduling(
    policy_name: str,
    requests_tokens: List[List[int]],
    cache_slots: int = 8192,
) -> Tuple[List[MockReq], float, int, int, int]:
    """Simulate the scheduling process:

    1. Pre-populate cache with some requests (first 30%) to build up prefix tree.
    2. Build a waiting queue from the remaining requests.
    3. Run calc_priority() to sort the waiting queue.
    4. Simulate prefill in sorted order: match_prefix → insert.
    5. Return metrics.

    Returns: (sorted_queue, scheduling_latency_sec, total_hit_tokens,
              total_requested_tokens, prefix_computed_count)
    """
    from sglang.srt.managers.schedule_policy import SchedulePolicy

    cache, allocator = make_cache(total_slots=cache_slots)

    # Phase 1: warm up cache with first 30% requests
    warmup_count = max(1, len(requests_tokens) * 30 // 100)
    for token_ids in requests_tokens[:warmup_count]:
        key = RadixKey(token_ids=list(token_ids), extra_key=None)
        match_result = cache.match_prefix(MatchPrefixParams(key=key))
        hit_len = len(match_result.device_indices)
        need = len(token_ids) - hit_len
        if need > 0:
            if allocator.available < need:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                cache.evict(EvictParams(num_tokens=need - allocator.available))
            value = allocator.alloc(need)
            full_value = (
                torch.cat([match_result.device_indices, value])
                if hit_len > 0
                else value
            )
            cache.insert(
                InsertParams(
                    key=RadixKey(token_ids=list(token_ids)), value=full_value
                )
            )

    # Phase 2: build waiting queue from remaining requests
    MockReq._counter = 0
    remaining = requests_tokens[warmup_count:]
    waiting_queue: List[MockReq] = []
    for i, token_ids in enumerate(remaining):
        req = MockReq(token_ids=token_ids, arrival_order=i)
        waiting_queue.append(req)

    # Phase 3: create SchedulePolicy and run calc_priority
    policy = SchedulePolicy(
        policy=policy_name,
        tree_cache=cache,
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
    )

    t0 = time.perf_counter()
    prefix_computed = policy.calc_priority(waiting_queue)
    scheduling_latency = time.perf_counter() - t0

    # Phase 4: simulate prefill in the sorted order
    total_hit_tokens = 0
    total_requested_tokens = 0
    for req in waiting_queue:
        token_ids = req.origin_input_ids
        total_requested_tokens += len(token_ids)
        key = RadixKey(token_ids=list(token_ids), extra_key=None)
        match_result = cache.match_prefix(MatchPrefixParams(key=key))
        hit_len = len(match_result.device_indices)
        total_hit_tokens += hit_len

        # Insert into cache for subsequent requests to benefit
        need = len(token_ids) - hit_len
        if need > 0:
            if allocator.available < need:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                cache.evict(EvictParams(num_tokens=need - allocator.available))
            if allocator.available >= need:
                value = allocator.alloc(need)
                full_value = (
                    torch.cat([match_result.device_indices, value])
                    if hit_len > 0
                    else value
                )
                cache.insert(
                    InsertParams(
                        key=RadixKey(token_ids=list(token_ids)), value=full_value
                    )
                )

    return (
        waiting_queue,
        scheduling_latency,
        total_hit_tokens,
        total_requested_tokens,
        1 if prefix_computed else 0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchMetrics:
    policy: str = ""
    workload: str = ""
    num_requests: int = 0
    total_tokens_requested: int = 0
    total_tokens_hit: int = 0
    scheduling_latency_ms: float = 0.0
    prefix_computed: bool = False
    # First N requests' prefix hit lengths after sorting (to show ordering quality)
    top_hits: List[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.total_tokens_requested == 0:
            return 0.0
        return self.total_tokens_hit / self.total_tokens_requested

    def summary_line(self) -> str:
        prefix_tag = "✓prefix" if self.prefix_computed else " no-pfx"
        return (
            f"[{self.policy:>12s}] {self.workload:<16s} | "
            f"reqs={self.num_requests:4d}  "
            f"hit_rate={self.hit_rate:.2%}  "
            f"sched_lat={self.scheduling_latency_ms:7.2f}ms  "
            f"[{prefix_tag}]"
        )


# ---------------------------------------------------------------------------
# Run a single benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    policy_name: str,
    workload_name: str,
    requests_tokens: List[List[int]],
    cache_slots: int = 8192,
) -> BenchMetrics:
    """Run a single policy × workload combination."""
    (
        sorted_queue,
        sched_lat,
        total_hit,
        total_req,
        pfx_computed,
    ) = simulate_scheduling(policy_name, requests_tokens, cache_slots)

    metrics = BenchMetrics(
        policy=policy_name,
        workload=workload_name,
        num_requests=len(sorted_queue),
        total_tokens_requested=total_req,
        total_tokens_hit=total_hit,
        scheduling_latency_ms=sched_lat * 1000,
        prefix_computed=bool(pfx_computed),
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WORKLOADS = {
    "shared_prefix": lambda n: gen_shared_prefix_requests(n),
    "no_sharing": lambda n: gen_unique_requests(n),
    "mixed": lambda n: gen_mixed_requests(n),
    "long_queue": lambda n: gen_long_queue_requests(n),
}

# Only benchmark policies that SchedulePolicy actually supports.
# "routing-key" requires running_batch context; "priority" is not a real enum value.
POLICIES = ["lpm", "dfs-weight", "fcfs", "lof", "random"]


def main():
    parser = argparse.ArgumentParser(description="Scheduling policy benchmark")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=200,
        help="Number of requests per workload (long_queue always >= 500)",
    )
    parser.add_argument(
        "--cache-slots",
        type=int,
        default=8192,
        help="Total KV cache slots",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICIES,
        help="Scheduling policies to benchmark",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOADS.keys()),
        help="Workload names to run",
    )
    args = parser.parse_args()

    print("=" * 100)
    print(
        f"Scheduling Policy Benchmark  |  "
        f"requests={args.num_requests}  cache_slots={args.cache_slots}"
    )
    print("=" * 100)

    all_metrics: List[BenchMetrics] = []

    for wl_name in args.workloads:
        if wl_name not in WORKLOADS:
            print(f"[WARN] Unknown workload: {wl_name}, skipping.")
            continue
        requests = WORKLOADS[wl_name](args.num_requests)
        for policy in args.policies:
            m = run_benchmark(policy, wl_name, requests, cache_slots=args.cache_slots)
            all_metrics.append(m)
            print(m.summary_line())
        print("-" * 100)

    # ---- Summary: Hit Rate ----
    print("\n" + "=" * 100)
    print("Summary: Cache Hit Rate by (Policy × Workload)")
    print("=" * 100)

    header = f"{'Policy':>12s}"
    for wl_name in args.workloads:
        header += f"  {wl_name:>16s}"
    print(header)

    metrics_map: Dict[Tuple[str, str], BenchMetrics] = {
        (m.policy, m.workload): m for m in all_metrics
    }
    for policy in args.policies:
        row = f"{policy:>12s}"
        for wl_name in args.workloads:
            m = metrics_map.get((policy, wl_name))
            if m:
                row += f"  {m.hit_rate:>15.2%}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    # ---- Summary: Scheduling Latency ----
    print("\n" + "=" * 100)
    print("Summary: Scheduling Latency (ms) by (Policy × Workload)")
    print("=" * 100)

    header = f"{'Policy':>12s}"
    for wl_name in args.workloads:
        header += f"  {wl_name:>16s}"
    print(header)

    for policy in args.policies:
        row = f"{policy:>12s}"
        for wl_name in args.workloads:
            m = metrics_map.get((policy, wl_name))
            if m:
                row += f"  {m.scheduling_latency_ms:>14.2f}ms"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    print("=" * 100)

    # ---- Key Findings ----
    print("\nKey Findings:")

    # Check LPM degradation on long_queue
    lpm_long = metrics_map.get(("lpm", "long_queue"))
    fcfs_long = metrics_map.get(("fcfs", "long_queue"))
    if lpm_long and fcfs_long:
        if not lpm_long.prefix_computed:
            print(
                f"  ⚠ LPM degraded to FCFS on long_queue "
                f"(queue_size > 128): prefix_computed=False"
            )
        else:
            print(f"  ✓ LPM kept prefix computation on long_queue")

    # Find best hit rate per workload
    for wl_name in args.workloads:
        best_policy = None
        best_rate = -1.0
        for policy in args.policies:
            m = metrics_map.get((policy, wl_name))
            if m and m.hit_rate > best_rate:
                best_rate = m.hit_rate
                best_policy = policy
        if best_policy:
            print(
                f"  ★ Best hit rate on {wl_name}: "
                f"{best_policy} ({best_rate:.2%})"
            )


if __name__ == "__main__":
    main()
