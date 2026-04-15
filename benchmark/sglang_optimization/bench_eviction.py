"""Benchmark: Radix Cache Eviction Strategy Comparison

Compare different eviction strategies (LRU, LFU, Adaptive) on simulated
workloads to measure cache hit rate, eviction counts, and shared-prefix
protection effectiveness.

Usage:
    python benchmark/sglang_optimization/bench_eviction.py

This benchmark does NOT require a running SGLang server. It directly
instantiates RadixCache objects in simulation mode and replays synthetic
request traces.

Workloads:
    1. shared_prefix  – Many requests share a long System Prompt.
    2. no_sharing     – All requests have unique prompts.
    3. mixed          – 50% shared + 50% unique.
    4. bursty_evict   – Rapid arrival that forces frequent eviction.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


# ---------------------------------------------------------------------------
# Helpers
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


def make_cache(policy: str, total_slots: int = 4096) -> Tuple[RadixCache, MockAllocator]:
    """Create a RadixCache in simulation mode with the given eviction policy."""
    allocator = MockAllocator(total_slots)
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy=policy,
    )
    cache = RadixCache(params)
    return cache, allocator


# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TOKENS = list(range(1, 201))  # 200-token shared prefix


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BenchMetrics:
    policy: str = ""
    workload: str = ""
    total_requests: int = 0
    total_tokens_requested: int = 0
    total_tokens_hit: int = 0
    total_tokens_inserted: int = 0
    total_tokens_evicted: int = 0
    eviction_calls: int = 0
    elapsed_sec: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total_tokens_requested == 0:
            return 0.0
        return self.total_tokens_hit / self.total_tokens_requested

    def summary_line(self) -> str:
        return (
            f"[{self.policy:>10s}] {self.workload:<16s} | "
            f"reqs={self.total_requests:4d}  "
            f"hit_rate={self.hit_rate:.2%}  "
            f"hit_tokens={self.total_tokens_hit:6d}/{self.total_tokens_requested:6d}  "
            f"evictions={self.total_tokens_evicted:6d}  "
            f"time={self.elapsed_sec:.3f}s"
        )


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

def run_workload(
    policy: str,
    workload_name: str,
    requests: List[List[int]],
    cache_slots: int = 2048,
) -> BenchMetrics:
    """Replay a list of token-id sequences through a fresh RadixCache and
    measure hit rate.

    For each request we:
      1. ``match_prefix`` – measure how many tokens are already cached.
      2. ``insert`` – insert the full sequence (simulating KV computation).
      3. If OOM, ``evict`` to make room, then retry insert.
    """
    cache, allocator = make_cache(policy, total_slots=cache_slots)
    metrics = BenchMetrics(policy=policy, workload=workload_name)

    t0 = time.perf_counter()

    for token_ids in requests:
        metrics.total_requests += 1
        metrics.total_tokens_requested += len(token_ids)

        # 1) match prefix
        key = RadixKey(token_ids=list(token_ids), extra_key=None)
        match_result = cache.match_prefix(MatchPrefixParams(key=key))
        hit_len = len(match_result.device_indices)
        metrics.total_tokens_hit += hit_len

        # 2) insert (the cache deduplicates automatically)
        need_tokens = len(token_ids) - hit_len
        if need_tokens > 0:
            # Ensure enough room
            free = allocator.available
            if free < need_tokens:
                evict_need = need_tokens - free
                result = cache.evict(EvictParams(num_tokens=evict_need))
                metrics.total_tokens_evicted += result.num_tokens_evicted
                metrics.eviction_calls += 1

            value = allocator.alloc(need_tokens)
            # Pad the value tensor so that it covers the full key length.
            # match_prefix already returned ``hit_len`` cached indices; we
            # only allocated ``need_tokens`` for the *new* portion.  The
            # insert helper walks the tree and skips the already-stored
            # prefix, so we need a value tensor of the same length as the
            # key.  Reuse the matched indices for the prefix part.
            full_value = torch.cat(
                [match_result.device_indices, value]
            ) if hit_len > 0 else value
            cache.insert(
                InsertParams(key=RadixKey(token_ids=list(token_ids)), value=full_value)
            )
            metrics.total_tokens_inserted += need_tokens

    metrics.elapsed_sec = time.perf_counter() - t0
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WORKLOADS = {
    "shared_prefix": lambda n: gen_shared_prefix_requests(n),
    "no_sharing": lambda n: gen_unique_requests(n),
    "mixed": lambda n: gen_mixed_requests(n),
}

POLICIES = ["lru", "lfu", "adaptive"]


def main():
    parser = argparse.ArgumentParser(description="Eviction strategy benchmark")
    parser.add_argument(
        "--num-requests", type=int, default=200, help="Number of requests per workload"
    )
    parser.add_argument(
        "--cache-slots",
        type=int,
        default=2048,
        help="Total KV cache slots (controls eviction pressure)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICIES,
        help="Eviction policies to benchmark",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOADS.keys()),
        help="Workload names to run",
    )
    args = parser.parse_args()

    print("=" * 90)
    print(f"Eviction Strategy Benchmark  |  requests={args.num_requests}  cache_slots={args.cache_slots}")
    print("=" * 90)

    all_metrics: List[BenchMetrics] = []

    for wl_name in args.workloads:
        if wl_name not in WORKLOADS:
            print(f"[WARN] Unknown workload: {wl_name}, skipping.")
            continue
        requests = WORKLOADS[wl_name](args.num_requests)
        for policy in args.policies:
            m = run_workload(policy, wl_name, requests, cache_slots=args.cache_slots)
            all_metrics.append(m)
            print(m.summary_line())
        print("-" * 90)

    # Summary table
    print("\n" + "=" * 90)
    print("Summary: Cache Hit Rate by (Policy × Workload)")
    print("=" * 90)

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

    print("=" * 90)


if __name__ == "__main__":
    main()
