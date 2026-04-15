# -*- coding: utf-8 -*-
"""Benchmark: Cache Warming Effectiveness

Measures the impact of cache warming on prefix cache hit rates by comparing
cold-start vs. warm-start scenarios with shared System Prompts.

Usage:
    python benchmark/sglang_optimization/bench_cache_warming.py
    python benchmark/sglang_optimization/bench_cache_warming.py --num-requests 200

This benchmark does NOT require a running SGLang server. It directly
instantiates RadixCache and CacheWarmingManager in simulation mode.

Test scenarios:
    1. cold_start  - No warming, requests arrive at a fresh cache.
    2. warm_start  - Cache pre-populated with System Prompts via CacheWarmingManager.

For each scenario, we measure:
    - First-request cache hit rate (does the very first request benefit?)
    - Overall cache hit rate across all requests.
    - Number of unique tree nodes created.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_PYTHON_DIR = os.path.join(_PROJECT_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)

# ---------------------------------------------------------------------------
# Try importing CacheWarmingManager
# ---------------------------------------------------------------------------
try:
    from sglang.srt.mem_cache.cache_warming import CacheWarmingManager
    HAS_WARMING = True
except ImportError:
    HAS_WARMING = False


# ---------------------------------------------------------------------------
# MockAllocator (reuse pattern from bench_eviction.py)
# ---------------------------------------------------------------------------
class MockAllocator:
    """Minimal mock for token_to_kv_pool_allocator."""

    def __init__(self, total_slots: int):
        self.total_slots = total_slots
        self._used = 0
        self.device = "cpu"
        self.page_size = 1

    def available_size(self) -> int:
        return self.total_slots - self._used

    def alloc(self, n: int):
        import torch
        if self._used + n > self.total_slots:
            return None
        self._used += n
        return torch.arange(self._used - n, self._used, dtype=torch.int64)

    def free(self, indices):
        if indices is not None:
            self._used -= len(indices)
            self._used = max(0, self._used)

    def get_kvcache(self):
        return None


def create_simulated_cache(num_slots: int = 16384) -> RadixCache:
    """Create a simulated RadixCache without GPU resources."""
    return RadixCache.create_simulated(
        disable=False,
        page_size=1,
        total_token_slots=num_slots,
    )


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    list(range(1, 401)),    # 400-token system prompt A
    list(range(401, 801)),  # 400-token system prompt B
    list(range(801, 1201)), # 400-token system prompt C
]


def generate_requests(
    num_requests: int,
    system_prompts: List[List[int]] = None,
    user_token_range: Tuple[int, int] = (50, 200),
    base_offset: int = 5000,
) -> List[List[int]]:
    """Generate requests that share system prompts.

    Each request = system_prompt + random_user_tokens.
    """
    if system_prompts is None:
        system_prompts = SYSTEM_PROMPTS

    requests = []
    for i in range(num_requests):
        sp = system_prompts[i % len(system_prompts)]
        user_len = random.randint(*user_token_range)
        user_tokens = list(range(base_offset + i * 300, base_offset + i * 300 + user_len))
        requests.append(sp + user_tokens)
    return requests


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_requests(
    cache: RadixCache,
    requests: List[List[int]],
) -> Dict[str, float]:
    """Simulate processing requests through the cache.

    For each request:
        1. match_prefix() to find cache hits
        2. insert() to populate the cache (simulating post-prefill)

    Returns metrics dict.
    """
    total_tokens = 0
    total_hits = 0
    first_req_hits = 0
    first_req_total = 0

    for i, token_ids in enumerate(requests):
        radix_key = RadixKey(token_ids=token_ids, extra_key=None)

        # Match prefix
        match_result = cache.match_prefix(
            MatchPrefixParams(key=radix_key)
        )
        hit_len = len(match_result.device_indices)

        total_tokens += len(token_ids)
        total_hits += hit_len

        if i == 0:
            first_req_hits = hit_len
            first_req_total = len(token_ids)

        # Insert (simulate post-prefill cache population)
        import torch
        dummy_value = torch.arange(len(token_ids), dtype=torch.int64)
        cache.insert(InsertParams(key=radix_key, value=dummy_value))

    hit_rate = total_hits / max(total_tokens, 1) * 100
    first_hit_rate = first_req_hits / max(first_req_total, 1) * 100

    return {
        "total_requests": len(requests),
        "total_tokens": total_tokens,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "first_req_hits": first_req_hits,
        "first_req_total": first_req_total,
        "first_hit_rate": first_hit_rate,
    }


def count_tree_nodes(node, depth=0) -> int:
    """Count total nodes in the radix tree."""
    count = 1
    for child in node.children.values():
        count += count_tree_nodes(child, depth + 1)
    return count


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(num_requests: int, num_slots: int):
    """Run the cold-start vs warm-start comparison."""
    print("=" * 100)
    print(f"Cache Warming Benchmark  |  requests={num_requests}  cache_slots={num_slots}")
    print("=" * 100)

    # Generate the same workload for both scenarios
    random.seed(42)
    requests = generate_requests(num_requests, SYSTEM_PROMPTS)

    # ---- Scenario 1: Cold Start ----
    print("\n--- Scenario 1: Cold Start (no warming) ---")
    cache_cold = create_simulated_cache(num_slots)
    t0 = time.perf_counter()
    metrics_cold = simulate_requests(cache_cold, requests)
    t1 = time.perf_counter()
    cold_nodes = count_tree_nodes(cache_cold.root_node)

    print(f"  Hit rate:        {metrics_cold['hit_rate']:>7.2f}%")
    print(f"  First-req hits:  {metrics_cold['first_req_hits']}/{metrics_cold['first_req_total']} "
          f"({metrics_cold['first_hit_rate']:.1f}%)")
    print(f"  Tree nodes:      {cold_nodes}")
    print(f"  Time:            {(t1 - t0) * 1000:.2f}ms")

    # ---- Scenario 2: Warm Start ----
    print("\n--- Scenario 2: Warm Start (system prompts pre-cached) ---")

    if not HAS_WARMING:
        print("  [SKIP] CacheWarmingManager not available")
        return

    cache_warm = create_simulated_cache(num_slots)

    # Create a temporary config file with the system prompts as token_ids
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        config = {"token_ids": SYSTEM_PROMPTS, "prompts": []}
        json.dump(config, f)
        config_path = f.name

    try:
        # Warm the cache
        manager = CacheWarmingManager(
            tree_cache=cache_warm,
            config_path=config_path,
            tokenizer=None,
        )
        manager.load_config()

        warm_t0 = time.perf_counter()
        while not manager.is_done:
            manager.maybe_warm()
        warm_t1 = time.perf_counter()

        warm_nodes_before = count_tree_nodes(cache_warm.root_node)
        print(f"  Warming time:    {(warm_t1 - warm_t0) * 1000:.2f}ms")
        print(f"  Prompts warmed:  {manager.total_prompts_warmed}")
        print(f"  Tokens warmed:   {manager.total_tokens_warmed}")
        print(f"  Tree nodes after warming: {warm_nodes_before}")

        # Now simulate requests
        t0 = time.perf_counter()
        metrics_warm = simulate_requests(cache_warm, requests)
        t1 = time.perf_counter()
        warm_nodes = count_tree_nodes(cache_warm.root_node)

        print(f"  Hit rate:        {metrics_warm['hit_rate']:>7.2f}%")
        print(f"  First-req hits:  {metrics_warm['first_req_hits']}/{metrics_warm['first_req_total']} "
              f"({metrics_warm['first_hit_rate']:.1f}%)")
        print(f"  Tree nodes:      {warm_nodes}")
        print(f"  Time:            {(t1 - t0) * 1000:.2f}ms")
    finally:
        os.unlink(config_path)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("Summary: Cold Start vs Warm Start")
    print("=" * 100)
    print(f"{'Metric':<30} {'Cold Start':>15} {'Warm Start':>15} {'Improvement':>15}")
    print("-" * 75)
    print(f"{'Overall hit rate':<30} {metrics_cold['hit_rate']:>14.2f}% {metrics_warm['hit_rate']:>14.2f}% "
          f"{metrics_warm['hit_rate'] - metrics_cold['hit_rate']:>+14.2f}%")
    print(f"{'First request hit rate':<30} {metrics_cold['first_hit_rate']:>14.1f}% {metrics_warm['first_hit_rate']:>14.1f}% "
          f"{metrics_warm['first_hit_rate'] - metrics_cold['first_hit_rate']:>+14.1f}%")
    print(f"{'Total cache hits (tokens)':<30} {metrics_cold['total_hits']:>15,} {metrics_warm['total_hits']:>15,} "
          f"{metrics_warm['total_hits'] - metrics_cold['total_hits']:>+15,}")

    print("\nKey Findings:")
    if metrics_warm["first_hit_rate"] > metrics_cold["first_hit_rate"]:
        print(f"  * Warm start eliminates cold-start penalty: first request hits "
              f"{metrics_warm['first_req_hits']} tokens (vs {metrics_cold['first_req_hits']} cold)")
    if metrics_warm["hit_rate"] > metrics_cold["hit_rate"]:
        print(f"  * Overall hit rate improved by "
              f"{metrics_warm['hit_rate'] - metrics_cold['hit_rate']:.2f} percentage points")
    else:
        print("  * Warming had minimal impact on overall hit rate (cache fills quickly during simulation)")
    print(f"  * Warming {len(SYSTEM_PROMPTS)} prompts took "
          f"{(warm_t1 - warm_t0) * 1000:.2f}ms (negligible overhead)")


def main():
    parser = argparse.ArgumentParser(description="Cache Warming Benchmark")
    parser.add_argument(
        "--num-requests", type=int, default=100,
        help="Number of requests to simulate (default: 100)",
    )
    parser.add_argument(
        "--cache-slots", type=int, default=16384,
        help="Total cache slots (default: 16384)",
    )
    args = parser.parse_args()

    run_benchmark(args.num_requests, args.cache_slots)


if __name__ == "__main__":
    main()
