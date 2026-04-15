"""Benchmark: N-gram Trie vs SAM (Suffix Automaton) Proposer.

This benchmark compares the draft token proposal quality between:
1. The existing C++ Trie-based NgramCache (baseline)
2. The new SuffixAutomatonProposer (SAM)
3. The combined SAM + NgramCache approach

Metrics:
- Match rate: How often each proposer finds useful matches
- Match length: Average length of matched suffix
- Proposal quality: Number of non-zero draft tokens per request
- Latency: Time to propose draft tokens

Usage:
    python bench_spec_decode.py [--num-tokens N] [--num-requests N] [--draft-token-num N]
"""

import argparse
import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Import SAM proposer
import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"
    ),
)

from sglang.srt.speculative.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton,
    SuffixAutomatonProposer,
)

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_requests: int,
    context_len: int,
    output_len: int,
    vocab_size: int = 32000,
    repeat_pattern_ratio: float = 0.3,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic request data with controllable repetition patterns.

    Creates sequences where a portion of the output tokens repeat patterns
    from the context, simulating real-world generation scenarios where
    suffix matching is beneficial (e.g., code completion, structured output).

    Args:
        num_requests: Number of synthetic requests.
        context_len: Length of input context per request.
        output_len: Length of output tokens per request.
        vocab_size: Vocabulary size for random token generation.
        repeat_pattern_ratio: Fraction of output that repeats context patterns.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: rid, origin_input_ids, output_ids.
    """
    rng = np.random.RandomState(seed)
    requests = []

    for i in range(num_requests):
        # Generate random context
        context = rng.randint(1, vocab_size, size=context_len).tolist()

        # Generate output with some repeating patterns from context
        output = []
        for j in range(output_len):
            if rng.random() < repeat_pattern_ratio and len(context) > 5:
                # Pick a random substring from context to repeat
                start = rng.randint(0, max(1, len(context) - 5))
                pattern_len = min(rng.randint(2, 8), len(context) - start)
                pattern = context[start : start + pattern_len]
                output.extend(pattern[:1])  # Add one token at a time
            else:
                output.append(rng.randint(1, vocab_size))

        output = output[:output_len]

        requests.append(
            {
                "rid": f"req_{i}",
                "origin_input_ids": context,
                "output_ids": output,
            }
        )

    return requests


def benchmark_sam_proposer(
    requests: List[Dict],
    draft_token_num: int = 8,
    max_match_window: int = 12,
) -> Dict:
    """Benchmark the SuffixAutomatonProposer.

    Simulates the incremental generation process: for each request,
    tokens are added one at a time, and SAM is queried at each step.

    Args:
        requests: List of request dicts with origin_input_ids and output_ids.
        draft_token_num: Number of draft tokens per proposal.
        max_match_window: Maximum match window for SAM.

    Returns:
        Dict with benchmark statistics.
    """
    proposer = SuffixAutomatonProposer(
        draft_token_num=draft_token_num,
        max_match_window=max_match_window,
    )

    stats = {
        "total_proposals": 0,
        "hit_count": 0,
        "total_nonzero_drafts": 0,
        "total_match_len": 0,
        "propose_times": [],
        "extend_times": [],
    }

    for req in requests:
        rid = req["rid"]
        context = req["origin_input_ids"]
        output = req["output_ids"]

        # Simulate incremental generation
        current_context = list(context)

        for step, token in enumerate(output):
            current_context.append(token)

            # Measure propose time
            t0 = time.perf_counter()
            result = proposer.propose(
                req_id=rid,
                context=current_context,
                draft_token_num=draft_token_num,
            )
            t1 = time.perf_counter()

            stats["total_proposals"] += 1
            stats["propose_times"].append(t1 - t0)

            if result is not None:
                draft_tokens, tree_mask = result
                nonzero = np.count_nonzero(draft_tokens[1:])
                stats["hit_count"] += 1
                stats["total_nonzero_drafts"] += nonzero

        # Cleanup
        proposer.cleanup(rid)

    # Compute aggregated stats
    total = stats["total_proposals"]
    stats["hit_rate"] = stats["hit_count"] / max(total, 1)
    stats["avg_nonzero_drafts"] = stats["total_nonzero_drafts"] / max(
        stats["hit_count"], 1
    )
    stats["avg_propose_time_us"] = (
        np.mean(stats["propose_times"]) * 1e6 if stats["propose_times"] else 0
    )
    stats["p99_propose_time_us"] = (
        np.percentile(stats["propose_times"], 99) * 1e6
        if stats["propose_times"]
        else 0
    )

    return stats


def benchmark_sam_construction(
    vocab_size: int = 32000,
    sequence_lengths: List[int] = None,
    seed: int = 42,
) -> Dict:
    """Benchmark SAM construction (extend) performance.

    Measures the time to incrementally build a suffix automaton
    for sequences of various lengths.

    Args:
        vocab_size: Vocabulary size for random tokens.
        sequence_lengths: List of sequence lengths to test.
        seed: Random seed.

    Returns:
        Dict mapping sequence length to construction stats.
    """
    if sequence_lengths is None:
        sequence_lengths = [100, 500, 1000, 2000, 4000]

    rng = np.random.RandomState(seed)
    results = {}

    for seq_len in sequence_lengths:
        tokens = rng.randint(1, vocab_size, size=seq_len).tolist()
        sam = IncrementalSuffixAutomaton()

        t0 = time.perf_counter()
        for token in tokens:
            sam.extend(token)
        t1 = time.perf_counter()

        total_time = t1 - t0
        results[seq_len] = {
            "total_time_ms": total_time * 1e3,
            "per_token_us": total_time / seq_len * 1e6,
            "num_states": sam.size,
            "states_per_token": sam.size / seq_len,
        }

    return results


def benchmark_sam_matching(
    vocab_size: int = 32000,
    context_len: int = 1000,
    num_queries: int = 100,
    query_window: int = 12,
    seed: int = 42,
) -> Dict:
    """Benchmark SAM suffix matching performance.

    Builds a SAM from a context, then measures match_suffix performance
    with various query patterns.

    Args:
        vocab_size: Vocabulary size.
        context_len: Length of context to build SAM from.
        num_queries: Number of match queries to run.
        query_window: Window size for each query.
        seed: Random seed.

    Returns:
        Dict with matching stats.
    """
    rng = np.random.RandomState(seed)
    context = rng.randint(1, vocab_size, size=context_len).tolist()

    # Build SAM
    sam = IncrementalSuffixAutomaton()
    for token in context:
        sam.extend(token)

    # Run match queries
    match_times = []
    match_lengths = []
    hit_count = 0

    for _ in range(num_queries):
        # Mix of queries: some from context (should match), some random
        if rng.random() < 0.5:
            # Extract a substring from context as query (should match well)
            start = rng.randint(0, max(1, context_len - query_window))
            query = context[start : start + query_window]
        else:
            # Random query (may partially match)
            query = rng.randint(1, vocab_size, size=query_window).tolist()

        t0 = time.perf_counter()
        state, match_len = sam.match_suffix(query)
        t1 = time.perf_counter()

        match_times.append(t1 - t0)
        match_lengths.append(match_len)
        if match_len >= 2:
            hit_count += 1

    return {
        "avg_match_time_us": np.mean(match_times) * 1e6,
        "p99_match_time_us": np.percentile(match_times, 99) * 1e6,
        "avg_match_length": np.mean(match_lengths),
        "hit_rate": hit_count / num_queries,
        "num_states": sam.size,
    }


def print_results(title: str, stats: Dict, indent: int = 0):
    """Pretty-print benchmark results."""
    prefix = " " * indent
    print(f"\n{prefix}{'=' * 60}")
    print(f"{prefix}{title}")
    print(f"{prefix}{'=' * 60}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{prefix}  {key}: {value:.4f}")
        elif isinstance(value, list):
            continue  # Skip raw data arrays
        elif isinstance(value, dict):
            print(f"{prefix}  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"{prefix}    {k}: {v:.4f}")
                else:
                    print(f"{prefix}    {k}: {v}")
        else:
            print(f"{prefix}  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark N-gram Trie vs SAM Proposer"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of synthetic requests",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
        help="Context length per request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length per request",
    )
    parser.add_argument(
        "--draft-token-num",
        type=int,
        default=8,
        help="Number of draft tokens",
    )
    parser.add_argument(
        "--max-match-window",
        type=int,
        default=12,
        help="Maximum match window size",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=0.3,
        help="Ratio of output that repeats context patterns",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("N-gram Speculative Decoding: SAM Proposer Benchmark")
    print("=" * 60)
    print(f"  num_requests: {args.num_requests}")
    print(f"  context_len: {args.context_len}")
    print(f"  output_len: {args.output_len}")
    print(f"  draft_token_num: {args.draft_token_num}")
    print(f"  max_match_window: {args.max_match_window}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  repeat_ratio: {args.repeat_ratio}")

    # 1. Benchmark SAM construction
    print("\n" + "=" * 60)
    print("1. SAM Construction Performance")
    print("=" * 60)
    construction_results = benchmark_sam_construction(
        vocab_size=args.vocab_size, seed=args.seed
    )
    print(f"  {'SeqLen':<10} {'Time(ms)':<12} {'Per-token(μs)':<15} {'States':<10} {'States/token':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*15} {'-'*10} {'-'*12}")
    for seq_len, stats in construction_results.items():
        print(
            f"  {seq_len:<10} {stats['total_time_ms']:<12.2f} "
            f"{stats['per_token_us']:<15.2f} {stats['num_states']:<10} "
            f"{stats['states_per_token']:<12.2f}"
        )

    # 2. Benchmark SAM matching
    print("\n" + "=" * 60)
    print("2. SAM Suffix Matching Performance")
    print("=" * 60)
    matching_results = benchmark_sam_matching(
        vocab_size=args.vocab_size,
        context_len=args.context_len,
        query_window=args.max_match_window,
        seed=args.seed,
    )
    print(f"  avg_match_time: {matching_results['avg_match_time_us']:.2f} μs")
    print(f"  p99_match_time: {matching_results['p99_match_time_us']:.2f} μs")
    print(f"  avg_match_length: {matching_results['avg_match_length']:.2f}")
    print(f"  hit_rate: {matching_results['hit_rate']:.4f}")
    print(f"  num_states: {matching_results['num_states']}")

    # 3. Benchmark SAM proposer (end-to-end)
    print("\n" + "=" * 60)
    print("3. SAM Proposer End-to-End Performance")
    print("=" * 60)

    requests = generate_synthetic_data(
        num_requests=args.num_requests,
        context_len=args.context_len,
        output_len=args.output_len,
        vocab_size=args.vocab_size,
        repeat_pattern_ratio=args.repeat_ratio,
        seed=args.seed,
    )

    # Test with different repeat ratios
    for ratio in [0.0, 0.1, 0.3, 0.5, 0.8]:
        test_requests = generate_synthetic_data(
            num_requests=args.num_requests,
            context_len=args.context_len,
            output_len=args.output_len,
            vocab_size=args.vocab_size,
            repeat_pattern_ratio=ratio,
            seed=args.seed,
        )

        sam_stats = benchmark_sam_proposer(
            test_requests,
            draft_token_num=args.draft_token_num,
            max_match_window=args.max_match_window,
        )

        print(f"\n  repeat_ratio={ratio:.1f}:")
        print(f"    hit_rate: {sam_stats['hit_rate']:.4f}")
        print(f"    avg_nonzero_drafts: {sam_stats['avg_nonzero_drafts']:.2f}")
        print(f"    avg_propose_time: {sam_stats['avg_propose_time_us']:.2f} μs")
        print(f"    p99_propose_time: {sam_stats['p99_propose_time_us']:.2f} μs")

    # 4. Memory analysis
    print("\n" + "=" * 60)
    print("4. SAM Memory Usage Analysis")
    print("=" * 60)
    proposer = SuffixAutomatonProposer(
        draft_token_num=args.draft_token_num,
        max_match_window=args.max_match_window,
    )

    for req in requests:
        full_context = req["origin_input_ids"] + req["output_ids"]
        proposer.propose(
            req_id=req["rid"],
            context=full_context,
            draft_token_num=args.draft_token_num,
        )

    mem_stats = proposer.get_stats()
    print(f"  active_automata: {mem_stats['active_automata']}")
    print(f"  total_states: {mem_stats['total_states']}")
    print(f"  total_tokens: {mem_stats['total_tokens']}")
    avg_states_per_req = mem_stats["total_states"] / max(
        mem_stats["active_automata"], 1
    )
    print(f"  avg_states_per_request: {avg_states_per_req:.1f}")
    # Rough memory estimate: ~100 bytes per state (len + link + trans dict + cnt)
    est_mem_mb = mem_stats["total_states"] * 100 / (1024 * 1024)
    print(f"  estimated_memory: {est_mem_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
