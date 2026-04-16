# -*- coding: utf-8 -*-
"""End-to-End Performance Benchmark Framework (Optimization 9)

A unified benchmark framework that covers all optimization points in the SGLang
optimization study. It automates:
  1. Launching SGLang servers with different configurations
  2. Running standardized workloads (single-turn, multi-turn, long-doc, code, mixed)
  3. Collecting metrics (TTFT, ITL, throughput, cache hit rate, spec decode accept rate)
  4. Generating comparison reports

Usage:
    # Quick: compare two configs on the "single_turn" scenario (requires a model)
    python benchmark/sglang_optimization/e2e_benchmark.py \
        --model-path meta-llama/Llama-3.2-1B-Instruct \
        --scenarios single_turn \
        --num-prompts 50

    # Full suite: all scenarios × baseline vs optimized
    python benchmark/sglang_optimization/e2e_benchmark.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --scenarios single_turn multi_turn long_doc code_complete mixed \
        --num-prompts 200

    # Custom config comparison via JSON files
    python benchmark/sglang_optimization/e2e_benchmark.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --config-files baseline.json optimized.json

    # Dry-run: print configs and scenarios without launching servers
    python benchmark/sglang_optimization/e2e_benchmark.py \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --dry-run

Prerequisites:
    - A GPU machine with sufficient VRAM for the target model
    - SGLang installed (pip install sglang[all])
    - The target model downloaded or accessible via HuggingFace

Note: Unlike the simulation-only benchmarks (bench_scheduling.py, bench_eviction.py),
this benchmark REQUIRES a GPU and launches real SGLang servers.
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: ensure the project's python/ dir is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

logger = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class ServerConfig:
    """A named configuration for a SGLang server.

    Each field maps directly to a ``ServerArgs`` attribute.  Only fields that
    differ from defaults need to be specified.
    """

    name: str = "baseline"
    # Scheduling
    schedule_policy: str = "fcfs"
    # Eviction
    radix_eviction_policy: str = "lru"
    # Overlap
    disable_overlap_schedule: bool = False
    enable_dynamic_overlap: bool = False
    # Cache
    enable_cache_report: bool = True
    # Metrics
    enable_metrics: bool = True
    # Speculative decoding (None = disabled)
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_draft_tokens: Optional[int] = None
    # Extra kwargs passed to ServerArgs
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_server_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs suitable for ``ServerArgs(**kwargs)``."""
        kwargs: Dict[str, Any] = {}
        # Direct mappings
        for attr in [
            "schedule_policy",
            "radix_eviction_policy",
            "disable_overlap_schedule",
            "enable_dynamic_overlap",
            "enable_cache_report",
            "enable_metrics",
            "speculative_algorithm",
            "speculative_draft_model_path",
            "speculative_num_draft_tokens",
        ]:
            val = getattr(self, attr)
            if val is not None:
                kwargs[attr] = val
        kwargs.update(self.extra)
        return kwargs


# Built-in config presets
BUILTIN_CONFIGS: Dict[str, ServerConfig] = {
    "baseline": ServerConfig(
        name="baseline",
        schedule_policy="fcfs",
        radix_eviction_policy="lru",
        disable_overlap_schedule=True,  # overlap OFF
        enable_dynamic_overlap=False,
    ),
    "optimized": ServerConfig(
        name="optimized",
        schedule_policy="lpm",
        radix_eviction_policy="lru",
        disable_overlap_schedule=False,  # overlap ON
        enable_dynamic_overlap=True,
    ),
    "lpm_adaptive": ServerConfig(
        name="lpm_adaptive",
        schedule_policy="lpm",
        radix_eviction_policy="adaptive",
        disable_overlap_schedule=False,
        enable_dynamic_overlap=True,
    ),
}


@dataclass
class ScenarioConfig:
    """Definition of a benchmark workload scenario."""

    name: str
    description: str
    # Dataset for bench_serving
    dataset_name: str = "random"
    num_prompts: int = 200
    # Random dataset params
    random_input_len: int = 256
    random_output_len: int = 128
    random_range_ratio: float = 0.5
    # ShareGPT params (when dataset_name == "sharegpt")
    sharegpt_output_len: Optional[int] = None
    # Request rate (requests/s).  None = send all at once (max throughput test)
    request_rate: Optional[float] = None
    # Extra bench_serving args
    extra_bench_args: Dict[str, Any] = field(default_factory=dict)


# Built-in scenario presets
BUILTIN_SCENARIOS: Dict[str, ScenarioConfig] = {
    "single_turn": ScenarioConfig(
        name="single_turn",
        description="Short single-turn conversations (256 input, 128 output)",
        dataset_name="random",
        random_input_len=256,
        random_output_len=128,
        random_range_ratio=0.5,
    ),
    "multi_turn": ScenarioConfig(
        name="multi_turn",
        description="Multi-turn with shared system prompt (prefix cache scenario)",
        dataset_name="generated-shared-prefix",
        extra_bench_args={
            "gsp_num_groups": 32,
            "gsp_prompts_per_group": 8,
            "gsp_system_prompt_len": 1024,
            "gsp_question_len": 128,
            "gsp_output_len": 256,
        },
    ),
    "long_doc": ScenarioConfig(
        name="long_doc",
        description="Long document summarization (2048 input, 512 output)",
        dataset_name="random",
        random_input_len=2048,
        random_output_len=512,
        random_range_ratio=0.25,
    ),
    "code_complete": ScenarioConfig(
        name="code_complete",
        description="Code completion — short input, moderate output",
        dataset_name="random",
        random_input_len=512,
        random_output_len=256,
        random_range_ratio=0.5,
    ),
    "mixed": ScenarioConfig(
        name="mixed",
        description="Mixed workload (random lengths, stress test)",
        dataset_name="random",
        random_input_len=1024,
        random_output_len=512,
        random_range_ratio=1.0,  # full range randomness
    ),
}


@dataclass
class RunResult:
    """Aggregated result of one (config × scenario) run."""

    config_name: str = ""
    scenario_name: str = ""

    # Client-side metrics (from bench_serving)
    num_completed: int = 0
    duration_s: float = 0.0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    total_throughput: float = 0.0

    # Latency (ms)
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0

    # Server-side metrics (from /server_info)
    cache_hit_rate: float = 0.0
    spec_accept_length: float = 0.0
    gen_throughput_server: float = 0.0

    # Raw output for further analysis
    raw_bench_result: Optional[Dict] = None

    def summary_line(self) -> str:
        return (
            f"[{self.config_name:>14s}] {self.scenario_name:<14s} | "
            f"reqs={self.num_completed:4d}  "
            f"out_tput={self.output_throughput:7.1f} tok/s  "
            f"ttft_p50={self.median_ttft_ms:7.1f}ms  "
            f"ttft_p99={self.p99_ttft_ms:7.1f}ms  "
            f"itl_p50={self.median_itl_ms:6.1f}ms  "
            f"cache_hit={self.cache_hit_rate:.2%}"
        )


# ===========================================================================
# Server management
# ===========================================================================


def _find_free_port() -> int:
    """Find an available port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_server(
    model_path: str,
    config: ServerConfig,
    port: int,
    tp_size: int = 1,
    mem_fraction: float = 0.88,
    extra_server_args: Optional[Dict[str, Any]] = None,
) -> multiprocessing.Process:
    """Launch a SGLang server process with the given config.

    Returns the Process object.  The caller is responsible for terminating it.
    """
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server_engine import launch_server_process

    kwargs = {
        "model_path": model_path,
        "port": port,
        "host": "127.0.0.1",
        "tp_size": tp_size,
        "mem_fraction_static": mem_fraction,
        "log_level": "warning",
    }
    kwargs.update(config.to_server_kwargs())
    if extra_server_args:
        kwargs.update(extra_server_args)

    server_args = ServerArgs(**kwargs)
    logger.info(f"Launching server [{config.name}] on port {port} ...")
    proc = launch_server_process(server_args)
    logger.info(f"Server [{config.name}] ready on port {port}")
    return proc


def shutdown_server(proc: multiprocessing.Process) -> None:
    """Gracefully shut down a server process."""
    if proc and proc.is_alive():
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(proc.pid)
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
    logger.info("Server shut down.")


# ===========================================================================
# Benchmark runner
# ===========================================================================


def _build_bench_serving_args(
    base_url: str,
    model_path: str,
    scenario: ScenarioConfig,
    num_prompts: Optional[int] = None,
    request_rate: Optional[float] = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace that mimics the CLI args of bench_serving.py."""
    ns = argparse.Namespace()

    # Backend
    ns.backend = "sglang"
    ns.base_url = base_url
    ns.host = None
    ns.port = None
    ns.model = model_path
    ns.tokenizer = None

    # Dataset
    ns.dataset_name = scenario.dataset_name
    ns.dataset_path = ""
    ns.num_prompts = num_prompts or scenario.num_prompts
    ns.sharegpt_output_len = scenario.sharegpt_output_len
    ns.sharegpt_context_len = None

    # Random dataset
    ns.random_input_len = scenario.random_input_len
    ns.random_output_len = scenario.random_output_len
    ns.random_range_ratio = scenario.random_range_ratio

    # Generated shared prefix (GSP)
    ns.gsp_num_groups = scenario.extra_bench_args.get("gsp_num_groups", 64)
    ns.gsp_prompts_per_group = scenario.extra_bench_args.get("gsp_prompts_per_group", 16)
    ns.gsp_system_prompt_len = scenario.extra_bench_args.get("gsp_system_prompt_len", 2048)
    ns.gsp_question_len = scenario.extra_bench_args.get("gsp_question_len", 128)
    ns.gsp_output_len = scenario.extra_bench_args.get("gsp_output_len", 256)

    # Request rate
    if request_rate is not None:
        ns.request_rate = request_rate
    elif scenario.request_rate is not None:
        ns.request_rate = scenario.request_rate
    else:
        ns.request_rate = float("inf")  # max throughput

    ns.max_concurrency = None
    ns.seed = 42

    # Streaming / misc
    ns.disable_stream = False
    ns.disable_ignore_eos = False
    ns.disable_tqdm = True
    ns.multi_modal_content_format = "openai"
    ns.extra_request_body = "{}"
    ns.apply_chat_template = False
    ns.profile = False
    ns.return_logprob = False
    ns.lora_name = None
    ns.warmup_requests = 3
    ns.result_filename = ""
    ns.tag = ""
    ns.header = []
    ns.show_percentiles = False
    ns.plot_throughput = False
    ns.enable_multi_turn = False
    ns.output_file = None

    # PD separated mode
    ns.pd_separated = False
    # LoRA
    ns.lora_request_distribution = "uniform"
    ns.lora_zipf_alpha = 2.0
    # Output details
    ns.output_details = False
    # Flush cache (we handle this ourselves before each run)
    ns.flush_cache = False
    # Tokenize prompt
    ns.tokenize_prompt = False
    # Trace timestamps
    ns.use_trace_timestamps = False
    ns.mooncake_slowdown_factor = 1.0
    ns.mooncake_num_rounds = 1
    # Served model name
    ns.served_model_name = None
    # Print requests
    ns.print_requests = False

    # Apply scenario-specific extra args
    for k, v in scenario.extra_bench_args.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)

    return ns


def run_scenario(
    base_url: str,
    model_path: str,
    config_name: str,
    scenario: ScenarioConfig,
    num_prompts: Optional[int] = None,
    request_rate: Optional[float] = None,
) -> RunResult:
    """Run a single benchmark scenario against a running server.

    This function wraps SGLang's ``bench_serving.run_benchmark()`` and extracts
    the relevant metrics.
    """
    import requests as http_requests

    from sglang.bench_serving import run_benchmark as _run_benchmark

    result = RunResult(config_name=config_name, scenario_name=scenario.name)

    bench_args = _build_bench_serving_args(
        base_url=base_url,
        model_path=model_path,
        scenario=scenario,
        num_prompts=num_prompts,
        request_rate=request_rate,
    )

    # Flush cache before each run for fair comparison
    try:
        http_requests.post(f"{base_url}/flush_cache", timeout=30)
        time.sleep(2)
    except Exception:
        pass

    # Run benchmark — run_benchmark() returns a dict with all metrics
    print(f"\n{'='*80}")
    print(f"  Running: config={config_name}  scenario={scenario.name}")
    print(f"  {scenario.description}")
    print(f"{'='*80}")

    bench_result: Optional[Dict] = None
    try:
        bench_result = _run_benchmark(bench_args)
    except SystemExit:
        pass
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return result

    # Extract client-side metrics from the returned dict
    if bench_result and isinstance(bench_result, dict):
        result.raw_bench_result = bench_result

        result.num_completed = bench_result.get("completed", 0)
        result.duration_s = bench_result.get("duration", 0.0)
        result.request_throughput = bench_result.get("request_throughput", 0.0)
        result.output_throughput = bench_result.get("output_throughput", 0.0)
        result.total_throughput = bench_result.get("total_throughput", 0.0)

        # TTFT
        result.mean_ttft_ms = bench_result.get("mean_ttft_ms", 0.0)
        result.median_ttft_ms = bench_result.get("median_ttft_ms", 0.0)
        result.p99_ttft_ms = bench_result.get("p99_ttft_ms", 0.0)

        # ITL
        result.mean_itl_ms = bench_result.get("mean_itl_ms", 0.0)
        result.median_itl_ms = bench_result.get("median_itl_ms", 0.0)
        result.p99_itl_ms = bench_result.get("p99_itl_ms", 0.0)

        # E2E latency
        result.mean_e2e_ms = bench_result.get("mean_e2e_latency_ms", 0.0)
        result.median_e2e_ms = bench_result.get("median_e2e_latency_ms", 0.0)
        result.p99_e2e_ms = bench_result.get("p99_e2e_latency_ms", 0.0)

        # Speculative decoding accept length (from bench_serving's server_info)
        result.spec_accept_length = bench_result.get("accept_length", 0.0) or 0.0

        # Server-side gen throughput (from server_info in bench_result)
        server_info = bench_result.get("server_info")
        if server_info and isinstance(server_info, dict):
            if "decode" in server_info:
                server_info = server_info["decode"][0]
            internal = server_info.get("internal_states", [{}])
            if internal and isinstance(internal, list):
                internal = internal[0]
            else:
                internal = {}
            result.cache_hit_rate = internal.get("prefix_cache_hit_rate", 0.0)
            result.gen_throughput_server = internal.get("gen_throughput", 0.0)

    # Fallback: collect server-side metrics directly if not obtained from bench_result
    if result.cache_hit_rate == 0.0:
        try:
            resp = http_requests.get(f"{base_url}/get_server_info", timeout=10)
            if resp.status_code == 200:
                info = resp.json()
                if "decode" in info:
                    info = info["decode"][0]

                internal = info.get("internal_states", [{}])
                if internal and isinstance(internal, list):
                    internal = internal[0]
                else:
                    internal = {}

                result.cache_hit_rate = internal.get("prefix_cache_hit_rate", 0.0)
                if result.spec_accept_length == 0.0:
                    result.spec_accept_length = internal.get(
                        "avg_spec_accept_length", 0.0
                    )
        except Exception as e:
            logger.warning(f"Failed to get server info: {e}")

    return result


# ===========================================================================
# Report generation
# ===========================================================================


def print_comparison_report(results: List[RunResult]) -> None:
    """Print a formatted comparison report across all runs."""
    if not results:
        print("No results to report.")
        return

    # Group by scenario
    scenarios = sorted(set(r.scenario_name for r in results))
    configs = sorted(set(r.config_name for r in results))

    results_map: Dict[Tuple[str, str], RunResult] = {
        (r.config_name, r.scenario_name): r for r in results
    }

    print("\n" + "=" * 120)
    print("  End-to-End Performance Comparison Report")
    print("=" * 120)

    # Per-scenario tables
    for scenario in scenarios:
        print(f"\n{'─'*120}")
        print(f"  Scenario: {scenario}")
        print(f"{'─'*120}")

        header = (
            f"  {'Config':>16s} │ "
            f"{'Reqs':>5s} │ "
            f"{'Out tput':>10s} │ "
            f"{'TTFT p50':>10s} │ "
            f"{'TTFT p99':>10s} │ "
            f"{'ITL p50':>9s} │ "
            f"{'ITL p99':>9s} │ "
            f"{'E2E p50':>9s} │ "
            f"{'Cache Hit':>10s} │ "
            f"{'Spec Len':>9s}"
        )
        print(header)
        print(f"  {'':>16s} │ "
              f"{'':>5s} │ "
              f"{'tok/s':>10s} │ "
              f"{'ms':>10s} │ "
              f"{'ms':>10s} │ "
              f"{'ms':>9s} │ "
              f"{'ms':>9s} │ "
              f"{'ms':>9s} │ "
              f"{'':>10s} │ "
              f"{'':>9s}")
        print(f"  {'─'*16}─┼─{'─'*5}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─"
              f"{'─'*9}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*9}")

        for cfg in configs:
            r = results_map.get((cfg, scenario))
            if r is None:
                continue
            row = (
                f"  {cfg:>16s} │ "
                f"{r.num_completed:>5d} │ "
                f"{r.output_throughput:>10.1f} │ "
                f"{r.median_ttft_ms:>10.1f} │ "
                f"{r.p99_ttft_ms:>10.1f} │ "
                f"{r.median_itl_ms:>9.1f} │ "
                f"{r.p99_itl_ms:>9.1f} │ "
                f"{r.median_e2e_ms:>9.1f} │ "
                f"{r.cache_hit_rate:>9.2%} │ "
                f"{r.spec_accept_length:>9.2f}"
            )
            print(row)

    # Improvement summary (first config = baseline)
    if len(configs) >= 2:
        baseline_name = configs[0]
        print(f"\n{'='*120}")
        print(f"  Improvement vs baseline ({baseline_name})")
        print(f"{'='*120}")

        for scenario in scenarios:
            base = results_map.get((baseline_name, scenario))
            if base is None:
                continue

            for cfg in configs[1:]:
                opt = results_map.get((cfg, scenario))
                if opt is None:
                    continue

                tput_delta = _pct_change(base.output_throughput, opt.output_throughput)
                ttft_delta = _pct_change(base.median_ttft_ms, opt.median_ttft_ms, lower_is_better=True)
                itl_delta = _pct_change(base.median_itl_ms, opt.median_itl_ms, lower_is_better=True)
                cache_delta = opt.cache_hit_rate - base.cache_hit_rate

                print(
                    f"  {scenario:<14s} │ {cfg:>14s} vs {baseline_name} │ "
                    f"tput: {tput_delta:>+7.1f}%  "
                    f"ttft: {ttft_delta:>+7.1f}%  "
                    f"itl: {itl_delta:>+7.1f}%  "
                    f"cache_hit: {cache_delta:>+7.2%}"
                )

    print("\n" + "=" * 120)


def _pct_change(base: float, new: float, lower_is_better: bool = False) -> float:
    """Compute percentage change.  Positive = improvement."""
    if base == 0:
        return 0.0
    delta = (new - base) / abs(base) * 100
    if lower_is_better:
        delta = -delta
    return delta


def save_results_json(results: List[RunResult], output_path: str) -> None:
    """Save results to a JSON file for later analysis."""
    data = []
    for r in results:
        d = dict(asdict(r))
        d.pop("raw_bench_result", None)
        data.append(d)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


# ===========================================================================
# Main orchestrator
# ===========================================================================


class BenchmarkSuite:
    """Orchestrates the full benchmark: server launch → run → collect → report."""

    def __init__(
        self,
        model_path: str,
        configs: List[ServerConfig],
        scenarios: List[ScenarioConfig],
        tp_size: int = 1,
        mem_fraction: float = 0.88,
        num_prompts: Optional[int] = None,
        request_rate: Optional[float] = None,
        output_dir: str = "benchmark_results",
        extra_server_args: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.configs = configs
        self.scenarios = scenarios
        self.tp_size = tp_size
        self.mem_fraction = mem_fraction
        self.num_prompts = num_prompts
        self.request_rate = request_rate
        self.output_dir = output_dir
        self.extra_server_args = extra_server_args or {}

    def run(self) -> List[RunResult]:
        """Run all (config × scenario) combinations and return results."""
        all_results: List[RunResult] = []

        for config in self.configs:
            port = _find_free_port()
            base_url = f"http://127.0.0.1:{port}"
            proc = None

            try:
                # Launch server
                proc = launch_server(
                    model_path=self.model_path,
                    config=config,
                    port=port,
                    tp_size=self.tp_size,
                    mem_fraction=self.mem_fraction,
                    extra_server_args=self.extra_server_args,
                )

                # Run each scenario
                for scenario in self.scenarios:
                    result = run_scenario(
                        base_url=base_url,
                        model_path=self.model_path,
                        config_name=config.name,
                        scenario=scenario,
                        num_prompts=self.num_prompts,
                        request_rate=self.request_rate,
                    )
                    all_results.append(result)
                    print(result.summary_line())

            except Exception as e:
                logger.error(f"Error with config [{config.name}]: {e}")
            finally:
                if proc:
                    shutdown_server(proc)
                    time.sleep(5)  # Allow port to be released

        return all_results

    def run_and_report(self) -> List[RunResult]:
        """Run benchmarks and print comparison report."""
        results = self.run()

        # Report
        print_comparison_report(results)

        # Save
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir, f"e2e_results_{timestamp}.json"
        )
        save_results_json(results, output_path)

        return results


# ===========================================================================
# CLI
# ===========================================================================


def load_configs_from_file(path: str) -> List[ServerConfig]:
    """Load server configs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    configs = []
    if isinstance(data, dict):
        data = [data]
    for item in data:
        name = item.pop("name", f"config_{len(configs)}")
        extra = {}
        known_fields = {f.name for f in ServerConfig.__dataclass_fields__.values()}
        for k in list(item.keys()):
            if k not in known_fields:
                extra[k] = item.pop(k)
        configs.append(ServerConfig(name=name, extra=extra, **item))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="SGLang End-to-End Performance Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism")
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.88,
        help="GPU memory fraction for KV cache",
    )

    # Configs
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "optimized"],
        help=f"Built-in config names: {list(BUILTIN_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--config-files",
        nargs="+",
        default=None,
        help="JSON files defining server configs (overrides --configs)",
    )

    # Scenarios
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["single_turn", "multi_turn"],
        help=f"Built-in scenario names: {list(BUILTIN_SCENARIOS.keys())}",
    )

    # Workload size
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Override num_prompts for all scenarios",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help="Override request rate (req/s) for all scenarios.  Default = inf (max throughput)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for result JSON files",
    )

    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs and scenarios without running",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve configs
    if args.config_files:
        configs: List[ServerConfig] = []
        for path in args.config_files:
            configs.extend(load_configs_from_file(path))
    else:
        configs = []
        for name in args.configs:
            if name in BUILTIN_CONFIGS:
                configs.append(BUILTIN_CONFIGS[name])
            else:
                logger.warning(f"Unknown config '{name}', skipping")

    # Resolve scenarios
    scenarios: List[ScenarioConfig] = []
    for name in args.scenarios:
        if name in BUILTIN_SCENARIOS:
            scenarios.append(BUILTIN_SCENARIOS[name])
        else:
            logger.warning(f"Unknown scenario '{name}', skipping")

    if not configs or not scenarios:
        logger.error("No valid configs or scenarios. Aborting.")
        sys.exit(1)

    # Print plan
    print("=" * 80)
    print("  SGLang E2E Benchmark Suite")
    print("=" * 80)
    print(f"  Model:      {args.model_path}")
    print(f"  TP size:    {args.tp_size}")
    print(f"  Configs:    {[c.name for c in configs]}")
    print(f"  Scenarios:  {[s.name for s in scenarios]}")
    print(f"  Num prompts: {args.num_prompts or 'per-scenario default'}")
    print(f"  Req rate:    {args.request_rate or 'inf (max throughput)'}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 80)

    for cfg in configs:
        print(f"\n  Config [{cfg.name}]:")
        for k, v in cfg.to_server_kwargs().items():
            print(f"    {k}: {v}")

    for sc in scenarios:
        print(f"\n  Scenario [{sc.name}]: {sc.description}")
        print(f"    dataset={sc.dataset_name}  input={sc.random_input_len}  output={sc.random_output_len}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running benchmarks.")
        return

    # Run
    suite = BenchmarkSuite(
        model_path=args.model_path,
        configs=configs,
        scenarios=scenarios,
        tp_size=args.tp_size,
        mem_fraction=args.mem_fraction,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        output_dir=args.output_dir,
    )

    suite.run_and_report()


if __name__ == "__main__":
    main()
