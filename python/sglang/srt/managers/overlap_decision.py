"""Dynamic Overlap Decision Maker for SGLang Scheduler.

This module provides an intelligent overlap decision mechanism that replaces
the current all-or-nothing approach in `is_disable_overlap_for_batch()`.

Instead of a static rule, it uses an Exponential Moving Average (EMA) of
historical GPU forward time and CPU post-processing time to decide whether
overlapping the current batch is beneficial.

Key insight:
  - GPU forward time >> CPU post-processing time → overlap is beneficial
  - GPU forward time ≈ CPU post-processing time → overlap gain is marginal
  - GPU forward time << CPU post-processing time → overlap hurts (CPU stalls next step)

Usage:
  Enable with --enable-dynamic-overlap flag.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class OverlapDecisionMaker:
    """Dynamic overlap decision maker based on runtime GPU/CPU time statistics.

    Tracks the ratio of GPU forward time to CPU post-processing time using
    Exponential Moving Average (EMA). When the GPU/CPU ratio is high enough,
    overlap is enabled to hide CPU processing behind GPU computation. When
    the ratio is low, overlap is disabled to avoid stalling.

    Attributes:
        gpu_time_ema: EMA of GPU forward time in seconds.
        cpu_time_ema: EMA of CPU post-processing time in seconds.
        ema_alpha: Smoothing factor for EMA (higher = more weight on recent).
        ratio_threshold: Minimum GPU/CPU ratio to enable overlap.
        min_batch_size: Minimum batch size to consider overlap worthwhile.
        warmup_steps: Number of initial steps to collect statistics before
            making dynamic decisions (uses heuristic during warmup).
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        ratio_threshold: float = 1.5,
        min_batch_size: int = 4,
        warmup_steps: int = 10,
    ):
        """Initialize the OverlapDecisionMaker.

        Args:
            ema_alpha: EMA smoothing factor in (0, 1]. Default 0.1.
            ratio_threshold: Minimum GPU/CPU time ratio to enable overlap.
                A value of 1.5 means GPU must be at least 1.5x slower than
                CPU for overlap to be worthwhile. Default 1.5.
            min_batch_size: Minimum batch size for overlap during warmup.
                Default 4.
            warmup_steps: Number of initial steps before using dynamic
                decisions. Default 10.
        """
        self.ema_alpha = ema_alpha
        self.ratio_threshold = ratio_threshold
        self.min_batch_size = min_batch_size
        self.warmup_steps = warmup_steps

        # EMA statistics
        self.gpu_time_ema: float = 0.0
        self.cpu_time_ema: float = 0.0

        # Step counter for warmup
        self._step_count: int = 0

        # Timing helpers
        self._gpu_start_time: Optional[float] = None
        self._cpu_start_time: Optional[float] = None

        # Decision history for logging
        self._total_decisions: int = 0
        self._overlap_decisions: int = 0

    def should_overlap(
        self,
        batch: Optional[ScheduleBatch],
        last_batch: Optional[ScheduleBatch],
    ) -> bool:
        """Decide whether to enable overlap for the current batch.

        This method is called in place of the static overlap decision.
        Hard constraints (None batches, spec+grammar incompatibility)
        are still checked first. Then dynamic statistics are used.

        Args:
            batch: The current batch to run. None if no batch.
            last_batch: The previous batch. None if no previous batch.

        Returns:
            True if overlap should be enabled, False otherwise.
        """
        # Hard constraint: need both batches
        if batch is None or last_batch is None:
            return False

        self._total_decisions += 1

        # During warmup, use heuristic based on batch size
        if self._step_count < self.warmup_steps:
            decision = batch.batch_size() >= self.min_batch_size
            if decision:
                self._overlap_decisions += 1
            return decision

        # After warmup: use EMA-based dynamic decision
        if self.gpu_time_ema > 0 and self.cpu_time_ema > 0:
            ratio = self.gpu_time_ema / self.cpu_time_ema
            decision = ratio > self.ratio_threshold
        else:
            # Fallback to heuristic
            decision = batch.batch_size() >= self.min_batch_size

        if decision:
            self._overlap_decisions += 1

        return decision

    def update_stats(self, gpu_time: float, cpu_time: float) -> None:
        """Update the EMA statistics with measured timings.

        Should be called after each batch completes with the measured
        GPU forward time and CPU post-processing time.

        Args:
            gpu_time: GPU forward time in seconds.
            cpu_time: CPU post-processing time in seconds.
        """
        self._step_count += 1
        alpha = self.ema_alpha

        if self._step_count == 1:
            # Initialize EMA with the first observation
            self.gpu_time_ema = gpu_time
            self.cpu_time_ema = cpu_time
        else:
            self.gpu_time_ema = alpha * gpu_time + (1 - alpha) * self.gpu_time_ema
            self.cpu_time_ema = alpha * cpu_time + (1 - alpha) * self.cpu_time_ema

    def mark_gpu_start(self) -> None:
        """Mark the start of GPU forward computation for timing."""
        self._gpu_start_time = time.perf_counter()

    def mark_gpu_end(self) -> float:
        """Mark the end of GPU forward computation.

        Returns:
            Elapsed GPU time in seconds, or 0.0 if start was not marked.
        """
        if self._gpu_start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._gpu_start_time
        self._gpu_start_time = None
        return elapsed

    def mark_cpu_start(self) -> None:
        """Mark the start of CPU post-processing for timing."""
        self._cpu_start_time = time.perf_counter()

    def mark_cpu_end(self) -> float:
        """Mark the end of CPU post-processing.

        Returns:
            Elapsed CPU time in seconds, or 0.0 if start was not marked.
        """
        if self._cpu_start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self._cpu_start_time
        self._cpu_start_time = None
        return elapsed

    @property
    def gpu_cpu_ratio(self) -> float:
        """Current GPU/CPU time ratio from EMA statistics."""
        if self.cpu_time_ema > 0:
            return self.gpu_time_ema / self.cpu_time_ema
        return 0.0

    @property
    def overlap_rate(self) -> float:
        """Fraction of decisions that chose overlap."""
        if self._total_decisions > 0:
            return self._overlap_decisions / self._total_decisions
        return 0.0

    def get_stats_summary(self) -> str:
        """Get a human-readable summary of current statistics.

        Returns:
            A string describing current EMA timings, ratio, and decisions.
        """
        return (
            f"OverlapDecision stats: "
            f"gpu_ema={self.gpu_time_ema * 1000:.2f}ms, "
            f"cpu_ema={self.cpu_time_ema * 1000:.2f}ms, "
            f"ratio={self.gpu_cpu_ratio:.2f}, "
            f"threshold={self.ratio_threshold:.1f}, "
            f"overlap_rate={self.overlap_rate:.1%}, "
            f"steps={self._step_count}"
        )
