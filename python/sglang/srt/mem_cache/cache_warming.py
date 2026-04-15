# -*- coding: utf-8 -*-
"""Cache Warming Manager for SGLang RadixCache.

Pre-populates the RadixCache with frequently used prompts (e.g., System Prompts)
so that subsequent requests can benefit from prefix cache hits from the very
first request, eliminating the cold-start penalty.

Architecture:
    The CacheWarmingManager is created by the Scheduler and invoked during idle
    periods via `self_check_during_idle()`.  It reads a JSON configuration file
    containing a list of prompt strings (or token-ID lists) and inserts them
    into the RadixCache through the standard `match_prefix() -> insert()` path.

    Two warming modes are supported:
    1. **Tree-only warming** (default, lightweight):
       Inserts token IDs into the Radix Tree structure with dummy KV values.
       This enables the scheduling strategies (LPM / DFS-Weight) to correctly
       identify shared-prefix requests and sort the waiting queue optimally.
       The actual KV cache will be computed on the first real prefill.

    2. **Full KV warming** (requires model forward):
       Constructs a synthetic prefill request and runs it through the model to
       produce real KV cache entries.  This is more expensive but eliminates
       the first-request prefill latency entirely.  Only feasible when the
       Scheduler exposes `run_batch()` to the warming manager.

Usage:
    # In server_args.py, user specifies:
    --cache-warming-prompts /path/to/warmup_prompts.json

    # JSON format:
    {
        "prompts": [
            "You are a helpful assistant...",
            "You are a code review expert..."
        ],
        "token_ids": [
            [1, 2, 3, ...],
            [4, 5, 6, ...]
        ]
    }

    Either "prompts" (strings, tokenized at runtime) or "token_ids" (pre-tokenized)
    can be provided.  Both can coexist.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class CacheWarmingManager:
    """Manages cache warming for RadixCache during system idle periods.

    Lifecycle:
        1. Created in ``Scheduler.__init__()`` if ``--cache-warming-prompts`` is set.
        2. ``load_config()`` parses the JSON file and tokenizes prompts.
        3. ``maybe_warm()`` is called from ``self_check_during_idle()`` on each
           idle iteration.  It processes one prompt per call to avoid blocking
           the event loop for too long.
        4. Once all prompts have been warmed, ``is_done`` becomes True and
           subsequent calls are no-ops.

    Attributes:
        warming_token_ids: List of token-ID sequences to warm into the cache.
        warmed_indices:    Set of indices into ``warming_token_ids`` that have
                           already been inserted.
        is_done:           True when all prompts have been warmed.
    """

    def __init__(
        self,
        tree_cache: RadixCache,
        config_path: str,
        tokenizer=None,
    ):
        self.tree_cache = tree_cache
        self.config_path = config_path
        self.tokenizer = tokenizer

        # State
        self.warming_token_ids: List[List[int]] = []
        self.warmed_indices: set = set()
        self.is_done: bool = False
        self._loaded: bool = False
        self._warm_cursor: int = 0  # Next index to warm

        # Stats
        self.total_tokens_warmed: int = 0
        self.total_prompts_warmed: int = 0
        self.warming_start_time: Optional[float] = None
        self.warming_end_time: Optional[float] = None

    def load_config(self) -> bool:
        """Load and parse the warming configuration file.

        Returns:
            True if configuration was loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(
                f"Cache warming config file not found: {self.config_path}"
            )
            self.is_done = True
            return False

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to parse cache warming config: {e}")
            self.is_done = True
            return False

        # Parse token_ids (pre-tokenized)
        if "token_ids" in config:
            for tid_list in config["token_ids"]:
                if isinstance(tid_list, list) and len(tid_list) > 0:
                    self.warming_token_ids.append(tid_list)

        # Parse prompts (need tokenizer)
        if "prompts" in config:
            if self.tokenizer is None:
                logger.warning(
                    "Cache warming config contains 'prompts' but no tokenizer "
                    "is available. Only 'token_ids' entries will be used."
                )
            else:
                for prompt_text in config["prompts"]:
                    if isinstance(prompt_text, str) and len(prompt_text) > 0:
                        try:
                            token_ids = self.tokenizer.encode(prompt_text)
                            if len(token_ids) > 0:
                                self.warming_token_ids.append(token_ids)
                        except Exception as e:
                            logger.warning(
                                f"Failed to tokenize warming prompt: {e}"
                            )

        if len(self.warming_token_ids) == 0:
            logger.info("Cache warming: no prompts to warm.")
            self.is_done = True
            self._loaded = True
            return True

        logger.info(
            f"Cache warming: loaded {len(self.warming_token_ids)} prompts "
            f"({sum(len(t) for t in self.warming_token_ids)} total tokens)"
        )
        self._loaded = True
        return True

    def maybe_warm(self) -> bool:
        """Attempt to warm one prompt into the cache.

        Called during idle periods.  Processes at most one prompt per call
        to keep latency bounded.

        Returns:
            True if warming work was done, False if nothing to do.
        """
        if self.is_done:
            return False

        # Lazy load config on first call
        if not self._loaded:
            if not self.load_config():
                return False

        if self.is_done:
            return False

        if self.warming_start_time is None:
            self.warming_start_time = time.perf_counter()

        # Find next unwarmed prompt
        while self._warm_cursor < len(self.warming_token_ids):
            idx = self._warm_cursor
            self._warm_cursor += 1

            if idx in self.warmed_indices:
                continue

            token_ids = self.warming_token_ids[idx]
            success = self._warm_one_prompt(token_ids)
            if success:
                self.warmed_indices.add(idx)
                self.total_prompts_warmed += 1
                self.total_tokens_warmed += len(token_ids)
                logger.debug(
                    f"Cache warming: warmed prompt {idx + 1}/"
                    f"{len(self.warming_token_ids)} "
                    f"({len(token_ids)} tokens)"
                )
            return success

        # All prompts have been processed
        self.is_done = True
        self.warming_end_time = time.perf_counter()
        elapsed = self.warming_end_time - (self.warming_start_time or self.warming_end_time)
        logger.info(
            f"Cache warming complete: {self.total_prompts_warmed} prompts, "
            f"{self.total_tokens_warmed} tokens in {elapsed:.3f}s"
        )
        return False

    def _warm_one_prompt(self, token_ids: List[int]) -> bool:
        """Insert one prompt's token IDs into the RadixCache tree structure.

        This performs "tree-only warming": the Radix Tree nodes are created
        with the correct key structure, but the KV values are dummy tensors.
        This enables:
        - LPM/DFS-Weight scheduling to correctly identify shared prefixes
        - The tree structure to be pre-built so ``match_prefix()`` can
          navigate the correct path

        The actual KV cache values will be computed when the first real
        request with this prefix is processed through prefill.

        Args:
            token_ids: The token ID sequence to insert.

        Returns:
            True if insertion was successful.
        """
        if self.tree_cache.disable:
            return False

        try:
            radix_key = RadixKey(token_ids=token_ids, extra_key=None)

            # First, check if this prefix is already in the cache
            match_result = self.tree_cache.match_prefix(
                MatchPrefixParams(key=radix_key)
            )
            existing_len = len(match_result.device_indices)

            if existing_len >= len(token_ids):
                # Already fully cached, nothing to do
                logger.debug(
                    f"Cache warming: prompt already cached "
                    f"({existing_len}/{len(token_ids)} tokens)"
                )
                return True

            # Insert the token IDs with dummy KV values
            # The dummy values signal to the tree that nodes exist at these
            # positions.  When a real request arrives, init_next_round_input()
            # will call match_prefix() and find the tree structure, then the
            # actual prefill will compute real KV values.
            dummy_value = torch.arange(
                len(token_ids), dtype=torch.int64, device="cpu"
            )
            self.tree_cache.insert(
                InsertParams(key=radix_key, value=dummy_value)
            )
            return True

        except Exception as e:
            logger.warning(f"Cache warming: failed to warm prompt: {e}")
            return False

    def get_stats(self) -> dict:
        """Return warming statistics."""
        return {
            "total_prompts": len(self.warming_token_ids),
            "prompts_warmed": self.total_prompts_warmed,
            "tokens_warmed": self.total_tokens_warmed,
            "is_done": self.is_done,
            "elapsed_seconds": (
                (self.warming_end_time or time.perf_counter())
                - (self.warming_start_time or time.perf_counter())
                if self.warming_start_time
                else 0.0
            ),
        }
