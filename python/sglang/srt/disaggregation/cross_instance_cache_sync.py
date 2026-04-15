"""
Cross-Instance Cache Synchronization for PD Disaggregation

In PD (Prefill-Decode) disaggregation mode, Prefill and Decode instances run
separately. After Prefill computes KV cache for a request, it transfers the
cache to the Decode instance. However, when multiple requests share common
prefixes (e.g., system prompts), the same KV cache blocks are transferred
repeatedly.

This module implements a lightweight cache-awareness mechanism:
- Prefill side: publishes block hash digests after computing KV cache
- Decode side: subscribes to hash digests and maintains a local hash registry
- On new requests: the Decode side can query the registry to determine which
  prefix blocks already exist, enabling the scheduler to skip redundant
  KV transfers for shared prefixes

Architecture:
    ┌─────────────────┐          ZMQ PUB/SUB         ┌─────────────────┐
    │  Prefill Instance│  ──── CacheSyncEvent ────>  │  Decode Instance │
    │                  │                              │                  │
    │  RadixCache      │                              │  CacheHashRegistry│
    │  _record_store   │                              │   block_hashes   │
    │  _record_remove  │                              │   prefix_chains  │
    └─────────────────┘                              └─────────────────┘

Note: The Decode side uses chunk cache (disable_radix_cache=True) by design,
so we maintain only a hash registry (not a full RadixCache) on the Decode side.
The registry enables prefix-aware transfer optimization without changing the
fundamental cache architecture.
"""

import hashlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import msgspec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Event types for cross-instance sync (separate from KV events)
# ─────────────────────────────────────────────────────────


class CacheSyncEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    """Base class for cross-instance cache sync events."""


class PrefixCacheStored(CacheSyncEvent):
    """Emitted when Prefill instance stores prefix blocks in RadixCache.

    Contains the hash chain for a sequence of blocks that form a prefix.
    The Decode side uses this to register which prefixes are "known" to
    exist on the Prefill side, enabling transfer optimization.
    """

    # Hash of each block in order (position-aware SHA256 chain)
    block_hashes: List[int]
    # Token IDs for each block (for verification)
    token_ids_per_block: List[List[int]]
    # Page size used for the blocks
    page_size: int
    # Request ID that triggered this cache store
    request_id: str
    # Timestamp of the store event
    timestamp: float


class PrefixCacheRemoved(CacheSyncEvent):
    """Emitted when Prefill instance evicts blocks from RadixCache."""

    block_hashes: List[int]
    timestamp: float


class PrefixCacheCleared(CacheSyncEvent):
    """Emitted when Prefill instance clears all cache."""

    timestamp: float


class CacheSyncBatch(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """Batch of sync events with timestamp."""

    ts: float
    events: List[Any]
    source_instance_id: str = ""


# ─────────────────────────────────────────────────────────
# Prefill side: Cache State Publisher
# ─────────────────────────────────────────────────────────


class PrefillCacheStatePublisher:
    """Publishes Prefill-side RadixCache state changes for Decode-side awareness.

    This class works alongside the existing KV event system. While KV events
    (BlockStored/BlockRemoved) are used for metrics, this publisher sends
    higher-level prefix cache state that the Decode side can use to optimize
    KV transfer scheduling.

    Usage in prefill.py's process_batch_result_disagg_prefill:
        After cache_unfinished_req(req), collect the prefix hash chain
        and publish it so Decode instances know what's cached.
    """

    def __init__(
        self,
        instance_id: str = "",
        enabled: bool = True,
    ):
        self.instance_id = instance_id
        self.enabled = enabled
        self._event_queue: List[CacheSyncEvent] = []
        self._lock = threading.Lock()

        # Statistics
        self.total_events_published = 0
        self.total_blocks_published = 0

    def record_prefix_stored(
        self,
        block_hashes: List[int],
        token_ids_per_block: List[List[int]],
        page_size: int,
        request_id: str,
    ) -> None:
        """Record that a prefix has been stored in the Prefill RadixCache.

        Called after RadixCache.cache_unfinished_req() or insert() in
        the prefill processing path.

        Args:
            block_hashes: Position-aware SHA256 hash chain (as int64)
            token_ids_per_block: Token IDs for each block
            page_size: Page size used for blocks
            request_id: The request ID
        """
        if not self.enabled or not block_hashes:
            return

        event = PrefixCacheStored(
            block_hashes=block_hashes,
            token_ids_per_block=token_ids_per_block,
            page_size=page_size,
            request_id=request_id,
            timestamp=time.time(),
        )

        with self._lock:
            self._event_queue.append(event)
            self.total_events_published += 1
            self.total_blocks_published += len(block_hashes)

    def record_prefix_removed(self, block_hashes: List[int]) -> None:
        """Record that blocks have been evicted from Prefill RadixCache."""
        if not self.enabled or not block_hashes:
            return

        event = PrefixCacheRemoved(
            block_hashes=block_hashes,
            timestamp=time.time(),
        )

        with self._lock:
            self._event_queue.append(event)

    def record_cache_cleared(self) -> None:
        """Record that the Prefill RadixCache has been fully cleared."""
        if not self.enabled:
            return

        event = PrefixCacheCleared(timestamp=time.time())

        with self._lock:
            self._event_queue.append(event)

    def take_events(self) -> List[CacheSyncEvent]:
        """Atomically take all pending events (called by scheduler's event publish loop)."""
        with self._lock:
            events = self._event_queue
            self._event_queue = []
        return events

    def get_stats(self) -> Dict[str, int]:
        """Return publisher statistics."""
        return {
            "total_events_published": self.total_events_published,
            "total_blocks_published": self.total_blocks_published,
        }


# ─────────────────────────────────────────────────────────
# Decode side: Cache Hash Registry
# ─────────────────────────────────────────────────────────


@dataclass
class CachedBlockInfo:
    """Information about a cached block known to exist on Prefill side."""

    block_hash: int
    # The token IDs that this block contains
    token_ids: List[int]
    # When this block was first registered
    registered_time: float = 0.0
    # How many times this block has been matched (for stats)
    match_count: int = 0
    # The source Prefill instance
    source_instance: str = ""


class CacheHashRegistry:
    """Maintains a registry of known prefix block hashes from Prefill instances.

    The Decode side uses this lightweight registry to determine which prefix
    blocks are already computed and cached on the Prefill side. When a new
    request arrives at the Decode side, the scheduler can check this registry
    to find how much of the request's prefix is already known, and potentially
    optimize the KV transfer:

    1. Skip transferring blocks that are already in Decode's local cache
       (from a previous request with the same prefix)
    2. Prioritize requests whose prefixes are already cached (better scheduling)
    3. Estimate transfer size before actual transfer begins

    The registry stores only hashes (not actual KV data), keeping it very
    lightweight (~16 bytes per block entry).
    """

    def __init__(
        self,
        max_entries: int = 1_000_000,
        ttl_seconds: float = 3600.0,
        enabled: bool = True,
    ):
        """
        Args:
            max_entries: Maximum number of block hashes to track
            ttl_seconds: Time-to-live for entries (auto-expire old entries)
            enabled: Whether the registry is active
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        # Primary hash set for O(1) lookup
        self._known_hashes: Set[int] = set()

        # Detailed block info (optional, for debugging/stats)
        self._block_info: Dict[int, CachedBlockInfo] = {}

        # Prefix chain tracking: maps (parent_hash) -> set of child hashes
        # This allows reconstructing prefix chains for match_prefix-like queries
        self._prefix_chains: Dict[int, Set[int]] = defaultdict(set)

        # Token-to-hash mapping for prefix matching
        # Maps tuple(token_ids[:page_size]) -> set of possible block hashes
        # Used for fast prefix lookup without full SHA256 computation
        self._token_prefix_index: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)

        # Statistics
        self.total_registered = 0
        self.total_removed = 0
        self.total_lookups = 0
        self.total_hits = 0
        self.total_misses = 0

        self._lock = threading.Lock()

    def register_blocks(
        self,
        block_hashes: List[int],
        token_ids_per_block: List[List[int]],
        source_instance: str = "",
    ) -> int:
        """Register a chain of block hashes from a Prefill instance.

        Args:
            block_hashes: Ordered list of block hashes (position-aware)
            token_ids_per_block: Token IDs for each corresponding block
            source_instance: ID of the Prefill instance

        Returns:
            Number of newly registered blocks (excludes duplicates)
        """
        if not self.enabled or not block_hashes:
            return 0

        now = time.time()
        new_count = 0

        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._known_hashes) + len(block_hashes) > self.max_entries:
                self._evict_oldest_locked(len(block_hashes))

            prev_hash = None
            for bh, token_ids in zip(block_hashes, token_ids_per_block):
                if bh not in self._known_hashes:
                    self._known_hashes.add(bh)
                    self._block_info[bh] = CachedBlockInfo(
                        block_hash=bh,
                        token_ids=token_ids,
                        registered_time=now,
                        source_instance=source_instance,
                    )
                    # Build token prefix index for fast lookup
                    token_key = tuple(token_ids)
                    self._token_prefix_index[token_key].add(bh)
                    new_count += 1

                # Track prefix chains
                if prev_hash is not None:
                    self._prefix_chains[prev_hash].add(bh)
                prev_hash = bh

            self.total_registered += new_count

        return new_count

    def remove_blocks(self, block_hashes: List[int]) -> int:
        """Remove block hashes that have been evicted from Prefill cache.

        Args:
            block_hashes: Block hashes to remove

        Returns:
            Number of blocks actually removed
        """
        if not self.enabled or not block_hashes:
            return 0

        removed_count = 0

        with self._lock:
            for bh in block_hashes:
                if bh in self._known_hashes:
                    self._known_hashes.discard(bh)
                    info = self._block_info.pop(bh, None)
                    if info is not None:
                        token_key = tuple(info.token_ids)
                        hash_set = self._token_prefix_index.get(token_key)
                        if hash_set:
                            hash_set.discard(bh)
                            if not hash_set:
                                del self._token_prefix_index[token_key]
                    # Clean up prefix chains
                    self._prefix_chains.pop(bh, None)
                    removed_count += 1

            self.total_removed += removed_count

        return removed_count

    def clear(self) -> None:
        """Clear all registered hashes (e.g., on Prefill cache clear)."""
        with self._lock:
            self._known_hashes.clear()
            self._block_info.clear()
            self._prefix_chains.clear()
            self._token_prefix_index.clear()

    def contains(self, block_hash: int) -> bool:
        """Check if a block hash is known to exist on Prefill side."""
        self.total_lookups += 1
        if block_hash in self._known_hashes:
            self.total_hits += 1
            return True
        self.total_misses += 1
        return False

    def estimate_cached_prefix_length(
        self,
        token_ids: List[int],
        page_size: int,
    ) -> int:
        """Estimate how many leading tokens of a sequence are already cached.

        This method computes the position-aware hash chain for the given token
        sequence and checks how many consecutive leading blocks are registered
        in the hash registry. This gives an estimate of how much KV cache
        transfer can be skipped for a new request.

        Args:
            token_ids: Full token ID sequence of the new request
            page_size: Page size to use for block boundaries

        Returns:
            Number of leading tokens that are likely cached (page-aligned)
        """
        if not self.enabled or not token_ids:
            return 0

        from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

        cached_len = 0
        parent_hash_str = None

        for start in range(0, len(token_ids), page_size):
            page_tokens = token_ids[start : start + page_size]
            if len(page_tokens) < page_size:
                break  # Partial page at the end

            # Compute position-aware hash (same algorithm as RadixCache)
            hash_str = get_hash_str(page_tokens, prior_hash=parent_hash_str)
            block_hash = hash_str_to_int64(hash_str)

            if self.contains(block_hash):
                cached_len = start + page_size
                parent_hash_str = hash_str
            else:
                break  # Chain broken, stop counting

        return cached_len

    def get_stats(self) -> Dict[str, Any]:
        """Return registry statistics."""
        return {
            "total_registered": self.total_registered,
            "total_removed": self.total_removed,
            "num_known_hashes": len(self._known_hashes),
            "num_prefix_chains": len(self._prefix_chains),
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": (
                self.total_hits / self.total_lookups
                if self.total_lookups > 0
                else 0.0
            ),
        }

    def _evict_oldest_locked(self, num_needed: int) -> None:
        """Evict oldest entries to make room. Must be called with lock held."""
        if not self._block_info:
            return

        # Sort by registration time, evict oldest
        sorted_entries = sorted(
            self._block_info.items(), key=lambda x: x[1].registered_time
        )

        num_to_evict = min(num_needed, len(sorted_entries))
        for i in range(num_to_evict):
            bh = sorted_entries[i][0]
            self._known_hashes.discard(bh)
            info = self._block_info.pop(bh)
            token_key = tuple(info.token_ids)
            hash_set = self._token_prefix_index.get(token_key)
            if hash_set:
                hash_set.discard(bh)
                if not hash_set:
                    del self._token_prefix_index[token_key]
            self._prefix_chains.pop(bh, None)


# ─────────────────────────────────────────────────────────
# Cross-Instance Cache Sync Coordinator
# ─────────────────────────────────────────────────────────


class CrossInstanceCacheSync:
    """Coordinates cache state synchronization between Prefill and Decode instances.

    This is the main entry point for the cross-instance cache sync feature.
    It wraps the Publisher (Prefill side) and Registry (Decode side) and
    provides a unified interface for the scheduler.

    Usage:
        # On Prefill side:
        sync = CrossInstanceCacheSync(mode="prefill", instance_id="prefill-0")
        sync.on_prefix_cached(block_hashes, token_ids_per_block, page_size, req_id)

        # On Decode side:
        sync = CrossInstanceCacheSync(mode="decode", instance_id="decode-0")
        cached_len = sync.estimate_prefix_hit(token_ids, page_size)
    """

    def __init__(
        self,
        mode: str,  # "prefill" or "decode"
        instance_id: str = "",
        max_registry_entries: int = 1_000_000,
        registry_ttl_seconds: float = 3600.0,
        enabled: bool = True,
    ):
        self.mode = mode
        self.instance_id = instance_id
        self.enabled = enabled

        # Prefill side: publisher
        self.publisher: Optional[PrefillCacheStatePublisher] = None
        if mode == "prefill" and enabled:
            self.publisher = PrefillCacheStatePublisher(
                instance_id=instance_id,
                enabled=True,
            )

        # Decode side: registry
        self.registry: Optional[CacheHashRegistry] = None
        if mode == "decode" and enabled:
            self.registry = CacheHashRegistry(
                max_entries=max_registry_entries,
                ttl_seconds=registry_ttl_seconds,
                enabled=True,
            )

        logger.info(
            f"CrossInstanceCacheSync initialized: mode={mode}, "
            f"instance_id={instance_id}, enabled={enabled}"
        )

    # ─── Prefill-side API ───

    def on_prefix_cached(
        self,
        block_hashes: List[int],
        token_ids_per_block: List[List[int]],
        page_size: int,
        request_id: str,
    ) -> None:
        """Called on Prefill side when prefix blocks are stored in RadixCache.

        This should be called in process_batch_result_disagg_prefill after
        cache_unfinished_req, to notify Decode instances about newly cached prefixes.
        """
        if self.publisher is not None:
            self.publisher.record_prefix_stored(
                block_hashes=block_hashes,
                token_ids_per_block=token_ids_per_block,
                page_size=page_size,
                request_id=request_id,
            )

    def on_prefix_removed(self, block_hashes: List[int]) -> None:
        """Called on Prefill side when blocks are evicted."""
        if self.publisher is not None:
            self.publisher.record_prefix_removed(block_hashes)

    def on_cache_cleared(self) -> None:
        """Called on Prefill side when cache is fully cleared."""
        if self.publisher is not None:
            self.publisher.record_cache_cleared()

    def take_sync_events(self) -> List[CacheSyncEvent]:
        """Take pending sync events (Prefill side, for publishing)."""
        if self.publisher is not None:
            return self.publisher.take_events()
        return []

    # ─── Decode-side API ───

    def process_sync_events(self, events: List[CacheSyncEvent]) -> None:
        """Process received sync events on Decode side to update registry."""
        if self.registry is None:
            return

        for event in events:
            if isinstance(event, PrefixCacheStored):
                self.registry.register_blocks(
                    block_hashes=event.block_hashes,
                    token_ids_per_block=event.token_ids_per_block,
                    source_instance=event.request_id,
                )
            elif isinstance(event, PrefixCacheRemoved):
                self.registry.remove_blocks(event.block_hashes)
            elif isinstance(event, PrefixCacheCleared):
                self.registry.clear()

    def estimate_prefix_hit(
        self,
        token_ids: List[int],
        page_size: int,
    ) -> int:
        """Estimate how many leading tokens are cached on Prefill side.

        Called on Decode side when scheduling new requests. The returned
        length indicates how much of the KV transfer could potentially
        be optimized (e.g., transferred from Prefill's RadixCache instead
        of recomputing).

        Args:
            token_ids: Full token ID sequence of the request
            page_size: Current page size

        Returns:
            Number of leading tokens that are likely cached (page-aligned)
        """
        if self.registry is not None:
            return self.registry.estimate_cached_prefix_length(token_ids, page_size)
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats: Dict[str, Any] = {
            "mode": self.mode,
            "instance_id": self.instance_id,
            "enabled": self.enabled,
        }
        if self.publisher is not None:
            stats["publisher"] = self.publisher.get_stats()
        if self.registry is not None:
            stats["registry"] = self.registry.get_stats()
        return stats
