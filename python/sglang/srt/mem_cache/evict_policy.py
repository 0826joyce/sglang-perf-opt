from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class AdaptiveStrategy(EvictionStrategy):
    """Multi-factor adaptive eviction strategy.

    Combines three signals into a single priority score:

    1. **Recency** (``last_access_time``): nodes accessed long ago get a
       lower score and are evicted first – similar to LRU.
    2. **Frequency** (``hit_count``): infrequently accessed nodes get a
       lower score – similar to LFU.  The hit count is normalised by the
       maximum observed hit count so that the three factors live on
       comparable scales.
    3. **Tree depth**: nodes deeper in the radix tree (closer to leaves)
       get a lower score.  Shallow nodes are more likely to be shared
       prefixes (e.g. system prompts) whose eviction would hurt many
       future requests.

    The final priority is the weighted sum::

        priority = w_recency  * last_access_time
                 + w_frequency * (hit_count / max_hit_count)
                 + w_depth     * (-depth)

    A *lower* priority means the node will be evicted *sooner* (heap-min
    order used by ``RadixCache.evict``).

    Args:
        w_recency:   Weight for the recency factor.  Default ``0.4``.
        w_frequency: Weight for the frequency factor. Default ``0.3``.
        w_depth:     Weight for the depth factor.     Default ``0.3``.
    """

    def __init__(
        self,
        w_recency: float = 0.4,
        w_frequency: float = 0.3,
        w_depth: float = 0.3,
    ):
        self.w_recency = w_recency
        self.w_frequency = w_frequency
        self.w_depth = w_depth
        # Tracks the running maximum hit_count across all nodes seen so
        # far.  Used to normalise frequency_score into roughly [0, 1].
        self._max_hit_count: int = 1

    def get_priority(self, node: "TreeNode") -> float:
        # --- Factor 1: recency (higher = more recent = less likely to evict) ---
        recency_score = node.last_access_time

        # --- Factor 2: frequency (higher = more hits = less likely to evict) ---
        if node.hit_count > self._max_hit_count:
            self._max_hit_count = node.hit_count
        frequency_score = node.hit_count / self._max_hit_count

        # --- Factor 3: tree depth (deeper = closer to leaf = easier to evict) ---
        depth = self._get_depth(node)
        depth_score = -depth  # negative so deeper nodes have lower priority

        return (
            self.w_recency * recency_score
            + self.w_frequency * frequency_score
            + self.w_depth * depth_score
        )

    @staticmethod
    def _get_depth(node: "TreeNode") -> int:
        """Return the depth of *node* in the radix tree (root = 0)."""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
