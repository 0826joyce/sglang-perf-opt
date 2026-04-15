"""Suffix Automaton Proposer for N-gram Speculative Decoding.

This module implements a per-request Suffix Automaton (SAM) that complements
SGLang's existing C++ Trie-based N-gram cache. The SAM provides:

1. **Variable-length suffix matching**: Unlike the fixed-window N-gram Trie,
   SAM can match any suffix of the generated context in O(m) time.
2. **Per-request context**: Each request maintains its own automaton, capturing
   local generation patterns that the global Trie might miss.
3. **O(1) amortized incremental updates**: New tokens are added to the SAM
   incrementally as they are generated.

Integration with NGRAMWorker:
    - SAM is queried first for per-request context matching.
    - On miss, falls back to the existing NgramCache (C++ Trie).
    - Results are merged with priority given to longer matches.

Usage:
    Enabled via --speculative-ngram-use-sam flag.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixAutomatonState:
    """A single state in the Suffix Automaton.

    Attributes:
        len: The length of the longest suffix ending at this state.
        link: The suffix link (index into the states array), pointing to the
              state representing the longest proper suffix that is a different
              equivalence class.
        trans: Transition map from token_id to state index.
        cnt: Number of times this state's endpos set has been visited (for
             frequency-based candidate ranking).
    """

    __slots__ = ["len", "link", "trans", "cnt"]

    def __init__(self, length: int = 0, link: int = -1):
        self.len = length
        self.link = link
        self.trans: Dict[int, int] = {}
        self.cnt: int = 0


class IncrementalSuffixAutomaton:
    """O(1) amortized incremental Suffix Automaton.

    Supports online construction: tokens are added one at a time,
    and the automaton always represents all suffixes of the tokens
    seen so far.

    Properties:
        - State count: at most 2n - 1 states for n tokens
        - Transition count: at most 3n - 4 transitions for n tokens
        - extend() is O(1) amortized

    The automaton can be used for:
        - Longest suffix matching: find the longest suffix of a query
          that appears in the constructed text.
        - Candidate collection: from a matched state, follow transitions
          to collect possible next tokens ranked by frequency.
    """

    def __init__(self, max_states: int = 0):
        """Initialize with the initial state (empty string).

        Args:
            max_states: Optional hint for pre-allocation. If 0, grows dynamically.
                       For n tokens, at most 2n states are needed.
        """
        init_state = SuffixAutomatonState(length=0, link=-1)
        self.states: List[SuffixAutomatonState] = [init_state]
        self.last: int = 0  # Index of the state corresponding to the entire string
        self.size: int = 1
        self._token_count: int = 0

    @property
    def token_count(self) -> int:
        """Number of tokens that have been inserted."""
        return self._token_count

    def extend(self, token_id: int) -> None:
        """Insert one token into the automaton. O(1) amortized.

        This extends the automaton so that it represents all suffixes of
        the string constructed so far plus the new token.

        Args:
            token_id: The token ID to append.
        """
        # Check if we can reuse an existing transition (for repeated extend)
        if token_id in self.states[self.last].trans:
            q = self.states[self.last].trans[token_id]
            if self.states[self.last].len + 1 == self.states[q].len:
                # Simple case: the transition is a direct suffix link extension
                self.states[q].cnt += 1
                self.last = q
                self._token_count += 1
                return
            else:
                # Need to clone
                clone = self.size
                cloned_state = SuffixAutomatonState(
                    length=self.states[self.last].len + 1,
                    link=self.states[q].link,
                )
                cloned_state.trans = dict(self.states[q].trans)
                cloned_state.cnt = self.states[q].cnt
                self.states.append(cloned_state)
                self.size += 1

                p = self.last
                while p != -1 and self.states[p].trans.get(token_id) == q:
                    self.states[p].trans[token_id] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[clone].cnt += 1
                self.last = clone
                self._token_count += 1
                return

        # Standard SAM extension
        cur = self.size
        new_state = SuffixAutomatonState(
            length=self.states[self.last].len + 1,
            link=-1,
        )
        new_state.cnt = 1
        self.states.append(new_state)
        self.size += 1

        p = self.last
        while p != -1 and token_id not in self.states[p].trans:
            self.states[p].trans[token_id] = cur
            p = self.states[p].link

        if p == -1:
            # No state has a transition on this token; link to initial state
            self.states[cur].link = 0
        else:
            q = self.states[p].trans[token_id]
            if self.states[p].len + 1 == self.states[q].len:
                # Direct link
                self.states[cur].link = q
            else:
                # Clone state q
                clone = self.size
                cloned_state = SuffixAutomatonState(
                    length=self.states[p].len + 1,
                    link=self.states[q].link,
                )
                cloned_state.trans = dict(self.states[q].trans)
                cloned_state.cnt = self.states[q].cnt
                self.states.append(cloned_state)
                self.size += 1

                while p != -1 and self.states[p].trans.get(token_id) == q:
                    self.states[p].trans[token_id] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self.last = cur
        self._token_count += 1

    def match_suffix(self, query: List[int]) -> Tuple[int, int]:
        """Find the longest suffix of `query` that exists in the automaton.

        Uses the suffix link structure to efficiently find the longest match
        by walking from the end of the query backward.

        Args:
            query: Token ID sequence to match against.

        Returns:
            (state_index, match_length): The state where the longest suffix
            match ends, and the length of that match. If no match, returns (0, 0).
        """
        if not query:
            return 0, 0

        state = 0
        matched = 0

        for token_id in query:
            while state != 0 and token_id not in self.states[state].trans:
                state = self.states[state].link
                matched = self.states[state].len

            if token_id in self.states[state].trans:
                state = self.states[state].trans[token_id]
                matched += 1

        return state, matched

    def collect_candidates(
        self, state: int, max_candidates: int, max_depth: int
    ) -> List[Tuple[List[int], float]]:
        """Collect candidate continuations from a matched state.

        Performs a BFS/DFS from the given state, collecting paths of tokens
        that could follow the matched suffix. Candidates are ranked by
        the frequency count stored in each state.

        Args:
            state: The SAM state to expand from (typically from match_suffix).
            max_candidates: Maximum number of candidate paths to return.
            max_depth: Maximum depth (number of tokens) to explore.

        Returns:
            List of (token_path, score) tuples, sorted by score descending.
            Each token_path is a list of token IDs forming a candidate continuation.
        """
        if state < 0 or state >= self.size:
            return []

        candidates: List[Tuple[List[int], float]] = []
        # BFS with priority based on frequency
        # Each entry: (negative_score, path, current_state)
        queue: List[Tuple[float, List[int], int]] = []

        trans = self.states[state].trans
        for token_id, next_state in trans.items():
            score = self.states[next_state].cnt
            queue.append((-score, [token_id], next_state))

        # Sort by score (highest first, since we negated)
        queue.sort()

        result_paths: List[Tuple[List[int], float]] = []

        while queue and len(result_paths) < max_candidates:
            neg_score, path, cur_state = queue.pop(0)
            score = -neg_score

            result_paths.append((path, score))

            if len(path) < max_depth:
                child_trans = self.states[cur_state].trans
                for token_id, next_state in child_trans.items():
                    child_score = self.states[next_state].cnt
                    new_path = path + [token_id]
                    queue.append((-child_score, new_path, next_state))
                queue.sort()

        return result_paths


class SuffixAutomatonProposer:
    """Per-request Suffix Automaton Proposer for N-gram speculative decoding.

    Manages one SAM per active request. The SAM captures the per-request
    generation pattern and proposes draft tokens based on suffix matching.

    Integration with NGRAMWorker:
        1. Before NgramCache.batch_get(), call propose() for each request.
        2. SAM results are merged into the existing draft token tree structure.
        3. On request completion, cleanup() frees the SAM state.

    The proposer maintains incremental state: each call to propose() only
    extends the SAM with newly generated tokens (not the entire context).
    """

    def __init__(self, draft_token_num: int, max_match_window: int):
        """Initialize the SAM proposer.

        Args:
            draft_token_num: Number of draft tokens to propose (matches
                           NGRAMWorker.draft_token_num).
            max_match_window: Maximum match window size for suffix matching.
        """
        self._automata: Dict[str, IncrementalSuffixAutomaton] = {}
        self._last_positions: Dict[str, int] = {}  # Track incremental position
        self.draft_token_num = draft_token_num
        self.max_match_window = max_match_window

    @property
    def active_count(self) -> int:
        """Number of active per-request automata."""
        return len(self._automata)

    def propose(
        self,
        req_id: str,
        context: List[int],
        draft_token_num: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate draft tokens for a request using suffix automaton matching.

        The SAM is incrementally updated with new tokens since the last call,
        then queried for the longest suffix match of the current context.
        From the matched state, candidate continuations are collected and
        formatted as a draft token tree with attention mask.

        Args:
            req_id: Unique request identifier.
            context: Current context token IDs (tail of origin_input_ids + output_ids).
            draft_token_num: Override for number of draft tokens. If None, uses default.

        Returns:
            (draft_tokens, tree_mask) as numpy arrays matching NgramCache.batch_get()
            format, or None if no useful match was found.
            - draft_tokens: shape (draft_token_num,), int64
            - tree_mask: shape (draft_token_num * draft_token_num,), uint8
        """
        n_drafts = draft_token_num or self.draft_token_num

        # 1. Get or create SAM for this request
        if req_id not in self._automata:
            self._automata[req_id] = IncrementalSuffixAutomaton()
            self._last_positions[req_id] = 0

        sam = self._automata[req_id]
        last_pos = self._last_positions[req_id]

        # 2. Incrementally extend SAM with new tokens only
        # context is the tail of the full sequence; we track how many tokens
        # we've already inserted
        context_len = len(context)
        if context_len > last_pos:
            # Only insert tokens that are new since last propose()
            new_tokens = context[last_pos:]
            for token_id in new_tokens:
                sam.extend(token_id)
            self._last_positions[req_id] = context_len

        # 3. Match: find longest suffix of context that exists in SAM
        # Use the last max_match_window tokens as the query
        query_window = min(self.max_match_window, context_len)
        query = context[-query_window:]
        state, match_len = sam.match_suffix(query)

        if match_len < 2:
            # Too short a match; not useful, fall back to NgramCache
            return None

        # 4. Collect candidate continuations from matched state
        candidates = sam.collect_candidates(
            state,
            max_candidates=n_drafts - 1,  # -1 because first slot is last_token
            max_depth=n_drafts - 1,
        )

        if not candidates:
            return None

        # 5. Build draft token tree in the same format as NgramCache.batch_get()
        # Format: BFS-ordered token array + n×n attention mask
        draft_tokens, tree_mask = self._build_draft_tree(
            last_token=context[-1],
            candidates=candidates,
            n_drafts=n_drafts,
        )

        return draft_tokens, tree_mask

    def _build_draft_tree(
        self,
        last_token: int,
        candidates: List[Tuple[List[int], float]],
        n_drafts: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a draft token tree from SAM candidates.

        Constructs a tree structure in the same format as C++ Ngram::fillResult():
        - draft_tokens: BFS-ordered flat array of shape (n_drafts,)
        - tree_mask: n_drafts × n_drafts attention mask

        The tree has last_token as root (index 0), and candidate paths
        as branches.

        Args:
            last_token: The last generated token (tree root).
            candidates: List of (path, score) from collect_candidates().
            n_drafts: Total number of draft token slots.

        Returns:
            (draft_tokens, tree_mask) as numpy arrays.
        """
        # Build a tree structure: node_id -> {token_id: child_node_id}
        # Node 0 is the root (last_token)
        tree_tokens = [last_token]  # BFS order
        tree_parents = [-1]  # Parent index for each node

        # Track tree structure for deduplication
        # Key: (parent_idx, token_id) -> node_idx
        node_map: Dict[Tuple[int, int], int] = {}

        for path, score in candidates:
            parent_idx = 0  # Start from root
            for token_id in path:
                key = (parent_idx, token_id)
                if key in node_map:
                    parent_idx = node_map[key]
                else:
                    if len(tree_tokens) >= n_drafts:
                        break
                    new_idx = len(tree_tokens)
                    tree_tokens.append(token_id)
                    tree_parents.append(parent_idx)
                    node_map[key] = new_idx
                    parent_idx = new_idx

        # Zero-pad to n_drafts
        while len(tree_tokens) < n_drafts:
            tree_tokens.append(0)
            tree_parents.append(0)

        # Build attention mask: tree_mask[i][j] = 1 iff node j is an ancestor
        # of node i (or j == i)
        n = n_drafts
        mask = np.zeros((n, n), dtype=np.uint8)
        mask[0][0] = 1  # Root attends to itself

        for i in range(1, n):
            if tree_parents[i] >= 0:
                # Copy parent's mask row
                mask[i, : tree_parents[i] + 1] = mask[
                    tree_parents[i], : tree_parents[i] + 1
                ]
            mask[i][i] = 1

        draft_tokens = np.array(tree_tokens, dtype=np.int64)
        tree_mask = mask.flatten()

        return draft_tokens, tree_mask

    def cleanup(self, req_id: str) -> None:
        """Remove the SAM state for a completed request.

        Should be called when a request finishes to free memory.

        Args:
            req_id: The request ID to clean up.
        """
        self._automata.pop(req_id, None)
        self._last_positions.pop(req_id, None)

    def cleanup_all(self) -> None:
        """Remove all SAM states. Called on cache pool clear."""
        self._automata.clear()
        self._last_positions.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the proposer state.

        Returns:
            Dict with keys: active_automata, total_states, total_tokens.
        """
        total_states = sum(sam.size for sam in self._automata.values())
        total_tokens = sum(sam.token_count for sam in self._automata.values())
        return {
            "active_automata": len(self._automata),
            "total_states": total_states,
            "total_tokens": total_tokens,
        }
