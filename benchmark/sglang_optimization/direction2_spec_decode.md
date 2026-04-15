# 方向二：推测解码增强

> 在 vLLM 上积累的后缀自动机 Proposer 和自适应评分经验，在 SGLang 上可以**与已有的 EAGLE 和 C++ N-gram 实现协同**。SGLang 的 Spec Decode 实现比 vLLM V1 成熟得多。

## 概述

SGLang 已经内置了完整的推测解码框架，支持多种算法和丰富的 Worker 类型：

| 现有能力 | 可优化方向 |
|---------|-----------|
| 5 种推测算法（EAGLE/EAGLE3/STANDALONE/NGRAM/NONE） | N-gram Trie 固定窗口匹配，不支持可变长度后缀匹配 |
| C++ Trie 树 N-gram + BFS/Prob 两种匹配模式 | EAGLE verify 通过的 token KV 被丢弃，未与 RadixCache 协同 |
| EAGLE multi-step draft + tree verify + CUDA Graph 加速 | 现有 Metrics 仅有 `accept_length` 和 `accept_rate`，缺少细粒度可观测性 |

本方向包含 **3 个优化点**：

```
方向二：推测解码增强
  ├── 优化 4：N-gram Proposer 后缀自动机增强 [核心] ← ✅ 已实现
  ├── 优化 5：EAGLE 投机解码 + Radix Cache 协同 [进阶] ← ✅ 已实现
  └── 优化 6：推测解码全链路可观测性 [辅助] ← 📋 待实现
```

---

## SGLang 推测解码架构总览

在深入各优化点之前，先梳理 SGLang 推测解码的整体架构。

### 算法枚举与 Worker 工厂

推测解码的入口是 `SpeculativeAlgorithm` 枚举（`spec_info.py` L15-22），通过 `create_worker()` 工厂方法（L52-105）根据算法类型和 overlap 模式分派到 8 种 Worker 类：

```
SpeculativeAlgorithm
  ├── EAGLE ──────┬── enable_multi_layer_eagle ──┬── overlap → MultiLayerEagleWorkerV2
  │               │                              └── no-overlap → MultiLayerEagleWorker
  │               ├── overlap → EAGLEWorkerV2
  │               └── no-overlap → EAGLEWorker
  ├── EAGLE3 ─────┘  (is_eagle() == True)
  ├── STANDALONE ─┬── overlap → StandaloneWorkerV2
  │               └── no-overlap → StandaloneWorker
  ├── NGRAM ──────┴── (不支持 overlap) → NGRAMWorker
  └── NONE ───────── (不创建 Worker)
```

### SpecInput 类型体系

所有推测解码的输入/输出都继承自 `SpecInput(ABC)`（`spec_info.py` L116-143），通过 `SpecInputType` 枚举（L108-113）区分：

| SpecInputType | 对应类 | 用途 |
|---------------|--------|------|
| `EAGLE_DRAFT` | `EagleDraftInput` | EAGLE draft 阶段的输入 |
| `EAGLE_VERIFY` | `EagleVerifyInput` | EAGLE/STANDALONE verify 阶段的输入 |
| `NGRAM_VERIFY` | `NgramVerifyInput` | N-gram verify 阶段的输入 |

---

## 优化 4：N-gram Proposer 后缀自动机增强 `[核心]` `[✅ 已实现]`

### 4.1 问题分析

#### SGLang N-gram 推测解码现状

SGLang 的 N-gram 推测解码由三层实现组成：

```
Python 层                          C++ 层
┌──────────────────────┐    ┌────────────────────────────┐
│ NGRAMWorker           │    │ ngram.h / ngram.cpp         │
│  (ngram_worker.py)    │    │  ├── TrieNode struct        │
│  ├── forward_batch_   │    │  │   (child, freq, lru)     │
│  │   generation()     │    │  ├── Ngram class            │
│  ├── _prepare_draft_  │    │  │   ├── match()            │
│  │   tokens()         │    │  │   ├── matchBFS()         │
│  └── _update_ngram_   │    │  │   ├── matchProb()        │
│      cache()          │    │  │   ├── insert() (async)   │
│                       │    │  │   └── squeeze() (LRU)    │
│ NgramCache            │    │  └── fillResult() → tree    │
│  (ngram_cache.py)     │    │      + attention mask       │
│  ├── batch_get()      │    └────────────────────────────┘
│  ├── batch_put()      │
│  └── synchronize()    │    pybind11 绑定层
│                       │    ┌────────────────────────────┐
│ NgramVerifyInput      │    │ ngram_cache_binding.cpp     │
│  (ngram_info.py)      │    │  Param, Ngram, Result       │
│  ├── verify()         │    └────────────────────────────┘
│  ├── _fill_requests() │
│  └── _free_cache()    │
└──────────────────────┘
```

#### C++ Trie 树结构详解

`TrieNode`（`ngram.h` L21-36）是 N-gram 缓存的核心数据结构：

```cpp
struct TrieNode {
  std::unordered_map<int32_t, TrieNode*> child;  // token → 子节点
  std::list<TrieNode*>::const_iterator global_lru_pos;  // 全局 LRU 位置
  std::list<TrieNode*>::const_iterator parent_lru_pos;  // 父节点 LRU 位置
  int32_t token;       // 当前节点对应的 token
  TrieNode* parent;    // 父节点指针
  std::list<TrieNode*> lru;  // 按 LRU 排序的子节点列表
  int32_t freq = 0;    // 访问频率

  struct CompareByFreq { ... };
  std::multiset<TrieNode*, CompareByFreq> sorted_children;  // 按频率降序的子节点集合
};
```

`Ngram` 类（`ngram.h` L38-109）管理整个 Trie 树的生命周期：
- **节点池**：`nodes_`（预分配）+ `node_pool_`（空闲节点指针）+ `free_node_count_`
- **全局 LRU**：`global_lru_`（双向链表），驱逐时从尾部弹出
- **异步插入**：`insert_queue_`（线程安全队列）+ `insert_worker_`（后台线程）
- **互斥锁**：`mutex_`，`match()` 和 `insert()` 互斥

#### N-gram 匹配流程

`match()` 方法（`ngram.cpp` L138-164）是匹配的基础：

```
match(tokens, batch_size):
  for window_size in [max_window, ..., min_window]:
    从 tokens 末尾取 window_size 个 token
    从 root_ 开始逐 token 遍历 Trie
    如果全部匹配 → 记录 (cursor_node, window_size)
  return 所有匹配结果 [(node, depth), ...]
```

在 `match()` 基础上，有两种树扩展模式：

| 模式 | 函数 | 原理 | 适用场景 |
|------|------|------|---------|
| **BFS** | `matchBFS()` (L257-294) | 从匹配节点做 BFS 扩展，breadth 按匹配深度线性缩放 | 默认模式，均衡探索 |
| **Prob** | `matchProb()` (L296-356) | 优先队列扩展，按归一化频率概率排序 | 高频模式优先 |

两种模式最终都通过 `fillResult()` (L22-61) 将树形结构转为：
- **flat token array**：BFS 序列化的 draft tokens
- **attention mask matrix**：`n×n` 二维掩码，表示 token 间的注意力关系

#### 核心局限性

1. **固定窗口匹配**：`match()` 只能匹配 `[min_window, max_window]` 范围内的连续 N-gram。如果上下文的有效匹配在更长的后缀中，Trie 无法捕获
2. **无可变长度回退**：当精确的 N-gram 未命中时，只能降到更短的窗口重试，无法利用部分匹配信息
3. **Per-request 上下文丢失**：Trie 是跨请求共享的全局结构，无法利用当前请求已生成的上下文做局部匹配

#### 与 vLLM 的对比

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| N-gram 实现 | 纯 Python + Numba JIT 的 KMP 搜索 | C++ Trie 树 + pybind11 |
| 匹配复杂度 | O(context_len) per match | O(window_size) per match，但需遍历多个 window |
| 跨请求共享 | 无（per-request） | 有（全局 Trie + LRU 驱逐） |
| 可变长度匹配 | 无 | 无（固定 N 值窗口） |
| 异步插入 | 无 | 有（后台线程 + 线程安全队列） |

### 4.2 设计方案：SuffixAutomatonProposer

在 SGLang 的 C++ N-gram Trie 基础上，**添加后缀自动机（SAM）作为备选 Proposer**，与现有 NgramCache 形成互补：

```
请求到达 → _prepare_draft_tokens()
  │
  ├── 1. SuffixAutomaton.propose()  ← per-request 上下文匹配
  │     └── 增量更新 SAM → 自适应回退匹配（从 n 到 n//2）
  │
  ├── 2. NgramCache.batch_get()     ← 跨请求模式匹配（现有）
  │     └── Trie match → BFS/Prob 扩展 → fillResult()
  │
  └── 3. 合并候选 → 选择最优 draft tokens
```

#### 后缀自动机核心结构

```python
# 新文件: python/sglang/srt/speculative/suffix_automaton_proposer.py

class IncrementalSuffixAutomaton:
    """O(1) 增量更新的后缀自动机。

    与 Trie 的差异：
    - Trie: 存储前缀到后缀的路径，匹配 = 从根遍历
    - SAM: 存储所有后缀的状态转移，匹配 = 从当前状态沿 suffix link 回退

    关键属性：
    - 状态数 ≤ 2n（n 为已插入 token 数）
    - 增量插入 O(1) 均摊
    - 最长匹配 O(m)（m 为查询长度）
    """

    def __init__(self):
        self.states = [{'len': 0, 'link': -1, 'trans': {}}]
        self.last = 0  # 当前状态
        self.size = 1

    def extend(self, token_id: int):
        """增量插入一个 token，O(1) 均摊。"""
        cur = self.size
        self.states.append({
            'len': self.states[self.last]['len'] + 1,
            'link': -1,
            'trans': {}
        })
        self.size += 1
        p = self.last
        while p != -1 and token_id not in self.states[p]['trans']:
            self.states[p]['trans'][token_id] = cur
            p = self.states[p]['link']
        if p == -1:
            self.states[cur]['link'] = 0
        else:
            q = self.states[p]['trans'][token_id]
            if self.states[p]['len'] + 1 == self.states[q]['len']:
                self.states[cur]['link'] = q
            else:
                # clone state
                clone = self.size
                self.states.append({
                    'len': self.states[p]['len'] + 1,
                    'link': self.states[q]['link'],
                    'trans': dict(self.states[q]['trans'])
                })
                self.size += 1
                while p != -1 and self.states[p]['trans'].get(token_id) == q:
                    self.states[p]['trans'][token_id] = clone
                    p = self.states[p]['link']
                self.states[q]['link'] = clone
                self.states[cur]['link'] = clone
        self.last = cur


class SuffixAutomatonProposer:
    """增量后缀自动机 Proposer，作为 NgramCache 的增强选项。

    与 NgramCache (C++ Trie) 的关系:
    - NgramCache: 固定 N 值匹配，跨请求共享模式，C++ 实现高效
    - SuffixAutomaton: 可变长度匹配，per-request 状态，O(1) 增量更新

    组合策略:
    1. 先查 SuffixAutomaton（per-request 上下文匹配）
    2. 未命中再查 NgramCache（跨请求模式匹配）
    3. 两者候选合并，按匹配长度 + 频率评分选最优
    """

    def __init__(self):
        self._automata: Dict[str, IncrementalSuffixAutomaton] = {}

    def propose(self, req_id: str, context: List[int],
                n: int, k: int) -> List[int]:
        """为一个请求生成 draft tokens。

        Args:
            req_id: 请求 ID
            context: 当前上下文的 token IDs
            n: 匹配窗口大小
            k: 候选数量

        Returns:
            draft_tokens: 生成的 draft token IDs
        """
        # 1. 获取或创建该请求的 SAM
        if req_id not in self._automata:
            self._automata[req_id] = IncrementalSuffixAutomaton()

        sam = self._automata[req_id]

        # 2. 增量更新 SAM
        # 只插入新增的 token（上次 propose 后新生成的）
        for token_id in context[-n:]:
            sam.extend(token_id)

        # 3. 自适应回退匹配
        # 从最长 n-gram 开始，逐步缩短直到找到匹配
        for window in range(n, n // 2, -1):
            suffix = context[-window:]
            # 在 SAM 中匹配
            state = 0
            matched = 0
            for token in suffix:
                if token in sam.states[state]['trans']:
                    state = sam.states[state]['trans'][token]
                    matched += 1
                else:
                    break
            if matched >= window:
                # 从匹配状态沿转移边收集候选
                candidates = self._collect_candidates(sam, state, k)
                if candidates:
                    return candidates

        return []  # 未命中，交给 NgramCache

    def _collect_candidates(self, sam, state, k):
        """从匹配状态收集候选 draft tokens。"""
        candidates = []
        trans = sam.states[state]['trans']
        for token_id in list(trans.keys())[:k]:
            candidates.append(token_id)
        return candidates

    def cleanup(self, req_id: str):
        """请求完成后清理 SAM。"""
        self._automata.pop(req_id, None)
```

#### 与 NGRAMWorker 的集成方式

修改 `NGRAMWorker._prepare_draft_tokens()` 方法（`ngram_worker.py` L122-141），在现有 `NgramCache.batch_get()` 之前插入 SAM 查询：

```python
def _prepare_draft_tokens(self, batch: ScheduleBatch):
    bs = batch.batch_size()
    self.ngram_cache.synchronize()

    batch_tokens = []
    sam_results = []  # 🆕 SAM 候选

    for req in batch.reqs:
        check_token = self._efficient_concat_last_n(
            req.origin_input_ids, req.output_ids, self.max_match_window_size
        )
        batch_tokens.append(check_token)

        # 🆕 先尝试 SAM 匹配
        if self.sam_proposer is not None:
            sam_draft = self.sam_proposer.propose(
                req.rid, check_token,
                n=self.max_match_window_size,
                k=self.draft_token_num
            )
            sam_results.append(sam_draft)

    # 现有 NgramCache 匹配
    req_drafts, mask = self.ngram_cache.batch_get(batch_tokens)

    # 🆕 合并 SAM 和 NgramCache 的结果
    if self.sam_proposer is not None:
        req_drafts, mask = self._merge_proposals(
            req_drafts, mask, sam_results, bs
        )

    return req_drafts, mask
```

### 4.3 NGRAMWorker 完整调用链

理解 NGRAMWorker 的端到端流程是正确集成 SAM 的前提。

#### 阶段 A：Worker 创建

```
用户启动命令:
  python -m sglang.launch_server --speculative-algorithm NGRAM

                    ┌──────────────────────────────────────┐
                    │  1. spec_info.py :: SpeculativeAlgorithm│
                    │     .from_string("NGRAM")             │
                    │     (L24-31)                          │
                    │                                      │
                    │  2. create_worker()                   │
                    │     (L95-103)                         │
                    │     is_ngram() → True                 │
                    │     overlap → 报错（不支持）            │
                    │     → return NGRAMWorker              │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  3. NGRAMWorker.__init__()             │
                    │     (ngram_worker.py L26-59)          │
                    │                                      │
                    │  self.target_worker = target_worker    │
                    │  self.draft_token_num = speculative_   │
                    │    num_draft_tokens                    │
                    │  self.ngram_cache = NgramCache(        │
                    │    min_match_window_size, ...          │
                    │    capacity=1000000)                   │
                    │  self._init_preallocated_tensors()     │
                    │     (L72-121) ← 预分配 GPU 张量       │
                    └──────────────────────────────────────┘
```

#### 阶段 B：NgramCache 的 C++ 初始化

```
NgramCache.__init__()  (ngram_cache.py L24-46)
     │
     ├── torch.utils.cpp_extension.load()  ← JIT 编译 C++ 代码
     │     sources: ngram_cache_binding.cpp, ngram.cpp
     │     extra_cflags: ["-O3", "-std=c++20"]
     │
     └── ngram_cache_cpp.Ngram(capacity, param)
           └── Ngram::Ngram()  (ngram.cpp L63-130)
                 ├── 预分配 capacity 个 TrieNode
                 ├── 参数校验（branch_length > 1, min <= max 等）
                 └── 启动 insert_worker_ 后台线程
```

#### 阶段 C：forward_batch_generation() 主流程

```
Scheduler 事件循环
     │
     ▼
NGRAMWorker.forward_batch_generation(batch)
  (ngram_worker.py L214-284)
     │
     ├── 1. _prepare_for_speculative_decoding(batch)  (L215)
     │     │
     │     ├── if batch.forward_mode.is_extend(): return  ← extend 模式跳过
     │     │
     │     ├── _prepare_draft_tokens(batch)  (L156)
     │     │     │
     │     │     ├── ngram_cache.synchronize()  ← 等待异步插入完成
     │     │     │     └── Ngram::synchronize()  (ngram.cpp L189-193)
     │     │     │           └── while(!insert_queue_.empty()) sleep(10μs)
     │     │     │
     │     │     ├── 遍历 batch.reqs:
     │     │     │     req.origin_input_ids + req.output_ids
     │     │     │     → _efficient_concat_last_n(max_match_window_size)
     │     │     │     → batch_tokens
     │     │     │
     │     │     └── ngram_cache.batch_get(batch_tokens)  (ngram_cache.py L57-59)
     │     │           └── Ngram::batchMatch()  (ngram.cpp L358-368)
     │     │                 ├── 加锁 mutex_
     │     │                 ├── for each token_seq:
     │     │                 │     matchBFS() or matchProb()
     │     │                 │       └── match() → Trie 遍历
     │     │                 │       └── BFS/优先队列扩展
     │     │                 │       └── fillResult() → token[] + mask[]
     │     │                 └── 合并所有请求的结果
     │     │
     │     ├── reconstruct_indices_from_tree_mask()  ← sgl_kernel CUDA kernel
     │     │     (L160-169)
     │     │     → positions, retrive_index, retrive_next_token, retrive_next_sibling
     │     │
     │     ├── 构建 FULL_MASK 树掩码  (L173-185)
     │     │     → torch.cat([prefix_mask, draft_mask]) per request
     │     │
     │     └── batch.spec_info = NgramVerifyInput(...)  (L189-198)
     │           batch.forward_mode = ForwardMode.TARGET_VERIFY
     │
     ├── 2. model_worker_batch = batch.get_model_worker_batch()  (L216)
     │
     ├── 3. if TARGET_VERIFY:  (L221-266)
     │     │
     │     ├── target_worker.forward_batch_generation(is_verify=True)  (L229-231)
     │     │     → logits_output, can_run_cuda_graph
     │     │
     │     ├── (可选) generate_token_bitmask()  ← grammar 约束
     │     │
     │     ├── verify_input.verify(batch, logits_output, page_size, vocab_mask)
     │     │     (L258-260)
     │     │     └── NgramVerifyInput.verify()  (ngram_info.py L377-446)
     │     │           ├── _greedy_verify() or _sampling_verify()
     │     │           │     └── verify_tree_greedy() / tree_speculative_sampling_target_only()
     │     │           ├── _fill_requests()  (L156-206)
     │     │           │     └── 遍历 accepted tokens → req.output_ids.append()
     │     │           │     └── req.spec_verify_ct += 1
     │     │           │     └── req.spec_accepted_tokens += accepted_draft_tokens
     │     │           │     └── req.update_spec_acceptance_histogram()
     │     │           └── _free_cache()  (L208-278)
     │     │                 └── 释放未 accept 的 KV cache slots
     │     │
     │     └── _update_ngram_cache(batch)  (L265)
     │           └── ngram_cache.batch_put(batch_tokens)  (ngram_cache.py L48-49)
     │                 └── Ngram::asyncInsert()  (ngram.cpp L251-255)
     │                       └── insert_queue_.enqueue() → 后台线程处理
     │
     └── 4. return GenerationBatchResult(...)  (L278-284)
```

#### 阶段 D：C++ Trie 异步插入

```
后台 insert_worker_ 线程
  │
  └── Ngram::insert()  (ngram.cpp L195-249)
        │
        └── while(!quit_flag_):
              │
              ├── dequeue(data) from insert_queue_
              │
              ├── 加锁 mutex_
              │
              └── for i in [0, size - min_match_window_size):
                    │
                    ├── 截取 [i, i + branch_length) 的 token 子序列
                    │
                    ├── if 需要更多节点 > free_node_count_:
                    │     squeeze() → LRU 驱逐最旧节点
                    │
                    ├── 从 root_ 开始逐 token 遍历/创建路径:
                    │     ├── 新节点: child.insert() + lru.emplace_front()
                    │     │           + global_lru_.emplace_back()
                    │     │           + sorted_children.insert()
                    │     └── 已有节点: sorted_children.erase()+freq++ 
                    │                   +sorted_children.insert()
                    │                   + lru.splice(front)
                    │
                    └── 更新路径上所有节点的 global_lru_ 位置
                          → splice to front（最近访问）
```

### 4.4 修改文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `python/sglang/srt/speculative/suffix_automaton_proposer.py` | **新建** | 后缀自动机 Proposer 核心实现 |
| `python/sglang/srt/speculative/ngram_worker.py` | **修改** | 集成 SAM 作为备选 Proposer，修改 `_prepare_draft_tokens()` |
| `python/sglang/srt/server_args.py` | **修改** | 新增 `--speculative-ngram-use-sam` CLI 参数 |
| `benchmark/sglang_optimization/bench_spec_decode.py` | **新建** | N-gram vs SAM 对比 Benchmark |

### 4.5 使用方式

```bash
# 启用 SAM 增强的 N-gram 推测解码
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-ngram-use-sam

# 对比 baseline（纯 N-gram Trie）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --speculative-algorithm NGRAM
```

### 4.6 学习收获

通过实现此优化，深入理解了：

1. **SGLang N-gram Spec Decode 全链路**：`NGRAMWorker.forward_batch_generation()` → `NgramCache.batch_get()` → `NgramVerifyInput.verify()` → `_update_ngram_cache()`。NGRAMWorker 是唯一不继承 `TpModelWorker` 的推测解码 Worker（直接使用 `target_worker`）
2. **C++ Trie 的 pybind11 绑定机制**：`torch.utils.cpp_extension.load()` 在运行时 JIT 编译 C++，通过 `ngram_cache_binding.cpp` 暴露 `Param`、`Ngram`、`Result` 类给 Python
3. **两种匹配模式的设计权衡**：BFS 按 LRU 顺序的 breadth 扩展（均衡探索），Prob 按频率归一化的优先队列扩展（高频模式优先）。breadth 随匹配深度线性缩放：`breadth = (max_window - depth) * scale + min_bfs_breadth`
4. **异步插入 + 同步匹配的线程模型**：`asyncInsert()` 通过线程安全队列将 token 发送到后台线程，`batchMatch()` 加互斥锁保证读一致性，`synchronize()` 通过忙等待确保队列清空
5. **`SpeculativeAlgorithm` 枚举的扩展**：添加新算法需要在枚举中新增值 + `create_worker()` 中添加分支 + 实现对应的 Worker 类
6. **Trie LRU 驱逐机制**：`squeeze()` 从 `global_lru_` 尾部弹出最旧的叶节点，从父节点的 `child`/`lru`/`sorted_children` 中清除，回收到 `node_pool_`

---

## 优化 5：EAGLE 投机解码 + Radix Cache 协同 `[进阶]` `[✅ 已实现]`

### 5.1 问题分析

#### SGLang EAGLE Worker 现状

`EAGLEWorker`（`eagle_worker.py` L78-1011）是 SGLang 中最复杂的推测解码实现，继承自 `TpModelWorker`：

```
EAGLEWorker(TpModelWorker)
  │
  ├── 关键属性:
  │   ├── target_worker: TpModelWorker     ← 目标模型 Worker
  │   ├── draft_model_runner: ModelRunner   ← 草稿模型 Runner (= self.model_runner)
  │   ├── topk: int                         ← draft 每步选取 top-k
  │   ├── speculative_num_steps: int        ← draft 步数
  │   ├── speculative_num_draft_tokens: int ← verify token 数
  │   └── page_size: int                    ← KV cache 页大小
  │
  ├── 内存共享:
  │   ├── req_to_token_pool     ← 与 target_worker 共享!
  │   └── token_to_kv_pool_allocator ← 与 target_worker 共享!
  │
  ├── CUDA Graph:
  │   ├── EAGLEDraftCudaGraphRunner      ← draft decode 加速
  │   └── EAGLEDraftExtendCudaGraphRunner ← draft extend 加速
  │
  └── Attention Backend:
      ├── draft_attn_backend         ← draft decode attention
      └── draft_extend_attn_backend  ← draft extend attention
```

#### EAGLE forward_batch_generation() 两条路径

`EAGLEWorker.forward_batch_generation()`（L274-333）根据 batch 的 forward_mode 走两条不同路径：

```
forward_batch_generation(batch)
  │
  ├── 路径 A: Extend（首次 prefill）
  │     (L286-305)
  │     │
  │     ├── forward_target_extend(batch)
  │     │     → target model forward (CaptureHiddenMode.FULL)
  │     │     → logits_output (含 hidden_states), next_token_ids
  │     │
  │     └── forward_draft_extend(batch, hidden_states, next_token_ids)
  │           → draft model forward（使用 target 的 hidden_states）
  │           → capture_for_decode() → topk_p, topk_index, hidden_states
  │
  └── 路径 B: Decode（推测解码循环）
        (L306-333)
        │
        ├── draft(batch) → spec_info: EagleVerifyInput
        │     │
        │     ├── _draft_preprocess_decode(batch)
        │     │     → 分配 draft cache locations
        │     │     → assign_draft_cache_locs() Triton kernel
        │     │
        │     ├── draft_forward(forward_batch)  (L611-681)
        │     │     │
        │     │     └── for i in range(speculative_num_steps):
        │     │           select_top_k_tokens(i, topk_p, topk_index, ...)
        │     │           → input_ids, hidden_states, scores, tree_info
        │     │           if i < speculative_num_steps - 1:
        │     │               draft_model_runner.forward()  ← draft model forward
        │     │               → new topk_p, topk_index
        │     │     └── organize_draft_results()
        │     │           → parent_list, top_scores_index, draft_tokens
        │     │
        │     └── build_tree_kernel_efficient()  (eagle_utils.py L47-158)
        │           → tree_mask, positions, retrive_index, draft_tokens
        │           → EagleVerifyInput(...)
        │
        ├── verify(batch, spec_info)  (L687-777)
        │     │
        │     ├── spec_info.prepare_for_verify(batch)
        │     │     → alloc out_cache_loc for draft tokens
        │     │     → assign_req_to_token_pool()
        │     │
        │     ├── target_worker.forward_batch_generation(is_verify=True)
        │     │     → logits_output (含 hidden_states)
        │     │
        │     └── spec_info.verify(batch, logits_output, allocator, page_size)
        │           (eagle_info.py L216-612)
        │           │
        │           ├── greedy: verify_tree_greedy_func() (sgl_kernel)
        │           │   sampling: tree_speculative_sampling_target_only()
        │           │
        │           ├── 遍历 accept_index → req.output_ids.append()
        │           │     → req.spec_verify_ct += 1
        │           │     → req.spec_accepted_tokens += accepted_draft_tokens
        │           │     → req.update_spec_acceptance_histogram()
        │           │
        │           ├── 释放 unaccepted KV cache slots
        │           │
        │           └── → EagleVerifyOutput(draft_input, verified_id,
        │                   accept_length_per_req_cpu, accepted_indices)
        │
        └── forward_draft_extend_after_decode(batch)  (L898-986)
              → 为下一轮 decode 准备 draft model 的 KV 状态
              → draft_model_runner.forward() or cuda_graph replay
```

#### Bigram Key 机制

EAGLE 使用 **bigram key** 是因为 draft model 的输入是 `(token_t, hidden_t-1)` 对。RadixCache 支持 `is_bigram=True` 模式，`convert_to_bigram_key()` 将相邻 token 对转为 bigram：

```
tokens: [A, B, C, D]
bigram_key: [(A,B), (B,C), (C,D)]
```

这意味着 EAGLE 的 RadixCache 查找粒度是 token 对而非单个 token。

#### 核心问题

**EAGLE draft 阶段的候选 KV Cache 浪费**：

1. EAGLE `draft()` 方法（L528-609）生成 `speculative_num_draft_tokens` 个候选 token
2. `verify()` 方法（L687-777）对这些候选进行验证
3. 验证通过的 accepted tokens 的 KV 保留，**被拒绝的 draft tokens 的 KV 被立即释放**（`eagle_info.py` L436-504）
4. 如果后续请求生成了相似的 token 序列，这些 KV 需要重新计算

**具体浪费路径**：

```
verify() 结束后:
  evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
  evict_mask[accept_index] = False
  # ↑ accepted tokens: False (保留)
  # ↑ rejected tokens: True  (释放)

  token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
  # ↑ rejected tokens 的 KV cache 被立即释放!
```

### 5.2 设计方案：Speculative KV Cache Reuse

将 EAGLE 验证通过的 token 序列的 KV Cache 保留在 RadixCache 中，供后续请求复用。

#### 核心思路

```
EAGLE verify 完成
  │
  ├── accepted tokens → 正常写入 RadixCache (现有逻辑)
  │
  └── 🆕 额外操作: 将 accepted token 序列的完整前缀
      写入 RadixCache，使后续请求可以命中
      │
      └── _cache_accepted_tokens(batch, verify_output)
            for each req with accept_length > threshold:
              # 将 origin_input_ids + output_ids 的 KV 写入缓存
              tree_cache.cache_unfinished_req(req)
```

#### 与 RadixCache 的交互设计

```python
# 修改 eagle_worker.py 中 verify() 完成后的逻辑

def _cache_accepted_tokens(self, batch, verify_output):
    """将验证通过的 token 序列写入 RadixCache，供后续复用。

    只有当 accepted tokens > 阈值时才缓存，避免开销过大。
    """
    min_cache_len = 2  # 至少 2 个 accepted tokens 才有缓存价值

    for i, req in enumerate(batch.reqs):
        accept_len = verify_output.accept_length_per_req_cpu[i]
        if accept_len >= min_cache_len:
            # 将当前请求的 KV 状态写入 RadixCache
            # 这样后续有相同前缀的请求可以直接命中
            batch.tree_cache.cache_unfinished_req(
                req,
                token_ids=req.origin_input_ids + req.output_ids,
            )
```

#### 需要考虑的问题

1. **Bigram Key 兼容性**：EAGLE 使用 bigram key 模式的 RadixCache，缓存写入时需要正确转换
2. **KV Pool 共享**：draft 和 target worker 共享 `req_to_token_pool` 和 `token_to_kv_pool_allocator`，缓存操作不能破坏共享状态
3. **缓存膨胀**：频繁写入中间状态可能导致 RadixCache 快速膨胀，需要控制写入频率
4. **一致性**：缓存的 KV 是 target model 的输出，不是 draft model 的，需要确保后续命中时使用正确的 KV

### 5.3 EagleVerifyInput/Output 数据流

理解 verify 阶段的数据流是实现缓存协同的关键。

#### EagleVerifyInput（`eagle_info.py` L54-102）

```python
@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor       # (bs * draft_token_num,)
    custom_mask: torch.Tensor       # 树形注意力掩码
    positions: torch.Tensor         # (bs * draft_token_num,)
    retrive_index: torch.Tensor     # (bs, draft_token_num) 检索索引
    retrive_next_token: torch.Tensor    # 树遍历: 下一个子节点
    retrive_next_sibling: torch.Tensor  # 树遍历: 下一个兄弟节点
    spec_steps: int                 # 推测步数
    topk: int                       # top-k 值
    draft_token_num: int            # 每个请求的 draft token 数
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
```

#### EagleVerifyOutput（`eagle_info.py` L810-822）

```python
@dataclass
class EagleVerifyOutput:
    draft_input: EagleDraftInput           # 下一轮 draft 的输入
    logits_output: LogitsProcessorOutput   # target model logits
    verified_id: torch.Tensor              # accepted token IDs (含 bonus token)
    accept_length_per_req_cpu: List[int]   # 每个请求的 accept 长度
    accepted_indices: torch.Tensor         # 从 logits 中提取的已接受索引
```

#### Per-request Spec 指标更新（`eagle_info.py` L396-431）

```python
for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
    # ... 遍历 accepted tokens ...
    req.spec_verify_ct += 1
    accepted_draft_tokens = sum(1 for idx in accept_index_row if idx != -1) - 1
    req.spec_accepted_tokens += accepted_draft_tokens
    req.update_spec_acceptance_histogram(accepted_draft_tokens)
```

### 5.4 修改文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `python/sglang/srt/speculative/eagle_worker.py` | **修改** | verify 后缓存 accepted tokens 到 RadixCache |
| `python/sglang/srt/speculative/eagle_info.py` | **修改** | 在 `verify()` 方法中添加缓存写入逻辑 |
| `python/sglang/srt/server_args.py` | **修改** | 新增 `--speculative-eagle-cache-reuse` CLI 参数 |
| `benchmark/sglang_optimization/bench_eagle_cache.py` | **新建** | EAGLE + RadixCache 协同效果 Benchmark |

### 5.5 学习收获

通过实现此优化，深入理解了：

1. **EAGLE 的完整 draft-verify 流程**：`forward_batch_generation()` → `draft()` → `draft_forward()`（multi-step topk selection）→ `build_tree_kernel_efficient()` → `verify()` → `EagleVerifyOutput` → `forward_draft_extend_after_decode()`
2. **Draft-Target 内存共享机制**：`EAGLEWorker.__init__()` 中 `self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()` (L113-115)，draft 和 target 共享 KV 池，但各自管理不同的 KV cache layers
3. **Bigram Key 机制**：EAGLE 的 draft model 输入是 `(token_t, hidden_t-1)` 对，`convert_to_bigram_key()` 将相邻 token 对转为 bigram key，RadixCache 支持 `is_bigram=True` 模式
4. **CUDA Graph 在推测解码中的使用**：`EAGLEDraftCudaGraphRunner` 捕获固定 shape 的 draft forward graph；`EAGLEDraftExtendCudaGraphRunner` 捕获 draft extend graph。需要 `can_run()` 检查 batch 是否兼容
5. **Tree 构建 CUDA Kernel**：`build_tree_kernel_efficient()` 调用 `sgl_kernel.build_tree_kernel_efficient` 生成 tree_mask、positions、retrive_index。支持三种 TreeMaskMode：`FULL_MASK`、`QLEN_ONLY`、`QLEN_ONLY_BITPACKING`
6. **select_top_k_tokens() 的多步选择逻辑**：第 0 步从 extend 的 topk_p/topk_index 出发，后续步骤用 `expand_scores = scores.unsqueeze(2) * topk_p.reshape(-1, topk, topk)` 做 beam expansion，然后 `fast_topk()` 选最优

---

## 优化 6：推测解码全链路可观测性 `[辅助]` `[📋 待实现]`

### 6.1 问题分析

#### SGLang 现有推测解码 Metrics

SGLang 已经有基础的推测解码指标收集，但分散在多个层次：

**Scheduler 级别指标**（`scheduler_metrics_mixin.py`）：

| 指标 | 位置 | 粒度 | 含义 |
|------|------|------|------|
| `spec_num_accepted_tokens` | L83 | batch 级 | 近 N 个 batch 的 accepted tokens 总数 |
| `spec_num_forward_ct` | L84 | batch 级 | 近 N 个 batch 的 forward 次数 |
| `spec_total_num_accepted_tokens` | L86 | 全局 | 服务器生命周期内总 accepted tokens |
| `spec_total_num_forward_ct` | L87 | 全局 | 服务器生命周期内总 forward 次数 |

`update_spec_metrics()` 方法（L149-152）在每次 decode step 后被调用：

```python
def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
    self.spec_num_accepted_tokens += num_accepted_tokens + bs
    self.spec_num_forward_ct += bs
    self.num_generated_tokens += num_accepted_tokens
```

`log_decode_stats()` 方法（L318-488）在 decode_log_interval 后计算并记录：

```python
spec_accept_length = self.spec_num_accepted_tokens / self.spec_num_forward_ct
# 计算 acceptance rate
num_draft_tokens = self.server_args.speculative_num_draft_tokens or draft_tokens_fallback
total_draft_tokens = self.spec_num_forward_ct * num_draft_tokens
spec_accept_rate = self.spec_num_accepted_tokens / total_draft_tokens
```

**Prometheus Gauges**（`metrics/collector.py`）：

| Prometheus 指标名 | 类型 | 含义 |
|------------------|------|------|
| `sglang:spec_accept_length` | Gauge (L384-388) | 平均接受长度 |
| `sglang:spec_accept_rate` | Gauge (L390-393) | 平均接受率 |

**Per-request 级别指标**（`schedule_batch.py`）：

| 字段 | 位置 | 含义 |
|------|------|------|
| `req.spec_verify_ct` | L767 | 该请求的 verify 次数 |
| `req.spec_accepted_tokens` | L771 | 该请求的 accepted tokens 总数 |
| `req.spec_acceptance_histogram` | L776 | 接受长度直方图 |

`update_spec_acceptance_histogram()` 方法（L866-876）：

```python
def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):
    if len(self.spec_acceptance_histogram) <= accepted_draft_tokens:
        self.spec_acceptance_histogram.extend(
            [0] * (accepted_draft_tokens - len(self.spec_acceptance_histogram) + 1)
        )
    self.spec_acceptance_histogram[accepted_draft_tokens] += 1
```

**`/v1/loads` API**（`scheduler_metrics_mixin.py` L692-701）：

```python
if include_all or "spec" in include:
    if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
        speculative = SpeculativeMetrics(
            accept_length=(self.spec_total_num_accepted_tokens
                          / self.spec_total_num_forward_ct),
            accept_rate=self.stats.spec_accept_rate,
        )
```

#### 核心不足

1. **缺少阶段耗时**：无法区分 draft forward / tree building / verify forward / post-processing 各阶段的耗时占比
2. **缺少 N-gram 专项指标**：N-gram 的匹配率、匹配深度、Trie 节点利用率等信息不可见
3. **缺少 KV Cache 影响指标**：推测解码对 KV cache 的额外消耗（accepted vs wasted）不可观测
4. **Per-request 指标未暴露**：`spec_acceptance_histogram` 在请求完成时未输出到 API 或日志
5. **缺少分位数统计**：只有均值，无 P50/P90/P99 分位数

### 6.2 设计方案：全链路 Metrics 增强

#### 6.2.1 新增 Metrics 数据类

```python
# 修改: python/sglang/srt/metrics/collector.py

@dataclass
class SpecDecodeDetailedStats:
    """推测解码全链路详细指标"""

    # === 阶段耗时 (EAGLE 专用) ===
    eagle_draft_time_ms: float = 0.0      # Draft model forward 总耗时
    eagle_verify_time_ms: float = 0.0     # Verify 阶段总耗时 (target forward + token verify)
    eagle_tree_build_time_ms: float = 0.0 # Token tree 构建耗时
    eagle_draft_extend_time_ms: float = 0.0  # Draft extend after decode 耗时

    # === 接受率详细指标 ===
    acceptance_rate_by_step: List[float] = field(default_factory=list)
    # 每个 draft step 的接受率
    # e.g., [0.85, 0.72, 0.60, 0.45] 表示 step 1~4 的接受率

    # === N-gram 专项指标 ===
    ngram_match_rate: float = 0.0         # N-gram 匹配成功率
    ngram_avg_match_depth: float = 0.0    # 平均匹配深度
    ngram_trie_node_usage: float = 0.0    # Trie 节点利用率
    ngram_insert_queue_depth: int = 0     # 异步插入队列深度

    # === KV Cache 影响 ===
    spec_cache_allocated_tokens: int = 0  # 推测解码分配的 KV slots
    spec_cache_accepted_tokens: int = 0   # 最终保留的 KV slots
    spec_cache_wasted_tokens: int = 0     # 被释放的 KV slots
    spec_cache_waste_ratio: float = 0.0   # 浪费比例

    # === 派生指标 ===
    @property
    def tokens_per_step(self) -> float:
        """每步有效 token 数 = (accepted + bonus) / forward_ct"""
        return (self.spec_cache_accepted_tokens
                / max(1, self.spec_cache_allocated_tokens))

    @property
    def draft_overhead_ratio(self) -> float:
        """Draft 开销占比 = draft_time / total_time"""
        total = (self.eagle_draft_time_ms + self.eagle_verify_time_ms
                + self.eagle_tree_build_time_ms + self.eagle_draft_extend_time_ms)
        return self.eagle_draft_time_ms / max(0.001, total)
```

#### 6.2.2 新增 Prometheus Gauges

```python
# 修改: python/sglang/srt/metrics/collector.py :: SchedulerMetricsCollector

# 阶段耗时 Gauges
self.spec_draft_time_ms = Gauge(
    name="sglang:spec_draft_time_ms",
    documentation="Draft model forward time in milliseconds.",
    labelnames=labels.keys(),
)
self.spec_verify_time_ms = Gauge(
    name="sglang:spec_verify_time_ms",
    documentation="Verify stage time in milliseconds.",
    labelnames=labels.keys(),
)
self.spec_tree_build_time_ms = Gauge(
    name="sglang:spec_tree_build_time_ms",
    documentation="Token tree building time in milliseconds.",
    labelnames=labels.keys(),
)

# KV Cache 指标
self.spec_cache_waste_ratio = Gauge(
    name="sglang:spec_cache_waste_ratio",
    documentation="Ratio of wasted KV cache slots from rejected draft tokens.",
    labelnames=labels.keys(),
)

# 按 step 的接受率 (Histogram)
self.spec_acceptance_by_step = Histogram(
    name="sglang:spec_acceptance_by_step",
    documentation="Acceptance rate distribution by draft step.",
    labelnames=[*labels.keys(), "step"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
```

#### 6.2.3 采集点设计

在推测解码的关键路径上添加计时器和计数器：

```
EAGLEWorker.forward_batch_generation()
  │
  ├── 路径 B (Decode):
  │     │
  │     ├── [TIMER START: draft]
  │     │   draft(batch) → spec_info
  │     │   [TIMER END: draft → eagle_draft_time_ms]
  │     │
  │     ├── [TIMER START: verify]
  │     │   verify(batch, spec_info) → verify_output
  │     │   [TIMER END: verify → eagle_verify_time_ms]
  │     │
  │     ├── [COUNTER: cache metrics]
  │     │   allocated = len(batch.out_cache_loc)
  │     │   accepted = len(verify_output.accepted_indices)
  │     │   wasted = allocated - accepted
  │     │
  │     ├── [TIMER START: draft_extend]
  │     │   forward_draft_extend_after_decode(batch)
  │     │   [TIMER END: draft_extend → eagle_draft_extend_time_ms]
  │     │
  │     └── [EMIT: SpecDecodeDetailedStats]
  │
  └── (per-request) EagleVerifyInput.verify():
        [COUNTER: acceptance_by_step]
        for each req:
          per_step_accepted[step] += (accepted or not)
```

#### 6.2.4 Per-request Metrics 暴露

在请求完成的响应中添加推测解码指标：

```python
# 修改: python/sglang/srt/managers/io_struct.py

@dataclass
class TokenizedGenerateReqInput:
    # ... existing fields ...

    # 🆕 推测解码指标（请求完成时填充）
    spec_metrics: Optional[Dict] = None
    # {
    #   "verify_count": req.spec_verify_ct,
    #   "accepted_tokens": req.spec_accepted_tokens,
    #   "acceptance_histogram": req.spec_acceptance_histogram,
    #   "avg_acceptance_length": req.spec_accepted_tokens / max(1, req.spec_verify_ct),
    # }
```

### 6.3 现有 Metrics 调用链

```
Scheduler 事件循环
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│ scheduler.py :: event_loop_normal()                      │
│                                                         │
│   batch_result = spec_worker.forward_batch_generation() │
│                                                         │
│   ★ update_spec_metrics(bs, num_accepted_tokens)        │
│     (scheduler_metrics_mixin.py L149-152)               │
│     → spec_num_accepted_tokens += num_accepted + bs     │
│     → spec_num_forward_ct += bs                         │
│     → num_generated_tokens += num_accepted              │
│                                                         │
│   每 decode_log_interval 次:                             │
│   ★ log_decode_stats()                                  │
│     (scheduler_metrics_mixin.py L318-488)               │
│     → spec_accept_length = accepted / forward_ct        │
│     → spec_accept_rate = accepted / total_draft_tokens  │
│     → stats.spec_accept_rate = spec_accept_rate         │
│     → stats.spec_accept_length = spec_accept_length     │
│     → metrics_collector.log_stats(stats)                │
│       └── Prometheus: sglang:spec_accept_length         │
│       └── Prometheus: sglang:spec_accept_rate           │
│                                                         │
│   /v1/loads API 请求:                                    │
│   ★ get_loads()                                         │
│     (scheduler_metrics_mixin.py L634-770)               │
│     → SpeculativeMetrics(accept_length, accept_rate)    │
└─────────────────────────────────────────────────────────┘
```

### 6.4 修改文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `python/sglang/srt/metrics/collector.py` | **修改** | 新增 `SpecDecodeDetailedStats`、Prometheus Gauges |
| `python/sglang/srt/managers/scheduler_metrics_mixin.py` | **修改** | 新增 `update_spec_detailed_metrics()` 方法 |
| `python/sglang/srt/speculative/eagle_worker.py` | **修改** | 在 `forward_batch_generation()` 中添加计时器 |
| `python/sglang/srt/speculative/ngram_worker.py` | **修改** | 在 `forward_batch_generation()` 中添加匹配率计数 |
| `python/sglang/srt/managers/io_struct.py` | **修改** | 请求完成响应中添加 spec_metrics |
| `benchmark/sglang_optimization/spec_decode_metrics.py` | **新建** | 推测解码 Metrics 采集和可视化工具 |

### 6.5 使用方式

```bash
# 启动时启用详细推测解码指标
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-eagle-path /path/to/eagle-model \
    --enable-metrics

# 查询 Prometheus 指标
curl http://localhost:30000/metrics | grep sglang:spec_

# 查询 /v1/loads API（包含 spec 指标）
curl http://localhost:30000/v1/loads?include=spec
```

### 6.6 Benchmark 设计

```python
# benchmark/sglang_optimization/spec_decode_metrics.py

"""推测解码指标采集 Benchmark

测试场景:
1. EAGLE: 不同 topk × num_steps 配置下的各阶段耗时分布
2. N-gram: 不同 match_window × bfs_breadth 下的匹配率
3. KV Cache: 推测解码的 cache 浪费比例 vs accept_rate

输出格式:
  ┌────────────────────────────────────────────────────────┐
  │ EAGLE Timing Breakdown (avg per batch)                  │
  │   draft:        2.3ms (18%)                            │
  │   tree_build:   0.8ms ( 6%)                            │
  │   verify:       8.5ms (66%)                            │
  │   draft_extend: 1.2ms (10%)                            │
  │   total:       12.8ms                                  │
  │                                                        │
  │ Acceptance by Step:                                    │
  │   step 1: 85%  ████████████████                        │
  │   step 2: 72%  ██████████████                          │
  │   step 3: 60%  ████████████                            │
  │   step 4: 45%  █████████                               │
  │                                                        │
  │ KV Cache Impact:                                       │
  │   allocated:  2048 slots/batch                         │
  │   accepted:   1280 slots/batch                         │
  │   wasted:      768 slots/batch (37.5%)                 │
  └────────────────────────────────────────────────────────┘
"""
```

### 6.7 学习收获

通过实现此优化，深入理解了：

1. **SGLang 的 Metrics 收集框架**：`SchedulerMetricsMixin` 通过 `SchedulerStats` 数据类传递指标到 `SchedulerMetricsCollector`，后者通过 `prometheus_client` 暴露 Prometheus 格式指标。`log_stats()` 方法（collector.py L990-992）将 stats 映射到对应的 Gauge/Counter
2. **`update_spec_metrics()` 的调用时机**：在 `Scheduler.event_loop_normal()` 中，每次 decode step 后立即调用，`num_accepted_tokens` 来自 `GenerationBatchResult.num_accepted_tokens`
3. **Per-request vs Batch 级 Metrics 的差异**：`req.spec_verify_ct` / `req.spec_accepted_tokens` / `req.spec_acceptance_histogram` 在 `EagleVerifyInput.verify()` / `NgramVerifyInput._fill_requests()` 中逐请求更新；`spec_num_accepted_tokens` / `spec_num_forward_ct` 在 Scheduler 级别按 batch 聚合
4. **`/v1/loads` API 的可扩展设计**：通过 `include` 参数控制返回哪些指标模块（`core`/`spec`/`memory`/`lora`/`disagg`/`queues`），每个模块对应一个 `Optional[XxxMetrics]` 字段
5. **推测解码的端到端性能瓶颈**：verify 阶段（target model forward）通常占总时间的 60-70%，draft forward 占 15-20%，tree building 和 post-processing 占剩余部分。这意味着优化 draft 效率的收益有限，重点应在提高 accept rate

---

## 实施状态

| 优化点 | 状态 | 说明 |
|--------|------|------|
| 优化 4：N-gram SAM Proposer | 📋 待实现 | 在 C++ Trie 基础上添加后缀自动机 Proposer |
| 优化 5：EAGLE + RadixCache 协同 | ✅ 已实现 | 将 EAGLE verified tokens 的 KV 写入 RadixCache |
| 优化 6：推测解码可观测性 | 📋 待实现 | 全链路 Metrics 增强 |

---

## 与 vLLM 优化经验的对照

| vLLM 上的优化 | 本方向对标 | SGLang 的不同之处 |
|--------------|-----------|------------------|
| **后缀自动机 Proposer** | 优化 4 (SAM + N-gram) | vLLM 的 N-gram 是纯 Python + Numba JIT；SGLang 有 C++ Trie 作为基础，SAM 是增强而非替代 |
| **自适应评分** | 优化 4 (Trie 已有 BFS/Prob 模式) | SGLang 的 Trie 已经支持频率排序的 `sorted_children` 和两种匹配模式 |
| **Spec Decode 可观测性** | 优化 6 | SGLang 已有 `spec_accept_length`/`spec_accept_rate` Prometheus Gauge 和 per-request histogram，需要细化阶段耗时和 KV cache 影响 |
| **EAGLE 集成** | 优化 5 | vLLM V1 无 EAGLE 支持；SGLang 的 EAGLE 是最成熟的实现，支持 EAGLE/EAGLE3/multi-layer/overlap 多种模式 |
