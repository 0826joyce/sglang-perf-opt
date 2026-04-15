# SGLang 深度优化学习计划

> 基于在 vLLM V1 上的 PD 分离、Prefix Cache 调度、后缀解码三大优化经验，针对 SGLang 的架构特点设计可落地的优化点，在实践中加深对 SGLang 核心机制的理解

## 前置说明

### 你已有的经验（vLLM 项目）

| 方向 | 在 vLLM 上做过的 | 核心收获 |
|------|-----------------|---------|
| **PD 分离** | V1 引擎适配 + 智能路由 + 调度器感知 + KV 传输优化 + Prefix Cache 协同 | 深入理解了 KV Cache 跨 GPU 传输、调度器与传输层的协调、Prefix Caching hash chain |
| **Prefix Cache 调度** | Cache-Aware Scheduling + Segmented LRU + 抢占缓存保护 + 预热 | 掌握了 KV Cache block 管理、引用计数、驱逐策略、调度器与缓存的交互时序 |
| **后缀解码** | 后缀数组 Proposer + 增量 SAM + 自适应评分 + 跨请求共享 | 理解了 Spec Decode 全链路、Proposer 接口、RejectionSampler 验证机制 |

### SGLang 与 vLLM 的关键架构差异

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **KV Cache 管理** | `KVCacheManager` — 基于 hash chain 的 block 管理，`FreeKVCacheBlockQueue` 双链表管理空闲 block | `RadixCache` — 基于 Radix Tree（基数树）管理 KV Cache，树节点带 `lock_ref` 引用计数 |
| **调度策略** | MLFQ + FCFS（无缓存感知） | 已有 **LPM（最长前缀匹配）** 和 **DFS-Weight** 缓存感知策略，以及 in-batch prefix caching |
| **驱逐策略** | 纯 LRU（`FreeKVCacheBlockQueue`） | 已支持 6 种策略（LRU/LFU/FIFO/MRU/FILO/Priority），通过 `EvictionStrategy` 抽象 |
| **Spec Decode** | V1 仅支持 N-gram Proposer（纯 Python KMP） | 支持 EAGLE/EAGLE3/NGRAM/Standalone，C++ N-gram 实现（Trie 树），CUDA Graph 加速 |
| **PD 分离** | V0 有基础实现，V1 需要从头适配 | 已有完整的 `disaggregation/` 模块，支持 Prefill/Decode 分离 |
| **Overlap 调度** | 无 | 已有 `event_loop_overlap()` — CUDA 多流 GPU/CPU 重叠 |
| **分页缓存** | 基于 hash chain 的 block，`FreeKVCacheBlockQueue` | 两级内存池：`ReqToTokenPool`（请求→token 位置映射）+ `TokenToKVPoolAllocator`（KV 索引分配） |
| **HiCache** | 无 | 有 `HiRadixCache` — GPU→CPU→Disk 三级层次化缓存 |

---

## 优化方向总览

根据 SGLang 的架构特点和你已有的经验，设计 **3 个方向、共 9 个优化点**：

```
方向一：调度与缓存协同优化（对标 vLLM Prefix Cache 调度经验）
  ├── 优化 1：Radix Cache 驱逐策略增强 — Adaptive Eviction
  ├── 优化 2：调度策略量化对比 Benchmark
  └── 优化 3：缓存预热与主动预取

方向二：推测解码增强（对标 vLLM 后缀解码经验）
  ├── 优化 4：N-gram Proposer 后缀自动机增强
  ├── 优化 5：EAGLE 投机解码 + Radix Cache 协同
  └── 优化 6：推测解码全链路可观测性

方向三：Overlap 调度与 PD 分离深度优化（对标 vLLM PD 分离经验）
  ├── 优化 7：Overlap 调度精细化 — 动态 Overlap 决策 ✅
  ├── 优化 8：PD 分离场景下的 Radix Cache 跨实例协同
  └── 优化 9：端到端性能分析 Benchmark 框架
```

---

## 方向一：调度与缓存协同优化

> 你在 vLLM 上做的 Segmented LRU、Cache-Aware Scheduling、抢占缓存保护等优化，在 SGLang 上需要**适配到 Radix Tree 结构**。SGLang 已经有 LPM/DFS-Weight 策略和 6 种驱逐策略，但仍有优化空间。

### 优化 1：Radix Cache 驱逐策略增强 — Adaptive Eviction `[核心]`

#### SGLang 现状分析

SGLang 的 `RadixCache` 驱逐逻辑位于 `radix_cache.py` 的 `evict()` 方法：

```python
# radix_cache.py L565-593
def evict(self, params: EvictParams) -> EvictResult:
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
```

驱逐策略在 `evict_policy.py` 中定义，每个策略返回一个优先级值用于堆排序：

```python
# evict_policy.py — 完整文件仅 47 行
class LRUStrategy:   get_priority → node.last_access_time
class LFUStrategy:   get_priority → (node.hit_count, node.last_access_time)
class FIFOStrategy:  get_priority → node.creation_time
class PriorityStrategy: get_priority → (node.priority, node.last_access_time)
```

**SGLang 与 vLLM 的关键差异**：
- vLLM 的驱逐是在 **flat block 级别**（`FreeKVCacheBlockQueue` 的双链表），O(1) popleft
- SGLang 的驱逐是在 **Radix Tree 叶子节点级别**，使用 heap 排序，O(n log n)
- SGLang 的 `TreeNode` 已经有 `hit_count`、`last_access_time`、`priority`、`creation_time` 等丰富的元数据

**可优化的问题**：
1. **缺少 Segmented 保护机制**：虽然有 LFU 策略考虑 `hit_count`，但没有类似 vLLM Segmented LRU 的"试用区/保护区"概念，高频节点仍可能被一次性大量驱逐
2. **驱逐是被动触发**：只在分配失败时才驱逐，没有后台预驱逐（proactive eviction）
3. **驱逐不区分节点的树位置**：根附近的共享前缀节点（如 System Prompt）和叶子节点的驱逐权重相同

#### 设计方案

**新增一个 Adaptive 驱逐策略**，综合考虑：
- 访问频率（`hit_count`）
- 最近访问时间（`last_access_time`）
- **树深度**（depth）——越靠近根的节点越可能是共享前缀，驱逐代价越高
- **子树大小**（subtree_size）——该节点被驱逐后影响的范围

```python
# 新文件: python/sglang/srt/mem_cache/evict_policy.py — 新增 AdaptiveStrategy

class AdaptiveStrategy(EvictionStrategy):
    """自适应驱逐策略：综合频率、时间、树结构的多因子评分。

    驱逐优先级 = w1 * recency_score + w2 * frequency_score + w3 * depth_score
    分数越低越先被驱逐。
    """

    def __init__(
        self,
        w_recency: float = 0.4,   # 最近访问时间权重
        w_frequency: float = 0.3,  # 访问频率权重
        w_depth: float = 0.3,      # 树深度权重（深度越大，越接近叶子，越容易驱逐）
    ):
        self.w_recency = w_recency
        self.w_frequency = w_frequency
        self.w_depth = w_depth
        self._max_hit_count = 1  # 动态追踪，防止除零

    def get_priority(self, node: "TreeNode") -> float:
        """返回驱逐优先级（值越小越先被驱逐）。

        与 LRU/LFU 不同，这里将多个因子归一化后加权组合，
        得到一个综合评分。
        """
        # 因子 1：最近访问时间（越久越先驱逐）
        recency_score = node.last_access_time  # 越大=越近=越不该驱逐

        # 因子 2：访问频率（越低越先驱逐）
        self._max_hit_count = max(self._max_hit_count, node.hit_count + 1)
        frequency_score = node.hit_count / self._max_hit_count  # [0, 1]

        # 因子 3：树深度（越深=越接近叶子=越容易驱逐）
        depth = self._get_depth(node)
        depth_score = -depth  # 越深分数越低

        return (
            self.w_recency * recency_score
            + self.w_frequency * frequency_score
            + self.w_depth * depth_score
        )

    @staticmethod
    def _get_depth(node: "TreeNode") -> int:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
```

#### 你将学到什么

1. **Radix Tree 的驱逐时序**：`evict()` → `_delete_leaf()` → 父节点递归检查 → `_update_leaf_status()`
2. **`lock_ref` 的作用**：`inc_lock_ref()`/`dec_lock_ref()` 沿着从叶到根的路径更新引用计数，`lock_ref > 0` 的节点不可驱逐
3. **`evictable_leaves` 集合的维护**：只有 `lock_ref == 0` 且无子节点的 TreeNode 才是驱逐候选
4. **SGLang 策略模式的扩展**：通过实现 `EvictionStrategy` 抽象类新增策略

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/mem_cache/evict_policy.py` | 新增 `AdaptiveStrategy` | 自适应驱逐策略 |
| `python/sglang/srt/mem_cache/radix_cache.py` | `__init__` 中注册 `"adaptive"` | 支持 `--eviction-policy adaptive` |
| `benchmark/sglang_optimization_study/bench_eviction.py` | 新建 | 对比不同驱逐策略的缓存命中率 |

#### 预期效果

- System Prompt 等高频共享前缀的驱逐概率大幅降低
- 在混合负载（短对话 + 长文档）下，缓存命中率提升 10-20%

---

### 优化 2：调度策略量化对比 Benchmark `[基础]`

#### SGLang 现状分析

SGLang 已有多种调度策略（`LPM`、`DFS-Weight`、`FCFS`、`LOF`、`RANDOM`、`ROUTING-KEY`），但缺少系统性的性能对比。

**关键代码路径**（`schedule_policy.py`）：

```python
# calc_priority 方法 — 调度排序入口
def calc_priority(self, waiting_queue, running_batch):
    if CacheAwarePolicy.LPM:
        # 1. 对每个 waiting 请求调用 tree_cache.match_prefix() 获取缓存命中长度
        # 2. 按 -len(prefix_indices) 排序（命中越多越优先）
        # 3. in-batch prefix caching: 降低与等待队列中其他请求共享前缀的请求优先级
    elif CacheAwarePolicy.DFS_WEIGHT:
        # 1. 按 Radix Tree 的 DFS 权重排序
        # 2. 使用 _calc_weight() 递归计算每个节点的权重（=引用该节点的 waiting 请求数）
```

**LPM 的一个有趣退化**：

```python
def _determine_active_policy(self, waiting_queue):
    if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
        # 当等待队列 > 128 时，LPM 退化为 FCFS
        # 原因：match_prefix() 需要对每个请求遍历 Radix Tree，O(n) 开销
        return CacheAgnosticPolicy.FCFS
```

#### 设计方案

编写一个 Benchmark 脚本，系统对比所有调度策略在不同工作负载下的表现：

```python
# benchmark/sglang_optimization_study/bench_scheduling.py

"""调度策略对比 Benchmark

测试维度:
1. 共享前缀场景（100 个请求共享 System Prompt）
2. 无前缀共享场景（100 个独立请求）
3. 混合场景（50 共享 + 50 独立）
4. 长队列压力（500+ 等待请求，测试 LPM→FCFS 退化）

度量指标:
- TTFT 分布（P50, P95, P99）
- 缓存命中率（RadixCache hit tokens / total tokens）
- 吞吐量（tokens/s）
- 调度延迟（calc_priority 耗时）
"""

WORKLOADS = {
    "shared_prefix": {
        "system_prompt": "你是一个有帮助的 AI 助手..." * 100,  # ~400 tokens
        "num_requests": 100,
        "user_prompt_len_range": (10, 50),
    },
    "no_sharing": {
        "system_prompt": None,
        "num_requests": 100,
        "user_prompt_len_range": (100, 500),
    },
    "mixed": {
        "shared_ratio": 0.5,
        "num_requests": 100,
    },
    "long_queue": {
        "num_requests": 500,
        "arrival_rate": "burst",  # 所有请求同时到达
    },
}

POLICIES = ["lpm", "dfs-weight", "fcfs", "lof", "random"]
```

#### 你将学到什么

1. **SGLang 的 Benchmark 框架**：`benchmark/` 下已有 38+ 场景，学习其标准化写法
2. **`match_prefix()` 的完整路径**：从 `_match_prefix_helper()` 到 `_split_node()`（节点分裂），理解 Radix Tree 的查找与结构变异
3. **In-batch prefix caching 的精妙设计**：使用一个**模拟的 `waiting_queue_radix_tree`**（无实际 KV 数据）来检测等待队列内部的前缀共享
4. **LPM 的性能天花板**：`len(waiting_queue) > 128` 时退化，理解为什么

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `benchmark/sglang_optimization_study/bench_scheduling.py` | 新建 | 调度策略对比 |
| `benchmark/sglang_optimization_study/workloads/` | 新建 | 标准化测试数据集 |

---

### 优化 3：缓存预热与主动预取 `[进阶]`

#### SGLang 现状分析

SGLang 的 `HiRadixCache`（`hiradix_cache.py`）已经有 **GPU→CPU→Disk 三级缓存**，以及 `load_back()` 将 CPU 上的 KV 数据加载回 GPU 的能力。但目前的缓存是完全被动的——只有请求到达时才触发 `match_prefix()`。

**HiRadixCache 的 load_back 逻辑**（`hiradix_cache.py`）：

```python
def load_back(self, node, mem_quota=None):
    """将 evicted 但 backuped（在 CPU/Disk 上）的节点加载回 GPU"""
    # 1. 找到所有需要加载的 evicted 节点链
    # 2. 保护祖先节点不被驱逐（inc_lock_ref）
    # 3. 如果 GPU 空间不足，先驱逐 → 再加载
    # 4. cache_controller.load(host_indices) → GPU
```

#### 设计方案

**主动缓存预热**：在系统冷启动或空闲时，提前将高频 System Prompt 的 KV Cache 计算好并存入 Radix Tree。

```python
# 新文件: python/sglang/srt/mem_cache/cache_warming.py

class CacheWarmingManager:
    """缓存预热管理器

    工作原理:
    1. 从配置文件加载常见 System Prompt（token IDs）
    2. 系统空闲时，构造虚拟请求执行 prefill
    3. 将计算好的 KV Cache 通过 RadixCache.insert() 写入缓存
    4. 后续真实请求命中预热缓存后可跳过 prefill

    触发条件:
    - 系统启动后的初始化阶段
    - 运行中检测到系统空闲（waiting_queue 和 running_batch 都为空）
    - HiCache 场景：从 CPU/Disk 预取即将到来的前缀
    """

    def __init__(self, warmup_config_path: str):
        self.warmup_prompts = self._load_config(warmup_config_path)

    def warmup(self, scheduler):
        """执行预热

        关键步骤:
        1. 为每个 warmup prompt 构造虚拟 Req
        2. 通过 scheduler 的标准路径执行 prefill
        3. 请求完成后 KV Cache 自动写入 RadixCache
        4. 标记为预热请求，不产生实际输出
        """
        for prompt in self.warmup_prompts:
            virtual_req = self._create_virtual_req(prompt)
            scheduler.add_warmup_request(virtual_req)
```

**与 HiCache 的协同**：如果启用了 HiCache，预热可以先在 CPU 上准备 KV 数据，然后按需加载到 GPU，避免启动时大量 GPU 计算。

#### 你将学到什么

1. **Scheduler 的请求生命周期**：从 `process_input_requests()` → `_add_request_to_queue()` → `get_new_batch_prefill()` → `run_batch()` → `cache_finished_req()`/`cache_unfinished_req()`
2. **RadixCache 的写入路径**：`cache_finished_req()` → `insert()` → `_insert_helper()` — 如何将 KV 索引写入 Radix Tree
3. **HiRadixCache 的三级缓存交互**：`write_backup()` → `cache_controller.write()` → `load_back()` → `cache_controller.load()`
4. **`req.fill_ids` 与 `req.origin_input_ids` 的区别**：前者是已填充的 token IDs（含部分 output），后者是原始输入

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/mem_cache/cache_warming.py` | 新建 | 预热管理器 |
| `python/sglang/srt/managers/scheduler.py` | 修改 | `self_check_during_idle()` 中触发预热 |
| `python/sglang/srt/server_args.py` | 修改 | 新增 `--warmup-config` 参数 |

---

## 方向二：推测解码增强

> 你在 vLLM 上做的后缀自动机 Proposer 和自适应评分，在 SGLang 上可以**与已有的 EAGLE 和 C++ N-gram 实现协同**，SGLang 的 Spec Decode 实现比 vLLM V1 成熟得多。

### 优化 4：N-gram Proposer 后缀自动机增强 `[核心]` `[✅ 已实现]`

#### SGLang 现状分析

SGLang 的 N-gram Proposer 已经有一个 **C++ 实现的 Trie 树**（`speculative/cpp_ngram/`）：

```
speculative/cpp_ngram/
├── ngram.cpp          # C++ Trie 树实现 (12.79 KB)
├── ngram.h            # Trie 节点定义
├── param.h            # 参数配置
├── queue.h            # 线程安全队列
└── ngram_cache.py     # Python 绑定 (NgramCache 类)
```

```cpp
// ngram.h
class Ngram {
    std::vector<TrieNode> nodes_;
    std::vector<TrieNode*> node_pool_;
    // ... Trie 树实现
};
```

**与 vLLM 的差异**：
- vLLM V1 的 N-gram 是纯 Python + Numba JIT 的 KMP 搜索，每次 O(context_len)
- SGLang 用 **C++ Trie 树** 做 N-gram 缓存，支持跨请求的模式共享
- 但 SGLang 的 Trie 是固定 N 值匹配，不支持可变长度

#### 设计方案

在 SGLang 的 C++ N-gram Trie 基础上，**添加后缀自动机作为备选 Proposer**，复用你在 vLLM 上的经验：

```python
# 新文件: python/sglang/srt/speculative/suffix_automaton_proposer.py

class SuffixAutomatonProposer:
    """增量后缀自动机 Proposer，作为 NgramCache 的增强选项。

    与 NgramCache (C++ Trie) 的关系:
    - NgramCache: 固定 N 值匹配，跨请求共享模式，C++ 实现高效
    - SuffixAutomaton: 可变长度匹配，per-request 状态，O(1) 增量更新

    两者可以组合使用:
    1. 先查 SuffixAutomaton（per-request 上下文匹配）
    2. 未命中再查 NgramCache（跨请求模式匹配）
    """

    def __init__(self):
        # 每个请求的 SAM
        self._automata: Dict[str, IncrementalSuffixAutomaton] = {}

    def propose(self, req_id: str, context: np.ndarray, n: int, k: int):
        # 1. 增量更新当前请求的 SAM
        # 2. 自适应回退匹配（从 n 到 n//2）
        # 3. 多候选评分选最优
        ...
```

**关键适配点**：SGLang 的 `NGRAMWorker`（`ngram_worker.py`）与 `EAGLEWorker` 不同，它是独立的 Worker 类，有自己的 `forward_batch_speculative_generation()` 和 `verify()` 方法。需要理解其接口才能正确接入。

#### 你将学到什么

1. **SGLang 的 N-gram Spec Decode 全链路**：`NGRAMWorker.forward_batch_speculative_generation()` → `NgramCache.get_match()` → 构建 verify batch → 验证
2. **C++ Trie 的 Python 绑定机制**：`ngram_cache_binding.cpp` 通过 pybind11 暴露给 Python
3. **SGLang Spec Decode 与 vLLM 的架构差异**：SGLang 用独立的 Worker 做 draft+verify，vLLM V1 将 proposer 集成在 ModelRunner 内部
4. **`SpeculativeAlgorithm` 枚举的扩展**：如何注册新的推测解码算法

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/speculative/suffix_automaton_proposer.py` | 新建 | 后缀自动机 Proposer |
| `python/sglang/srt/speculative/ngram_worker.py` | 修改 | 集成 SAM 作为备选 Proposer |
| `benchmark/sglang_optimization_study/bench_spec_decode.py` | 新建 | N-gram vs SAM 对比 |

---

### 优化 5：EAGLE 投机解码 + Radix Cache 协同 `[进阶]` `[✅ 已实现]`

#### SGLang 现状分析

SGLang 的 EAGLE Worker（`eagle_worker.py`，40.6KB）是整个项目中最复杂的模块之一。关键特征：

1. **Draft 模型与 Target 模型共享 `req_to_token_pool` 和 `token_to_kv_pool_allocator`**
2. **EAGLE 使用 bigram key**：RadixCache 支持 `is_bigram=True` 模式，`convert_to_bigram_key()` 将 token IDs 转为 bigram 对
3. **EAGLE3 多层 Draft**：`multi_layer_eagle_worker.py` (30KB) 实现多层 draft 模型
4. **CUDA Graph 加速**：`eagle_draft_cuda_graph_runner.py` 和 `eagle_draft_extend_cuda_graph_runner.py`

**当前的问题**：EAGLE draft 阶段生成的候选 token 在验证被拒绝后，其 KV Cache 被丢弃。但如果多个请求在 decode 阶段生成了相似的后续内容，这些 KV Cache 可能是可复用的。

#### 设计方案

**Speculative KV Cache Reuse**：将 EAGLE 验证通过的 token 序列的 KV Cache 保留在 RadixCache 中，供后续请求复用。

```python
# 核心思路：
# 1. EAGLE verify 通过的 accepted tokens → 写入 RadixCache
# 2. 下一步 EAGLE draft 时，先查 RadixCache 是否有命中
# 3. 命中的部分跳过 draft model forward，直接使用缓存的 KV

# 修改 eagle_worker.py 中 verify 完成后的逻辑：
def _cache_accepted_tokens(self, batch, accepted_token_ids):
    """将验证通过的 token 序列写入 RadixCache，供后续复用。"""
    for req_idx, accepted in enumerate(accepted_token_ids):
        if len(accepted) > 1:  # 至少 2 个 accepted tokens 才有缓存价值
            req = batch.reqs[req_idx]
            # 将 origin_input_ids + output_ids + accepted_ids 的 KV 写入缓存
            self.tree_cache.cache_unfinished_req(req)
```

#### 你将学到什么

1. **EAGLE 的完整 draft-verify 流程**：`forward_batch_speculative_generation()` → `draft_extend()` → `build_tree_kernel_efficient()` → `verify()` → `EagleVerifyOutput`
2. **Bigram Key 机制**：为什么 EAGLE 需要 bigram（因为 EAGLE 的 draft model 输入是 (token_t, hidden_t-1) 对）
3. **RadixCache 与 EAGLE 的交互**：`cache_finished_req()` 中 `convert_to_bigram_key()` 的转换逻辑
4. **CUDA Graph 在推测解码中的使用**：`EAGLEDraftCudaGraphRunner` 如何捕获固定 shape 的 draft forward

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/speculative/eagle_worker.py` | 修改 | verify 后缓存 accepted tokens |
| `benchmark/sglang_optimization_study/bench_eagle_cache.py` | 新建 | EAGLE + RadixCache 协同效果 |

---

### 优化 6：推测解码全链路可观测性 `[辅助]`

#### 设计方案

为 SGLang 的推测解码添加详细的性能指标：

```python
# 新文件: benchmark/sglang_optimization_study/spec_decode_metrics.py

@dataclass
class SpecDecodeMetrics:
    """推测解码全链路指标"""

    # EAGLE 指标
    eagle_draft_time_ms: float = 0.0      # Draft model forward 耗时
    eagle_verify_time_ms: float = 0.0     # Verify 阶段耗时
    eagle_tree_build_time_ms: float = 0.0 # Token tree 构建耗时

    # 接受率指标
    total_draft_tokens: int = 0           # 总 draft tokens
    total_accepted_tokens: int = 0        # 总 accepted tokens
    acceptance_rate_by_step: List[float]  # 每步的接受率

    # N-gram 指标
    ngram_match_rate: float = 0.0         # N-gram 匹配成功率
    ngram_avg_match_len: float = 0.0      # 平均匹配长度

    # KV Cache 影响
    spec_cache_hit_tokens: int = 0        # 因推测解码而额外命中的缓存 tokens
    spec_cache_waste_tokens: int = 0      # 被拒绝的 draft tokens 的 KV 浪费

    @property
    def tokens_per_step(self) -> float:
        """每步有效 token 数"""
        return (self.total_accepted_tokens + self.total_draft_tokens) / max(1, self.total_draft_tokens)

    @property
    def draft_overhead_ratio(self) -> float:
        """Draft 开销占比 = draft_time / (draft_time + verify_time)"""
        total = self.eagle_draft_time_ms + self.eagle_verify_time_ms
        return self.eagle_draft_time_ms / max(0.001, total)
```

#### 你将学到什么

1. **SGLang 的 Metrics 收集框架**：`SchedulerMetricsMixin` 和 `SchedulerMetricsCollector` 的工作方式
2. **推测解码的端到端性能瓶颈**：Draft forward 通常只占总时间的 10-20%，主要开销在 tree building 和 verify

---

## 方向三：Overlap 调度与 PD 分离深度优化

> 你在 vLLM 上做的 PD 分离全套优化，在 SGLang 上可以**利用已有的 disaggregation 模块和 overlap 调度**，聚焦更高级的优化。

### 优化 7：Overlap 调度精细化 — 动态 Overlap 决策 `[核心]` `[✅ 已实现]`

#### SGLang 现状分析

SGLang 的 `event_loop_overlap()` 已经实现了 GPU/CPU 重叠调度：

```python
# scheduler.py L1109-1151
def event_loop_overlap(self):
    """A scheduler loop that overlaps the CPU processing and GPU computation."""
    self.result_queue = deque()

    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run()
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

        # 如果不需要 overlap，立即处理上一个 batch 的结果
        if disable_overlap_for_batch:
            pop_and_process()

        # 启动当前 batch
        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))

        # 处理上一个 batch 的结果（与当前 batch 的 GPU 计算重叠）
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
```

**`is_disable_overlap_for_batch()`** 决定是否禁用 overlap：

```python
def is_disable_overlap_for_batch(self, batch):
    # 以下场景禁用 overlap:
    # 1. batch 为 None（空闲）
    # 2. 上一个 batch 为 None
    # 3. 推测解码模式（spec decode 有额外依赖）
    # 4. DP attention（数据并行 attention 有同步需求）
    # 5. Pipeline 并行
    # 6. Prefill-only batch（prefill 的后处理可能很重）
```

**可优化的问题**：
- 当前是 **全有或全无** 的 overlap 决策——要么完全 overlap，要么完全同步
- 没有根据 **batch 大小和类型** 做动态调整
- Prefill batch 被一刀切禁用了 overlap，但小 prefill batch 其实可以 overlap

#### 设计方案

**动态 Overlap 决策**：根据 batch 的特征（大小、类型、预估耗时）决定 overlap 策略。

```python
class OverlapDecisionMaker:
    """动态 Overlap 决策器

    核心思路:
    - GPU forward 耗时 >> CPU 后处理耗时 → overlap 收益大
    - GPU forward 耗时 ≈ CPU 后处理耗时 → overlap 收益小
    - GPU forward 耗时 << CPU 后处理耗时 → 不 overlap（CPU 会拖慢下一步）

    决策因子:
    1. batch_size: batch 越大，GPU forward 越慢，overlap 收益越大
    2. forward_mode: decode vs prefill（prefill 的 CPU 后处理更重）
    3. 历史统计: 动态追踪 GPU/CPU 耗时比
    """

    def __init__(self):
        self._gpu_time_ema = 0.0  # GPU forward 时间的指数移动平均
        self._cpu_time_ema = 0.0  # CPU 后处理时间的指数移动平均
        self._alpha = 0.1  # EMA 系数

    def should_overlap(self, batch, last_batch) -> bool:
        """决定是否对当前 batch 启用 overlap"""
        if batch is None or last_batch is None:
            return False

        # 基本禁用条件
        if self._has_hard_constraints(batch):
            return False

        # 动态决策：GPU/CPU 耗时比 > 阈值时启用
        if self._gpu_time_ema > 0 and self._cpu_time_ema > 0:
            ratio = self._gpu_time_ema / self._cpu_time_ema
            return ratio > 1.5  # GPU 至少比 CPU 慢 1.5x 才值得 overlap

        # 启发式默认决策
        return batch.batch_size() >= 4  # 小 batch 不值得 overlap

    def update_stats(self, gpu_time_ms: float, cpu_time_ms: float):
        self._gpu_time_ema = self._alpha * gpu_time_ms + (1 - self._alpha) * self._gpu_time_ema
        self._cpu_time_ema = self._alpha * cpu_time_ms + (1 - self._alpha) * self._cpu_time_ema
```

#### 你将学到什么

1. **Overlap 调度的完整时序**：`run_batch()` 启动 GPU → `result_queue.append()` 存储 future → `pop_and_process()` CPU 处理上一个结果
2. **`batch.copy()` 的作用**：overlap 模式下 batch 对象会被下一步修改，需要深拷贝
3. **`FutureMap` 机制**：Overlap 模式下 sampling 结果通过 future 占位符传递，异步获取
4. **CUDA 多流编程**：`torch.cuda.Stream` 的使用，forward_stream 和 copy_stream 的分离

#### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/managers/scheduler.py` | 修改 | 集成 `OverlapDecisionMaker` |
| `benchmark/sglang_optimization_study/bench_overlap.py` | 新建 | Overlap 效果量化 |

---

### 优化 8：PD 分离场景下的 Radix Cache 跨实例协同 `[进阶]`

#### SGLang 现状分析

SGLang 已有完整的 PD 分离实现（`srt/disaggregation/`），包括：
- `prefill.py` — Prefill 端逻辑
- `decode.py` — Decode 端逻辑
- `kv_events.py` — KV Cache 事件系统（`BlockStored`、`BlockRemoved`、`AllBlocksCleared`）

**SGLang 的 KV 事件系统**（`kv_events.py`）非常适合做跨实例缓存协同：

```python
# kv_events.py
@dataclass
class BlockStored:
    """Emitted when a block is stored in the cache."""
    token_ids: List[int]
    block_hash: str
    medium: str  # "gpu", "cpu", "disk"

@dataclass
class BlockRemoved:
    """Emitted when a block is removed from the cache."""
    block_hash: str
```

#### 设计方案

利用 SGLang 已有的 KV 事件系统，实现 **Prefill→Decode 的缓存状态同步**：

```python
# 核心思路：
# 1. Prefill 实例 prefill 完成后，发出 BlockStored 事件
# 2. Decode 实例接收事件，在本地 RadixCache 中注册对应的 hash
# 3. 下一个具有相同前缀的请求到达 Decode 时，跳过 KV 传输

class CrossInstanceCacheSync:
    """跨实例缓存状态同步器

    利用 SGLang 的 kv_event_queue 机制：
    - Prefill 实例：RadixCache.insert() 后发出 BlockStored 事件
    - Decode 实例：接收事件，在本地 RadixCache 中注册 hash
    - 后续请求：match_prefix() 命中已注册的 hash → 跳过传输
    """
```

#### 你将学到什么

1. **SGLang 的 KV 事件系统**：`_record_store_event()` / `_record_remove_event()` / `_record_all_cleared_event()` 的触发时机
2. **PD 分离的完整数据流**：Prefill bootstrap → KV 传输 → Decode preallocate → Decode transfer
3. **`hash_value` 在 TreeNode 中的作用**：延迟计算的页哈希，用于跨实例缓存一致性验证

---

### 优化 9：端到端性能分析 Benchmark 框架 `[基础]`

#### 设计方案

构建一个统一的 Benchmark 框架，覆盖所有优化点的效果验证：

```python
# benchmark/sglang_optimization_study/e2e_benchmark.py

"""端到端性能分析框架

目标: 一键运行所有优化点的对比测试，生成统一的报告

测试场景:
1. 单轮对话（短 prompt + 短 output）
2. 多轮对话（长 context，前缀缓存命中场景）
3. 长文档总结（长 prompt，chunked prefill 场景）
4. 代码补全（重复模式多，推测解码场景）
5. 混合负载（上述场景随机混合）

输出指标:
- TTFT (Time To First Token) 分布
- ITL (Inter-Token Latency) 分布
- Throughput (tokens/s)
- GPU 利用率
- 缓存命中率
- 推测解码接受率
"""

class BenchmarkSuite:
    def __init__(self, model_path: str, scenarios: List[str]):
        self.scenarios = scenarios

    def run_scenario(self, name: str, config: dict):
        """运行单个场景"""
        # 1. 启动 SGLang server（指定优化参数）
        # 2. 发送测试请求
        # 3. 收集指标
        # 4. 生成报告

    def compare_configs(self, baseline: dict, optimized: dict):
        """对比基线和优化后的性能"""
        # 两组配置分别运行，对比关键指标
```

---

## 实施路线图

### Phase 1：基础理解与量化（1-2 周）

| 优化点 | 目标 | 优先级 |
|--------|------|--------|
| **优化 2**：调度策略 Benchmark | 量化现有策略差异，建立 baseline | P0 |
| **优化 9**：e2e Benchmark 框架 | 建立统一的效果验证工具 | P0 |
| **优化 6**：Spec Decode 指标 | 理解推测解码全链路 | P1 |

### Phase 2：核心优化实现（2-3 周）

| 优化点 | 目标 | 优先级 |
|--------|------|--------|
| **优化 1**：Adaptive Eviction | 深入理解 RadixCache 驱逐机制 | P0 |
| **优化 4**：SAM Proposer | 复用 vLLM 经验，适配 SGLang | P0 |
| **优化 7**：动态 Overlap | 理解 Overlap 调度时序 | P1 |

### Phase 3：进阶优化（2-3 周）

| 优化点 | 目标 | 优先级 |
|--------|------|--------|
| **优化 3**：缓存预热 | 理解请求全生命周期 | P1 |
| **优化 5**：EAGLE + RadixCache | 理解 EAGLE 的复杂架构 | P2 |
| **优化 8**：PD 跨实例缓存 | 理解 PD 分离完整链路 | P2 |

### 依赖关系

```
优化 9 (Benchmark 框架) ──→ 贯穿所有优化点的效果验证
    │
    ├──→ 优化 2 (调度策略 Benchmark) ──→ 优化 1 (Adaptive Eviction) ──→ 优化 3 (缓存预热)
    │
    ├──→ 优化 6 (Spec Decode 指标) ──→ 优化 4 (SAM Proposer) ──→ 优化 5 (EAGLE + Cache)
    │
    └──→ 优化 7 (动态 Overlap) ──→ 优化 8 (PD 跨实例缓存)
```

---

## 核心学习收获预期

通过这 9 个优化点，你将**系统性地掌握 SGLang 的核心机制**：

| 模块 | 对应优化点 | 深入程度 |
|------|-----------|---------|
| **RadixCache** | 优化 1, 2, 3, 5, 8 | 驱逐策略 → 树结构变异 → lock_ref → HiCache 三级缓存 |
| **Scheduler** | 优化 2, 3, 7 | 调度策略 → PrefillAdder 资源分配 → Overlap 时序 → 抢占机制 |
| **Spec Decode** | 优化 4, 5, 6 | EAGLE draft-verify 链路 → N-gram Trie → CUDA Graph → Bigram Key |
| **PD 分离** | 优化 8 | KV 事件系统 → Bootstrap → Transfer → 跨实例缓存 |
| **内存池** | 优化 1, 5 | ReqToTokenPool → TokenToKVPoolAllocator → 两级内存池架构 |
| **请求生命周期** | 优化 3, 7 | 从 HTTP 请求到 tokenize → schedule → forward → detokenize 全流程 |

---

## 与 vLLM 优化经验的对照

| vLLM 上的优化 | SGLang 对标 | SGLang 的不同之处 |
|--------------|-----------|------------------|
| Segmented LRU | 优化 1 (Adaptive Eviction) | vLLM 是 flat block 级别的双链表；SGLang 是 Radix Tree 叶子节点级别的 heap |
| Cache-Aware Scheduling | 优化 2 (已有 LPM/DFS-Weight) | SGLang 已内置！你的任务是量化对比和微调 |
| 抢占缓存保护 | N/A | SGLang 的抢占通过 priority scheduling 做，释放的是 req_to_token 级别资源而非 block 级别 |
| 缓存预热 | 优化 3 | 类似思路，但 SGLang 有 HiCache 可以利用 CPU/Disk 预存 |
| SuffixTreeProposer | 优化 4 | SGLang 已有 C++ Trie N-gram，SAM 是其增强 |
| 增量 SAM | 优化 4 | 直接复用，接口适配 SGLang 的 proposer 框架 |
| PD V1 适配 | 优化 8 | SGLang 已有完整 PD 分离，聚焦缓存协同 |
| 智能路由 | N/A | SGLang 已有 DataParallelController |
