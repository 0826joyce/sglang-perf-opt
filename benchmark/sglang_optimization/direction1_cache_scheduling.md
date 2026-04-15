# 方向一：调度与缓存协同优化

> 对标 vLLM Prefix Cache 调度经验（Segmented LRU、Cache-Aware Scheduling、抢占缓存保护），针对 SGLang 的 Radix Tree 结构进行适配优化。

## 概述

SGLang 已经内置了 LPM/DFS-Weight 等缓存感知调度策略和 6 种驱逐策略，但在以下方面仍有优化空间：

| 现有能力 | 可优化方向 |
|---------|-----------|
| 6 种驱逐策略（LRU/LFU/FIFO/MRU/FILO/Priority） | 缺少多因子自适应策略，不考虑树结构位置 |
| LPM/DFS-Weight/FCFS/LOF/RANDOM 调度策略 | 缺少系统性量化对比，不清楚各场景最优策略 |
| HiRadixCache GPU→CPU→Disk 三级缓存 | 缓存完全被动，无预热和主动预取机制 |

本方向包含 **3 个优化点**：

```
方向一：调度与缓存协同优化
  ├── 优化 1：Radix Cache 驱逐策略增强 — Adaptive Eviction [核心] ← 已实现
  ├── 优化 2：调度策略量化对比 Benchmark [基础] ← 已实现
  └── 优化 3：缓存预热与主动预取 [进阶]
```

---

## 优化 1：Radix Cache 驱逐策略增强 — Adaptive Eviction `[核心]` `[已实现]`

### 1.1 问题分析

#### SGLang 驱逐机制现状

SGLang 的 `RadixCache.evict()` 方法（`radix_cache.py` L565-592）使用 **heap 排序**对可驱逐叶子节点按策略优先级排序，然后逐个驱逐：

```python
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

现有的 6 种策略（`evict_policy.py`）各有局限：

| 策略 | 驱逐依据 | 局限性 |
|------|---------|--------|
| LRU | `last_access_time` | 不考虑访问频率，一次性长请求可能挤掉高频前缀 |
| LFU | `(hit_count, last_access_time)` | 不考虑树结构，深层叶子和浅层共享前缀同等对待 |
| FIFO | `creation_time` | 完全不考虑访问模式 |
| MRU | `-last_access_time` | 反直觉，仅适用特定场景 |
| FILO | `-creation_time` | 反直觉，仅适用特定场景 |
| Priority | `(priority, last_access_time)` | 需要外部指定 priority 值 |

#### 核心问题

1. **缺少 Segmented 保护机制**：高频共享前缀（如 System Prompt）可能被一次性大量驱逐冲刷掉
2. **不区分树位置**：根附近的共享前缀节点（被大量请求复用）和深层叶子节点（仅被单个请求使用）的驱逐权重相同
3. **单因子决策**：每种策略只依赖一个维度的信息

#### 与 vLLM 的对比

| 维度 | vLLM | SGLang |
|------|------|--------|
| 驱逐粒度 | Flat block 级别（`FreeKVCacheBlockQueue` 双链表），O(1) | Radix Tree 叶子节点级别，heap 排序 O(n log n) |
| 驱逐信息 | 仅 LRU | 丰富元数据（hit_count, last_access_time, creation_time, priority） |
| 保护机制 | Segmented LRU（试用区/保护区） | `lock_ref` 引用计数（粗粒度） |

**结论**：SGLang 有更丰富的节点元数据和树结构信息，完全可以设计比 vLLM 更精细的驱逐策略。

### 1.2 设计方案：AdaptiveStrategy

新增一个 **Adaptive 驱逐策略**，综合考虑多个因子：

```
驱逐优先级 = w_recency × recency_score + w_frequency × frequency_score + w_depth × depth_score
```

| 因子 | 来源 | 含义 | 效果 |
|------|------|------|------|
| **recency_score** | `node.last_access_time` | 最近访问时间 | 越久未访问 → 分数越低 → 越先驱逐 |
| **frequency_score** | `node.hit_count / max_hit_count` | 归一化访问频率 | 访问越少 → 分数越低 → 越先驱逐 |
| **depth_score** | `-depth(node)` | 树深度的负值 | 越深（越接近叶子）→ 分数越低 → 越先驱逐 |

**核心思想**：
- **recency**：保留近期活跃的缓存（类似 LRU）
- **frequency**：保留高频访问的缓存（类似 LFU）
- **depth**：保留根部附近的共享前缀（树结构感知，SGLang 独有优势）

权重默认值 `(0.4, 0.3, 0.3)` 可通过 `--adaptive-eviction-weights` 配置。

### 1.3 实现文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `python/sglang/srt/mem_cache/evict_policy.py` | **修改** | 新增 `AdaptiveStrategy` 类 |
| `python/sglang/srt/mem_cache/radix_cache.py` | **修改** | 注册 `"adaptive"` 策略到 `__init__` 分支 |
| `python/sglang/srt/mem_cache/cache_init_params.py` | 无需改动 | `eviction_policy` 字段已支持字符串传递 |
| `python/sglang/srt/server_args.py` | **修改** | `RADIX_EVICTION_POLICY_CHOICES` 添加 `"adaptive"` |
| `benchmark/sglang_optimization/bench_eviction.py` | **新建** | 对比不同驱逐策略的缓存命中率 |

### 1.4 代码变更详解

#### 1.4.1 `evict_policy.py` — 新增 AdaptiveStrategy

```python
class AdaptiveStrategy(EvictionStrategy):
    """自适应多因子驱逐策略。

    综合考虑：
    - 最近访问时间（recency）
    - 访问频率（frequency）
    - 树深度（depth）——越接近根的节点越可能是共享前缀，驱逐代价越高

    驱逐优先级 = w_recency * recency + w_frequency * frequency + w_depth * depth_score
    值越小越先被驱逐。
    """

    def __init__(self, w_recency=0.4, w_frequency=0.3, w_depth=0.3):
        self.w_recency = w_recency
        self.w_frequency = w_frequency
        self.w_depth = w_depth
        self._max_hit_count = 1

    def get_priority(self, node: "TreeNode") -> float:
        # 因子 1：最近访问时间
        recency_score = node.last_access_time

        # 因子 2：归一化访问频率
        self._max_hit_count = max(self._max_hit_count, node.hit_count + 1)
        frequency_score = node.hit_count / self._max_hit_count

        # 因子 3：树深度（越深越先驱逐）
        depth = self._get_depth(node)
        depth_score = -depth

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

#### 1.4.2 `radix_cache.py` — 注册新策略

```python
# 在 __init__ 的策略分支中添加
elif self.eviction_policy == "adaptive":
    self.eviction_strategy: EvictionStrategy = AdaptiveStrategy()
```

#### 1.4.3 `server_args.py` — 支持命令行参数

```python
RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "adaptive"]
```

### 1.5 使用方式

```bash
# 启动时指定 adaptive 驱逐策略
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --radix-eviction-policy adaptive

# 对比 baseline（默认 LRU）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --radix-eviction-policy lru
```

### 1.6 学习收获

通过实现此优化，深入理解了：

1. **Radix Tree 驱逐时序**：`evict()` → `_delete_leaf()` → 父节点递归检查 → `_update_leaf_status()`
2. **`lock_ref` 引用计数**：`inc_lock_ref()`/`dec_lock_ref()` 沿叶到根路径更新，`lock_ref > 0` 不可驱逐
3. **`evictable_leaves` 集合维护**：只有 `lock_ref == 0` 且无子节点的 TreeNode 是驱逐候选
4. **SGLang 策略模式扩展**：实现 `EvictionStrategy` 抽象类即可新增策略
5. **`TreeNode` 元数据**：`last_access_time`、`hit_count`、`creation_time`、`priority` 的更新时机

### 1.7 Adaptive Eviction 端到端调用链

下面梳理从 **启动服务 → 请求到达 → 触发驱逐** 的完整路径，标注 `AdaptiveStrategy.get_priority()` 在哪一步被调用。

#### 阶段 A：服务启动 — 策略对象创建

```
用户启动命令:
  python -m sglang.launch_server --radix-eviction-policy adaptive

                    ┌──────────────────────────────────────┐
                    │  1. server_args.py                    │
                    │     RADIX_EVICTION_POLICY_CHOICES     │
                    │     = ["lru", "lfu", "adaptive"]     │
                    │     (L172)                            │
                    │                                      │
                    │  argparse 解析 → ServerArgs 数据类     │
                    │     radix_eviction_policy = "adaptive"│
                    │     (L339, L3191-3197)                │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  2. scheduler.py :: Scheduler.__init__│
                    │     构造 CacheInitParams              │
                    │     eviction_policy=                  │
                    │       server_args.radix_eviction_policy│
                    │     (L632-652)                        │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  3. cache_init_params.py              │
                    │     @dataclass CacheInitParams        │
                    │       eviction_policy: str = "lru"    │
                    │     (L22) — 中转层，透传字符串          │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  4. radix_cache.py :: RadixCache.__init__│
                    │     (L262-310)                        │
                    │                                      │
                    │  self.eviction_policy = "adaptive"    │
                    │  ...                                  │
                    │  elif self.eviction_policy == "adaptive":│
                    │      self.eviction_strategy =         │
                    │          AdaptiveStrategy()  ← 创建!  │
                    │  (L302-303)                           │
                    │                                      │
                    │  self.evictable_leaves = set()        │
                    └──────────────────────────────────────┘
```

**关键文件链**：`server_args.py` → `scheduler.py` → `cache_init_params.py` → `radix_cache.py` → `evict_policy.py`

#### 阶段 B：请求到达 — Prefill 路径触发驱逐

当一个 HTTP 请求到达后，经过 TokenizerManager 分词，进入 Scheduler 的 waiting_queue。Scheduler 在事件循环中调用 `get_new_batch_prefill()` 组装 prefill batch 时，可能触发驱逐：

```
  Scheduler 事件循环
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │ 5. scheduler.py :: get_new_batch_prefill()           │
  │    → _get_new_batch_prefill_raw()                    │
  │    (L1929-1944, L1946-2078)                          │
  │                                                     │
  │    5a. 对 waiting_queue 按调度策略排序                  │
  │        policy.calc_priority(waiting_queue, ...)      │
  │        (L1982)                                       │
  │                                                     │
  │    5b. 遍历 waiting_queue, 对每个 req:                │
  │        req.init_next_round_input(self.tree_cache)    │
  │        (L2064)                                       │
  │           ↓                                          │
  │    ┌─────▼──────────────────────────────────┐        │
  │    │ schedule_batch.py :: Req.init_next_round_input│  │
  │    │ (L888-941)                              │        │
  │    │                                         │        │
  │    │ tree_cache.match_prefix(                │        │
  │    │   MatchPrefixParams(key=RadixKey(...))) │  ← 查缓存│
  │    │ (L904-910)                              │        │
  │    │                                         │        │
  │    │ → self.prefix_indices = 命中的 KV 索引   │        │
  │    │ → self.last_node = 匹配到的最深树节点    │        │
  │    └─────────────────────────────────────────┘        │
  │                                                     │
  │    5c. adder.add_one_req(req, ...)                   │
  │        (L2065)                                       │
  │        → PrefillAdder 检查 rem_total_tokens          │
  │          (可用 + 可驱逐 token 是否够用)                │
  │        → inc_lock_ref(req.last_node) 锁定缓存节点     │
  │        → 将 req 加入 can_run_list                    │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 6. schedule_batch.py :: ScheduleBatch.prepare_for_extend│
  │    → 调用 alloc_for_extend(batch)                    │
  │    (L1490-1508)                                      │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 7. common.py :: alloc_for_extend()                   │
  │    (L330-391)                                        │
  │                                                     │
  │    page_size == 1 时:                                │
  │      alloc_token_slots(tree_cache, extend_num_tokens)│
  │      (L359)                                          │
  │    page_size > 1 时:                                 │
  │      alloc_paged_token_slots_extend(...)              │
  │      (L366)                                          │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 8. common.py :: alloc_token_slots() / alloc_paged_...│
  │    (L201-226 / L255-294)                             │
  │                                                     │
  │    evict_from_tree_cache(tree_cache, num_tokens)      │
  │    (L207 / L268)                                     │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 9. common.py :: evict_from_tree_cache()              │
  │    (L229-252)                                        │
  │                                                     │
  │    if allocator.available_size() < num_tokens:       │
  │        tree_cache.evict(                             │
  │            EvictParams(num_tokens=num_tokens))       │
  │    (L251-252) ← 仅当内存不足时才真正驱逐              │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 10. radix_cache.py :: RadixCache.evict()             │
  │     (L568-595)                                       │
  │                                                     │
  │  ★★★ AdaptiveStrategy 在这里被调用 ★★★               │
  │                                                     │
  │  leaves = list(self.evictable_leaves)                │
  │  eviction_heap = [                                   │
  │    (self.eviction_strategy.get_priority(node), node) │ ← 调用 AdaptiveStrategy.get_priority()
  │    for node in leaves                                │
  │  ]                                                   │
  │  heapq.heapify(eviction_heap)                        │
  │                                                     │
  │  while num_evicted < num_tokens:                     │
  │      _priority, x = heapq.heappop(eviction_heap)     │ ← 优先级最低的先出堆
  │      allocator.free(x.value)     # 释放 KV 显存       │
  │      self._delete_leaf(x)        # 从树中删除叶子      │
  │                                                     │
  │      # 父节点变成新叶子时，重新计算优先级               │
  │      if x.parent.children == {} and lock_ref == 0:   │
  │          new_priority =                              │
  │            self.eviction_strategy.get_priority(       │ ← 再次调用!
  │              x.parent)                               │
  │          heapq.heappush(eviction_heap,               │
  │            (new_priority, x.parent))                  │
  └──────────────────────────────────────────────────────┘
```

#### 阶段 C：Decode 路径触发驱逐

除了 Prefill，Decode 阶段每一步也需要分配 1 个 token 的 KV 存储：

```
  Scheduler :: run_batch() → 每个 decode step
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │ 11. schedule_batch.py :: ScheduleBatch               │
  │     .prepare_for_decode()                            │
  │     → alloc_for_decode(batch, token_per_req=1)       │
  │     (L1993)                                          │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │ 12. common.py :: alloc_for_decode()                  │
  │     (L423-453)                                       │
  │     → alloc_token_slots(tree_cache, bs)              │
  │       → evict_from_tree_cache(...)                   │ ← 同阶段B的步骤8-10
  │         → tree_cache.evict(...)                      │
  │           → AdaptiveStrategy.get_priority()          │
  └─────────────────────────────────────────────────────┘
```

#### 阶段 D：Decode 期间内存检查触发驱逐

每次 decode step 结束后，Scheduler 会检查下一步是否还有足够内存：

```
  ┌─────────────────────────────────────────────────────┐
  │ 13. schedule_batch.py :: ScheduleBatch               │
  │     .check_decode_mem()                              │
  │     (L1835-1837)                                     │
  │                                                     │
  │     num_tokens = new_tokens_required_next_decode()   │
  │     evict_from_tree_cache(self.tree_cache, num_tokens)│ ← 同步骤9
  │     return allocator.available_size() >= num_tokens  │
  └─────────────────────────────────────────────────────┘
```

#### 阶段 E：Retract（抢占回退）触发驱逐

当内存严重不足需要回退请求时：

```
  ┌─────────────────────────────────────────────────────┐
  │ 14. schedule_batch.py :: ScheduleBatch.release_req() │
  │     (L1913-1918)                                     │
  │                                                     │
  │     release_kv_cache(req, tree_cache, is_insert=False)│
  │     evict_from_tree_cache(tree_cache, num_tokens)    │ ← 同步骤9
  └─────────────────────────────────────────────────────┘
```

#### 完整调用链总结

```
                         启动阶段（一次性）
┌──────────────────────────────────────────────────────────────┐
│ ServerArgs("adaptive")                                        │
│   → Scheduler.__init__()                                      │
│     → CacheInitParams(eviction_policy="adaptive")             │
│       → RadixCache.__init__()                                 │
│         → self.eviction_strategy = AdaptiveStrategy()         │
└──────────────────────────────────────────────────────────────┘

                       请求处理阶段（每次请求）
┌──────────────────────────────────────────────────────────────┐
│ 1. Req.init_next_round_input()                                │
│    → tree_cache.match_prefix() ← 查缓存命中（更新 last_access_time）│
│                                                              │
│ 2. PrefillAdder.add_one_req()                                 │
│    → inc_lock_ref(last_node) ← 锁定命中节点防驱逐               │
│    → 检查 rem_total_tokens（available + evictable）             │
│                                                              │
│ 3. alloc_for_extend() / alloc_for_decode()                    │
│    → alloc_token_slots() / alloc_paged_token_slots_*()        │
│      → evict_from_tree_cache()                                │
│        → if available < needed:                               │
│            tree_cache.evict(EvictParams(...))                  │
│              ┌──────────────────────────────────────────┐     │
│              │ ★ AdaptiveStrategy.get_priority(node)    │     │
│              │   = 0.4 * last_access_time               │     │
│              │   + 0.3 * (hit_count / max_hit_count)    │     │
│              │   + 0.3 * (-depth)                       │     │
│              │                                          │     │
│              │ → 值最小的节点优先被驱逐                    │     │
│              │ → 深层 + 低频 + 久未访问 = 最先驱逐        │     │
│              │ → 浅层 + 高频 + 近期活跃 = 最后驱逐（保护）│     │
│              └──────────────────────────────────────────┘     │
│                                                              │
│ 4. 请求完成 → cache_finished_req()                            │
│    → insert() 将 KV 写回 RadixCache                           │
│    → dec_lock_ref(last_node) ← 解锁，节点可能进入 evictable_leaves│
└──────────────────────────────────────────────────────────────┘
```

#### 关键结论

| 问题 | 答案 |
|------|------|
| **AdaptiveStrategy 在哪一步被创建？** | 服务启动时 `RadixCache.__init__()` 根据 `eviction_policy="adaptive"` 创建，全局唯一实例 |
| **什么时候触发 `get_priority()`？** | 当 KV cache 空间不足时，由 `evict_from_tree_cache()` → `RadixCache.evict()` 调用 |
| **Prefill 和 Decode 都会触发吗？** | 是的。Prefill 通过 `alloc_for_extend()`，Decode 通过 `alloc_for_decode()`，都经过 `evict_from_tree_cache()` |
| **如果内存充足会调用策略吗？** | **不会**。`evict_from_tree_cache()` 先检查 `available_size() < num_tokens`，充足则直接返回 |
| **`match_prefix()` 与驱逐的关系？** | `match_prefix()` 不触发驱逐，但它更新 `last_access_time` 和 `hit_count`，影响后续驱逐的优先级计算 |
| **`lock_ref` 如何保护节点？** | `inc_lock_ref()` 将节点从 `evictable_leaves` 移除；只有 `lock_ref == 0` 的叶子节点才是驱逐候选 |

### 1.8 预期效果

| 场景 | LRU | LFU | Adaptive |
|------|-----|-----|----------|
| 共享 System Prompt + 短对话 | System Prompt 可能被冲刷 | 频率保护，但不区分树位置 | 频率 + 树深度双重保护 |
| 混合负载（短对话 + 长文档） | 长文档挤掉短对话缓存 | 短对话频率高但节点深 | 综合权衡，缓存命中率↑ 10-20% |
| 多模型/多 LoRA（extra_key 隔离） | 各 namespace 独立 LRU | 各 namespace 独立 LFU | 各 namespace 独立 Adaptive |

---

## 优化 2：调度策略量化对比 Benchmark `[基础]` `[已实现]`

### 2.1 目标

系统对比 SGLang 所有调度策略在不同工作负载下的表现，建立性能 baseline。

### 2.2 SGLang 调度策略现状

调度策略定义在 `schedule_policy.py` 的 `calc_priority()` 方法中，分两大类：

**Cache-Aware（缓存感知）策略**：

| 策略 | 排序依据 | 适用场景 |
|------|---------|---------|
| **LPM** (Longest Prefix Match) | `match_prefix()` 返回的前缀命中长度，降序 | 共享前缀场景 |
| **DFS-Weight** | Radix Tree 的 DFS 权重（子树引用计数） | 高缓存复用场景 |

**Cache-Agnostic（缓存无关）策略**：

| 策略 | 排序依据 | 适用场景 |
|------|---------|---------|
| **FCFS** (First Come First Served) | 到达顺序（`wait_queue_entry_time`） | 通用/低开销场景 |
| **LOF** (Longest Output First) | 预估输出长度（`max_new_tokens`） | 减少 HoL 阻塞 |
| **RANDOM** | 随机 | Baseline |
| **ROUTING-KEY** | 路由键频率 | 多租户场景 |

**关键发现**：
- LPM 在队列 > 128 时退化为 FCFS（`_determine_active_policy()` L158-162），因为 `match_prefix()` 的 O(n) 开销
- Cache-Aware 策略会在 `calc_priority()` 中调用 `_compute_prefix_matches()`，同时执行 in-batch prefix caching 检测
- 如果 tree_cache 被禁用，所有 CacheAware 策略自动降级为 FCFS

### 2.3 调度策略调用链

```
Scheduler 事件循环
     │
     ▼
 get_new_batch_prefill()  (scheduler.py L1929)
   └─ _get_new_batch_prefill_raw()  (L1946)
        │
        ├─ self.policy.calc_priority(self.waiting_queue, self.running_batch)
        │    (L1982) ← 对 waiting_queue 就地排序
        │    │
        │    ├─ FCFS: 不排序（保持 FIFO）或按 priority + wait_queue_entry_time
        │    ├─ LPM:  _compute_prefix_matches() → _sort_by_longest_prefix()
        │    │         对每个 req 调用 tree_cache.match_prefix() 获取命中长度
        │    │         按 -len(prefix_indices) 排序（命中最长的优先）
        │    ├─ DFS-Weight: _compute_prefix_matches() → _sort_by_dfs_weight()
        │    │         按树的 DFS 遍历权重排序
        │    ├─ LOF:   按 -max_new_tokens 排序
        │    ├─ RANDOM: random.shuffle()
        │    └─ ROUTING-KEY: 按与 running_batch 共享 routing_key 的频率排序
        │
        ├─ 创建 PrefillAdder  (L1999)
        │
        └─ for req in self.waiting_queue:  (L2022)
             req.init_next_round_input(self.tree_cache)  # 查缓存
             adder.add_one_req(req, ...)  # 加入 batch
```

### 2.4 Benchmark 设计

```python
# benchmark/sglang_optimization/bench_scheduling.py

测试维度（4 种工作负载）:
1. shared_prefix  — 100 个请求共享 400-token System Prompt
2. no_sharing     — 100 个独立请求（100~300 tokens）
3. mixed          — 50% 共享 + 50% 独立，随机打乱
4. long_queue     — 500+ 请求，测试 LPM→FCFS 退化

度量指标:
- 缓存命中率（hit tokens / total tokens）
- 调度延迟（calc_priority 耗时，ms）
- 是否触发 prefix_computed（CacheAware 策略是否实际计算前缀）

对比策略:
- lpm, dfs-weight, fcfs, lof, random
```

### 2.5 Benchmark 实现

Benchmark 的核心模拟流程：

```
Phase 1: 缓存预热（前 30% 请求）
  ├─ 对每个 warmup 请求: match_prefix() → insert()
  └─ 建立前缀树基础结构

Phase 2: 构建等待队列（剩余 70% 请求）
  └─ 创建 MockReq 对象，携带 origin_input_ids, sampling_params 等

Phase 3: 调度排序
  ├─ 实例化 SchedulePolicy(policy_name, tree_cache)
  ├─ 计时: calc_priority(waiting_queue)
  └─ 测量 prefix_computed（LPM/DFS-Weight 会为 True）

Phase 4: 模拟 Prefill
  ├─ 按排序后的顺序遍历 waiting_queue
  ├─ 对每个 req: match_prefix() 测量实际命中
  ├─ insert() 写回缓存（后续请求可以受益）
  └─ 累计 hit_tokens / total_tokens
```

输出示例：

```
====================================================================================================
Scheduling Policy Benchmark  |  requests=200  cache_slots=8192
====================================================================================================
[         lpm] shared_prefix    | reqs= 140  hit_rate=82.15%  sched_lat=   1.23ms  [prefix_computed]
[  dfs-weight] shared_prefix    | reqs= 140  hit_rate=80.42%  sched_lat=   1.45ms  [prefix_computed]
[        fcfs] shared_prefix    | reqs= 140  hit_rate=65.30%  sched_lat=   0.02ms  [ no-pfx]
[         lof] shared_prefix    | reqs= 140  hit_rate=63.88%  sched_lat=   0.15ms  [ no-pfx]
[      random] shared_prefix    | reqs= 140  hit_rate=64.10%  sched_lat=   0.05ms  [ no-pfx]
...
Summary: Cache Hit Rate by (Policy x Workload)
      Policy    shared_prefix       no_sharing            mixed       long_queue
         lpm          82.15%           12.18%           48.63%           42.10%
  dfs-weight          80.42%           11.54%           47.20%           44.30%
        fcfs          65.30%           12.18%           39.87%           38.50%
         lof          63.88%           11.90%           38.40%           37.80%
      random          64.10%           12.05%           39.10%           38.20%

Key Findings:
  ! LPM degraded to FCFS on long_queue (queue_size > 128): prefix_computed=False
  * Best hit rate on shared_prefix: lpm (82.15%)
```

（以上数值为示意，实际需运行测得）

### 2.6 实现文件

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `benchmark/sglang_optimization/bench_scheduling.py` | **新建** | 调度策略对比 Benchmark |

### 2.7 实现要点

1. **MockReq**：轻量级请求 mock，只携带 `SchedulePolicy` 排序所需字段（`rid`, `origin_input_ids`, `prefix_indices`, `last_node`, `sampling_params.max_new_tokens`, `time_stats.wait_queue_entry_time`）
2. **不需要 SGLang Server**：直接实例化 `RadixCache` + `SchedulePolicy`，与 `bench_eviction.py` 模式一致
3. **真实 `calc_priority()` 调用**：使用 SGLang 原生的 `SchedulePolicy` 类，CacheAware 策略会真正调用 `tree_cache.match_prefix()` 和排序
4. **In-batch prefix caching**：LPM 策略内部会使用 `waiting_queue_radix_tree` 检测队列内部的前缀共享，并对重复前缀请求降优先级

### 2.8 学习收获

1. **`match_prefix()` 完整路径**：`_match_prefix_helper()` → `_split_node()`（节点分裂）→ 返回 `MatchResult(device_indices, last_device_node, last_host_node)`
2. **In-batch prefix caching 机制**：`_compute_prefix_matches()` 中使用 `waiting_queue_radix_tree`（模拟 RadixCache，无实际 KV）检测队列内部前缀共享，当前缀命中 > `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD`(32) 时降优先级
3. **LPM 退化机制**：`_determine_active_policy()` 在 `len(waiting_queue) > 128` 时退化为 FCFS
4. **策略分类设计**：`CacheAwarePolicy` vs `CacheAgnosticPolicy` 枚举分离，`_validate_and_adjust_policy()` 在 tree_cache 禁用时自动降级
5. **DFS-Weight 排序**：通过 `_calc_weight()` 递归计算子树中等待请求数量，`_get_dfs_priority()` 按 DFS 顺序展开——权重大的子树先遍历，实现缓存局部性最大化
6. **FCFS 的时间来源**：使用 `req.time_stats.wait_queue_entry_time` 而非独立的 arrival_time 字段

---

## 优化 3：缓存预热与主动预取 `[进阶]`

### 3.1 目标

在系统冷启动或空闲时，主动将高频 System Prompt 的 KV Cache 预计算并存入 RadixCache。

### 3.2 SGLang HiRadixCache 现状

SGLang 的 `HiRadixCache`（`hiradix_cache.py`）支持 GPU→CPU→Disk 三级缓存，有 `load_back()` 方法将 CPU 上的 KV 数据加载回 GPU。但缓存完全被动——只有请求到达时才触发 `match_prefix()`。

### 3.3 设计方案

```python
# 新文件: python/sglang/srt/mem_cache/cache_warming.py

class CacheWarmingManager:
    """缓存预热管理器

    工作流程:
    1. 从配置文件加载常见 System Prompt（token IDs）
    2. 系统空闲时，构造虚拟请求执行 prefill
    3. 将计算好的 KV Cache 写入 RadixCache
    4. 后续真实请求命中预热缓存后跳过 prefill

    触发条件:
    - 系统启动后初始化阶段
    - 运行中检测到空闲（waiting_queue 和 running_batch 都为空）
    - HiCache 场景：从 CPU/Disk 预取即将到来的前缀
    """
```

### 3.4 与 HiCache 协同

如果启用了 HiCache，预热可以先在 CPU 上准备 KV 数据，然后按需加载到 GPU，避免启动时大量 GPU 计算。

### 3.5 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `python/sglang/srt/mem_cache/cache_warming.py` | 新建 | 预热管理器 |
| `python/sglang/srt/managers/scheduler.py` | 修改 | `self_check_during_idle()` 中触发预热 |
| `python/sglang/srt/server_args.py` | 修改 | 新增 `--warmup-config` 参数 |

### 3.6 学习收获

1. **Scheduler 请求全生命周期**：`process_input_requests()` → `_add_request_to_queue()` → `get_new_batch_prefill()` → `run_batch()` → `cache_finished_req()`
2. **RadixCache 写入路径**：`cache_finished_req()` → `insert()` → `_insert_helper()`
3. **HiRadixCache 三级缓存交互**：`write_backup()` → `cache_controller.write()` → `load_back()` → `cache_controller.load()`

---

## 实施状态

| 优化点 | 状态 | 说明 |
|--------|------|------|
| 优化 1：Adaptive Eviction | ✅ 已实现 | 新增 `AdaptiveStrategy`，修改 3 个文件 |
| 优化 2：调度策略 Benchmark | ✅ 已实现 | 新建 `bench_scheduling.py`，对比 5 种策略 × 4 种工作负载 |
| 优化 3：缓存预热 | 📋 待实现 | 依赖对 Scheduler 空闲检测的深入理解 |

---

## 与 vLLM 优化经验的对照

| vLLM 上的优化 | 本方向对标 | SGLang 的不同之处 |
|--------------|-----------|------------------|
| **Segmented LRU** | 优化 1 (Adaptive Eviction) | vLLM 是 flat block 级别的双链表；SGLang 是 Radix Tree 叶子节点级别的 heap，拥有更丰富的节点元数据（hit_count, depth, tree structure） |
| **Cache-Aware Scheduling** | 优化 2 (已有 LPM/DFS-Weight) | SGLang 已内置！任务是量化对比和微调 |
| **抢占缓存保护** | N/A | SGLang 通过 `lock_ref` 引用计数保护活跃节点，抢占通过 `retract_decode()` 释放 `req_to_token` 级别资源 |
| **缓存预热** | 优化 3 | 类似思路，但 SGLang 有 HiCache 可利用 CPU/Disk 预存 |
