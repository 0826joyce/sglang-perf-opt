# 方向三：Overlap 调度与 PD 分离深度优化

> 在 vLLM 上积累的 PD 分离全套优化经验，在 SGLang 上可以**利用已有的 disaggregation 模块和 overlap 调度**，聚焦更高级的优化。

## 概述

SGLang 已经内置了完整的 Overlap 调度和 PD 分离框架：

| 现有能力 | 可优化方向 |
|---------|-----------|
| `event_loop_overlap()` — CUDA 多流 GPU/CPU 重叠调度 | 当前是全有或全无的 overlap 决策，没有根据 batch 特征做动态调整 |
| `FutureMap` — 基于环形缓冲区的异步采样结果传递 | Prefill batch 被一刀切禁用 overlap，但小 prefill batch 可以 overlap |
| 完整的 `disaggregation/` 模块（prefill/decode 分离） | PD 分离场景下 Prefill→Decode 的 KV 状态缺少 RadixCache 级别的协同 |
| KV 事件系统（`BlockStored`/`BlockRemoved`/`AllBlocksCleared`） | 缺少端到端的性能分析 Benchmark 框架 |

本方向包含 **3 个优化点**：

```
方向三：Overlap 调度与 PD 分离深度优化
  ├── 优化 7：Overlap 调度精细化 — 动态 Overlap 决策 [核心] ← ✅ 已实现
  ├── 优化 8：PD 分离场景下的 Radix Cache 跨实例协同 [进阶] ← ✅ 已实现
  └── 优化 9：端到端性能分析 Benchmark 框架 [基础] ← 📋 待实现
```

---

## SGLang Overlap 调度架构总览

### 事件循环选择

SGLang 根据运行模式选择不同的事件循环：

```
run_scheduler_process()
  └── 选择事件循环:
      ├── pdmux          → event_loop_pdmux()
      ├── pp > 1         → event_loop_pp()
      ├── enable_overlap → event_loop_overlap()        ← 默认启用
      └── else           → event_loop_normal()
```

默认情况下 `disable_overlap_schedule = False`，即 overlap 模式是**默认开启**的。

### Normal 模式 vs Overlap 模式

**Normal 模式**（同步）：

```
while True:
    recv_reqs → process_input_requests
    batch = get_next_batch_to_run()
    if batch:
        result = run_batch(batch)          # GPU forward（阻塞等待）
        process_batch_result(batch, result) # CPU 后处理（GPU 空闲）
    last_batch = batch
```

**Overlap 模式**（异步重叠）：

```
while True:
    recv_reqs → process_input_requests
    batch = get_next_batch_to_run()

    if disable_overlap_for_batch:
        pop_and_process()                  # 立即处理上一个结果

    if batch:
        result = run_batch(batch)          # 当前 batch 启动 GPU forward
        result_queue.append((batch, result))

    if last_batch and not disable_overlap:
        pop_and_process()                  # 与 GPU forward 重叠处理上一个结果

    launch_batch_sample_if_needed()        # 延迟采样（依赖上一个 batch）
    last_batch = batch
```

### 关键组件

#### 1. FutureMap（`overlap_utils.py`）

环形缓冲区，存储异步的采样结果。Overlap 模式下，当前 batch 的 `input_ids` 中可能包含"未来值"（负数索引），在 GPU forward 开始时通过 `resolve_future()` 替换为真实值。

```python
class FutureMap:
    def alloc_future_indices(bs)    # 分配环形缓冲区位置
    def resolve_future(batch)       # 在 forward stream 中解析未来值
    def store_to_map(indices, result) # 存储结果到缓冲区
```

#### 2. CUDA 多流

```python
self.default_stream    # CPU 调度流（准备 batch、处理结果）
self.forward_stream    # GPU 前向计算流
self.copy_stream       # GPU→CPU 数据拷贝流
```

#### 3. `is_disable_overlap_for_batch()` — 当前的禁用逻辑

```python
def is_disable_overlap_for_batch(self, batch):
    # 条件 1：两个连续 prefill batch 禁用 overlap（改善 TTFT）
    disable = (SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
               and batch.is_extend() and last_batch.is_extend())

    # 条件 2：spec + grammar 不支持 overlap
    need_grammar_sync = (batch.is_spec_v2 and batch.has_grammar
                        and batch.is_decode() and len(result_queue) > 0)

    return disable or need_grammar_sync
```

**问题**：当前决策是"全有或全无"，没有根据 batch 大小、GPU/CPU 耗时比等动态因素做细粒度调整。

---

## 优化 7：Overlap 调度精细化 — 动态 Overlap 决策 `[核心]` `[✅ 已实现]`

### 问题分析

当前 `is_disable_overlap_for_batch()` 的决策逻辑过于粗糙：

1. **全有或全无**：要么完全 overlap，要么完全同步
2. **不考虑 batch 大小**：小 batch（如 1-2 个请求）的 GPU forward 很快，overlap 的收益可能不抵 FutureMap 的开销
3. **不考虑历史统计**：不同负载模式下 GPU/CPU 耗时比差异很大

### 设计方案

引入 `OverlapDecisionMaker`，基于以下因子做动态决策：

1. **batch 大小**：batch 越大 GPU forward 越慢，overlap 收益越大
2. **forward mode**：decode 通常比 prefill 的 CPU 后处理更轻
3. **历史 GPU/CPU 耗时比**：使用 EMA（指数移动平均）动态追踪

### 实现

**新增文件**: `python/sglang/srt/managers/overlap_decision.py`

核心类 `OverlapDecisionMaker`：
- `should_overlap(batch, last_batch)` — 综合硬约束和动态统计做决策
- `update_stats(gpu_time_ms, cpu_time_ms)` — 用 EMA 更新耗时统计
- 可通过 `--enable-dynamic-overlap` 启用

**修改文件**:
- `python/sglang/srt/server_args.py` — 新增 `enable_dynamic_overlap` 参数
- `python/sglang/srt/managers/scheduler.py` — 集成 `OverlapDecisionMaker`

### 工作原理

```
┌─────────────────────────────────────┐
│      OverlapDecisionMaker           │
│                                     │
│  硬约束检查:                         │
│  ├── batch/last_batch 为 None?      │
│  ├── 连续 prefill overlap 禁用?     │
│  └── spec + grammar 不兼容?         │
│                                     │
│  动态决策 (通过硬约束后):             │
│  ├── gpu_time_ema / cpu_time_ema    │
│  │   > ratio_threshold? → overlap   │
│  ├── batch_size >= min_bs? → overlap│
│  └── 否则 → 同步执行                │
│                                     │
│  统计更新 (每次 batch 完成后):       │
│  └── EMA 更新 gpu_time, cpu_time    │
└─────────────────────────────────────┘
```

### 预期效果

| 场景 | 当前行为 | 优化后 |
|------|---------|-------|
| 大 decode batch (bs≥8) | overlap ✅ | overlap ✅ （不变） |
| 小 decode batch (bs=1-2) | overlap ✅ （可能浪费） | 同步执行 ✅ （减少 FutureMap 开销） |
| 大 prefill batch | 同步（连续prefill禁用） | 同步 ✅ （不变） |
| 小 prefill + 大 decode | overlap ✅ | overlap ✅ （不变） |
| GPU 快 CPU 慢 | overlap ✅ （CPU 拖慢下一步） | 同步执行 ✅ （避免拖延） |

---

## 优化 8：PD 分离场景下的 Radix Cache 跨实例协同 `[进阶]` `[✅ 已实现]`

### SGLang 现状分析

SGLang 已有完整的 PD 分离实现（`srt/disaggregation/`），包括：
- `prefill.py` — Prefill 端逻辑
- `decode.py` — Decode 端逻辑
- `kv_events.py` — KV Cache 事件系统（`BlockStored`、`BlockRemoved`、`AllBlocksCleared`）

**关键发现**：
1. **Decode 侧强制使用 chunk cache**（`disable_radix_cache = True`），不使用 RadixCache
2. **KV 事件系统有完整的 ZMQ PUB/SUB 基础设施**，但目前仅用于 metrics 上报
3. **Prefill 侧 RadixCache 的 `_record_store_event` 已有完善的 SHA256 链式哈希机制**

### 设计方案

由于 Decode 侧使用 chunk cache（非 RadixCache），我们采用**轻量级哈希注册表**方案：

```
┌─────────────────────┐        CacheSyncEvent        ┌─────────────────────┐
│   Prefill Instance   │  ─── PrefixCacheStored ───>  │   Decode Instance    │
│                      │  ─── PrefixCacheRemoved ──>  │                      │
│ PrefillCacheState    │  ─── PrefixCacheCleared ──>  │ CacheHashRegistry    │
│    Publisher         │                              │   _known_hashes      │
│                      │                              │   _prefix_chains     │
│ RadixCache           │                              │   _token_prefix_index│
│  cache_unfinished_req│                              │                      │
│  → _publish_cross_*  │                              │ estimate_prefix_hit()│
└─────────────────────┘                              └─────────────────────┘
```

核心类 `CrossInstanceCacheSync`：
- **Prefill 侧**：`PrefillCacheStatePublisher` — 在 `process_batch_result_disagg_prefill` 中，`cache_unfinished_req` 之后，遍历 RadixCache 节点路径，提取 SHA256 哈希链并发布
- **Decode 侧**：`CacheHashRegistry` — 维护轻量级哈希集合（~16 bytes/block），支持 `estimate_cached_prefix_length()` 查询，在 `process_decode_queue` 中注解请求的 prefix hit 长度

### 实现

**新增文件**: `python/sglang/srt/disaggregation/cross_instance_cache_sync.py`

核心组件：
- `PrefixCacheStored`/`PrefixCacheRemoved`/`PrefixCacheCleared` — 同步事件类型
- `PrefillCacheStatePublisher` — Prefill 侧事件收集器
- `CacheHashRegistry` — Decode 侧哈希注册表
  - `register_blocks()` — 注册来自 Prefill 的哈希链
  - `remove_blocks()` — 响应驱逐事件
  - `estimate_cached_prefix_length()` — 计算请求的 prefix 命中长度
- `CrossInstanceCacheSync` — 统一入口（根据 mode 选择 Publisher 或 Registry）

**修改文件**:
- `python/sglang/srt/server_args.py` — 新增 `enable_cross_instance_cache_sync` 和 `cross_instance_cache_sync_max_entries`
- `python/sglang/srt/managers/scheduler.py` — 在 `init_disaggregation` 中初始化 `CrossInstanceCacheSync`
- `python/sglang/srt/disaggregation/prefill.py` — 在 `process_batch_result_disagg_prefill` 中添加 `_publish_cross_instance_cache_state()`，遍历 TreeNode 路径提取哈希链
- `python/sglang/srt/disaggregation/decode.py` — 在 `process_decode_queue` 中添加 `_annotate_prefix_cache_hits()`，为传输完成的请求注解 prefix hit 长度
- `python/sglang/srt/managers/schedule_batch.py` — `Req` 类新增 `cross_instance_prefix_hit_len` 属性
- `python/sglang/srt/managers/scheduler_metrics_mixin.py` — 新增 `_publish_cross_instance_cache_sync_events()`
- `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py` — 调用发布方法

### 工作原理

```
Prefill 侧（process_batch_result_disagg_prefill）:
┌──────────────────────────────────────────────────┐
│ 1. req.output_ids.append(next_token_id)          │
│ 2. tree_cache.cache_unfinished_req(req)          │
│    └── RadixCache.insert() → _record_store_event │
│ 3. _publish_cross_instance_cache_state(req)  ← NEW│
│    ├── 遍历 req.last_node → root 收集节点路径     │
│    ├── 计算/获取每个节点的 hash_value              │
│    └── 调用 sync.on_prefix_cached(hashes, tokens) │
│ 4. send_kv_chunk(req, last_chunk=True)           │
└──────────────────────────────────────────────────┘

Decode 侧（process_decode_queue）:
┌──────────────────────────────────────────────────┐
│ 1. pop_preallocated() → extend transfer queue    │
│ 2. pop_transferred() → alloc_reqs                │
│ 3. _annotate_prefix_cache_hits(alloc_reqs)   ← NEW│
│    ├── 对每个 req 调用 estimate_prefix_hit()      │
│    ├── 计算 origin_input_ids 的 SHA256 哈希链     │
│    ├── 逐 page 检查 registry 中是否存在           │
│    └── 设置 req.cross_instance_prefix_hit_len     │
│ 4. waiting_queue.extend(alloc_reqs)              │
└──────────────────────────────────────────────────┘
```

### 哈希计算一致性

Prefill 和 Decode 侧使用完全相同的哈希算法（`get_hash_str` + `hash_str_to_int64`），确保一致性：

```python
# 位置感知的 SHA256 链式哈希
hash_str = get_hash_str(page_tokens, prior_hash=parent_hash_str)
block_hash = hash_str_to_int64(hash_str)  # 取前 16 hex 字符转 int64
```

### 可通过参数控制

```bash
# 启用跨实例缓存同步
python -m sglang.launch_server \
    --disaggregation-mode prefill \
    --enable-cross-instance-cache-sync \
    --cross-instance-cache-sync-max-entries 1000000
```

### 预期效果

| 场景 | 当前行为 | 优化后 |
|------|---------|-------|
| 相同 system prompt 的请求 | 每次都完整传输 KV | Decode 侧知道 prefix 已缓存，可优化调度 |
| 多轮对话（共享前缀） | 无感知，每次重传 | prefix hit 统计可用于调度优先级 |
| 缓存驱逐 | 无感知 | Decode 侧同步移除过期哈希 |
| 监控和分析 | 无跨实例缓存信息 | 提供 hit rate、命中长度等统计 |

### 学习收获

1. **SGLang 的 KV 事件系统**：`_record_store_event()` / `_record_remove_event()` / `_record_all_cleared_event()` 的触发时机
2. **PD 分离的完整数据流**：Prefill bootstrap → KV 传输 → Decode preallocate → Decode transfer
3. **`hash_value` 在 TreeNode 中的作用**：延迟计算的页哈希，用于跨实例缓存一致性验证
4. **Decode 侧的 chunk cache 限制**：不能直接使用 RadixCache，需要轻量级的哈希注册表替代
5. **Mixin 模式的代码组织**：`SchedulerDisaggregationPrefillMixin` 和 `SchedulerDisaggregationDecodeMixin` 通过 `self: Scheduler` 类型注解实现方法绑定

---

## 优化 9：端到端性能分析 Benchmark 框架 `[基础]` `[📋 待实现]`

### 设计方案

构建一个统一的 Benchmark 框架，覆盖所有优化点的效果验证：

```python
# benchmark/sglang_optimization/e2e_benchmark.py

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

### 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `benchmark/sglang_optimization/e2e_benchmark.py` | 新建 | 端到端 Benchmark 框架 |
| `benchmark/sglang_optimization/workloads/` | 新建 | 标准化测试数据集 |

---

## 汇总

| 优化点 | 状态 | 核心改动 |
|--------|------|---------|
| 优化 7：动态 Overlap 决策 | ✅ 已实现 | 基于 EMA 统计的 GPU/CPU 耗时比动态 overlap 决策 |
| 优化 8：PD 跨实例缓存协同 | ✅ 已实现 | 轻量级哈希注册表 + Prefill→Decode 缓存状态同步 |
| 优化 9：e2e Benchmark 框架 | 📋 待实现 | 统一的性能对比和回归测试工具 |
