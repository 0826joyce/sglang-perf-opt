# SGLang Performance Optimization Study

> 基于 [SGLang](https://github.com/sgl-project/sglang) 推理引擎的深度性能优化研究项目。在 vLLM V1 上积累的 PD 分离、Prefix Cache 调度、后缀解码三大优化经验基础上，针对 SGLang 的 Radix Tree 架构特点设计并实现了 **3 个方向、9 个优化点**。

## 项目背景

### 已有经验（vLLM 项目）

| 方向 | 在 vLLM 上做过的 | 核心收获 |
|------|-----------------|---------|
| **PD 分离** | V1 引擎适配 + 智能路由 + 调度器感知 + KV 传输优化 + Prefix Cache 协同 | KV Cache 跨 GPU 传输、调度器与传输层的协调 |
| **Prefix Cache 调度** | Cache-Aware Scheduling + Segmented LRU + 抢占缓存保护 + 预热 | KV Cache block 管理、引用计数、驱逐策略 |
| **后缀解码** | 后缀数组 Proposer + 增量 SAM + 自适应评分 + 跨请求共享 | Spec Decode 全链路、Proposer 接口、RejectionSampler |

### SGLang 与 vLLM 的关键架构差异

| 维度 | vLLM V1 | SGLang |
|------|---------|--------|
| **KV Cache 管理** | `KVCacheManager` — hash chain block 管理 | `RadixCache` — 基数树管理，节点带 `lock_ref` 引用计数 |
| **调度策略** | MLFQ + FCFS（无缓存感知） | LPM / DFS-Weight 缓存感知策略 + in-batch prefix caching |
| **驱逐策略** | 纯 LRU（双链表） | 6 种策略（LRU/LFU/FIFO/MRU/FILO/Priority） |
| **Spec Decode** | V1 仅 N-gram（Python KMP） | EAGLE/EAGLE3/NGRAM/Standalone，C++ Trie + CUDA Graph |
| **PD 分离** | V0 有基础实现 | 完整 `disaggregation/` 模块，支持 RDMA KV 传输 |
| **Overlap 调度** | 无 | `event_loop_overlap()` — CUDA 多流 GPU/CPU 重叠 |

---

## 优化方向总览

```
方向一：调度与缓存协同优化
  ├── 优化 1：Radix Cache 驱逐策略增强 — Adaptive Eviction     ✅ 已实现
  ├── 优化 2：调度策略量化对比 Benchmark                        ✅ 已实现
  └── 优化 3：缓存预热与主动预取                                ✅ 已实现

方向二：推测解码增强
  ├── 优化 4：N-gram Proposer 后缀自动机增强                    ✅ 已实现
  ├── 优化 5：EAGLE 投机解码 + Radix Cache 协同                 ✅ 已实现
  └── 优化 6：推测解码全链路可观测性                            📋 待实现

方向三：Overlap 调度与 PD 分离深度优化
  ├── 优化 7：Overlap 调度精细化 — 动态 Overlap 决策            ✅ 已实现
  ├── 优化 8：PD 分离场景下的 Radix Cache 跨实例协同            ✅ 已实现
  └── 优化 9：端到端性能分析 Benchmark 框架                     ✅ 已实现
```

---

## 方向一：调度与缓存协同优化

> 对标 vLLM Prefix Cache 调度经验，针对 SGLang 的 Radix Tree 结构进行适配。
> 
> 📄 详细文档：[`benchmark/sglang_optimization/direction1_cache_scheduling.md`](benchmark/sglang_optimization/direction1_cache_scheduling.md)

### 优化 1：Adaptive Eviction — 自适应多因子驱逐策略

**问题**：现有 6 种驱逐策略均为单因子决策，不区分树位置（根部共享前缀 vs 深层叶子），高频共享前缀可能被冲刷。

**方案**：新增 `AdaptiveStrategy`，综合三个因子加权评分：

```
驱逐优先级 = w_recency × last_access_time + w_frequency × (hit_count / max) + w_depth × (-depth)
```

- **recency**：保留近期活跃缓存（类 LRU）
- **frequency**：保留高频访问缓存（类 LFU）
- **depth**：保留根部共享前缀（Radix Tree 结构感知，SGLang 独有优势）

**改动文件**：
| 文件 | 说明 |
|------|------|
| `python/sglang/srt/mem_cache/evict_policy.py` | 新增 `AdaptiveStrategy` |
| `python/sglang/srt/mem_cache/radix_cache.py` | 注册 `"adaptive"` 策略 |
| `python/sglang/srt/server_args.py` | 支持 `--radix-eviction-policy adaptive` |

**使用**：
```bash
python -m sglang.launch_server --model-path <model> --radix-eviction-policy adaptive
```

### 优化 2：调度策略量化对比 Benchmark

系统对比 5 种调度策略（LPM / DFS-Weight / FCFS / LOF / RANDOM）× 4 种工作负载（共享前缀 / 无共享 / 混合 / 长队列）的缓存命中率和调度延迟。

**关键发现**：LPM 在队列 > 128 时退化为 FCFS（`_determine_active_policy()` 的性能保护机制）。

**运行（无需 GPU）**：
```bash
python benchmark/sglang_optimization/bench_scheduling.py
```

### 优化 3：缓存预热与主动预取

**问题**：RadixCache 完全被动，冷启动时第一批请求无法命中任何缓存，LPM/DFS-Weight 策略失效。

**方案**：`CacheWarmingManager` 在系统空闲时将高频 System Prompt 的 token IDs 预插入 Radix Tree（Tree-only warming），调度策略可立即感知共享前缀。

- **非阻塞设计**：每次空闲迭代只处理一个 prompt，不阻塞事件循环
- **惰性加载**：首次 `maybe_warm()` 时才读取 JSON 配置
- **渐进生效**：第一个真实请求完成 prefill 后，KV 值自动覆盖 dummy 值

**改动文件**：
| 文件 | 说明 |
|------|------|
| `python/sglang/srt/mem_cache/cache_warming.py` | **新建** CacheWarmingManager |
| `python/sglang/srt/server_args.py` | 新增 `--cache-warming-prompts` |
| `python/sglang/srt/managers/scheduler.py` | 初始化预热管理器 |
| `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py` | 空闲时触发预热 |

**使用**：
```bash
python -m sglang.launch_server --model-path <model> \
    --cache-warming-prompts benchmark/sglang_optimization/example_warmup_prompts.json
```

---

## 方向二：推测解码增强

> 在 vLLM 后缀自动机经验基础上，与 SGLang 已有的 EAGLE 和 C++ N-gram Trie 实现协同。
>
> 📄 详细文档：[`benchmark/sglang_optimization/direction2_spec_decode.md`](benchmark/sglang_optimization/direction2_spec_decode.md)

### 优化 4：N-gram Proposer 后缀自动机增强

**问题**：C++ Trie 的 N-gram 匹配是固定窗口 `[min_window, max_window]`，无可变长度回退，且无 per-request 上下文利用。

**方案**：新增 `SuffixAutomatonProposer`，与现有 `NgramCache` 形成互补：

```
请求到达 → _prepare_draft_tokens()
  ├── 1. SuffixAutomaton.propose()   ← per-request 上下文匹配（可变长度）
  ├── 2. NgramCache.batch_get()      ← 跨请求模式匹配（固定窗口）
  └── 3. 合并候选 → 选择最优 draft tokens
```

- **O(1) 增量更新**：`IncrementalSuffixAutomaton` 每个新 token 均摊 O(1)
- **自适应回退**：从最长 n-gram 逐步缩短直到命中

**使用**：
```bash
python -m sglang.launch_server --model-path <model> \
    --speculative-algorithm NGRAM --speculative-ngram-use-sam
```

### 优化 5：EAGLE + Radix Cache 协同

**问题**：EAGLE verify 被拒绝的 draft tokens 的 KV Cache 被立即释放，后续相似请求需重新计算。

**方案**：将 EAGLE 验证通过的 accepted token 序列写入 RadixCache，供后续请求复用：
- 只有 `accept_length > threshold` 时才缓存，避免开销过大
- 兼容 EAGLE 的 bigram key 模式

### 优化 6：推测解码全链路可观测性（📋 待实现）

新增阶段耗时（draft / tree_build / verify / draft_extend）、per-step 接受率、KV Cache 浪费比等细粒度 Prometheus 指标。

---

## 方向三：Overlap 调度与 PD 分离深度优化

> 利用 SGLang 已有的 disaggregation 模块和 overlap 调度，聚焦更高级的优化。
>
> 📄 详细文档：[`benchmark/sglang_optimization/direction3_overlap_pd.md`](benchmark/sglang_optimization/direction3_overlap_pd.md)

### 优化 7：动态 Overlap 决策

**问题**：当前 `is_disable_overlap_for_batch()` 是"全有或全无"的决策，不考虑 batch 大小和 GPU/CPU 耗时比。

**方案**：`OverlapDecisionMaker` 基于 EMA 动态追踪 GPU/CPU 耗时比，做精细化决策：
- **硬约束优先**：连续 prefill、spec+grammar 等场景强制同步
- **动态决策**：`gpu_time_ema / cpu_time_ema > ratio_threshold` 时启用 overlap
- **启发式默认**：`batch_size >= min_bs_threshold` 时启用

| 场景 | 优化前 | 优化后 |
|------|-------|-------|
| 大 decode batch (bs≥8) | overlap ✅ | overlap ✅ |
| 小 decode batch (bs=1-2) | overlap ✅（可能浪费） | 同步 ✅（减少开销） |
| GPU 快 CPU 慢 | overlap ✅（CPU 拖慢下一步） | 同步 ✅ |

**使用**：
```bash
python -m sglang.launch_server --model-path <model> --enable-dynamic-overlap
```

### 优化 8：PD 分离 — Radix Cache 跨实例协同

**问题**：PD 分离场景下，Decode 侧无法感知 Prefill 侧的 RadixCache 状态，相同前缀每次都重传 KV。

**方案**：轻量级哈希注册表 + SHA256 链式哈希同步：

```
Prefill 实例                               Decode 实例
RadixCache                                 CacheHashRegistry
  │                                           │
  ├── cache_unfinished_req()                  │
  ├── _publish_cross_instance_cache_state()   │
  │   └── PrefixCacheStored ─────────────────►│ register_blocks()
  │                                           │ estimate_cached_prefix_length()
  └── evict() ──── PrefixCacheRemoved ──────►│ remove_blocks()
```

- **Decode 侧**：维护 `_known_hashes` 集合（~16 bytes/block），支持 `estimate_cached_prefix_length()` 查询
- **可用于调度优化**：prefix hit 长度可作为调度优先级参考

**使用**：
```bash
# Prefill 实例
python -m sglang.launch_server --disaggregation-mode prefill \
    --enable-cross-instance-cache-sync

# Decode 实例
python -m sglang.launch_server --disaggregation-mode decode \
    --enable-cross-instance-cache-sync
```

### 优化 9：端到端性能分析 Benchmark 框架

全自动化的 benchmark 框架，一键完成：启动服务器 → 发送请求 → 收集指标 → 关闭服务器 → 换配置重复 → 输出对比报告。

**特性**：
- 支持多配置对比（baseline vs optimized）
- 支持多场景（single_turn / multi_turn / long_document / code_completion / mixed）
- 自动收集 TTFT / ITL / E2E / Throughput / Cache Hit Rate / Spec Accept Length
- 输出格式化表格 + JSON 报告

**使用**：
```bash
python benchmark/sglang_optimization/e2e_benchmark.py \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --configs baseline optimized \
    --scenarios single_turn multi_turn \
    --num-prompts 50
```

---

## 实施状态

| # | 优化点 | 方向 | 状态 | 核心改动 |
|---|--------|------|:----:|---------|
| 1 | Adaptive Eviction | 缓存调度 | ✅ | 多因子加权驱逐策略 |
| 2 | 调度策略 Benchmark | 缓存调度 | ✅ | 5 策略 × 4 负载对比 |
| 3 | 缓存预热 | 缓存调度 | ✅ | Tree-only warming + 非阻塞设计 |
| 4 | SAM Proposer | 推测解码 | ✅ | 后缀自动机 + N-gram Trie 互补 |
| 5 | EAGLE + Cache 协同 | 推测解码 | ✅ | Accepted tokens KV 复用 |
| 6 | Spec Decode 可观测性 | 推测解码 | 📋 | 全链路 Metrics 增强 |
| 7 | 动态 Overlap | PD 分离 | ✅ | EMA 统计 + 动态决策 |
| 8 | 跨实例缓存协同 | PD 分离 | ✅ | 轻量级哈希注册表 |
| 9 | E2E Benchmark | PD 分离 | ✅ | 全自动化性能对比框架 |

## Benchmark 文件

所有 benchmark 脚本位于 `benchmark/sglang_optimization/`：

| 文件 | 说明 | 需要 GPU |
|------|------|:-------:|
| `bench_eviction.py` | 驱逐策略对比（LRU / LFU / Adaptive） | ❌ |
| `bench_scheduling.py` | 调度策略对比（LPM / DFS-Weight / FCFS / LOF / RANDOM） | ❌ |
| `bench_cache_warming.py` | 冷启动 vs 预热缓存命中率对比 | ❌ |
| `bench_spec_decode.py` | N-gram vs SAM Proposer 对比 | ❌ |
| `e2e_benchmark.py` | 端到端全自动性能对比框架 | ✅ |

## 项目结构

```
benchmark/sglang_optimization/
├── optimization_plan.md              # 总体优化计划（9 个优化点概述）
├── direction1_cache_scheduling.md    # 方向一详细文档（调度与缓存协同）
├── direction2_spec_decode.md         # 方向二详细文档（推测解码增强）
├── direction3_overlap_pd.md          # 方向三详细文档（Overlap 与 PD 分离）
├── bench_eviction.py                 # 驱逐策略 Benchmark
├── bench_scheduling.py               # 调度策略 Benchmark
├── bench_cache_warming.py            # 缓存预热 Benchmark
├── bench_spec_decode.py              # 推测解码 Benchmark
├── e2e_benchmark.py                  # 端到端 Benchmark 框架
└── example_warmup_prompts.json       # 预热配置示例
```

## 与 vLLM 优化经验的对照

| vLLM 上的优化 | SGLang 对标优化 | 关键差异 |
|--------------|---------------|---------|
| Segmented LRU | 优化 1: Adaptive Eviction | vLLM 是 flat block 双链表；SGLang 是 Radix Tree 叶子 heap，有更丰富的节点元数据 |
| Cache-Aware Scheduling | 优化 2: 已有 LPM/DFS-Weight | SGLang 已内置，重点是量化对比和调优 |
| 缓存预热 | 优化 3: Tree-only Warming | SGLang 有 HiCache 三级缓存可利用 |
| 后缀自动机 Proposer | 优化 4: SAM + N-gram Trie | SGLang 有 C++ Trie 基础，SAM 是增强 |
| PD V1 适配 | 优化 8: 跨实例缓存协同 | SGLang 已有完整 PD 分离，聚焦缓存协同 |

---

## 基于

- [SGLang](https://github.com/sgl-project/sglang) — 高性能 LLM 推理服务框架
- SGLang 版本基线：main 分支

## License

Apache 2.0（同 SGLang）
