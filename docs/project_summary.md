# SGLang 项目要点总结

## 一、项目概述

SGLang 是由 [LMSYS](https://lmsys.org/about/) 组织维护的**高性能大语言模型（LLM）和多模态模型推理服务框架**。目标是在从单 GPU 到大规模分布式集群的各种场景下，提供**低延迟、高吞吐**的推理服务。

- **开源协议**：Apache 2.0
- **生产规模**：已部署在全球超 40 万个 GPU 上，每天生成数万亿 token
- **行业采用**：xAI、AMD、NVIDIA、Intel、LinkedIn、Cursor、Oracle Cloud、Google Cloud、Microsoft Azure、AWS 等

---

## 二、顶层目录结构

```
sglang/
├── python/                    # 核心 Python 包（SGLang Runtime）
│   └── sglang/srt/           # SRT（SGLang Runtime）核心代码
├── sgl-kernel/                # C++/CUDA 高性能内核库
│   ├── csrc/                  # CUDA/C++ 源码（attention, gemm, moe, allreduce 等）
│   └── python/                # Python 绑定
├── sgl-model-gateway/         # 模型网关（Rust + Go 实现）
├── benchmark/                 # 基准测试（38+ 场景）
├── docs/                      # 文档
├── examples/                  # 示例代码
├── test/                      # 测试用例（521+ Python 测试）
├── scripts/                   # 工具脚本
├── docker/                    # Docker/K8s 部署配置
└── 3rdparty/                  # 第三方依赖补丁
```

---

## 三、核心架构

### 3.1 进程架构

```
                     ┌──────────────────────┐
                     │   HTTP/gRPC Server    │  FastAPI + Uvicorn
                     │  (http_server.py)     │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  TokenizerManager     │  独立进程
                     │  (tokenizer_manager)  │  负责 tokenize/detokenize
                     └──────────┬───────────┘
                                │ ZMQ 通信
                     ┌──────────▼───────────┐
                     │  DataParallelController│ （可选，多副本调度）
                     └──────────┬───────────┘
                                │
                  ┌─────────────▼─────────────┐
                  │        Scheduler           │  每 TP Group 一个
                  │   (scheduler.py, 126KB)    │  核心调度器
                  │   ├── 请求队列管理          │
                  │   ├── Batch 组装            │
                  │   ├── KV Cache 管理         │
                  │   └── 输出处理              │
                  └─────────────┬──────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │       TpModelWorker         │  张量并行 Worker
                  │      (tp_worker.py)         │
                  └─────────────┬──────────────┘
                                │
                  ┌─────────────▼──────────────┐
                  │       ModelRunner           │  模型执行器
                  │    (model_runner.py)        │  管理前向推理 + CUDA Graph
                  └────────────────────────────┘
```

### 3.2 调度与推理函数调用链

#### 3.2.1 请求完整生命周期概览

```
HTTP 请求 → Engine.generate() → TokenizerManager.generate_request()
  → _tokenize_one_request()   (分词)
  → _send_one_request()       (ZMQ 发送到 Scheduler)
  → _wait_one_response()      (异步等待结果)

Scheduler 事件循环:
  recv_requests()              → ZMQ 接收请求
  process_input_requests()     → 分发到 handle_generate_request()
  handle_generate_request()    → 创建 Req 对象，加入 waiting_queue
  get_next_batch_to_run()      → 组装 prefill/decode 批次
  run_batch()                  → 驱动 GPU 前向推理
  process_batch_result()       → 处理输出，stream_output() 发送结果

结果返回:
  Scheduler → DetokenizerManager (ZMQ) → TokenizerManager.handle_loop()
  → _handle_batch_output() → 设置 ReqState.event → _wait_one_response() 返回
```

#### 3.2.2 服务启动链

```python
# engine.py
Engine.__init__()
  └→ _launch_subprocesses()
       ├→ _set_envs_and_config(server_args)          # 设置环境变量、检查版本
       ├→ _launch_scheduler_processes()               # 启动 Scheduler 子进程
       │    └→ mp.Process(target=run_scheduler_process)
       ├→ mp.Process(target=run_detokenizer_process)  # 启动 Detokenizer 子进程
       ├→ init_tokenizer_manager()                    # 主进程初始化 TokenizerManager
       └→ _wait_for_scheduler_ready()                 # 等待模型加载完成

# scheduler.py — Scheduler 进程入口
run_scheduler_process()
  └→ Scheduler.__init__()
       ├→ init_model_config()                # 解析模型配置
       ├→ init_ipc_channels()                # 建立 ZMQ 通信通道
       ├→ init_tp_worker()                   # 初始化 TpModelWorker（含 ModelRunner）
       ├→ init_memory_pool()                 # 初始化 KV Cache 内存池 + RadixCache
       ├→ SchedulePolicy(...)                # 创建调度策略
       └→ init_request_dispatcher()          # 注册请求类型→处理函数的分发映射
  └→ 进入事件循环:
       if enable_overlap:  event_loop_overlap()
       else:               event_loop_normal()
```

#### 3.2.3 TokenizerManager 请求处理链

```python
# tokenizer_manager.py — 主进程
class TokenizerManager:

  async def generate_request(obj, request):
      """请求入口，async generator"""
      auto_create_handle_loop()                  # 确保 handle_loop 已启动
      obj.normalize_batch_and_arguments()         # 规范化参数
      if obj.is_single:
          tokenized_obj = await _tokenize_one_request(obj)   # ① 分词
          state = _send_one_request(obj, tokenized_obj)       # ② 发送
          async for response in _wait_one_response(obj, state): # ③ 等待
              yield response

  async def _tokenize_one_request(obj):
      """分词：text → token_ids，处理多模态输入"""
      if obj.input_ids:       input_ids = obj.input_ids
      elif obj.input_embeds:  input_embeds = obj.input_embeds
      else:                   input_ids = await _tokenize_texts(input_text)
      # 处理多模态: mm_data_processor.process(image/audio/video)
      return TokenizedGenerateReqInput(...)

  def _send_one_request(obj, tokenized_obj, created_time):
      """通过 ZMQ 发送到 Scheduler"""
      tokenized_obj = wrap_shm_features(tokenized_obj)       # 共享内存包装
      self.send_to_scheduler.send_pyobj(tokenized_obj)        # ZMQ 发送
      state = ReqState([], False, asyncio.Event(), obj)       # 创建等待状态
      self.rid_to_state[obj.rid] = state
      return state

  async def _wait_one_response(obj, state, request):
      """通过 asyncio.Event 等待结果"""
      while True:
          await asyncio.wait_for(state.event.wait(), timeout=...)
          out = state.out_list[-1]
          if state.finished:
              yield out; break
          else:
              yield out  # streaming

  async def handle_loop():
      """后台事件循环：接收 Detokenizer 返回的结果"""
      while True:
          recv_obj = await self.recv_from_detokenizer.recv_pyobj()
          _result_dispatcher(recv_obj)  # → _handle_batch_output()

  def _handle_batch_output(recv_obj: BatchStrOutput):
      """将结果写入对应请求的 ReqState，触发 event"""
      for i, rid in enumerate(recv_obj.rids):
          state = self.rid_to_state[rid]
          state.out_list.append(out)
          if finished: state.finished = True
          state.event.set()  # 唤醒 _wait_one_response
```

#### 3.2.4 Scheduler 事件循环

```python
# scheduler.py

def event_loop_normal():
    """常规调度循环"""
    while True:
        recv_reqs = recv_requests()            # ① ZMQ 接收请求
        process_input_requests(recv_reqs)       # ② 分发处理
        batch = get_next_batch_to_run()         # ③ 组装批次
        if batch:
            result = run_batch(batch)           # ④ GPU 前向推理
            process_batch_result(batch, result) # ⑤ 处理结果
        else:
            self_check_during_idle()
        self.last_batch = batch

def event_loop_overlap():
    """Overlap 调度循环：GPU forward 与 CPU 后处理并行"""
    # 核心区别：run_batch 在 forward_stream 上异步执行
    # process_batch_result 处理上一个 batch（在 default_stream 上）
    result_queue = deque()
    while True:
        recv_reqs = recv_requests()
        process_input_requests(recv_reqs)
        batch = get_next_batch_to_run()
        # 如果需要，先处理上一个 batch 的结果
        if batch:
            batch_result = run_batch(batch)     # forward_stream 异步执行
            result_queue.append((batch.copy(), batch_result))
        if self.last_batch:
            tmp_batch, tmp_result = result_queue.popleft()
            process_batch_result(tmp_batch, tmp_result)  # CPU 后处理
        # 运行当前 batch 的采样（依赖上一 batch 的 grammar 结果）
        launch_batch_sample_if_needed(batch_result)
```

#### 3.2.5 请求接收与分发

```python
# scheduler.py

def recv_requests():
    """从 TokenizerManager 接收请求（ZMQ non-blocking poll）"""
    recv_reqs = []
    while True:
        try:
            recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            recv_req = unwrap_shm_features(recv_req)
        except zmq.ZMQError:
            break
        recv_reqs.append(recv_req)
    return recv_reqs

def process_input_requests(recv_reqs):
    """按类型分发请求"""
    for recv_req in recv_reqs:
        output = self._request_dispatcher(recv_req)
        # _request_dispatcher 是 TypeBasedDispatcher，映射：
        #   TokenizedGenerateReqInput  → handle_generate_request
        #   TokenizedEmbeddingReqInput → handle_embedding_request
        #   FlushCacheReqInput         → flush_cache_wrapped
        #   AbortReq                   → abort_request
        #   ...

def handle_generate_request(recv_req: TokenizedGenerateReqInput):
    """创建 Req 对象并加入等待队列"""
    req = Req(
        rid=recv_req.rid,
        origin_input_ids=recv_req.input_ids,
        sampling_params=recv_req.sampling_params,
        ...
    )
    req.tokenizer = self.tokenizer
    # 处理多模态输入、session 等
    validate_input_length(req, ...)
    init_req_max_new_tokens(req)
    _add_request_to_queue(req)  # → self.waiting_queue.append(req)
```

#### 3.2.6 批次组装（核心调度逻辑）

```python
# scheduler.py

def get_next_batch_to_run() -> Optional[ScheduleBatch]:
    """决定下一步运行 prefill 还是 decode"""
    # 1. 合并上一个 prefill 批次完成的请求到 running_batch
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.last_batch.filter_batch(...)
        self.running_batch.merge_batch(self.last_batch)

    # 2. 尝试创建新的 prefill 批次
    new_batch = get_new_batch_prefill()

    # 3. 优先运行 prefill；否则运行 decode
    if new_batch is not None:
        return new_batch              # → prefill 批次
    elif not self.running_batch.is_empty():
        self.running_batch = update_running_batch(self.running_batch)
        return self.running_batch     # → decode 批次
    return None                        # 空闲

def get_new_batch_prefill() -> Optional[ScheduleBatch]:
    """从 waiting_queue 组装 prefill 批次"""
    └→ _get_new_batch_prefill_raw()
        # 1. 调度策略排序
        self.policy.calc_priority(self.waiting_queue, self.running_batch)
        #   LPM:   按 prefix_indices 长度降序（最长前缀匹配优先）
        #   FCFS:  先到先服务
        #   DFS-Weight: 基数树深度优先权重
        #   LOF:   最长输出优先
        #   ROUTING-KEY: 按路由键频率优先

        # 2. 创建 PrefillAdder（预算管理器）
        adder = PrefillAdder(
            page_size, tree_cache, token_to_kv_pool_allocator,
            running_batch, new_token_ratio, max_prefill_tokens,
            chunked_prefill_size, ...
        )

        # 3. 逐个添加请求，检查预算
        for req in self.waiting_queue:
            req.init_next_round_input(self.tree_cache)
            #   → tree_cache.match_prefix() → 获取前缀命中的 KV 索引
            res = adder.add_one_req(req, ...)
            #   → 检查 rem_total_tokens, rem_input_tokens, rem_chunk_tokens
            #   → 可能触发 chunked prefill（长 prompt 分块）
            if res != AddReqResult.CONTINUE:
                break

        # 4. 创建 ScheduleBatch
        new_batch = ScheduleBatch.init_new(can_run_list, ...)
        new_batch.prepare_for_extend()
        #   → 构造 input_ids tensor, seq_lens tensor
        #   → alloc_for_extend(): 分配 KV Cache 内存
        return new_batch

def update_running_batch(batch) -> ScheduleBatch:
    """更新 decode 批次：检查 OOM、执行 retraction"""
    batch.filter_batch(...)  # 过滤已完成请求
    if not batch.check_decode_mem():
        retracted_reqs, new_token_ratio, _ = batch.retract_decode(...)
        # → 内存不足时，抢占低优先级请求回 waiting_queue
    batch.prepare_for_decode()
    #   → alloc_for_decode(): 为每个请求分配 1 个新 token 的 KV 槽
    return batch
```

#### 3.2.7 GPU 推理执行链

```python
# scheduler.py → tp_worker.py → model_runner.py

def run_batch(batch: ScheduleBatch):
    """调度器驱动 GPU 前向推理"""
    if self.enable_overlap:
        model_worker_batch = batch.get_model_worker_batch()
        # Overlap 模式：在 forward_stream 上异步执行
        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.default_stream)
            batch_result = self.model_worker.forward_batch_generation(
                model_worker_batch
            )
            batch_result.copy_to_cpu(...)  # 异步拷贝结果到 CPU
    else:
        worker_batch = batch.get_model_worker_batch()
        batch_result = self.model_worker.forward_batch_generation(worker_batch)
    return batch_result

# tp_worker.py
class TpModelWorker:
  def forward_batch_generation(model_worker_batch, ...):
      """TpModelWorker: 转换为 ForwardBatch 并调用 ModelRunner"""
      forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
      #   ForwardBatch: 底层 GPU tensor 数据结构
      out = self.model_runner.forward(forward_batch)
      logits_output = out.logits_output
      # 采样下一个 token
      batch_result.next_token_ids = self.model_runner.sample(
          logits_output, forward_batch
      )
      return batch_result

# model_runner.py
class ModelRunner:
  def forward(forward_batch) -> ModelRunnerOutput:
      """ModelRunner: 分发到 CUDA Graph 或 Eager 模式"""
      └→ _forward_raw(forward_batch)
          if can_run_graph:
              return self.graph_runner.replay(forward_batch)
              # → CUDA Graph 重放（decode 阶段常用）
          elif forward_batch.forward_mode.is_decode():
              return self.forward_decode(forward_batch)
          elif forward_batch.forward_mode.is_extend():
              return self.forward_extend(forward_batch)
          elif forward_batch.forward_mode.is_idle():
              return self.forward_idle(forward_batch)

  def forward_decode(forward_batch):
      """Decode 前向：单 token 推理"""
      self.attn_backend.init_forward_metadata(forward_batch)
      return self.model.forward(
          forward_batch.input_ids,   # [batch_size]
          forward_batch.positions,
          forward_batch,
      )

  def forward_extend(forward_batch):
      """Prefill/Extend 前向：多 token 推理"""
      # 可能使用 piecewise CUDA Graph
      if piecewise_cuda_graph_runner.can_run(forward_batch):
          return piecewise_cuda_graph_runner.replay(forward_batch)
      self.attn_backend.init_forward_metadata(forward_batch)
      return self.model.forward(
          forward_batch.input_ids,   # [total_extend_tokens]
          forward_batch.positions,
          forward_batch,
      )
```

#### 3.2.8 结果后处理与输出

```python
# scheduler.py + scheduler_output_processor_mixin.py

def process_batch_result(batch, result):
    """分发到对应的处理函数"""
    if batch.forward_mode.is_decode():
        process_batch_result_decode(batch, result)
    elif batch.forward_mode.is_extend():
        process_batch_result_prefill(batch, result)

def process_batch_result_prefill(batch, result):
    """处理 Prefill 结果"""
    next_token_ids = result.next_token_ids.tolist()
    for req, next_token_id in zip(batch.reqs, next_token_ids):
        if req.is_chunked <= 0:
            req.output_ids.append(next_token_id)      # 记录输出 token
            req.check_finished()                        # 检查停止条件
            if req.finished():
                release_kv_cache(req, self.tree_cache)  # 释放 KV 缓存
            else:
                self.tree_cache.cache_unfinished_req(req) # 更新 RadixCache
    stream_output(batch.reqs, ...)  # 发送结果

def process_batch_result_decode(batch, result):
    """处理 Decode 结果"""
    next_token_ids = result.next_token_ids.tolist()
    for req, next_token_id in zip(batch.reqs, next_token_ids):
        req.output_ids.append(next_token_id)
        req.check_finished()
        if req.finished():
            release_kv_cache(req, self.tree_cache)
    stream_output(batch.reqs, ...)

def stream_output(reqs, ...):
    """构造 BatchStrOutput 发送到 Detokenizer"""
    # 收集所有请求的 rid, output_ids, finished_reason 等
    output = BatchStrOutput(rids=..., decoded_texts=..., ...)
    self.send_to_detokenizer.send_output(output)
    # → DetokenizerManager 解码 → TokenizerManager.handle_loop() 接收
```

#### 3.2.9 数据结构流转

```
ScheduleBatch          →  ModelWorkerBatch       →  ForwardBatch
(scheduler.py)            (schedule_batch.py)        (forward_batch_info.py)
┌───────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ reqs: List[Req]│      │ forward_mode     │      │ forward_mode     │
│ forward_mode   │  →   │ input_ids (CPU)  │  →   │ input_ids (GPU)  │
│ tree_cache     │      │ seq_lens         │      │ seq_lens (GPU)   │
│ sampling_info  │      │ out_cache_loc    │      │ out_cache_loc    │
│ prefix_lens    │      │ sampling_info    │      │ req_pool_indices │
│ extend_lens    │      │ req_pool_indices │      │ attn_backend     │
└───────────────┘      └──────────────────┘      └──────────────────┘
  高层调度数据             GPU Worker 子集             底层 GPU Tensor
  （大部分在 CPU）         （CPU→GPU 转换）            （全部在 GPU）
```

#### 3.2.10 RadixCache 在调度中的关键调用点

```python
# 1. 调度排序阶段 — 前缀匹配
SchedulePolicy.calc_priority() → _compute_prefix_matches()
  → tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(...)))
  #   返回: prefix_indices (已缓存的 KV 索引), last_node (匹配终点)

# 2. 请求加入 prefill 批次
PrefillAdder.add_one_req()
  → tree_cache.inc_lock_ref(req.last_node)
  #   锁定前缀节点，防止被淘汰

# 3. Prefill 完成后 — 缓存未完成请求
process_batch_result_prefill()
  → tree_cache.cache_unfinished_req(req)
  #   → insert(): 将新计算的 KV 插入基数树
  #   → match_prefix(): 更新 prefix_indices
  #   → dec_lock_ref() + inc_lock_ref(): 转移锁

# 4. 请求完成 — 缓存最终结果
release_kv_cache(req, tree_cache)
  → tree_cache.cache_finished_req(req)
  #   → insert(): 插入完整的 token_ids → KV 映射
  #   → free(): 释放重复的 KV 索引
  #   → dec_lock_ref(): 释放锁

# 5. 内存不足 — 淘汰缓存
alloc_for_extend() / alloc_for_decode() 内存不足时
  → tree_cache.evict(EvictParams(num_tokens=...))
  #   → 按淘汰策略(LRU/LFU/FIFO/...)选择叶节点
  #   → token_to_kv_pool_allocator.free(): 释放 KV 内存
  #   → _delete_leaf(): 从树中删除节点
```

### 3.3 完整端到端请求生命周期函数调用链

下面以一个 **`/v1/chat/completions` 非流式请求**为例，追踪从 HTTP 到达到最终结果返回的**每一个关键函数调用**，标注文件名和行号（基于当前代码版本）。

---

#### 阶段一：HTTP 入口与 OpenAI 兼容层

```
用户发送 HTTP POST /v1/chat/completions
        │
        ▼
┌─ http_server.py ──────────────────────────────────────────────────────┐
│ @app.post("/v1/chat/completions")                        (line 1324) │
│ async def openai_v1_chat_completions(raw_request: Request):          │
│   → openai_serving_chat.handle_request(raw_request)                  │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
                       ▼
┌─ serving_chat.py ─────────────────────────────────────────────────────┐
│ class OpenAIServingChat(OpenAIServingBase):                           │
│                                                                       │
│ async def handle_request(raw_request):                   (line →继承) │
│   # 继承自 serving_base.py OpenAIServingBase.handle_request()        │
│   │                                                                   │
│   ▼                                                                   │
│ _convert_to_internal_request(chat_request, raw_request)  (line 240)  │
│   # 1. 应用 chat template，将 messages → prompt text                 │
│   # 2. 创建 GenerateReqInput(text=prompt, sampling_params=...)       │
│   # 3. 处理 tools/function_call、response_format 等                  │
│   return GenerateReqInput(...)                                        │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
                       ▼
┌─ serving_base.py ─────────────────────────────────────────────────────┐
│ class OpenAIServingBase:                                              │
│                                                                       │
│ async def handle_request(raw_request):                    (line 86)  │
│   adapted_request, gen_input = _convert_to_internal_request(...)     │
│   if adapted_request.stream:                                         │
│       → _handle_streaming_request(gen_input, adapted_request)        │
│   else:                                                              │
│       → _handle_non_streaming_request(gen_input, adapted_request)    │
│                                                                       │
│ async def _handle_non_streaming_request(gen_input, ...): (line 203)  │
│   async for response in self.tokenizer_manager.generate_request(     │
│       gen_input, raw_request                                         │
│   ):                                                                  │
│       pass  # 收集最终响应                                            │
│   return ChatCompletionResponse(...)  # 构造 OpenAI 格式响应         │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
                       ▼
           进入 TokenizerManager（主进程）
```

---

#### 阶段二：TokenizerManager — 分词与请求分发

```
┌─ tokenizer_manager.py ────────────────────────────────────────────────┐
│                                                                       │
│ async def generate_request(obj: GenerateReqInput, request):           │
│                                                      (line 490)      │
│   ① auto_create_handle_loop()                        (line 510)      │
│   │  # 首次调用时启动后台 handle_loop() 协程，                        │
│   │  # 用于接收 DetokenizerManager 返回的结果                        │
│   │                                                                   │
│   ② obj.normalize_batch_and_arguments()              (line 518)      │
│   │  # 规范化参数，处理 batch 请求拆分                               │
│   │                                                                   │
│   ③ tokenized_obj = await _tokenize_one_request(obj) (line 539)      │
│   │  ┌──────────────────────────────────────────────────────┐        │
│   │  │ _tokenize_one_request(obj)                (line 667) │        │
│   │  │                                                      │        │
│   │  │ if obj.input_ids 不为空:                              │        │
│   │  │     input_ids = obj.input_ids  # 已有 token_ids      │        │
│   │  │ elif obj.input_embeds 不为空:                         │        │
│   │  │     input_embeds = obj.input_embeds                   │        │
│   │  │ else:                                                 │        │
│   │  │     input_ids = await _tokenize_texts(input_text)     │        │
│   │  │     ┌─────────────────────────────────────────┐      │        │
│   │  │     │ _tokenize_texts(texts)       (line 587) │      │        │
│   │  │     │ if async_batch_tokenizer:               │      │        │
│   │  │     │   # 动态批量分词（高吞吐优化）          │      │        │
│   │  │     │   → async_batch_tokenizer.tokenize(...)  │      │        │
│   │  │     │ else:                                    │      │        │
│   │  │     │   → tokenizer.encode(text)               │      │        │
│   │  │     └─────────────────────────────────────────┘      │        │
│   │  │                                                      │        │
│   │  │ # 处理多模态输入（如果有 image/video/audio）         │        │
│   │  │ if has_multimodal:                                    │        │
│   │  │   mm_data_processor.process(input_ids, mm_data, ...)  │        │
│   │  │                                                      │        │
│   │  │ return TokenizedGenerateReqInput(                     │        │
│   │  │     rid=obj.rid,                                      │        │
│   │  │     input_ids=input_ids,                              │        │
│   │  │     sampling_params=sampling_params,                  │        │
│   │  │     ...                                               │        │
│   │  │ )                                                     │        │
│   │  └──────────────────────────────────────────────────────┘        │
│   │                                                                   │
│   ④ state = _send_one_request(obj, tokenized_obj)    (line 541)      │
│   │  ┌──────────────────────────────────────────────────────┐        │
│   │  │ _send_one_request(obj, tokenized_obj)     (line 1058)│        │
│   │  │                                                      │        │
│   │  │ tokenized_obj = wrap_shm_features(tokenized_obj)     │        │
│   │  │ # 将大的 embedding/image tensor 放入共享内存         │        │
│   │  │                                                      │        │
│   │  │ self.send_to_scheduler.send_pyobj(tokenized_obj)     │        │
│   │  │ # ★ ZMQ 发送到 Scheduler 子进程                     │        │
│   │  │                                                      │        │
│   │  │ state = ReqState(                                     │        │
│   │  │     out_list=[],                                      │        │
│   │  │     finished=False,                                   │        │
│   │  │     event=asyncio.Event(),  # 用于异步等待            │        │
│   │  │     obj=obj,                                          │        │
│   │  │ )                                                     │        │
│   │  │ self.rid_to_state[obj.rid] = state                    │        │
│   │  │ return state                                          │        │
│   │  └──────────────────────────────────────────────────────┘        │
│   │                                                                   │
│   ⑤ async for response in _wait_one_response(obj, state):            │
│      ┌──────────────────────────────────────────────────────┐        │
│      │ _wait_one_response(obj, state, request)   (line 1101)│        │
│      │                                                      │        │
│      │ while True:                                           │        │
│      │     await asyncio.wait_for(                           │        │
│      │         state.event.wait(),                           │        │
│      │         timeout=self.server_args.api_timeout          │        │
│      │     )                                                 │        │
│      │     # ★ 阻塞在此，直到 handle_loop() 设置 event     │        │
│      │                                                      │        │
│      │     state.event.clear()                               │        │
│      │     out = state.out_list[-1]                          │        │
│      │     if state.finished:                                │        │
│      │         del self.rid_to_state[obj.rid]                │        │
│      │         yield out  # 最终结果                         │        │
│      │         break                                         │        │
│      │     else:                                             │        │
│      │         yield out  # 流式中间结果                     │        │
│      └──────────────────────────────────────────────────────┘        │
│                                                                       │
│   yield response  → 返回给 serving_base.py                           │
└───────────────────────────────────────────────────────────────────────┘
```

> **跨进程分界线**：`send_to_scheduler.send_pyobj()` 通过 ZMQ IPC 将 `TokenizedGenerateReqInput` 从主进程发送到 Scheduler 子进程。

---

#### 阶段三：Scheduler — 请求接收与入队

```
┌─ scheduler.py ── Scheduler 子进程事件循环 ────────────────────────────┐
│                                                                       │
│ def event_loop_normal():                             (line 1082)     │
│   while True:                                                         │
│       recv_reqs = recv_requests()                    # ← STEP 1      │
│       process_input_requests(recv_reqs)              # ← STEP 2      │
│       batch = get_next_batch_to_run()                # ← STEP 3      │
│       if batch:                                                       │
│           result = run_batch(batch)                  # ← STEP 4      │
│           process_batch_result(batch, result)        # ← STEP 5      │
│                                                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│ STEP 1: recv_requests()                              (line 1191)     │
│ ═══════════════════════════════════════════════════════════════════    │
│   recv_reqs = []                                                      │
│   while True:                                                         │
│       try:                                                            │
│           recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)│
│           recv_req = unwrap_shm_features(recv_req)                   │
│           # 如果 TP > 1，广播给其他 TP rank                          │
│           if self.tp_size > 1:                                        │
│               broadcast_recv_input(recv_req)                          │
│       except zmq.ZMQError:                                            │
│           break                                                       │
│       recv_reqs.append(recv_req)                                      │
│   return recv_reqs                                                    │
│                                                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│ STEP 2: process_input_requests(recv_reqs)            (line 1329)     │
│ ═══════════════════════════════════════════════════════════════════    │
│   for recv_req in recv_reqs:                                          │
│       self._request_dispatcher(recv_req)                              │
│       # TypeBasedDispatcher 路由表:                                   │
│       #   TokenizedGenerateReqInput  → handle_generate_request()     │
│       #   TokenizedEmbeddingReqInput → handle_embedding_request()    │
│       #   FlushCacheReqInput         → flush_cache_wrapped()         │
│       #   AbortReq                   → abort_request()               │
│       #   GetMemPoolSizeReq          → handle_get_mem_pool_size()    │
│                                                                       │
│ ┌─ handle_generate_request(recv_req)                 (line 1446) ──┐ │
│ │                                                                   │ │
│ │ req = Req(                                                        │ │
│ │     rid = recv_req.rid,                                           │ │
│ │     origin_input_ids = recv_req.input_ids,                        │ │
│ │     sampling_params = recv_req.sampling_params,                   │ │
│ │     origin_input_text = recv_req.text,                            │ │
│ │     return_logprob = recv_req.return_logprob,                     │ │
│ │     session = recv_req.session_params,                            │ │
│ │     ...                                                           │ │
│ │ )                                                                 │ │
│ │ req.tokenizer = self.tokenizer                                    │ │
│ │                                                                   │ │
│ │ # 多模态数据迁移到 req                                            │ │
│ │ if recv_req.mm_inputs:                                            │ │
│ │     req.mm_inputs = recv_req.mm_inputs                            │ │
│ │                                                                   │ │
│ │ # 输入长度校验                                                    │ │
│ │ validate_input_length(req, max_context_len)                       │ │
│ │                                                                   │ │
│ │ # 初始化最大新 token 数                                           │ │
│ │ init_req_max_new_tokens(req)                                      │ │
│ │                                                                   │ │
│ │ # ★ 加入等待队列                                                 │ │
│ │ _add_request_to_queue(req)                                        │ │
│ │   → self.waiting_queue.append(req)                                │ │
│ │   → self.waiting_queue_unready.remove(req)  # 如果是预准备请求    │ │
│ │                                                                   │ │
│ └───────────────────────────────────────────────────────────────────┘ │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
            请求现在在 self.waiting_queue 中等待调度
                       │
                       ▼
```

---

#### 阶段四：Scheduler — 批次组装（Prefill）

```
┌─ scheduler.py ── STEP 3: get_next_batch_to_run() ────(line 1840) ────┐
│                                                                       │
│ # 1. 合并上一个 prefill 批次的请求到 running_batch                   │
│ if self.last_batch and self.last_batch.forward_mode.is_extend():     │
│     self.last_batch.filter_batch(filter_finished_reqs)               │
│     self.running_batch.merge_batch(self.last_batch)                  │
│                                                                       │
│ # 2. 尝试创建新的 prefill 批次                                       │
│ new_batch = get_new_batch_prefill()                  (line 1929)     │
│                                                                       │
│ if new_batch is not None:                                             │
│     return new_batch          # → 进入 STEP 4 执行 prefill           │
│ elif not self.running_batch.is_empty():                               │
│     batch = update_running_batch(self.running_batch)                  │
│     return batch              # → 进入 STEP 4 执行 decode            │
│ return None                   # → 空闲                               │
│                                                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│ get_new_batch_prefill() → _get_new_batch_prefill_raw()               │
│                                                      (line 1946)     │
│ ═══════════════════════════════════════════════════════════════════    │
│                                                                       │
│ ┌─ Phase A: 调度策略排序 ─────────────────────────────────────────┐  │
│ │                                                                  │  │
│ │ self.policy.calc_priority(self.waiting_queue, self.running_batch)│  │
│ │                                                                  │  │
│ │ ┌─ schedule_policy.py ── calc_priority()       (line 114) ─────┐│  │
│ │ │                                                               ││  │
│ │ │ # 计算可用 token 数                                          ││  │
│ │ │ available_tokens = available_size - running_batch.token_count ││  │
│ │ │ # 如果队列过长(>128)，退化为 FCFS 避免排序开销              ││  │
│ │ │ if len(waiting_queue) > 128 and policy == "lpm":             ││  │
│ │ │     policy = "fcfs"                                          ││  │
│ │ │                                                               ││  │
│ │ │ if policy == "lpm":                                          ││  │
│ │ │   # 最长前缀匹配优先                                        ││  │
│ │ │   _compute_prefix_matches(waiting_queue)                     ││  │
│ │ │   # → 对每个 req 调用 tree_cache.match_prefix()             ││  │
│ │ │   # → 获取 prefix_indices（已缓存 KV 的索引）               ││  │
│ │ │   waiting_queue.sort(key=lambda r: -len(r.prefix_indices))   ││  │
│ │ │                                                               ││  │
│ │ │ elif policy == "fcfs":                                       ││  │
│ │ │   # 先来先服务                                               ││  │
│ │ │   pass  # 保持原始入队顺序                                   ││  │
│ │ │                                                               ││  │
│ │ │ elif policy == "dfs-weight":                                 ││  │
│ │ │   # 深度优先搜索权重：优先复用热门前缀                      ││  │
│ │ │   waiting_queue.sort(key=lambda r: tree_weight(r))           ││  │
│ │ │                                                               ││  │
│ │ │ return SchedulePolicyResult(available_tokens)                 ││  │
│ │ └──────────────────────────────────────────────────────────────┘│  │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ ┌─ Phase B: PrefillAdder 构建与预算管理 ──────────────────────────┐  │
│ │                                                                  │  │
│ │ adder = PrefillAdder(                              (line 372)   │  │
│ │     page_size = tree_cache.page_size,                            │  │
│ │     tree_cache = tree_cache,                                     │  │
│ │     token_to_kv_pool_allocator = kv_allocator,                   │  │
│ │     running_batch = running_batch,                               │  │
│ │     new_token_ratio = new_token_ratio,                           │  │
│ │     max_prefill_tokens = server_args.max_prefill_tokens,         │  │
│ │     chunked_prefill_size = server_args.chunked_prefill_size,     │  │
│ │     ...                                                          │  │
│ │ )                                                                │  │
│ │ # PrefillAdder 内部计算初始预算：                                │  │
│ │ #   rem_total_tokens = available_slots - (running * new_token_r) │  │
│ │ #   rem_input_tokens = max_prefill_tokens                        │  │
│ │ #   rem_chunk_tokens = chunked_prefill_size (如果启用)           │  │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ ┌─ Phase C: 逐请求添加到批次 ─────────────────────────────────────┐  │
│ │                                                                  │  │
│ │ for i, req in enumerate(self.waiting_queue):                     │  │
│ │                                                                  │  │
│ │   ┌─ C.1: req.init_next_round_input()  ── schedule_batch.py ──┐│  │
│ │   │                                               (line 888)   ││  │
│ │   │ # 构建完整的输入 token 序列                                ││  │
│ │   │ fill_ids = self.origin_input_ids + self.output_ids         ││  │
│ │   │                                                            ││  │
│ │   │ # ★ RadixCache 前缀匹配                                   ││  │
│ │   │ match_result = tree_cache.match_prefix(                    ││  │
│ │   │     MatchPrefixParams(key=RadixKey(fill_ids))              ││  │
│ │   │ )                                                          ││  │
│ │   │ ┌─ radix_cache.py ── match_prefix()       (line 352) ────┐││  │
│ │   │ │ # 沿基数树匹配最长已缓存前缀                           │││  │
│ │   │ │ # 返回 MatchResult:                                     │││  │
│ │   │ │ #   device_indices: 已缓存 KV 的内存索引 (torch.Tensor) │││  │
│ │   │ │ #   last_device_node: 匹配到的最后一个树节点            │││  │
│ │   │ └─────────────────────────────────────────────────────────┘││  │
│ │   │                                                            ││  │
│ │   │ self.prefix_indices = match_result.device_indices          ││  │
│ │   │ self.last_node = match_result.last_device_node             ││  │
│ │   │ # extend_input_len = 总长度 - 前缀命中长度                ││  │
│ │   │ self.extend_input_len = len(fill_ids) - len(prefix_indices)││  │
│ │   └───────────────────────────────────────────────────────────┘│  │
│ │                                                                  │  │
│ │   ┌─ C.2: adder.add_one_req(req, ...)  ── schedule_policy.py ─┐│  │
│ │   │                                               (line 719)   ││  │
│ │   │ total_tokens = extend_input_len + max_new_tokens           ││  │
│ │   │                                                            ││  │
│ │   │ # 检查全局 token 预算                                      ││  │
│ │   │ if total_tokens > self.rem_total_tokens:                   ││  │
│ │   │     return AddReqResult.NO_TOKEN  # 总预算不足，停止       ││  │
│ │   │                                                            ││  │
│ │   │ # 检查 prefill token 预算                                  ││  │
│ │   │ if extend_input_len > self.rem_input_tokens:               ││  │
│ │   │     return AddReqResult.NO_TOKEN                           ││  │
│ │   │                                                            ││  │
│ │   │ # Chunked Prefill: 长 prompt 分块处理                     ││  │
│ │   │ if self.rem_chunk_tokens is not None:                      ││  │
│ │   │     if extend_input_len > self.rem_chunk_tokens:           ││  │
│ │   │         # 截断本次处理长度                                 ││  │
│ │   │         trunc_len = min(extend_input_len, rem_chunk_tokens)││  │
│ │   │         req.is_chunked = extend_input_len - trunc_len      ││  │
│ │   │         extend_input_len = trunc_len                       ││  │
│ │   │                                                            ││  │
│ │   │ # ★ 锁定 RadixCache 节点                                  ││  │
│ │   │ tree_cache.inc_lock_ref(req.last_node)                     ││  │
│ │   │                                                            ││  │
│ │   │ # 更新预算                                                 ││  │
│ │   │ self.rem_total_tokens -= total_tokens                      ││  │
│ │   │ self.rem_input_tokens -= extend_input_len                  ││  │
│ │   │ self.can_run_list.append(req)                              ││  │
│ │   │ return AddReqResult.CONTINUE                               ││  │
│ │   └───────────────────────────────────────────────────────────┘│  │
│ │                                                                  │  │
│ │   if res != AddReqResult.CONTINUE:                               │  │
│ │       break  # 预算耗尽，停止添加                               │  │
│ │                                                                  │  │
│ └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│ ┌─ Phase D: 创建 ScheduleBatch 并准备 Prefill ────────────────────┐  │
│ │                                                                  │  │
│ │ # 从 waiting_queue 移除已加入批次的请求                         │  │
│ │ for req in adder.can_run_list:                                   │  │
│ │     self.waiting_queue.remove(req)                               │  │
│ │                                                                  │  │
│ │ # 创建批次                                                      │  │
│ │ new_batch = ScheduleBatch.init_new(                 (line 1328) │  │
│ │     reqs = adder.can_run_list,                                   │  │
│ │     forward_mode = ForwardMode.EXTEND,                           │  │
│ │     ...                                                          │  │
│ │ )                                                                │  │
│ │                                                                  │  │
│ │ # ★ 准备 extend（分配 KV Cache 内存 + 构造 tensor）            │  │
│ │ new_batch.prepare_for_extend()                     (line 1449)  │  │
│ │ ┌────────────────────────────────────────────────────────────┐  │  │
│ │ │ prepare_for_extend():                                      │  │  │
│ │ │   self.forward_mode = ForwardMode.EXTEND                   │  │  │
│ │ │                                                            │  │  │
│ │ │   # 构造 GPU input tensors                                 │  │  │
│ │ │   for req in self.reqs:                                    │  │  │
│ │ │       input_ids += req.fill_ids[len(req.prefix_indices):]  │  │  │
│ │ │       seq_lens.append(len(req.fill_ids))                   │  │  │
│ │ │       prefix_lens.append(len(req.prefix_indices))          │  │  │
│ │ │       extend_lens.append(req.extend_input_len)             │  │  │
│ │ │                                                            │  │  │
│ │ │   # ★★ 分配 KV Cache 内存                                │  │  │
│ │ │   alloc_for_extend(self, mem_cache, ...)         (common.py│  │  │
│ │ │                                                  line 328) │  │  │
│ │ │   ┌────────────────────────────────────────────────────┐  │  │  │
│ │ │   │ alloc_for_extend():                                │  │  │  │
│ │ │   │                                                    │  │  │  │
│ │ │   │ # A. 分配 req 槽位                                 │  │  │  │
│ │ │   │ req_pool_indices = alloc_req_slots(batch_size)     │  │  │  │
│ │ │   │ # → ReqToTokenPool.alloc(batch_size)               │  │  │  │
│ │ │   │                                                    │  │  │  │
│ │ │   │ # B. 分配 KV token 槽位                            │  │  │  │
│ │ │   │ for req in reqs:                                   │  │  │  │
│ │ │   │   need_tokens = seq_len - prefix_len               │  │  │  │
│ │ │   │   out_cache_loc = kv_allocator.alloc(need_tokens)  │  │  │  │
│ │ │   │   # 内存不足时 → evict_from_tree_cache()           │  │  │  │
│ │ │   │   #   → tree_cache.evict()                         │  │  │  │
│ │ │   │   #     → 淘汰最不常用的叶节点                     │  │  │  │
│ │ │   │   #     → kv_allocator.free(evicted_indices)        │  │  │  │
│ │ │   │                                                    │  │  │  │
│ │ │   │ # C. 写入 req_to_token_pool 映射                   │  │  │  │
│ │ │   │ #   将 prefix_indices + out_cache_loc 写入         │  │  │  │
│ │ │   │ #   req_to_token_pool[req_idx, :seq_len] = ...     │  │  │  │
│ │ │   │ write_req_to_token_pool_triton(...)  # Triton 内核 │  │  │  │
│ │ │   └────────────────────────────────────────────────────┘  │  │  │
│ │ └────────────────────────────────────────────────────────────┘  │  │
│ │                                                                  │  │
│ │ return new_batch  # ForwardMode.EXTEND 的 ScheduleBatch         │  │
│ └────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
            ScheduleBatch (ForwardMode.EXTEND) 就绪
                       │
                       ▼
```

---

#### 阶段五：GPU 前向推理

```
┌─ scheduler.py ── STEP 4: run_batch(batch) ───────────(line 2247) ────┐
│                                                                       │
│ # 将 ScheduleBatch 转换为 ModelWorkerBatch（CPU→GPU 桥梁）           │
│ model_worker_batch = batch.get_model_worker_batch()  (line 2165)     │
│                                                                       │
│ # 调用 TpModelWorker                                                 │
│ result = self.model_worker.forward_batch_generation(                  │
│     model_worker_batch                                                │
│ )                                                                     │
│                                                                       │
│ # [Overlap 模式的区别]:                                               │
│ #   在 forward_stream 上异步执行，不阻塞 CPU                         │
│ #   result.copy_to_cpu(...)  异步拷贝到 CPU                          │
│ #   future_map.store_to_map()  存入 FutureMap                        │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
                       ▼
┌─ tp_worker.py ── TpModelWorker.forward_batch_generation() ───────────┐
│                                                      (line 426)      │
│                                                                       │
│ # 1. 将 ModelWorkerBatch → ForwardBatch（GPU tensor 格式）           │
│ forward_batch = ForwardBatch.init_new(                               │
│     model_worker_batch, self.model_runner                            │
│ )                                                                     │
│ ┌─ forward_batch_info.py ── ForwardBatch.init_new()  (line 379) ───┐ │
│ │                                                                   │ │
│ │ # 将 CPU tensor → GPU tensor                                     │ │
│ │ ret.input_ids = model_worker_batch.input_ids.to(device)          │ │
│ │ ret.seq_lens = model_worker_batch.seq_lens.to(device)            │ │
│ │ ret.out_cache_loc = model_worker_batch.out_cache_loc.to(device)  │ │
│ │ ret.req_pool_indices = ...                                        │ │
│ │ ret.positions = compute_positions(seq_lens, prefix_lens)          │ │
│ │                                                                   │ │
│ │ # 设置 attention backend 引用                                    │ │
│ │ ret.attn_backend = model_runner.attn_backend                     │ │
│ │ ret.token_to_kv_pool = model_runner.token_to_kv_pool             │ │
│ │ ret.req_to_token_pool = model_runner.req_to_token_pool           │ │
│ │                                                                   │ │
│ │ return ret  # ForwardBatch (全部在 GPU)                          │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ # 2. ★ 模型前向推理                                                 │
│ out = self.model_runner.forward(forward_batch)                       │
│ ┌─ model_runner.py ── forward()                      (line 2346) ──┐ │
│ │                                                                   │ │
│ │ return _forward_raw(forward_batch)                 (line 2402)   │ │
│ │                                                                   │ │
│ │ ┌─ _forward_raw() 分发逻辑 ──────────────────────────────────┐  │ │
│ │ │                                                              │  │ │
│ │ │ # Case A: CUDA Graph 重放（Decode 阶段常用）                │  │ │
│ │ │ if graph_runner and graph_runner.can_run(forward_batch):     │  │ │
│ │ │     return graph_runner.replay(forward_batch)                │  │ │
│ │ │     # → 直接重放预捕获的 GPU kernel 序列                    │  │ │
│ │ │     # → 极低 CPU 开销（避免每次 kernel launch）             │  │ │
│ │ │                                                              │  │ │
│ │ │ # Case B: Decode Eager 模式                                 │  │ │
│ │ │ elif forward_batch.forward_mode.is_decode():                 │  │ │
│ │ │     return forward_decode(forward_batch)          (line 2243)│  │ │
│ │ │                                                              │  │ │
│ │ │ # Case C: Extend/Prefill                                    │  │ │
│ │ │ elif forward_batch.forward_mode.is_extend():                 │  │ │
│ │ │     return forward_extend(forward_batch)          (line 2266)│  │ │
│ │ │                                                              │  │ │
│ │ │ # Case D: Idle（overlap 模式占位）                          │  │ │
│ │ │ elif forward_batch.forward_mode.is_idle():                   │  │ │
│ │ │     return forward_idle(forward_batch)            (line 2306)│  │ │
│ │ └──────────────────────────────────────────────────────────────┘  │ │
│ │                                                                   │ │
│ │ ┌─ forward_extend(forward_batch)                   (line 2266) ┐ │ │
│ │ │                                                               │ │ │
│ │ │ # 尝试 Piecewise CUDA Graph（分段图重放）                   │ │ │
│ │ │ if piecewise_graph_runner.can_run(forward_batch):             │ │ │
│ │ │     return piecewise_graph_runner.replay(forward_batch)       │ │ │
│ │ │                                                               │ │ │
│ │ │ # Eager 模式                                                 │ │ │
│ │ │ self.attn_backend.init_forward_metadata(forward_batch)        │ │ │
│ │ │ # → 初始化注意力计算所需的元数据（page table, 掩码等）      │ │ │
│ │ │                                                               │ │ │
│ │ │ logits_output = self.model.forward(                           │ │ │
│ │ │     input_ids = forward_batch.input_ids,                      │ │ │
│ │ │     positions = forward_batch.positions,                      │ │ │
│ │ │     forward_batch = forward_batch,                            │ │ │
│ │ │ )                                                             │ │ │
│ │ │ # → 经过全部 Transformer 层:                                │ │ │
│ │ │ #   Embedding → [RMSNorm → Attention → RMSNorm → MLP] × N  │ │ │
│ │ │ #     Attention 层:                                          │ │ │
│ │ │ #       Q, K, V = Linear(hidden_states)                      │ │ │
│ │ │ #       K, V → 写入 KV Cache (out_cache_loc 指定位置)       │ │ │
│ │ │ #       attn_output = attention_backend.forward(Q, K_cache,  │ │ │
│ │ │ #                                              V_cache, ...) │ │ │
│ │ │ #   → LM Head → logits [batch_size, vocab_size]             │ │ │
│ │ │                                                               │ │ │
│ │ │ return ModelRunnerOutput(logits_output=logits_output)         │ │ │
│ │ └───────────────────────────────────────────────────────────────┘ │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ logits_output = out.logits_output                                     │
│                                                                       │
│ # 3. ★ 采样下一个 token                                             │
│ next_token_ids = self.model_runner.sample(                            │
│     logits_output, forward_batch                                     │
│ )                                                                     │
│ ┌─ model_runner.py ── sample()                       (line 2487) ──┐ │
│ │                                                                   │ │
│ │ # 预处理 logits（温度缩放、top-p/top-k 过滤等）                 │ │
│ │ logits = _preprocess_logits(logits_output, forward_batch)        │ │
│ │                                                                   │ │
│ │ # 执行采样                                                       │ │
│ │ next_token_ids = self.sampler(                                    │ │
│ │     logits_output,                                                │ │
│ │     forward_batch.sampling_info,                                  │ │
│ │     ...                                                           │ │
│ │ )                                                                 │ │
│ │ # → 根据 temperature, top_p, top_k, min_p 等参数采样            │ │
│ │ # → 支持 greedy (argmax), random sampling, beam search          │ │
│ │ # → 返回 next_token_ids: [batch_size]                           │ │
│ │                                                                   │ │
│ │ return next_token_ids                                             │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ # 4. 构造返回结果                                                    │
│ batch_result = BatchResult(                                           │
│     next_token_ids = next_token_ids,                                  │
│     logits_output = logits_output,                                    │
│     ...                                                               │
│ )                                                                     │
│ return batch_result                                                   │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
            BatchResult (next_token_ids) 就绪
                       │
                       ▼
```

---

#### 阶段六：结果后处理与输出流

```
┌─ scheduler.py ── STEP 5: process_batch_result() ─────(line 2416) ────┐
│                                                                       │
│ if batch.forward_mode.is_decode():                                    │
│     process_batch_result_decode(batch, result)                        │
│ elif batch.forward_mode.is_extend():                                  │
│     process_batch_result_prefill(batch, result)                       │
│                                                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│ (以 Prefill 首次完成为例)                                             │
│ ═══════════════════════════════════════════════════════════════════    │
│                                                                       │
│ ┌─ scheduler_output_processor_mixin.py ──────────────────────────┐   │
│ │ process_batch_result_prefill(batch, result)     (line 124)     │   │
│ │                                                                 │   │
│ │ # 同步 GPU→CPU 拷贝完成                                       │   │
│ │ if self.enable_overlap:                                         │   │
│ │     batch.copy_done.synchronize()                               │   │
│ │                                                                 │   │
│ │ next_token_ids = result.next_token_ids.tolist()                 │   │
│ │                                                                 │   │
│ │ for i, (req, next_token_id) in enumerate(                       │   │
│ │     zip(batch.reqs, next_token_ids)                             │   │
│ │ ):                                                              │   │
│ │     # Chunked Prefill 未完成 → 不生成 token，继续下一块       │   │
│ │     if req.is_chunked > 0:                                      │   │
│ │         req.is_chunked -= 1                                     │   │
│ │         continue                                                │   │
│ │                                                                 │   │
│ │     # 记录输出 token                                            │   │
│ │     req.output_ids.append(next_token_id)                        │   │
│ │                                                                 │   │
│ │     # 检查停止条件                                              │   │
│ │     req.check_finished()                                        │   │
│ │     # → 检查: EOS token? max_new_tokens? stop_strings?         │   │
│ │                                                                 │   │
│ │     if req.finished():                                          │   │
│ │         ┌─ 请求完成 ──────────────────────────────────────────┐│   │
│ │         │ # ★ 释放 KV Cache 并缓存到 RadixCache              ││   │
│ │         │ release_kv_cache(req, self.tree_cache)               ││   │
│ │         │ ┌─ common.py ── release_kv_cache()   (line 465) ──┐││   │
│ │         │ │                                                  │││   │
│ │         │ │ tree_cache.cache_finished_req(req)                │││   │
│ │         │ │ ┌─ radix_cache.py  (line 446) ──────────────────┐│││   │
│ │         │ │ │ # 将完整 token_ids→KV 映射插入基数树          ││││   │
│ │         │ │ │ insert(key=req.origin_input_ids+output_ids,    ││││   │
│ │         │ │ │        value=kv_indices)                       ││││   │
│ │         │ │ │ # 释放重复的 KV 索引 (如果树中已有)           ││││   │
│ │         │ │ │ # dec_lock_ref(): 解除对前缀节点的锁定        ││││   │
│ │         │ │ └───────────────────────────────────────────────┘│││   │
│ │         │ │                                                  │││   │
│ │         │ │ # 释放 req_to_token_pool 中的 req 槽位          │││   │
│ │         │ │ req_to_token_pool.free(req.req_pool_idx)         │││   │
│ │         │ └──────────────────────────────────────────────────┘││   │
│ │         └─────────────────────────────────────────────────────┘│   │
│ │     else:                                                       │   │
│ │         ┌─ 请求未完成（进入 Decode 循环）────────────────────┐ │   │
│ │         │ # 缓存当前已计算的 KV 到 RadixCache               │ │   │
│ │         │ tree_cache.cache_unfinished_req(req)                │ │   │
│ │         │ ┌─ radix_cache.py  (line 493) ────────────────────┐│ │   │
│ │         │ │ # 插入到目前为止的 token_ids→KV 映射            ││ │   │
│ │         │ │ insert(key=fill_ids, value=kv_indices)           ││ │   │
│ │         │ │ # 重新 match_prefix 更新 prefix_indices          ││ │   │
│ │         │ │ # 转移锁引用: dec_lock_ref(old) + inc_lock_ref  ││ │   │
│ │         │ └─────────────────────────────────────────────────┘│ │   │
│ │         │ # → 下一轮循环中进入 running_batch 参与 Decode     │ │   │
│ │         └────────────────────────────────────────────────────┘ │   │
│ │                                                                 │   │
│ │ # ★ 发送结果到 DetokenizerManager                             │   │
│ │ stream_output(batch.reqs, ...)                     (line 858)  │   │
│ │ ┌────────────────────────────────────────────────────────────┐  │   │
│ │ │ stream_output_generation(reqs)                  (line 886) │  │   │
│ │ │                                                            │  │   │
│ │ │ # 收集所有请求的输出信息                                   │  │   │
│ │ │ output = BatchTokenIDOutput(                               │  │   │
│ │ │     rids = [req.rid for req in reqs],                      │  │   │
│ │ │     output_ids = [req.output_ids[-1] for req in reqs],     │  │   │
│ │ │     finished_reasons = [req.finished_reason for ...],       │  │   │
│ │ │     read_offsets = [...],                                   │  │   │
│ │ │     logprob_metadata = [...],                               │  │   │
│ │ │     ...                                                    │  │   │
│ │ │ )                                                          │  │   │
│ │ │                                                            │  │   │
│ │ │ self.send_to_detokenizer.send_output(output)               │  │   │
│ │ │ # ★ ZMQ 发送到 DetokenizerManager 子进程                  │  │   │
│ │ └────────────────────────────────────────────────────────────┘  │   │
│ └────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
        BatchTokenIDOutput 通过 ZMQ 发送到 DetokenizerManager
                       │
                       ▼
```

---

#### 阶段七：DetokenizerManager — 增量解码

```
┌─ detokenizer_manager.py ── DetokenizerManager 子进程 ────────────────┐
│                                                                       │
│ async def event_loop():                              (line 144)      │
│   while True:                                                         │
│       recv_obj = await self.recv_from_scheduler.recv_pyobj()         │
│       # 按类型分发                                                    │
│       if isinstance(recv_obj, BatchTokenIDOutput):                    │
│           → handle_batch_token_id_out(recv_obj)                       │
│       elif isinstance(recv_obj, BatchEmbeddingOut):                   │
│           → handle_batch_embedding_out(recv_obj)                      │
│                                                                       │
│ ┌─ handle_batch_token_id_out(recv_obj)               (line 360) ───┐ │
│ │                                                                   │ │
│ │ # 增量解码: token_ids → text                                     │ │
│ │ output = _decode_batch_token_id_output(recv_obj)   (line 225)    │ │
│ │ ┌────────────────────────────────────────────────────────────┐   │ │
│ │ │ _decode_batch_token_id_output(batch_output):               │   │ │
│ │ │                                                            │   │ │
│ │ │ for rid, output_id, read_offset, ... in zip(...):          │   │ │
│ │ │   decode_status = self.decode_status[rid]                  │   │ │
│ │ │   # 维护每个请求的解码状态:                                │   │ │
│ │ │   #   decode_status.output_ids  (累积的所有输出 token)     │   │ │
│ │ │   #   decode_status.read_offset (已解码到的位置)           │   │ │
│ │ │   decode_status.output_ids.append(output_id)               │   │ │
│ │ │                                                            │   │ │
│ │ │ # ★ 批量增量解码（高效分词器批量调用）                    │   │ │
│ │ │ decoded_texts = _grouped_batch_decode(                      │   │ │
│ │ │     tokenizer,                                              │   │ │
│ │ │     output_ids_list,                                        │   │ │
│ │ │     read_offsets,                                           │   │ │
│ │ │     skip_special_tokens                                     │   │ │
│ │ │ )                                                          │   │ │
│ │ │ # → 使用 tokenizer.batch_decode() 进行增量解码            │   │ │
│ │ │ # → 只解码新增的 token，不重复已解码的部分                 │   │ │
│ │ │ # → 处理 UTF-8 多字节字符的边界情况                       │   │ │
│ │ │                                                            │   │ │
│ │ │ return BatchStrOutput(                                      │   │ │
│ │ │     rids = recv_obj.rids,                                  │   │ │
│ │ │     decoded_texts = decoded_texts,                          │   │ │
│ │ │     output_ids = output_ids_list,                           │   │ │
│ │ │     finished_reasons = recv_obj.finished_reasons,           │   │ │
│ │ │     meta_info = meta_info_list,                             │   │ │
│ │ │     ...                                                    │   │ │
│ │ │ )                                                          │   │ │
│ │ └────────────────────────────────────────────────────────────┘   │ │
│ │                                                                   │ │
│ │ # ★ 发送回 TokenizerManager 主进程                               │ │
│ │ self.send_to_tokenizer.send_pyobj(output)                        │ │
│ │ # ZMQ 发送 BatchStrOutput                                       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
        BatchStrOutput 通过 ZMQ 返回到 TokenizerManager 主进程
                       │
                       ▼
```

---

#### 阶段八：TokenizerManager — 结果接收与请求完成

```
┌─ tokenizer_manager.py ── 主进程后台协程 ─────────────────────────────┐
│                                                                       │
│ async def handle_loop():                             (line 1474)     │
│   while True:                                                         │
│       recv_obj = await self.recv_from_detokenizer.recv_pyobj()       │
│       self._result_dispatcher(recv_obj)                               │
│       # TypeBasedDispatcher 路由:                                     │
│       #   BatchStrOutput → _handle_batch_output()                    │
│                                                                       │
│ ┌─ _handle_batch_output(recv_obj: BatchStrOutput)    (line 1483) ──┐ │
│ │                                                                   │ │
│ │ for i, rid in enumerate(recv_obj.rids):                          │ │
│ │     state = self.rid_to_state.get(rid)                           │ │
│ │     if state is None:                                             │ │
│ │         continue  # 已被 abort 的请求                            │ │
│ │                                                                   │ │
│ │     # 构造输出字典                                                │ │
│ │     out_dict = {                                                  │ │
│ │         "text": recv_obj.decoded_texts[i],                       │ │
│ │         "meta_info": recv_obj.meta_info[i],                      │ │
│ │         "output_ids": recv_obj.output_ids[i],                    │ │
│ │         "finished_reason": recv_obj.finished_reasons[i],         │ │
│ │     }                                                             │ │
│ │                                                                   │ │
│ │     state.out_list.append(out_dict)                               │ │
│ │                                                                   │ │
│ │     if recv_obj.finished_reasons[i] is not None:                  │ │
│ │         state.finished = True                                     │ │
│ │                                                                   │ │
│ │     # ★★★ 唤醒等待中的 _wait_one_response() ★★★              │ │
│ │     state.event.set()                                             │ │
│ │                                                                   │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│ _wait_one_response() 被唤醒后：                                       │
│ ═══════════════════════════════════════════════════════════════════    │
│   state.event.wait() 返回                                             │
│   out = state.out_list[-1]                                            │
│   state.finished == True  →  yield out; break                        │
│   # 清理: del self.rid_to_state[rid]                                 │
│                                                                       │
│ → yield response 传回 serving_base.py                                │
└──────────────────────┬────────────────────────────────────────────────┘
                       │
                       ▼
```

---

#### 阶段九：HTTP 响应返回

```
┌─ serving_base.py ─────────────────────────────────────────────────────┐
│                                                                       │
│ _handle_non_streaming_request():                                      │
│   async for response in tokenizer_manager.generate_request(...):     │
│       final_response = response                                       │
│   # response = {"text": "...", "output_ids": [...], ...}             │
│                                                                       │
│   # 构造 OpenAI 格式的 ChatCompletionResponse                        │
│   return ChatCompletionResponse(                                      │
│       id = f"chatcmpl-{uuid}",                                       │
│       object = "chat.completion",                                     │
│       created = timestamp,                                            │
│       model = model_name,                                             │
│       choices = [ChatCompletionChoice(                                │
│           index = 0,                                                  │
│           message = ChatMessage(                                      │
│               role = "assistant",                                     │
│               content = response["text"],                             │
│           ),                                                          │
│           finish_reason = response["finished_reason"],                │
│       )],                                                             │
│       usage = UsageInfo(                                              │
│           prompt_tokens = ...,                                        │
│           completion_tokens = ...,                                    │
│           total_tokens = ...,                                         │
│       ),                                                              │
│   )                                                                   │
│                                                                       │
│ → FastAPI 序列化为 JSON，HTTP 200 返回给用户                         │
└───────────────────────────────────────────────────────────────────────┘
```

---

#### 阶段十：Decode 循环（如果请求未完成）

```
请求在 Prefill 后未完成（req.finished() == False）
  → req 被加入 self.running_batch (Scheduler.get_next_batch_to_run 中合并)
  → 进入 Decode 循环

┌─ scheduler.py ── Decode 循环 ─────────────────────────────────────────┐
│                                                                       │
│ # 在 event_loop_normal() 的下一个迭代中:                             │
│                                                                       │
│ get_next_batch_to_run():                                              │
│   # 无新 prefill 请求 → 运行 decode                                  │
│   batch = update_running_batch(self.running_batch)                    │
│                                                                       │
│ ┌─ update_running_batch(batch)                       (line 2172) ──┐ │
│ │                                                                   │ │
│ │ # 1. 过滤已完成的请求                                            │ │
│ │ batch.filter_batch(filter_finished_reqs)                          │ │
│ │                                                                   │ │
│ │ # 2. 检查 Decode 内存是否足够                                    │ │
│ │ if not batch.check_decode_mem():                                  │ │
│ │     # 内存不足 → retract（抢占低优先级请求）                     │ │
│ │     retracted_reqs, new_token_ratio, _ = batch.retract_decode()  │ │
│ │     # → 将被抢占的请求释放 KV Cache，放回 waiting_queue          │ │
│ │     for req in retracted_reqs:                                    │ │
│ │         self.waiting_queue.appendleft(req)                        │ │
│ │                                                                   │ │
│ │ # 3. ★ 准备 Decode 批次                                         │ │
│ │ batch.prepare_for_decode()                         (line 1948)   │ │
│ │ ┌────────────────────────────────────────────────────────────┐   │ │
│ │ │ prepare_for_decode():                                      │   │ │
│ │ │   self.forward_mode = ForwardMode.DECODE                   │   │ │
│ │ │                                                            │   │ │
│ │ │   # 为每个请求分配 1 个新 KV token 槽位                   │   │ │
│ │ │   alloc_for_decode(self, mem_cache)             (common.py │   │ │
│ │ │                                                 line 423)  │   │ │
│ │ │   # → kv_allocator.alloc(batch_size)                       │   │ │
│ │ │   # → 写入 req_to_token_pool[req_idx, seq_len] = new_slot │   │ │
│ │ │                                                            │   │ │
│ │ │   # 更新 seq_lens += 1                                    │   │ │
│ │ │   self.seq_lens += 1                                       │   │ │
│ │ └────────────────────────────────────────────────────────────┘   │ │
│ │                                                                   │ │
│ │ return batch  # ForwardMode.DECODE                                │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ # 然后执行:                                                          │
│ result = run_batch(batch)                                             │
│   → TpModelWorker.forward_batch_generation()                         │
│     → ForwardBatch.init_new() (DECODE 模式)                          │
│     → ModelRunner.forward()                                           │
│       → _forward_raw() → graph_runner.replay() 或 forward_decode()   │
│       → model.forward(input_ids=[batch_size], ...)  # 每请求 1 token │
│     → ModelRunner.sample() → next_token_ids                          │
│                                                                       │
│ process_batch_result(batch, result)                                   │
│   → process_batch_result_decode()                                     │
│     → req.output_ids.append(next_token_id)                            │
│     → req.check_finished()                                            │
│     → if finished: release_kv_cache(req)                              │
│     → stream_output() → DetokenizerManager → TokenizerManager        │
│                                                                       │
│ ════════════════════════════════════════════════════════════════       │
│ 循环继续，直到所有请求 finished                                       │
│ ════════════════════════════════════════════════════════════════       │
└───────────────────────────────────────────────────────────────────────┘
```

---

#### 端到端完整数据流总结

```
┌─────────┐  HTTP POST   ┌───────────┐ GenerateReqInput  ┌────────────────┐
│  Client │ ──────────→ │ FastAPI   │ ──────────────→  │ OpenAIServing  │
│         │              │ Router    │                   │ Chat/Base      │
└─────────┘              └───────────┘                   └───────┬────────┘
                                                                 │
                    ┌────────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────┐     ZMQ (IPC)
    │      TokenizerManager (主进程)       │ ─────────────────┐
    │                                     │                    │
    │  tokenize → send → wait(event)      │                    │
    │       ▲                             │                    ▼
    │       │ event.set()                 │   ┌───────────────────────────┐
    │       │                             │   │    Scheduler (子进程)      │
    │  handle_loop() ← recv ← ZMQ        │   │                           │
    └──────────────┬──────────────────────┘   │  recv → dispatch          │
                   │                          │  → handle_generate_req    │
                   │                          │  → waiting_queue          │
                   │ ZMQ                      │                           │
                   │                          │  get_next_batch_to_run:   │
    ┌──────────────┴──────────────────────┐   │  ├→ calc_priority         │
    │  DetokenizerManager (子进程)         │   │  ├→ match_prefix (Radix) │
    │                                     │   │  ├→ PrefillAdder.add_req │
    │  recv(BatchTokenIDOut)              │   │  ├→ alloc KV Cache       │
    │  → batch_decode (token→text)        │   │  └→ prepare_for_extend   │
    │  → send(BatchStrOutput)             │   │                           │
    └─────────────────────────────────────┘   │  run_batch:               │
                   ▲                          │  └→ TpModelWorker         │
                   │ ZMQ                      │     └→ ModelRunner        │
                   │                          │       ├→ forward()        │
    ┌──────────────┴──────────────────────┐   │       │  (CUDA Graph /   │
    │    stream_output()                   │   │       │   Eager mode)    │
    │    → BatchTokenIDOutput              │   │       └→ sample()        │
    │    → send_to_detokenizer             │←──│                           │
    └─────────────────────────────────────┘   │  process_batch_result:    │
                                              │  ├→ output_ids.append    │
                                              │  ├→ check_finished       │
                                              │  ├→ release/cache KV     │
                                              │  └→ stream_output()      │
                                              └───────────────────────────┘

进程边界:
  ═══════════════════════════════════════
  主进程:  FastAPI + TokenizerManager
  子进程1: Scheduler + TpModelWorker + ModelRunner (per TP group)
  子进程2: DetokenizerManager
  通信:    ZMQ IPC (pyobj 序列化)
  ═══════════════════════════════════════

关键异步机制:
  1. asyncio.Event    — TokenizerManager 中 send ↔ wait 的同步
  2. ZMQ non-blocking — Scheduler 轮询式接收（不阻塞事件循环）
  3. CUDA Stream      — Overlap 模式下 forward_stream / default_stream 并行
  4. FutureMap        — Overlap 模式下延迟采样结果占位
```

### 3.4 核心模块说明

| 模块路径 | 说明 |
|---------|------|
| `srt/entrypoints/` | 入口点：HTTP Server、gRPC Server、OpenAI 兼容 API、Ollama API |
| `srt/managers/scheduler.py` | **核心调度器**（126KB）：事件循环、batch 调度、overlap 调度、推测解码协调 |
| `srt/managers/schedule_batch.py` | Batch 数据结构（Req、ScheduleBatch 等） |
| `srt/managers/schedule_policy.py` | 调度策略（Prefill/Decode 优先级、内存管理） |
| `srt/managers/tokenizer_manager.py` | Tokenizer 管理器（独立进程，处理 tokenize/detokenize） |
| `srt/model_executor/model_runner.py` | 模型执行器（加载模型、前向推理、CUDA Graph 捕获） |
| `srt/model_executor/cuda_graph_runner.py` | CUDA Graph 捕获和重放 |
| `srt/mem_cache/` | 内存/KV Cache 管理（Radix Cache、Memory Pool 等） |
| `srt/layers/` | 模型层实现（Attention、Linear、MoE、LayerNorm、Sampler 等） |
| `srt/models/` | **150+ 模型实现**（Llama、Qwen、DeepSeek、Gemma、GPT 等） |
| `srt/speculative/` | 推测解码（EAGLE、EAGLE3、NGRAM、Standalone） |
| `srt/disaggregation/` | Prefill-Decode 分离架构 |
| `srt/sampling/` | 采样策略和惩罚机制 |
| `srt/constrained/` | 受约束解码（结构化输出，xgrammar/outlines/llguidance） |
| `srt/lora/` | LoRA 适配器管理 |
| `srt/distributed/` | 分布式通信 |

---

## 四、核心性能优化技术

### 4.1 RadixAttention（前缀缓存）

**核心思想**：用 Radix Tree（基数树）管理 KV Cache，使得具有**相同前缀的请求可以共享 KV Cache**，避免重复计算。

- **实现**：`srt/mem_cache/radix_cache.py` — `RadixCache` 类
- **效果**：多轮对话、共享 system prompt 场景下，**最高 5x 加速**
- **驱逐策略**：支持 LRU / LFU

```
请求1: [system_prompt] + [user_msg_A]     ← 计算完整 KV
请求2: [system_prompt] + [user_msg_B]     ← 复用 system_prompt 的 KV Cache！
```

### 4.2 Continuous Batching（连续批处理）

**核心思想**：不等一个 batch 中最长的请求完成再处理下一批，而是**请求完成即释放资源、立即插入新请求**。

- GPU 始终有活干，利用率大幅提升
- 配合 Paged Attention 管理动态变化的 KV Cache

### 4.3 Overlap 调度（GPU/CPU 重叠执行）

**核心思想**：通过 CUDA 多流，让当前 batch 的 GPU 前向计算与上一个 batch 的 CPU 后处理并行。

- **实现**：`scheduler.py` 中的 `event_loop_overlap()`
- **关键机制**：`FutureMap` — 用 future 占位符代替尚未就绪的采样结果

```
GPU: [===batch1 forward===][===batch2 forward===][===batch3 forward===]
CPU:                       [batch1 后处理]       [batch2 后处理]
```

### 4.4 推测解码（Speculative Decoding）

**核心思想**：用轻量 Draft 模型**连续多步**猜测候选 token，然后 Target 模型**一次验证**，接受多个 token。

- **算法**：EAGLE / EAGLE3 / Standalone / NGRAM
- **配置**：`speculative_num_steps`（草稿步数）、`speculative_eagle_topk`（每步 top-k）
- **实现**：`srt/speculative/eagle_worker.py`、`eagle_worker_v2.py`（overlap 版本）

### 4.5 Chunked Prefill（分块预填充）

**核心思想**：长 prompt 分块处理，避免一次 prefill 过大导致 decode 请求饿死。

### 4.6 CUDA Graph

**核心思想**：捕获固定 shape 的 GPU kernel 调用图，后续重放避免 CPU launch 开销。

- **实现**：`srt/model_executor/cuda_graph_runner.py`

### 4.7 Paged Attention（分页注意力）

**核心思想**：KV Cache 按固定大小 page 管理，避免连续内存分配，支持动态增长。

- **实现**：`srt/mem_cache/memory_pool.py`（72KB）

### 4.8 结构化输出（Constrained Decoding）

**核心思想**：通过 grammar（如 JSON Schema）约束解码输出，用压缩有限状态机加速。

- **后端**：xgrammar / outlines / llguidance
- **效果**：JSON 解码 **3x 加速**

---

## 五、并行策略

| 策略 | 说明 | 关键文件 |
|------|------|---------|
| **Tensor Parallelism (TP)** | 模型按 head/hidden 维度切分到多 GPU | `srt/distributed/`, `srt/layers/communicator.py` |
| **Pipeline Parallelism (PP)** | 模型按层切分，流水线执行 | `srt/managers/scheduler_pp_mixin.py` |
| **Expert Parallelism (EP)** | MoE 模型的专家切分 | `srt/elastic_ep/`, `srt/eplb/` |
| **Data Parallelism (DP)** | 多副本处理不同请求 | `srt/managers/data_parallel_controller.py` |
| **DP Attention** | 注意力层的数据并行 | `srt/layers/dp_attention.py` |
| **Prefill-Decode 分离 (PD)** | Prefill 和 Decode 使用不同 GPU/节点 | `srt/disaggregation/` |

---

## 六、Attention Backend（注意力后端）

SGLang 支持 **15+ 种注意力后端**，覆盖多种硬件和优化策略：

| 后端 | 说明 |
|------|------|
| `flashinfer` | 默认后端，高性能 FlashInfer 库 |
| `fa3` / `fa4` | Flash Attention 3/4 |
| `flashmla` | Flash MLA（DeepSeek Multi-Latent Attention） |
| `cutlass_mla` | CUTLASS 实现的 MLA |
| `trtllm_mla` / `trtllm_mha` | TensorRT-LLM 实现 |
| `triton` | Triton kernel 实现（通用性好） |
| `aiter` / `wave` | AMD GPU 专用 |
| `intel_amx` | Intel AMX 加速 |
| `nsa` | Native Sparse Attention（稀疏注意力） |
| `dual_chunk_flash_attn` | 双块 Flash Attention |

---

## 七、支持的模型（150+）

### 语言模型
Llama (1/2/3/4)、Qwen (1/2/2.5/3/3.5)、DeepSeek (V2/V3/R1)、Gemma (1/2/3/3N)、GLM-4、Mistral/Mixtral、GPT-2/J/BigCode、Grok、Kimi、MiniCPM、OLMo、Phi (1/2/3/4)、DBRX、Falcon 等

### 多模态模型
LLaVA、InternVL、Qwen-VL、DeepSeek-VL2、Pixtral、MiniCPM-V、Phi-4-MM、NVILA 等

### 特殊模型
- **嵌入模型**：e5-mistral, gte, mcdse
- **奖励模型**：Skywork, InternLM2-Reward, Gemma2-Reward
- **扩散模型**：WAN, Qwen-Image, LLaDA2
- **Mamba/SSM 模型**：Falcon-H1, NemotronH

---

## 八、量化支持

| 量化方式 | 说明 |
|---------|------|
| FP8 | 8 位浮点量化 |
| FP4 / MXFP4 / MXFP8 | 4 位/微缩浮点 |
| INT4 (W4A16) | 4 位权重量化 |
| AWQ / AWQ-Marlin | 激活感知权重量化 |
| GPTQ / GPTQ-Marlin | 训练后量化 |
| BitsAndBytes | 混合精度量化 |
| GGUF | llama.cpp 格式 |
| ModelOpt FP8/FP4 | NVIDIA ModelOpt 框架 |
| W8A8 INT8/FP8 | 权重和激活都量化 |

---

## 九、sgl-kernel（高性能内核库）

独立的 C++/CUDA 内核包，提供底层高性能算子：

| 内核模块 | 说明 |
|---------|------|
| `csrc/attention/` | 注意力计算 kernel |
| `csrc/gemm/` | 矩阵乘法（FP8/FP4/INT4 等） |
| `csrc/moe/` | MoE 专家路由和计算 |
| `csrc/allreduce/` | 多 GPU 全归约通信 |
| `csrc/elementwise/` | 逐元素操作 |
| `csrc/speculative/` | 推测解码相关 kernel |
| `csrc/memory/` | 内存管理 |
| `csrc/quantization/` | 量化相关 |
| `csrc/kvcacheio/` | KV Cache I/O |
| `csrc/mamba/` | Mamba/SSM 模型 kernel |

---

## 十、API 兼容性

### OpenAI 兼容 API
- `/v1/chat/completions` — 对话补全
- `/v1/completions` — 文本补全
- `/v1/embeddings` — 嵌入
- `/v1/models` — 模型列表
- `/v1/chat/completions`（Responses API）— 响应式 API

### 其他 API
- **Ollama 兼容 API**
- **gRPC API** — 高性能 RPC 接口
- **Native SGLang API** — 原生 Python SDK

---

## 十一、部署方式

| 方式 | 说明 |
|------|------|
| pip/uv 安装 | `uv pip install sglang` |
| 源码安装 | `pip install -e "python"` |
| Docker | `lmsysorg/sglang:latest` |
| Kubernetes | OME Operator / StatefulSet |
| Docker Compose | 生产级编排 |
| SkyPilot | 跨云部署 |
| AWS SageMaker | 托管推理 |

---

## 十二、硬件支持

| 平台 | 型号 |
|------|------|
| **NVIDIA GPU** | GB200, B300, H100, A100, L40S, A10, T4, Spark |
| **AMD GPU** | MI355, MI300 |
| **Intel CPU** | Xeon (AMX 加速) |
| **Google TPU** | SGLang-Jax 后端 |
| **Ascend NPU** | 华为昇腾 |
| **Intel XPU** | Intel 独立 GPU |
| **NVIDIA Jetson** | 边缘推理 |

---

## 十三、关键配置参数速览

```python
# 模型和分词器
--model-path            # 模型路径（HuggingFace 或本地）
--context-length        # 上下文长度

# 并行
--tp-size               # 张量并行度
--dp-size               # 数据并行度
--pp-size               # 流水线并行度

# 内存和缓存
--mem-fraction-static   # 静态内存占比
--chunked-prefill-size  # 分块预填充大小

# 注意力
--attention-backend     # 注意力后端（flashinfer/triton/fa3/...）

# 量化
--quantization          # 量化方式（fp8/awq/gptq/...）

# 推测解码
--speculative-algorithm           # 推测算法（EAGLE/EAGLE3/NGRAM/...）
--speculative-num-steps           # 草稿步数
--speculative-eagle-topk          # 每步 top-k
--speculative-num-draft-tokens    # 草稿 token 数

# 调度
--schedule-policy       # 调度策略
--enable-overlap        # 启用 overlap 调度
```

---

## 十四、性能亮点

- **RadixAttention 前缀缓存**：共享前缀场景 **5x 加速**
- **DeepSeek MLA 优化**：**7x 加速**（v0.3）
- **JSON 结构化输出**：**3x 加速**（压缩 FSM）
- **torch.compile**：**1.5x 加速**（v0.3）
- **Zero-Overhead Scheduler**：v0.4 零开销调度器
- **GB200 大规模部署**：Prefill **3.8x**，Decode **4.8x** 吞吐提升
- **大规模 EP（96 H100）**：DeepSeek 高效 Expert Parallelism

---

## 十五、与其他框架对比

| 特性 | SGLang | vLLM | TensorRT-LLM |
|------|--------|------|--------------|
| RadixAttention 前缀缓存 | ✅ | ❌（APC 类似但不同） | ❌ |
| Overlap 调度 | ✅ | ❌ | ❌ |
| 推测解码 | ✅ EAGLE/EAGLE3/NGRAM | ✅ 部分 | ✅ |
| PD 分离 | ✅ | ✅ | ✅ |
| 结构化输出加速 | ✅ 压缩 FSM | ✅ | ❌ |
| RL 后训练集成 | ✅ 原生支持 | ❌ | ❌ |
| 多硬件支持 | ✅ 6+ 平台 | ✅ 有限 | ❌ NVIDIA only |
| 模型数量 | 150+ | 100+ | 30+ |

---

## 十六、RL 与后训练

SGLang 作为 RL rollout 后端被广泛采用：

- **AReaL**（Inclusion AI）
- **verl**（字节跳动/火山引擎）
- **Miles**（RadixArk）
- **slime**（清华 THUDM）
- **Tunix**（Google）

支持在线 RL 训练中的高效推理采样，提供权重同步（`weight_sync/`）和在线策略目标（`rl_on_policy_target`）。

---

## 十七、总结

SGLang 的核心竞争力在于**系统级优化的深度和广度**：

1. **调度层**：零开销调度器 + Overlap GPU/CPU 重叠 + Continuous Batching
2. **缓存层**：RadixAttention 前缀缓存 + Paged Attention + 分层缓存（HiCache）
3. **计算层**：15+ 注意力后端 + CUDA Graph + 高性能自定义 kernel
4. **算法层**：推测解码 + 结构化输出 FSM 加速
5. **分布式层**：TP/PP/EP/DP 全并行 + PD 分离
6. **生态层**：150+ 模型 + 6+ 硬件平台 + OpenAI API 兼容 + RL 框架集成
