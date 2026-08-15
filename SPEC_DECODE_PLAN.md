# Spec Decode 实施计划

> 以 vLLM V1 (https://github.com/vllm-project/vllm) 核心架构为对齐目标。
> 本文档是后续阶段性优化的唯一依据，完成一项勾掉一项。

---

## 0. vLLM V1 架构要点（对齐基准）

以下来自 vLLM 源码分析，是后续所有改动的参照标准。

### 0.1 统一调度

- `Scheduler.schedule()` 不区分 prefill/decode 分支。每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`（= prompt + output + spec_token_ids）。
- Scheduler 尝试让 `num_computed_tokens` 追上 `num_tokens_with_spec`，天然覆盖 chunked prefill、prefix caching、spec decode。
- spec token 通过 `request.spec_token_ids` 挂在请求上，Scheduler 输出 `scheduled_spec_decode_tokens: dict[str, list[int]]`。
- chunked prefill + spec decode 天然共存：同一 batch 里可以有 prefill chunk 行和 decode+spec 行。

### 0.2 Draft 模型 KV cache

- Draft 模型是完整的 vLLM model，注册在独立 KV cache group，由同一个 `KVCacheManager` 统一管理。
- Draft 的 attention 走和目标模型相同的 paged attention kernel、`block_table`、`slot_mapping`。
- **没有独立的 `extend()` 步骤**——draft KV cache 是分页的，已确认 token 的 draft KV 天然正确，不需要手动 catch-up。

### 0.3 Proposer 接口

```python
# vllm/v1/spec_decode/llm_base_proposer.py
def propose(
    self,
    num_speculative_tokens,
    target_token_ids: torch.Tensor,        # [num_tokens] 扁平
    target_positions: torch.Tensor,        # [num_tokens]
    target_hidden_states: torch.Tensor,    # [num_tokens, hidden_size]
    next_token_ids: torch.Tensor,          # [batch_size] target 的采样结果
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    ...
) -> torch.Tensor:  # [batch_size, num_spec_tokens]
```

目标模型的 hidden states 直接作为 `propose()` 的参数传入，不需要跨调用暂存。

### 0.4 执行流程

```
execute_model():
  1. target model forward → hidden_states, logits, aux_hidden_states
  2. if spec_decode_metadata is not None:
       drafter.propose(target_token_ids, target_positions,
                       target_hidden_states, next_token_ids, ...)
       rejection_sampler(...)
  3. return sampled_token_ids
```

没有独立的 `run_spec()` 入口，spec decode 集成在 `execute_model()` 内部。

### 0.5 EAGLE3 aux hidden states

- `GPUModelRunner.__init__` 里设置 `self.use_aux_hidden_state_outputs = True`（当 method == "eagle3"）。
- 模型 forward 返回 `aux_hidden_states: list[torch.Tensor] | None`，作为 `ExecuteModelState` 的字段。
- 在同一个 `execute_model()` 调用内直接传给 `propose()`。

### 0.5.1 ⚠️ aux/token 配对约定（重要，曾搞反）

**正确约定（vLLM `eagle.py` + EAGLE 论文 3.2 节双重证实）：shifted-token 错位配对**。

vLLM `EagleProposer.propose()` 开头：
```python
# Shift the input ids by one token.      # token 前移一位
self.input_ids[:num_tokens - 1] = target_token_ids[1:]
self.input_ids[last_token_indices] = next_token_ids
self.positions[:num_tokens] = target_positions          # positions 不移位
self.hidden_states[:num_tokens] = target_hidden_states  # hidden 不移位
```

EAGLE 论文训练损失：`L_reg = SmoothL1(f_{i+1}, Draft_Model(T_{2:i+1}, F_{1:i}))` —— token 序列 T₂..T_{i+1} 配特征序列 F₁..F_i。

**结论：draft 第 p 行 = (token t_{p+1}, position p, hidden f_p)——每个 token 配的是"预测出它"的 hidden（它前一个位置的 hidden），RoPE position 也是 hidden 的位置（token 位置 - 1）。**

推论：
- verify 窗口 `[last@start, d_0@start+1, ..., d_{K-1}@start+K]` 中，accepted[i]（位置 start+1+i）配窗口第 **i** 行（不是 i+1）
- bonus token（位置 start+K+1）配窗口最后一行（start+K）——**天然存在，无需 dummy/pad**
- prefill 暂存：token_ids[start+1 .. start+n]，且最后一个 chunk 的收尾 token 是**本步采样出的 token**（它配 hidden@P-1）——所以 stash 必须在采样之后调用
- draft committed KV 长度恒等于 len(seq) - 1（最后那个已采样 token 还没有配对的 hidden，要等下一次 forward）

**反例教训**：曾错误地按"同位置配对 (token_p, hidden_p)"实现，导致 verify stash 多跳一行、bonus 需要 pad 近似、首轮缺 1 token 的"bug"——这些都是错误约定下的伪问题，约定修正后全部自然消解。

### 0.6 Tree Attention

- vLLM **不支持** tree attention。GitHub issue #18327 被 closed as not planned。
- vLLM 的 EAGLE 只做链式（chain）drafting。

### 0.7 Rejection Sampler

- `RejectionSampler(sampler, speculative_config, device)` 三个参数。
- 支持 greedy（argmax 比对）和 probabilistic（比率接受 + 残差采样）两种模式。

---

## 1. 当前状态盘点

### 1.1 已完成

| 模块 | 文件 | 内容 |
|------|------|------|
| 目标模型 | `models/llama.py` | `LlamaModel.forward` 支持 `aux_layer_ids`，产出 `_aux_hidden_states`（3×hidden_size 拼接） |
| Draft 模型 | `models/eagle3_draft.py` | `Eagle3DraftModel` 完整实现：fc(3H→H)、单层 decoder、共享 embed_tokens、d2t 映射 |
| 元数据 | `spec_decode/metadata.py` | `SpecDecodeMetadata` + `make_spec_decode_metadata()` |
| 拒绝采样 | `spec_decode/rejection_sampler.py` | `RejectionSampler`，greedy + probabilistic 两种模式 |
| Proposer | `spec_decode/proposer.py` | `extend()` / `propose()` / `drop()` 方法（但未接线） |
| 权重加载 | `utils/loader.py` | `load_eagle3_weights()` |
| 调度器 | `engine/scheduler.py` | `schedule()`/`schedule_chunked()` decode 通道分配 K+1 block |
| 调度器 | `engine/scheduler.py` | `postprocess_spec()` 消费 accepted token 列表 |
| 调度器 | `engine/scheduler.py` | prefix-cache 规避：spec 开启时 `can_allocate` 强制全量分配 |
| Block 管理 | `engine/block_manager.py` | `can_allocate()` 支持 `enable_prefix_cache` 参数 |
| 模型运行器 | `engine/model_runner.py` | `run_spec()` 骨架、`prepare_spec_decode()` |
| 模型运行器 | `engine/model_runner.py` | Prefill CUDA graph 与 spec decode 互斥 |
| 引擎 | `engine/llm_engine.py` | `step()` 路由到 `postprocess_spec` |

### 1.2 未完成的缺口

| 维度 | vLLM | nano-vllm 现状 | 差距 |
|------|------|---------------|------|
| 调度 | 统一 schedule()，无 prefill/decode 分支 | `schedule()` / `schedule_chunked()` 两条路径，`run()` / `run_spec()` 两个入口 | 根本性架构差异 |
| Draft KV cache | 独立 KV cache group，paged attention | `Proposer._kv` dict，稠密张量 | 需要重构 Eagle3Attention |
| extend | 不存在 | `extend()` 手动 catch-up，需要跨调用暂存 | 架构差异的下游影响 |
| propose 接口 | 接收 target_hidden_states 作为参数 | 需要先 extend 再 propose，两步分离 | 接口不对齐 |
| aux hidden | 模型 forward 返回字段，同调用内传递 | 挂在模型上，需手动提取+暂存 | 流通方式不同 |
| chunked prefill + spec | 天然支持（统一调度） | 混批降级为单 token | 需要统一调度才能解决 |
| tree attention | 不支持 | 不支持 | 一致，不需要做 |
| rejection sampling | `RejectionSampler(sampler, config, device)` | `RejectionSampler()` 无参数 | 接口差异 |

---

## 2. 阶段一：最小可行链式 spec decode（MVP）

**目标**：让链式 spec decode 在当前架构下端到端跑通。不追求和 vLLM 完全一致，但核心数据流对齐。

### 2a. 接口对齐

- [x] **2a-1** `model_runner.py`: `self.speculative_config = config.speculative_config` 挪到 `warmup_model()` 之前
  - 原因：warmup 也跑 prefill，后续 prefill 路径需要用 `self.speculative_config` 判断是否传 `aux_layer_ids`
  - vLLM 对照：`__init__` 开头就设置 `self.speculative_config`

- [x] **2a-2** `model_runner.py:73-76`: Proposer 构造加 `target_model=self.model`
  - 当前只传 2 个参数，签名要求 3 个
  - vLLM 对照：`EagleProposer(vllm_config, device, runner=self)`

- [x] **2a-3** `model_runner.py:331`: `self.proposer.propose(seqs)` 直接传 `seqs`
  - 当前传 `[seq.token_ids for seq in seqs]`（`list[list[int]]`），但 `propose(seqs: list[Sequence])` 内部访问 `seq.seq_id`、`len(seq)`

### 2b. aux hidden 流通

- [x] **2b-1** `model_runner.py` `run_model()`: prefill 分支在 spec 开启时传 `aux_layer_ids=self.proposer.aux_layer_ids`
  - vLLM 对照：`use_aux_hidden_state_outputs=True` 时模型 forward 自动返回 aux

- [x] **2b-2** `model_runner.py` `run_spec()`: verify 分支传 `aux_layer_ids=self.proposer.aux_layer_ids`
  - 当前 `self.model(input_ids, positions)` 没传，`_aux_hidden_states` 永远是 None

- [x] **2b-3** `model_runner.py`: 新增 `_pending_extend: dict[int, tuple[list[int], list[int], torch.Tensor]]`
  - key=seq_id, value=(token_ids, positions, aux_tensor)
  - **nano-vllm 特有**，vLLM 不需要（draft KV 是分页的，不需要 extend）
  - 后续阶段三完成 draft paged attention 后可删除
  - 实际挪到了 `warmup_model()` 之前初始化（消除 proposer 创建顺序依赖，见 `_spec_aux_layer_ids` 的 `getattr` 保护）

- [x] **2b-4** `model_runner.py` `run()`: prefill 末尾提取 `self.model.model._aux_hidden_states`，按 `cu_seqlens_q` 拆分成 per-seq，写入 `_pending_extend`
  - 实现为 `_stash_prefill_aux`：不再区分"是否完整 prefill"，改为无条件按 chunk 累积（`start==0` 覆盖，否则拼接），修复了原计划"只处理 prefill 完成的 seq"会丢弃中间 chunk aux 的 bug

- [x] **2b-5** `model_runner.py` `run_spec()`: rejection sampling 后提取被接受位置的 aux hidden，写入 `_pending_extend` 供下一轮 extend
  - 实现为 `_stash_verify_aux`；全接受时 bonus token 越界，用窗口最后一行近似（见 3-4）

### 2c. 生命周期接线

- [x] **2c-1** `run_spec()` 开头：从 `_pending_extend` 读出数据，调 `self.proposer.extend(...)`
  - 实现为 `_extend_pending`：对 batch 里每个 seq `pop` 出 `_pending_extend` 条目喂 extend；缺条目直接 `KeyError`（视为生命周期上游 bug，不静默兜底）
  - **nano-vllm 特有**，vLLM 的 propose 一步到位（draft KV 分页，不需要 extend）

- [x] **2c-2** `run_spec()` 中间：extend 之后调 `self.proposer.propose(seqs)`
  - 此时 `_kv` 已填充，propose 不会 KeyError

- [x] **2c-3** `run_spec()` 尾部：rejection sampling 后暂存 aux hidden 到 `_pending_extend`
  - 与 2b-5 是同一个改动（`_stash_verify_aux`）

### 2d. run_chunked 路径适配

- [x] **2d-1** `model_runner.py` `run_chunked()`: prefill chunk 也传 `aux_layer_ids`，aux hidden 追加到 `_pending_extend`
  - `run_model(..., is_prefill=True)` 内部已统一走 `_spec_aux_layer_ids()`，无需额外改
  - 新增 `_stash_chunked_aux`：只累积 `seq.is_prefill` 的 seq；混批里的 decode seq（`is_prefill=False`）不 stash——它们本轮走的是 plain argmax，没有经过 extend/propose，见下方遗留问题
  - **遗留问题（对应表格第 105 行"混批降级为单 token"）**：混批中的 decode seq 本轮 token 不经过 extend，其 `_pending_extend` 条目留着上一轮 verify 的旧数据，下次真正进入 `run_spec` 时会比 `len(seq)` 少 1 个位置。当前不修（需要统一调度，见 4b），只在代码注释里标注，不静默掩盖。

---

## 3. 阶段二：正确性兜底

**目标**：不崩了之后，消除内存泄漏和潜在正确性问题。

- [x] **3-1** seq 完成后清理 proposer 状态
  - `model_runner.py` 新增 `drop_proposer_state(seq_ids)`：清 `proposer._kv/_aux/_draft0` + `_pending_extend`
  - `llm_engine.step()` 两个分支在 postprocess 后调 `_drop_spec_state(seqs)`（新增辅助方法）：收集 `is_finished` 的 seq + scheduler 记录的 preempted ids，统一 `model_runner.call("drop_proposer_state", ids)`
  - vLLM 对照：`_update_states()` 里 `for req_id in finished_req_ids: self.requests.pop(req_id)`

- [x] **3-2** `scheduler.py` `postprocess_spec`: `num_scheduled_tokens = len(accepted_tokens) - 1`
  - 排除 bonus/recovered 的 stale KV：verify 窗口里最后一个位置写的是被拒 draft 的 KV（或全接受时根本没写），它下一轮作为 last_token 由窗口第 0 行重写
  - 不变量：spec decode 下 `num_cached_tokens == len(seq) - 1` 恒成立
  - vLLM 对照：`request.num_computed_tokens -= num_rejected` 在 `update_from_output`

- [x] **3-3** `scheduler.py` `preempt()`: 被抢占时清理 proposer 状态 + `_pending_extend`
  - `preempt()` 记录 `preempted_seq_ids`（spec 开启时），由 `llm_engine.step()` 末尾统一 drain 并调 `drop_proposer_state`
  - 必要性问题：重新 prefill 后首次 extend 若拿到旧 `_kv` 会在其上拼接 → draft 上下文错乱，不只是泄漏
  - vLLM 对照：`_preempt_request()` 里 `request.spec_token_ids = []`

- [x] **3-4** ~~bonus token aux 精确化~~ **已消解**：基于错误的同位置配对约定提出的伪问题。约定修正（见 0.5.1）后，bonus token 配对的 hidden 恰好是 verify 窗口最后一行，`_stash_verify_aux` 直接 `aux[offset:offset+len(out)]`，无越界、无近似、无需 dummy token

- [x] **3-5** ~~首轮 draft 缺 1 token 上下文~~ **已消解**：同样基于错误约定的伪问题。约定修正后，prefill stash 的收尾 token 就是本步采样出的第一个生成 token（配对 hidden@P-1），draft 上下文完整，无缺口

---

## 4. 阶段三：对齐 vLLM 核心架构

**目标**：把 nano-vllm 的 spec decode 架构改成和 vLLM 一致。

> **依赖说明（重要）**：本节三个子项（4a / 4b / 4c）中，`4a` 和 `4b` 是**正交的两个重构**，没有强依赖，可独立安排：
>
> - **4a（draft paged attention）**：改 draft 模型侧（`Eagle3Attention` → paged kernel），消除 `extend()`。**收益是架构一致性 + 正确性**（draft KV 与 target 共享 block 空间，draft 也能吃 prefix cache）。做完后调度入口完全不用动。
> - **4b（统一调度）**：改 scheduler/runner 侧，合并 prefill/decode 分支。**收益是吞吐**（chunked prefill + spec 混批按行验证），不改功能正确性，但**动调度主流程，回归面最大**。
> - **4c（propose 接口对齐）**：依赖 4a。
>
> **建议顺序**：`4a → 4c` 先做（架构一致性核心），`4b` 放最后单独评估（吞吐优化，风险最高，可独立推迟）。

### 4a. Draft 模型接入统一 paged attention

**vLLM 最核心的架构决策**：draft model 注册在独立 KV cache group，走 paged attention，由 KVCacheManager 统一管理。这消除了 `extend()` 的需求。

- [ ] **4a-1** `models/eagle3_draft.py`: `Eagle3Attention` 改为 paged attention
  - 用项目现有的 paged attention kernel（`flash_attn_with_kvcache` 或等效）
  - 替代 `F.scaled_dot_product_attention` + 显式 mask
  - 认识 `block_table`/`slot_mapping`

- [ ] **4a-2** `model_runner.py` `allocate_kv_cache()`: draft 的 KV 占用 `kv_cache` 显存中的一块（一层）
  - draft 只有 1 层 transformer，KV cache 大小 = 1 层 target 的量

- [ ] **4a-3** `engine/block_manager.py`: hash/allocate 逻辑覆盖 draft 层
  - draft 和 target 共享 block id 空间
  - 同一个 block 除了 target 的 K/V 还多存 draft 的 K/V

- [ ] **4a-4** `spec_decode/proposer.py`: 删除 `_kv`/`_aux`/`_draft0` dict 和 `extend()`/`drop()` 方法
  - draft KV 由 BlockManager 自动管理
  - 已确认 token 的 draft KV 天然正确

- [ ] **4a-5** `spec_decode/proposer.py`: `propose()` 改为接收 target hidden states 直接作为参数
  - 对齐 vLLM 接口：`propose(target_token_ids, target_positions, target_hidden_states, next_token_ids, ...)`

- [ ] **4a-6** `model_runner.py`: 删除 `_pending_extend` 暂存机制
  - 不再需要跨调用暂存 aux hidden

### 4b. 统一调度（可推迟到 4c 之后）

> **注意**：本子项会重构调度主流程，回归面最大，收益仅为吞吐（非正确性）。建议在 `4a → 4c` 完成后、作为独立的吞吐优化单独评估是否值得做。

**vLLM 的 schedule() 设计**：不区分 prefill/decode，spec token 挂在 request 上。天然支持 chunked prefill + spec decode 混批。

- [ ] **4b-1** `engine/scheduler.py`: `schedule()` 和 `schedule_chunked()` 合并为统一入口
  - 参考 vLLM：每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`
  - 不再有 `has_prefill` 分支判断

- [ ] **4b-2** `engine/scheduler.py`: spec token 通过 `seq.spec_token_ids` 挂在 Sequence 上
  - Scheduler 输出 `scheduled_spec_decode_tokens: dict[int, list[int]]`
  - 对齐 vLLM 的 `SchedulerOutput.scheduled_spec_decode_tokens`

- [ ] **4b-3** `engine/scheduler.py`: 删除混批降级规则（`schedule_chunked` 第 205-212 行）
  - 统一调度后天然支持混批按行粒度验证

- [ ] **4b-4** `engine/model_runner.py`: `run()` / `run_spec()` / `run_chunked()` 合并为统一入口
  - 参考 vLLM 的 `execute_model()`：一次 forward，然后根据 `spec_decode_metadata` 决定是否走 spec 路径

- [ ] **4b-5** `engine/model_runner.py`: `prepare_chunked()` 和 `prepare_spec_decode()` 合并
  - 统一按行处理：prefill chunk 行（0 draft）和 decode verify 行（K draft）在同一个 varlen batch 里

- [ ] **4b-6** `engine/llm_engine.py`: `step()` 简化为单次调用
  - 不再有 `has_prefill` / `use_spec` 分支

- [ ] **4b-7** `engine/scheduler.py`: `postprocess_chunked()` 和 `postprocess_spec()` 合并
  - 统一按行类型分别处理

### 4c. Proposer 接口对齐

- [ ] **4c-1** `spec_decode/proposer.py`: `propose()` 签名对齐 vLLM
  ```python
  def propose(
      self,
      target_token_ids: torch.Tensor,
      target_positions: torch.Tensor,
      target_hidden_states: torch.Tensor,
      next_token_ids: torch.Tensor,
      ...
  ) -> torch.Tensor:  # [batch_size, num_spec_tokens]
  ```

- [ ] **4c-2** `model_runner.py`: `run_spec` 内直接调 `propose(target_hidden_states=...)`
  - 不再先 extend 再 propose

---

## 5. 阶段四：功能增强

- [ ] **5-1** 非 greedy 采样支持
  - `run_spec` 传 `draft_probs`（draft 侧的 softmax 输出），调 `_probabilistic` 路径
  - bonus token 从 `bonus_logits` 采样而非 argmax
  - vLLM 对照：`RejectionSampler` 支持 `rejection_sample_method="standard"`

- [ ] **5-2** `aux_layer_ids` 可配置
  - 从 `speculative_config["aux_layer_ids"]` 读取，而非硬编码 `{1, N//2-1, N-4}`
  - vLLM 对照：从 draft model config 的 `eagle_config` 读取

- [ ] **5-3** `Sequence.__getstate__/__setstate__` 补全 `spec_token_ids`
  - TP > 1 时跨进程 pickle/unpickle 需要

- [ ] **5-4** `RejectionSampler` 接口对齐
  - 当前 `RejectionSampler()` 无参数，vLLM 是 `RejectionSampler(sampler, spec_config, device)`

---

## 6. 阶段五：Tree Attention（可选）

**注意**：vLLM 不支持 tree attention（issue #18327 closed as not planned）。如果严格对齐 vLLM，此项可不做。如果需要，它是阶段三完成后的独立增强。

- [ ] **6-1** `config.py`: `speculative_config` 新增 tree 参数
  - `tree_mode: bool`, `tree_top_k: int`, `tree_depth: int`, `tree_total_tokens: int`

- [ ] **6-2** `proposer.py`: 新增 `propose_tree()` 方法
  - top-k 逐层扩展，构建候选树
  - 返回 `draft_tokens`, `tree_mask`, `tree_position_ids`, `retrieve_indices`
  - 参考 EAGLE `topK_genrate` 实现

- [ ] **6-3** `sequence.py`: 新增树相关字段
  - `spec_tree_mask`, `spec_tree_position_ids`, `spec_retrieve_indices`

- [ ] **6-4** `layers/attention.py`: 支持 custom tree mask
  - tree chunk 节点数少（≤64），用 dense attention + 显式 mask
  - past KV 仍走 paged cache

- [ ] **6-5** `model_runner.py` `prepare_spec_decode()`: 适配树结构
  - positions 用 `tree_position_ids`（深度值）
  - 传递 `tree_mask` 给 attention 层

- [ ] **6-6** `rejection_sampler.py`: 新增树模式
  - 遍历每条 root-to-leaf 路径（`retrieve_indices`），找最长被接受的前缀

- [ ] **6-7** `metadata.py` + `scheduler.py`: 适配树节点数量
  - `num_draft_tokens[i] = tree_total_tokens`（而非 K）
  - block 分配从 K+1 变为 tree_total_tokens+1

---

## 7. 依赖关系

```
阶段一 (MVP)
  │ 2a → 2b → 2c → 2d
  ▼
  ✅ 链式 spec decode 跑通（当前架构）
  │
阶段二 (正确性)
  │ 3-1, 3-2, 3-3（独立，可与阶段三并行准备）
  ▼
阶段三 (对齐 vLLM 架构)
  │ 主线：4a (draft paged attention) → 4c (propose 接口对齐)
  │   ├─ 4a 完成后：删除 _pending_extend、extend()、drop()
  │   └─ 4c 依赖 4a，完成后 propose 接口与 vLLM 一致
  │
  │ 支线：4b (统一调度) ← 与 4a/4c 正交，可推迟到 4c 之后单独做
  │   └─ 收益仅吞吐，动调度主流程，回归面最大
  ▼
  ✅ 架构与 vLLM 对齐（4a + 4c）
  │
阶段四 (功能增强)
  │ 5-1, 5-2, 5-3, 5-4（独立）
  ▼
阶段五 (可选，按需穿插)
  ├─ 4b 统一调度：chunked prefill + spec 混批按行验证（吞吐优化）
  └─ 6 Tree Attention：vLLM 不支持，严格对齐可跳过
  ▼
  ✅ 完整实现
```

---

## 8. 每阶段验证方式

| 阶段 | 验证方法 |
|------|----------|
| 2a | `LLMEngine(model=..., speculative_config={...})` 不抛 TypeError |
| 2b | prefill 后 `_pending_extend` 有数据；verify 后 aux hidden 非 None |
| 2c/2d | `engine.generate(["hello"], ...)` 产出非空文本，不 crash |
| 3 | 长对话不 OOM；多请求结果正确；被抢占的 seq 恢复后结果正确 |
| 4a | draft 模型的 KV cache 在 `kv_cache` 显存中；prefix cache 命中时 drafter 状态无缺口 |
| 4b | chunked prefill 混批时 decode 行仍走 K+1 verify；吞吐不降 |
| 4c | `propose()` 接口签名与 vLLM 一致 |
| 5 | temperature=0.7 输出非 greedy 结果 |
| 6 | tree 模式接受率 > 链式；吞吐提升 |

---

## 9. 已完成的历史变更

以下为在本计划制定前已完成的改动：

- [x] `models/llama.py`: `aux_layer_ids` 支持 + `_aux_hidden_states` 产出
- [x] `models/eagle3_draft.py`: EAGLE3 draft head 完整实现
- [x] `spec_decode/metadata.py`: `SpecDecodeMetadata` + `make_spec_decode_metadata()`
- [x] `spec_decode/rejection_sampler.py`: greedy + probabilistic 两种模式
- [x] `spec_decode/proposer.py`: `extend()` / `propose()` / `drop()` 方法（未接线）
- [x] `utils/loader.py`: `load_eagle3_weights()`
- [x] `engine/scheduler.py`: decode 通道 K+1 block 分配
- [x] `engine/scheduler.py`: `postprocess_spec()` 消费 accepted token
- [x] `engine/scheduler.py`: prefix-cache 规避（spec 开启时强制全量分配）
- [x] `engine/block_manager.py`: `can_allocate()` 支持 `enable_prefix_cache`
- [x] `engine/model_runner.py`: `run_spec()` 骨架 + `prepare_spec_decode()`
- [x] `engine/model_runner.py`: prefill CUDA graph 与 spec decode 互斥
- [x] `engine/llm_engine.py`: `step()` 路由到 `postprocess_spec`
