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
| extend | 不存在 | `extend()` 手动 catch-up，draft KV 已分页（4a），无跨调用暂存（`_pending_extend` 已删） | **主动保留，非待清理项**——见下方说明 |
| propose 接口 | 接收 target_hidden_states 作为参数、一步完成 catch-up+自链 | `extend`（消费 verify hidden）+ `propose`（自链）两步分离 | 接口不对齐，但**4c 已废弃**，不再追求合并（见下方说明） |
| aux hidden | 模型 forward 返回字段，同调用内传递 | 挂在模型上，需手动提取；`extend` 在产生的同一轮内立即消费（无暂存） | 流通方式不同，风险已消除 |
| tree attention | 不支持 | 不支持 | 一致，不需要做 |
| rejection sampling | `RejectionSampler(sampler, config, device)` | `RejectionSampler()` 无参数 | 接口差异 |

> **调度差异已主动保留**：nano-vllm 保留 `schedule()` / `schedule_chunked()` 两条路径的设计，
> 不对齐 vLLM V1 的统一调度（见阶段三 4b 已删除）。混批中 prefill 行与 spec decode 行
> 共存时降级为单 token 验证是已知限制，作为架构差异保留，不视为待修复的缺陷。

> **`extend`/`propose` 两步分离已主动保留，不是"未完成"**：EAGLE3 论文的训练配对
> （`L_reg = SmoothL1(f_{i+1}, Draft(T_{2:i+1}, F_{1:i}))`）决定了 draft 预测 `t_{p+1}`
> 必须用 target 在 position p 的真实 hidden `f_p`，而 `f_p` 只能来自"验证 forward"，
> 且验证发生在 draft 自链**之前**的轮次——这是算法本身的因果链，不是 nano-vllm 实现的缺陷。
> vLLM 能把两者合成一次 `propose(target_hidden_states=...)` 调用，前提是统一调度让
> "补算已确认 token" 和"验证本轮 draft" 合并进同一次 target forward；nano-vllm 保留双路径
> 调度（未做 4b），所以 `extend`（消费上一轮验证 hidden）与 `propose`（自链）必须是两个
> 独立步骤。**已做的收敛**是删除 `_pending_extend` 跨轮暂存，让 `extend` 从"下一轮开头消费
> stash"改为"验证产生 hidden 后立即执行"（见 4a-6），因果顺序不变，只是把多余的中转去掉。

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
  - **遗留问题**：混批中的 decode seq 本轮 token 不经过 extend，其 `_pending_extend` 条目留着上一轮 verify 的旧数据，下次真正进入 `run_spec` 时会比 `len(seq)` 少 1 个位置。当前不修（需要混批按行 spec 验证，但 nano-vllm 保留双路径调度，不计划做统一调度对齐），只在代码注释里标注，不静默掩盖。

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

> **依赖说明（重要，已更新）**：`4a` 已完成并验证通过；`4c` 已废弃（见下方 4c 小节的理由），不再是
> "4a 完成后接着做"的下一步。**`extend()` 不会被消除**——EAGLE3 算法本身要求它，详见 4a 小节末尾的说明。

### 4a. Draft 模型接入统一 paged attention

**vLLM 最核心的架构决策**：draft model 注册在独立 KV cache group，走 paged attention，由 KVCacheManager 统一管理。

**完成情况（已验证）**：`example.py`/`bench.py` 端到端跑通，`temperature=0` 下 `spec == non-spec` 逐 token 一致，`3.53x` 加速。多请求并发（首次 extend + 续接 extend 混合 batch）、preempt 语义均已在改动时逐项推导确认。

- [x] **4a-1** `models/eagle3_draft.py`: `Eagle3Attention` 改为 paged attention
  - 内部换成项目现有的 `layers.attention.Attention` 子模块，签名从 `(positions, hidden, past_key_values, cache_seqlens)` 简化为 `(positions, hidden_states)`，KV 读写走全局 `context`（`slot_mapping`/`block_tables`/`context_lens`），与 `LlamaAttention` 完全一致

- [x] **4a-2** `model_runner.py` `allocate_kv_cache()`: draft 的 KV 占用独立 `draft_kv_cache` 张量（一层）
  - 与 target `kv_cache` **相同的 `num_kvcache_blocks`/`block_size`**（block id 空间共享的前提），按 draft 自己的 `num_kv_heads`/`head_dim` 计算大小，始终全精度（不受 `kv_quant` 影响）
  - 挂载到 `proposer.draft.midlayer.self_attn.attn.k_cache/v_cache`；显存统计并入 `allocate_kv_cache` 的日志打印

- [x] **4a-3** ~~`engine/block_manager.py`: hash/allocate 逻辑覆盖 draft 层~~ **确认不需要改**
  - `BlockManager` 只发放/追踪 block id，不知道物理内容；target 和 draft 各自在同一个 block id 上维护自己的物理张量（`kv_cache` vs `draft_kv_cache`），无需 BlockManager 感知 draft 的存在
  - preempt 时两者用同一批 block id，天然同生命周期失效

- [x] **4a-4**（范围调整）`spec_decode/proposer.py`: 删除 dense `_kv` dict，改为 paged `draft_kv_cache`
  - `_kv`（稠密 per-seq KV 张量）已删除，KV 进 `draft_kv_cache`（4a-2），按 `seq.block_table` 寻址
  - `_draft_ctx_len`/`_aux`/`_draft0` 三个字典**保留**——`_draft_ctx_len` 是 KV 长度信息的替代（原来隐含在 `_kv` 张量长度里，paged 之后必须单独记录），`_aux`/`_draft0` 记录自链起点，三者都是**必要的调度状态**，不是待清理的暂存
  - `extend()`/`drop()` 方法**保留**（原计划要求删除，见下方"为什么不删"说明）——重写为 batched varlen forward + paged cache 读写，语义不变

- [x] **4a-6** `model_runner.py`: 删除 `_pending_extend` 跨轮暂存机制
  - `extend()` 的调用时机从"下一轮 `run_spec` 开头消费 stash"改为"prefill 采样后 / verify rejection sampling 后立即执行"（`_extend_prefill_aux`/`_extend_verify_aux`）
  - 因果顺序不变（extend 始终在为下一次 propose 准备状态），只是去掉了"暂存 → 下一轮取回"的中转
  - 多 chunk prefill 不再手动拼接跨 chunk 的 tokens/positions/aux——每个 chunk 直接调一次 `extend()`，靠其内部 `assert start == _draft_ctx_len` 保证连续

**为什么不做 4a-5（`propose()` 直接接收 target hidden states）/ 不追求删除 `extend()`**：

EAGLE3 的训练配对（`L_reg = SmoothL1(f_{i+1}, Draft(T_{2:i+1}, F_{1:i}))`）意味着 draft 预测 `t_{p+1}` 依赖 target 在 position p 的真实 hidden `f_p`，而 `f_p` 只产生于"验证 forward"，且验证发生在下一次 draft 自链**之前**的轮次——这是算法本身的因果链。vLLM 能一步做到 `propose(target_hidden_states=...)`，是因为统一调度把"补算已确认 token"和"验证本轮 draft"合并进了同一次 target forward，propose 直接吃这次 forward 产出的 hidden；nano-vllm 保留双路径调度（未做已废弃的 4b），没有这次"合并 forward"，所以 `extend`（消费上一轮验证 hidden，更新 committed 状态）与 `propose`（从 committed 状态自链）必须是两个独立步骤——这是 EAGLE3 算法在双路径调度下的**正确且自然**的实现形态，不是需要清理的技术债。4a-6 已经做了这个范围内唯一值得做的收敛（去掉跨轮暂存）。

### 4c. Proposer 接口对齐 —— **已废弃**

**结论：不做。** 4c 的目标（`propose(target_hidden_states=...)` 一步完成、无独立 `extend`）等价于要求 nano-vllm 采用 vLLM 的统一调度（4b），而 4b 已经因为"保留双路径调度"的决策被废弃。强行只合并 `extend`+`propose` 而不做统一调度，会打破"propose 在 verify 之前、只能读上一轮 hidden"的因果链，是"为对齐而对齐"，不产生正确性或性能收益。理由详见上方 4a"为什么不做 4a-5"的说明。

原计划内容（存档，不再执行）：
- `propose()` 签名改为 `propose(target_token_ids, target_positions, target_hidden_states, next_token_ids, ...) -> torch.Tensor`
- `run_spec` 内直接调 `propose(target_hidden_states=...)`，不再先 extend 再 propose

---

## 5. 阶段四：功能增强

- [x] **5-1** 非 greedy 采样支持
  - `propose()` 按 seq 选 token：greedy（temperature≤0）取 argmax；采样 seq 从 `q = softmax(draft_logits / T)` 中 **multinomial 采样**（投机采样正确性要求 draft token ~ q 而非 argmax）；`extend()` 改存 step-0 logits（`_draft0` → `_draft0_logits`），采样/argmax 推迟到 propose 时按 temperature 决定
  - 任何 seq 采样时，每步 q 行 scatter 到 target vocab 空间（hot set 之外为 0），以 `draft_probs [B*K, V]`（seq-major 展开，与 `draft_token_ids` 对齐）传给 `run_spec`
  - `RejectionSampler._probabilistic` 全部向量化：一次过滤 softmax、一次 gather 算 `p(x)/q(x)`、一次 uniform 抽取、一次批量 multinomial 残差采样，然后一批 `.tolist()` 同步；**混批按 req 分流**——greedy req 保持 argmax 验收（其 probs 行被忽略）
  - bonus token：greedy req argmax；采样 req 复用引擎的 `Sampler`（temperature/top-k/top-p 与非投机路径完全一致）
  - 目标分布 p 经 `_filtered_probs`（temperature + top-k/top-p 过滤，逐步镜像 `layers/sampler.Sampler`），保证投机采样输出分布 == 非投机采样分布
  - vLLM 对照：`RejectionSampler` 支持 `rejection_sample_method="standard"`

- [x] **5-2** `aux_layer_ids` 可配置
  - `Proposer.__init__` 新增 `aux_layer_ids` 参数，`model_runner.py` 传入 `self.speculative_config.get("aux_layer_ids")`；未配置时回退到 `{1, N//2-1, N-4}` heuristic
  - vLLM 对照：从 draft model config 的 `eagle_config` 读取（此处改为显式配置项，语义等价）

- [x] **5-3** `Sequence.__getstate__/__setstate__` 补全 `spec_token_ids`
  - TP > 1 时跨进程 pickle/unpickle 需要；当前流程里 `run_spec` 在 broadcast 后立即覆写 `spec_token_ids`，所以不是活跃 bug，但补全后消除了隐患

- [x] **5-4** `RejectionSampler` 接口对齐
  - 改为 `RejectionSampler(sampler)`——注入引擎的 `Sampler` 用于非 greedy 的 bonus token 采样（5-1 的配套依赖）
  - 相对 vLLM 的 `(sampler, spec_config, device)` 有意精简：张量自带 device，且本实现没有需要查询的 spec-config 对象

---

## 6. 依赖关系

```
阶段一 (MVP)
  │ 2a → 2b → 2c → 2d
  ▼
  ✅ 链式 spec decode 跑通（当前架构）
  │
阶段二 (正确性)
  │ 3-1, 3-2, 3-3（独立，可与阶段三并行准备）
  ▼
阶段三 (对齐 vLLM 核心架构)
  │ 4a (draft paged attention) —— ✅ 已完成并验证
  │   ├─ draft KV 接入分页存储，与 target 共享 block id 空间
  │   └─ 删除跨轮暂存 `_pending_extend`（extend 时机提前到数据产生的同一轮）
  │ 4c (propose 接口对齐) —— ❌ 已废弃，见 4c 小节理由
  ▼
  ✅ 4a 完成；extend()/propose() 两步分离是 EAGLE3 双路径调度下的正确形态，非待办
  │
阶段四 (功能增强)
  │ 5-1 (非 greedy 采样) + 5-4 (RejectionSampler 注入 sampler) —— ✅ 已完成
  │ 5-2 (aux_layer_ids 可配置)、5-3 (Sequence pickle 补全) —— ✅ 已完成
  ▼
  ✅ 完整实现
```

> **Tree Attention 已废弃**：vLLM 官方未实现（issue #18327 closed as "not planned"，
> 90天无活动被 stale bot 自动关闭，无 maintainer 回复，无 linked PR）。原因：
> (1) PagedAttention block size > 1 与 tree mask 不兼容；(2) 链式 EAGLE3 已有
> 3x+ 加速，tree 的边际收益有限；(3) 实现维护复杂度高。严格对齐 vLLM 不做此项。
>
> **统一调度已废弃**：nano-vllm 保留 `schedule()` / `schedule_chunked()` 双路径设计
> 与 `run()` / `run_spec()` / `run_chunked()` 三入口，不对齐 vLLM V1 的统一调度。
> 理由：动调度主流程，回归面最大，收益仅为吞吐（非正确性）。混批中 prefill 行与
> spec decode 行共存时降级为单 token 验证是已知限制，作为架构差异保留。

---

## 7. 每阶段验证方式

| 阶段 | 验证方法 |
|------|----------|
| 2a | `LLMEngine(model=..., speculative_config={...})` 不抛 TypeError |
| 2b | prefill 后 `_pending_extend` 有数据；verify 后 aux hidden 非 None |
| 2c/2d | `engine.generate(["hello"], ...)` 产出非空文本，不 crash |
| 3 | 长对话不 OOM；多请求结果正确；被抢占的 seq 恢复后结果正确 |
| 4a | draft 模型的 KV cache 在 `kv_cache` 显存中；prefix cache 命中时 drafter 状态无缺口 |
| 4c | `propose()` 接口签名与 vLLM 一致 |
| 5 | greedy 回归（`bench.py` spec == non-spec）；temperature=0.7 多次运行输出互不相同且不 crash；混批（greedy + 采样）正常 |

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
