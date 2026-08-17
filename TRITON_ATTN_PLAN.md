# Triton Attention 内核实施计划

> 目标：用自研 Triton 内核替换 `flash_attn` 包在 prefill/decode 两个路径上的调用,
> 并最终把 INT8 KV cache 反量化融合进内核,从而**同时拿到「显存减半 + 带宽不浪费 + CUDA Graph 解锁」**三项收益。
> 本文档是后续阶段性优化的唯一依据,完成一项勾掉一项。

---

## 0. 参考实现分析（移植蓝本）

### 0.1 移植目标矩阵

| 路径 | 主要参考 | 辅助参考 | 原因 |
|------|----------|----------|------|
| **decode paged attention** | `linshi-w/nano-vllm-paged-attention` | datawhale PagedAttention 教程 | 前者就是 nano-vllm 的 drop-in 替换,接口和 `context.block_tables/context.context_lens` 完全一致,且已对 `flash_attn_with_kvcache` 做过精度对齐(GQA 2:1 / MHA 均通过) |
| **prefill FA2** | `Wenyueh/MinivLLM` 的 `layers/attention.py` | OpenAI Triton `06-fused-attention.py` | MinivLLM 同样基于 nano-vllm,自研了 prefill 的 online-softmax FA2,架构和 context 约定同源,移植成本最低 |
| **INT8 融合反量化** | 无现成参考,自研 | 当前 `kv_quant.py` 的反量化 kernel | 没有任何参考仓库做了 INT8 KV 融合,这是本项目的核心增量,也是反超 `flash_attn` 包的支点 |

### 0.2 各参考仓库定位

**① `linshi-w/nano-vllm-paged-attention` —— decode 路径的主蓝本**
- 引擎完全不动,只替换 `nanovllm/layers/attention.py` 的 decode 分支为 `paged_attention(...)` 调用;
- 内核文件 `nanovllm/layers/paged_attention.py`,grid = `(num_seqs, num_heads)`,`BLOCK_SIZE=64`,`kv_block_size=256`(保证单个 tile 不跨 block,只需一次 `block_table` 查表);
- GQA 支持:`kv_head = head_idx // num_queries_per_kv`;
- 实测 RTX 5090 上比 `flash_attn_with_kvcache` 慢约 12.6%(404 vs 359 μs),但这是 from-scratch Triton 在纯 decode 内存受限场景下的合理水平;
- **不支持 INT8**,需要自己加融合反量化;
- 有 `test_paged_attention.py` 做精度对齐,可直接抄来做回归。

**② `Wenyueh/MinivLLM` —— prefill FA2 的主蓝本**
- 基于 nano-vllm,自研 prefill FA2(online-softmax,O(N) 显存)+ decode paged attention;
- 文件 `src/myvllm/layers/attention.py`;
- 有 `benchmark_prefilling.py` / `benchmark_decoding.py` 对比 naive PyTorch / 优化 PyTorch / Triton,可作为压测模板;
- 同源架构,context 约定几乎一致,移植摩擦最小。

**③ datawhale PagedAttention 教程 —— 理解原理用**
- 完整列出 decode paged attention kernel 代码,讲清了 `block_table` 间接寻址、online softmax、mask 处理;
- **不含 GQA**(假设 MHA),需要自己加 `kv_head_idx = head_idx // group`;
- 适合用来理解机制,不适合直接抄。

**④ `hkproj/triton-flash-attention` —— 仅作 FA2 机制学习**
- 纯 FA2 forward,不支持 varlen/paged/GQA;
- 教学性质,不作为移植蓝本。

### 0.3 为什么自己写而不是直接抄

**通用 prefill 场景**,Triton FA2 一般只能到 `flash_attn`(Dao 手写 CUTLASS)的 80~95%,这是公认的——`flash_attn` 是工业级 SOTA,想全面反超不现实。

但本项目有两处**专项场景几乎必赢**,这才是自研的价值所在:

1. **INT8 KV 融合反量化**:`flash_attn` 包**根本不支持 INT8 KV**,当前 `attention.py:84-90` 是「整块 INT8→BF16 反量化(读+写)→ 再整块读 BF16 算 attention」,等于缓存读 2 遍写 1 遍。融合后「读 INT8 → 内部反量化 → 算 attention」一趟完成,**确定性提速**,且 `flash_attn` 做不到。
2. **decode 单 query 小 batch**:`flash_attn_with_kvcache` 是通用实现,decode 场景固定开销偏大;专项 paged attention 在小 batch 下有机会持平甚至反超。
3. **CUDA Graph 解锁**(无形收益):`model_runner.py:73` 当前 `kv_quant=True` 时禁用 CUDA graph(动态反量化不兼容图)。融合内核做图内静态反量化后,**INT8 和 CUDA graph 可同时打开**。

---

## 0.4 开关设计（渐进式替换,默认零行为变更）

所有自研 Triton 内核**默认关闭**,通过 Config 开关控制;不开启时行为与当前完全一致(走 `flash_attn` 包)。这样每个阶段都能独立验证,出问题立即回退,不污染现有 baseline。

### 开关字段

`nanovllm/config.py` `Config` 新增:

```python
use_triton_attn: bool = False   # False=走 flash_attn 包(默认);True=走自研 Triton 内核
```

### 传递路径

```
Config.use_triton_attn
  → ModelRunner.__init__: 传给 model 的每个 Attention 子模块
  → Attention.__init__(..., use_triton_attn=False)
  → Attention.forward(): if self.use_triton_attn: 走自研 else 走 flash_attn
```

### 分支策略

`attention.py` 的 forward 在每个阶段逐步增加 Triton 分支,**flash_attn 分支保留不删**:

| 阶段 | `use_triton_attn=False`(默认) | `use_triton_attn=True` |
|------|------------------------------|----------------------|
| 阶段一完成后 | prefill: `flash_attn_varlen_func` | prefill: `triton_flash_attn_varlen` |
|  | decode: `flash_attn_with_kvcache` | decode: `flash_attn_with_kvcache`(未替换) |
| 阶段二完成后 | prefill: `flash_attn_varlen_func` | prefill: `triton_flash_attn_varlen` |
|  | decode: `flash_attn_with_kvcache` | decode: `triton_paged_attention`(BF16) |
| 阶段三完成后 | INT8 decode: `dequant + flash_attn` | INT8 decode: `triton_paged_attention_int8`(融合) |

### 验证方式

- **A/B 对比**:同一脚本跑两次,一次 `use_triton_attn=False`(baseline),一次 `True`,对比精度与性能;
- **回退**:任何阶段出问题,设 `use_triton_attn=False` 即恢复原行为,无需改代码;
- **最终验证通过后**,可考虑把默认值改为 `True`(或删除 flash_attn 分支),但**非必须**——保留双路径反而方便长期回归。

> **与 `kv_quant` 的关系**:两个开关正交。`use_triton_attn` 控制用哪个 attention 内核,`kv_quant` 控制 KV cache 存储精度。阶段三之前,`use_triton_attn=True` + `kv_quant=True` 时 decode 仍走 BF16 paged(不读 INT8 cache);阶段三完成后才走融合 INT8 内核。

---

## 1. 当前状态盘点

### 1.1 现有 attention 路径(`nanovllm/layers/attention.py`)

```python
# prefill: flash_attn_varlen_func(支持 block_table 前缀缓存)
# decode (kv_quant=False): flash_attn_with_kvcache(BF16 直读)
# decode (kv_quant=True):  dequant_kvcache → flash_attn_with_kvcache  ← 性能损耗点
```

### 1.2 现有 INT8 KV 量化(`nanovllm/layers/kv_quant.py`)

- `store_kvcache_int8`:per-(token, head) 对称 Min-Max,BF16→INT8 + FP32 scale;
- `dequant_kvcache`:**整块** INT8→BF16,独立 kernel,产出临时 BF16 张量;
- `dequant_kvcache_to_buf`:同上但写入预分配 buffer(无 alloc,但仍是两趟读写)。

### 1.3 现有 context 约定(`nanovllm/utils/context.py`)

`Context` 已携带所有 paged attention 所需字段:`cu_seqlens_q/k`、`max_seqlen_q/k`、`slot_mapping`、`context_lens`、`block_tables`、`is_prefill`。**无需扩展 context 接口**。

### 1.4 现有限制

- `model_runner.py:73`:`kv_quant=True` 时禁用 CUDA graph(注释明写"动态反量化与 CUDA graph 不兼容");
- prefill CUDA graph 与 spec decode 互斥(已有,不在本计划范围)。

---

## 2. 阶段一:BF16 prefill FA2(对齐精度与性能 baseline)

**目标**:用自研 Triton FA2 替换 `flash_attn_varlen_func`,**不追求超越**,只要精度对齐(误差 <1e-2)、性能在 80~95% 区间即可。这步的目的是拿到一个可靠的 baseline 和正确性参照,为阶段三的 INT8 融合打底。

### 2a. 内核骨架

- [x] **2a-1** 新建 `nanovllm/layers/triton_attn.py`,实现 `_fwd_kernel()` Triton kernel
  - 标准 FlashAttention-2 online-softmax 前向:`(m_i, l_i, acc)` running state,逐 KV block 更新,全程不materialize 完整注意力矩阵
  - 支持:causal mask 内联(块内 elementwise mask + 块级循环上界共同保证)、GQA(`kv_head_idx = head_idx // (num_heads // num_kv_heads)`)、varlen(`cu_seqlens`,同一个数组用于 q 和 k/v,对应"无 prefix cache"场景)
  - `HEAD_DIM` 作为 constexpr,由 wrapper 按 `q.shape[-1]` 动态传入(非硬编码,支持 64/128 等 2 的幂)
  - 输入输出 dtype:BF16(或调用方传入的其它 dtype),QK/softmax/累加器全程 fp32

- [x] **2a-2** Python wrapper `triton_flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, scale)`
  - 精简签名(仅覆盖本阶段场景:`cu_seqlens_q == cu_seqlens_k`,无 `block_table`);双 cu_seqlens + paged 寻址留给阶段五 `triton_flash_attn_varlen_paged`
  - `block_table`(前缀缓存)与 sliding window 场景均不在此函数处理,由 `attention.py` 分支决定是否调用

### 2b. 开关接线与对齐

- [x] **2b-0** 开关打通
  - `config.py`:新增 `use_triton_attn: bool = False` 字段
  - `model_runner.py`:模型构造后、`warmup_model()` 前,遍历 `hasattr(module, "k_cache")` 的子模块设置 `module.use_triton_attn = config.use_triton_attn`(与 `allocate_kv_cache()` 里 `kv_quant` 下发方式同一套模板)
  - `attention.py` `Attention.__init__`:新增 `use_triton_attn=False` 参数并存为属性

- [x] **2b-1** `attention.py` prefill 分支:增加 `use_triton` 判定(`self.use_triton_attn and block_tables is None and sliding_window is None`)后调用 `triton_flash_attn_varlen(...)`,**原 `flash_attn_varlen_func` 调用完整保留在 else 分支**
  - 保持 `context.is_prefill` + `context.block_tables is not None` 的外层分支逻辑不变

- [x] **2b-2** sliding window:**本阶段不支持**,判定条件里显式排除(`sliding_window is not None` 时强制走 else/flash_attn),避免静默错误;留给后续阶段按需补充 window mask

### 2c. 精度与性能对齐

- [x] **2c-1** 编写 `tests/test_triton_attn.py`:对比 Triton FA2 vs `flash_attn_varlen_func`
  - 用例:MHA / GQA 2:1 / GQA 4:1,seq_len 128/1024/4096(多序列不等长 varlen),causal=True
  - 精度阈值:`atol=1e-2, rtol=1e-2`(BF16 量级)
  - ✅ **已验证通过**(RTX 4090):4 个用例全部 PASSED,`max_abs_err=3.9e-3`,远小于阈值

- [x] **2c-2** 端到端回归:`python example.py` 输出文本与 baseline 一致(greedy)
  - ✅ 已验证:Llama-3.2-3B 下 `use_triton_attn=True/False` 两次 greedy 输出逐字一致
  - ✅ 性能对比(`bench_triton_prefill.py`,RTX 4090,Llama-3.2-3B GQA 3:1):
    - 定向调优 `BLOCK_M 64→128` + `num_warps 4→8` 后,长序列从 66% 提升到 80%+
    - 最终:`512→94.3%` / `2048→110%` / `4096→83.5%` / `8192→80.6%`
    - `1024→141.8%` 为 flash_attn 该 seqlen 档位自身偏慢(非 Triton 特别快),归因需注明
  - **阶段一达标收尾**:短序列打平/反超,长序列落在 80~95% 预期区间

---

## 3. 阶段二:BF16 decode paged attention

**目标**:用自研 Triton paged attention 替换 `flash_attn_with_kvcache`(BF16 路径),精度对齐。

### 3a. 内核实现

- [x] **3a-1** `triton_attn.py` 新增 `_paged_attn_decode_kernel()` Triton kernel
  - 借鉴 `linshi-w/nano-vllm-paged-attention` 的设计,自行实现:grid = `(num_seqs, num_heads)`,每 program 处理一个 (seq, q_head)
  - `BLOCK_N=64` 整除 cache `block_size=256` → **每个 tile 恰好落在一个物理块内,单次查表**(`n_start // BLOCK_SIZE` 即逻辑块号)
  - 单 query → softmax 状态是标量(`m_i/l_i` 为 0-d),累加器是 `(HEAD_DIM,)` 向量;无需 causal mask(decode 的 q 是最后一个 token,全历史可见),仅需 `context_len` 边界 mask
  - GQA:`kv_head_idx = head_idx // (num_heads // num_kv_heads)`;物理块号转 int64 防大 cache 溢出
  - online softmax:`(m_i, l_i, acc)` running state,QK/PV 全程 fp32 累加

- [x] **3a-2** Python wrapper `triton_paged_attention(q, k_cache, v_cache, block_tables, context_lens, scale)`
  - 接口对齐 `attention.py` decode 调用:直接接收 `(num_seqs, num_heads, head_dim)` 的 q(无需 unsqueeze)
  - block_size 从 `k_cache.shape[1]` 推导,不额外传参

### 3b. 开关接线与对齐

- [x] **3b-1** `attention.py` decode 分支接线:`kv_quant=False` + `use_triton_attn=True` + 无 sliding window 时走 `triton_paged_attention(...)`,**原 `flash_attn_with_kvcache` 保留在 else 分支**
  - `kv_quant=True` 时无视开关仍走 `dequant + flash_attn`(阶段三融合),符合计划
  - 已确认旁路安全:spec verify / chunked prefill 走 `is_prefill=True` + block_tables → prefill 分支自动回退 flash_attn;decode CUDA graph 按固定 bs 捕获,Triton kernel 静态 shape 可正常捕获

- [x] **3b-2** 精度对齐测试:`tests/test_triton_attn.py` 新增 4 个 decode 用例
  - 对比 Triton paged vs `flash_attn_with_kvcache`
  - 用例:MHA partial block(300) / 多序列不等长(128,256,300,511) / GQA 3:1(1024,Llama 配置) / GQA 4:1(4096)
  - **物理块随机洗牌**(`torch.randperm`),专抓"假设物理块连续/按序"的寻址 bug
  - ✅ **已验证通过**(RTX 4090):4 个 decode 用例全 PASSED,`max_abs_err=4.9e-4~2.0e-3`(单 query 累加更浅,比 prefill 还低)
  - 调试记录:曾踩"块内偏移 vs 全局位置"混淆的越界 bug(首块碰巧正确、跨块读越界 → 100% NaN),已修复并回归通过

- [ ] **3b-3** 端到端回归:`python example.py` greedy 输出与 baseline 逐 token 一致
  - **待办(需 CUDA 环境执行)**:`use_triton_attn=True` 跑一次对比 `False` 的 baseline

---

## 4. 阶段三:INT8 KV 融合反量化(核心增量)

**目标**:把 `dequant_kvcache` 整块反量化消融进 prefill 和 decode 内核,实现「读 INT8 → 内部反量化 → 算 attention」一趟完成。这是反超 `flash_attn` 包的关键,也是「INT8 性能体现上去」的完整实现。

### 4a. 内核改造:decode 路径(优先,收益最大)

- [x] **4a-1** `triton_attn.py` 新增 `_paged_attn_decode_int8_kernel()` kernel
  - 基于 3a-1 的 BF16 版本改造,KV cache 读取:`tl.load` INT8 → fp32 点积 → 乘 scale
  - **scale 后置**(per-token 对称量化,scale 与 head_dim 无关,可从点积提出):`qk[t] = k_scale[t]·softmax_scale·Σ q·k_int8`,每 token 只乘 1 次而非每元素
  - V 侧合并标量:`acc[d] += (p[t]·v_scale[t])·v_int8[t,d]`
  - scale 寻址:`(blocks, block_size, kv_heads)` 布局,block_table + 块内偏移,mask 加载防未初始化 NaN(`0.0` other + `p=0` 双保险)
  - **不再调用 `dequant_kvcache`**,无中间 BF16 buffer

- [x] **4a-2** Python wrapper `triton_paged_attention_int8(q, k_cache_int8, v_cache_int8, k_scale, v_scale, block_tables, context_lens, scale)`

- [x] **4a-3** `attention.py` decode `kv_quant=True` 分支:嵌套 `use_triton_attn` 开关走融合内核,**原 `dequant_kvcache + flash_attn_with_kvcache` 保留在 else 分支**
  - 四象限:`kv_quant × use_triton_attn` 独立组合,默认行为不变,出问题立即回退

### 4b. 内核改造:prefill 路径(次要,依赖阶段五)

> **注意**:prefill 的 INT8 融合需要「prefill 读 paged cache」这个能力先行,而这正是阶段五(prefix cache paged prefill)要做的事。所以 4b 排在阶段五之后,不阻塞阶段三/四的主线。
>
> 当前 `kv_quant.py` 注释写明"Prefill: unaffected (uses freshly computed K/V directly)"——普通 prefill 用新算的 BF16 K/V,不走 INT8 cache。所以 prefill 的 INT8 融合**仅对 prefix cache 命中时有意义**(读历史 INT8 cache)。

- [ ] **4b-1**(条件性,依赖阶段五)若 prefix cache 的历史 KV 也以 INT8 存储,则在阶段五的 paged prefill FA2 内核里加 INT8 读取分支(读 INT8 → 反量化 → 进 QK 计算)
  - 判定依据:`store_kvcache_int8` 是否对 prefix cache 的历史 KV 生效;若 prefix cache 始终 BF16,则 4b 整体跳过

- [ ] **4b-2**(条件性)若 4b-1 判定需要,在 `_flash_attn_varlen_paged` 里加 `kv_quant=True` 的 INT8 读取路径

### 4c. 精度与性能验证

- [ ] **4c-1** 精度对齐:INT8 融合内核 vs 当前 `dequant_kvcache + flash_attn_with_kvcache`
  - 阈值:`atol=1e-2, rtol=1e-2`(两者数值路径相同:都是 int8 × scale,仅 dequant 路径多一次中间 BF16 round,融合路径理论上更精确)
  - 已添加 4 个用例(`tests/test_triton_attn.py`):跨块 partial(300) / 多序列(128,256,511) / GQA 3:1(2048) / GQA 4:1(4096),量化模拟 + 物理块洗牌
  - **待办(需 CUDA 环境执行)**:运行 `python -u tests/test_triton_attn.py`

- [ ] **4c-2** 端到端精度:`python example.py` 用 `kv_quant=True` + `use_triton_attn=True` 跑
  - 对比 `kv_quant=True` + `use_triton_attn=False`(dequant 路径)的生成结果 token 一致率
  - **待办(需 CUDA 环境执行)**

- [ ] **4c-3** 性能对比:`bench_triton_decode.py` 新增 INT8 四路对比(`dequant+flash` 现状 / `flash BF16` 上限 / `triton BF16` / `triton INT8 融合`)
  - 已实现,输出 `int8 vs dequant`(融合收益)和 `int8 vs flash_bf16`(是否反超)两个比值
  - **待办(需 CUDA 环境执行)**

---

## 5. 阶段四:CUDA Graph 解锁与最终验证

**目标**:利用融合内核的静态反量化特性,把 `kv_quant=True` 时被禁用的 CUDA graph 重新打开,拿到最后一笔收益。

### 5a. CUDA Graph 兼容性修复

- [ ] **5a-1** `model_runner.py:73`:把 CUDA graph 禁用条件从 `not config.kv_quant` 改为 `not config.kv_quant and not config.use_triton_attn`
  - 即:**仅当 `kv_quant=True` 且 `use_triton_attn=False`(默认走 dequant 路径)时才禁用**;`use_triton_attn=True` 时走融合内核,无动态反量化,可正常捕获图
  - 前提:4a-3 已完成,`use_triton_attn=True` + `kv_quant=True` 的 decode 走融合内核
  - 默认行为不变:`use_triton_attn=False` + `kv_quant=True` 仍禁用 CUDA graph(与当前一致)

- [ ] **5a-2** warmup 路径覆盖 INT8 融合内核
  - `warmup_model()` 在 `use_triton_attn=True` 时确保捕获走 `triton_paged_attention_int8` 路径

### 5b. 最终回归与简历数据采集

- [ ] **5b-1** 全量回归:`python example.py` / `python bench.py` / `python serving_bench.py` 三个脚本均通过
  - 覆盖 `kv_quant={True, False}` × `enforce_eager={True, False}` 四种组合

- [ ] **5b-2** 采集简历可用数据
  - RTX 4090 + Qwen3-0.6B,记录:
    - INT8 KV 显存占用 vs BF16(期望约 50%)
    - decode TPOT:融合内核 vs 当前 INT8 路径 vs BF16(期望融合 > 当前 INT8,接近或优于 BF16)
    - CUDA graph 开关对 INT8 路径的吞吐影响(期望开启后明显提升)
    - 精度:token 一致率 / perplexity

---

## 5.5 阶段五:prefix cache paged prefill(补充项)

**目标**:打通「prefill 读 paged cache」——当 `block_tables is not None`(prefix cache 命中)时,prefill 的 k/v 从分页 KV cache 读历史前缀,而非新算的稠密张量。这是阶段一坑 2 绕开的场景。

**为什么排在 decode 之后**:核心难点是 `block_table` 间接寻址,这正是阶段二 decode paged attention 已经写好的逻辑,直接复用,避免重复劳动。

### 本质

```
prefix cache prefill = FA2 内核 + paged k/v 寻址 + 错位 causal + 双 cu_seqlens
```

与各阶段的差异(causal 是唯一的 prefill 特有增量):

| 场景 | q 位置 | k 位置 | causal 条件 |
|------|--------|--------|-------------|
| 无 prefix cache prefill(阶段一) | `0 + offs_m` | `offs_n` | `offs_m >= offs_n` |
| decode(阶段二) | 最后 1 token | `0..ctx_len` | 天然全满足,无 mask |
| **prefix cache prefill(阶段五)** | `start + offs_m` | `offs_n` | `(start + offs_m) >= offs_n` |

其中 `start = 前缀长度`(`seq.num_cached_tokens`)。需同时用 `cu_seqlens_q`(定位 q)和 `cu_seqlens_k`(定位 k)两个边界序列,阶段一的 MinivLLM 内核只用一个,必须改。

### 实现要点

- [ ] **5.5-1** `triton_attn.py` 新增 `_flash_attn_varlen_paged()` kernel
  - 基于阶段一 `_flash_attn_varlen_forward`(FA2)改造
  - k/v 读取:从 paged cache 按 `block_table` 间接寻址(复用阶段二 decode 的 block_table 逻辑)
  - 双 cu_seqlens + 错位 causal(上表)

- [ ] **5.5-2** Python wrapper `triton_flash_attn_varlen_paged(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, block_tables, scale)`

- [ ] **5.5-3** `attention.py` prefill 分支:`use_triton_attn=True` 且 `block_tables is not None` 时走新内核(此时 k/v 已是 `k_cache/v_cache`)

- [ ] **5.5-4** 精度对齐测试:prefix cache 场景对比 `flash_attn_varlen_func(block_table=...)`

### 触发场景与优先级

- 共享 prompt 前缀(如 system prompt 复用)、chunked prefill 的续算
- 默认 `enable_chunked_prefill=False`,主要出现在共享前缀场景,**优先级低于 INT8 融合(阶段三)**;完成后可再叠加 4b 的 prefill INT8 融合

---

## 6. 依赖关系

```
阶段一 (prefill BF16 FA2)
  │ 2a (内核) → 2b (接线) → 2c (对齐)
  ▼
  ✅ prefill 走自研 Triton,精度对齐 flash_attn_varlen_func
  │
阶段二 (decode BF16 paged)
  │ 3a (内核) → 3b (接线+对齐)
  ▼
  ✅ decode 走自研 Triton,精度对齐 flash_attn_with_kvcache
  │  此时 flash_attn 包仍保留(用于对齐参照)
  │
阶段三 (INT8 融合) ← 核心
  │ 4a (decode INT8 融合) —— 必做,收益最大
  │ 4b (prefill INT8 融合) —— 条件性,看 prefix cache 是否存 INT8
  │ 4c (验证)
  ▼
  ✅ INT8 KV 显存减半 + 带宽不浪费
  │
阶段四 (CUDA graph 解锁)
  │ 5a (解除禁用) → 5b (回归 + 数据采集)
  ▼
  ✅ INT8 + CUDA graph 同时打开,简历数据齐备
  │
阶段五 (prefix cache paged prefill) ← 补充项,不阻塞主线
  │ 5.5 (复用阶段二 block_table 寻址 + FA2 错位 causal)
  ▼
  ✅ prefill 读 paged cache 打通,可再叠加 4b 的 prefill INT8 融合
```

> **关键依赖**:阶段三 4a 依赖阶段二 3a 的 BF16 decode 内核(在其基础上改造);阶段四 5a 依赖阶段三 4a-3 完成(消除动态反量化);阶段五 5.5 依赖阶段二 3a 的 block_table 寻址逻辑(避免重复);4b 依赖阶段五 5.5(先有 paged prefill 才有 prefill INT8 融合)。
>
> **可并行项**:阶段一(2a)与阶段二(3a)的内核开发可并行,prefill 和 decode 是两个独立 kernel;阶段五与阶段三/四可并行(不同 kernel,不同文件)。

---

## 7. 每阶段验证方式

| 阶段 | 验证方法 |
|------|----------|
| 2b-0 | `Config(use_triton_attn=False)` 行为与当前完全一致(回归);`use_triton_attn=True` 走 Triton 分支 |
| 2a/2b | `tests/test_triton_attn.py` prefill 用例对齐 `flash_attn_varlen_func`(atol=1e-2);A/B 对比 `use_triton_attn` 两种值 |
| 2c | `example.py use_triton_attn=True` greedy 输出与 baseline 一致;`bench.py` 记录 prefill 吞吐比值 |
| 3a/3b | decode 用例对齐 `flash_attn_with_kvcache`;`example.py use_triton_attn=True` greedy 逐 token 一致 |
| 4a | INT8 融合内核 vs `dequant + flash_attn`(else 分支)输出一致;`example.py use_triton_attn=True kv_quant=True` 输出合理 |
| 4c | 四维指标:显存/TPOT/prefill吞吐/精度,数据写入 README;A/B 对比开关两种值 |
| 5a | `use_triton_attn=True kv_quant=True` 下 CUDA graph 成功捕获(无 dynamic shape 报错);默认值仍禁用(行为不变) |
| 5b | 四组合(`kv_quant × enforce_eager`)× `use_triton_attn` 两值,全量回归通过 |
| 5.5 | prefix cache 场景对齐 `flash_attn_varlen_func(block_table=...)`;共享前缀端到端 greedy 一致 |

---

## 8. 风险与回退

1. **Triton FA2 prefill 性能不达 80%**:回退到 `flash_attn_varlen_func`,只保留 decode + INT8 融合的成果。prefill BF16 不是必赢项,INT8 融合才是。
2. **INT8 融合内核精度异常**:先确认 `kv_quant.py` 的 scale 粒度(per-token-per-head 对称 Min-Max)是否足够;必要时升级到 per-channel 或非对称量化。
3. **CUDA graph 捕获失败**:回退到 `enforce_eager=True`,保留融合内核的带宽收益(不带图加速)。
4. **Triton 版本兼容**:Triton API 变动较快,锁定一个验证过的版本(参考 MinivLLM 的 `requirements.txt`)。

---

## 9. 不在本计划范围

- **Tree Attention / Spec Decode 的 Triton 化**:spec decode 已有独立计划(`SPEC_DECODE_PLAN.md`),其 draft 模型 attention 走同一个 `Attention` 模块,本计划完成后自动受益,不单独处理。
- **FP8 KV cache**:INT8 跑通后再考虑,量化路径不同(scale 是 FP32 单值 vs FP8 E4M3),先不做。
- **prefill chunked 的分段 CUDA Graph**:独立优化项,见后续计划。
- **多卡 TP 下的 attention**:当前 attention kernel 是单卡内的,TP 通信在 attention 之外,本计划不涉及。
