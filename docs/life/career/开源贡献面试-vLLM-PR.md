---
title: "开源贡献面试：vLLM PR"
---

# 开源贡献面试：vLLM PR

围绕简历中的 vLLM 开源贡献，按 AI Infra 技术面组织问题。每题的「简要回答」适合面试时先用约 30～60 秒交代核心结论；「完整阐述」展开机制、实现选择、具体例子和验证依据，供后续追问使用。回答采用第一人称，技术依据来自对应 PR 记录与源码笔记。

本文整理于 2026 年 10 月 9 日。下表的状态沿用仓库中 2026 年 10 月 5～6 日的记录，性能数字采用各 PR 已报告的实验结果。

| PR | 贡献内容 | 已有记录中的状态 |
|---|---|---|
| [#60040](https://github.com/vllm-project/vllm/pull/60040) | ROCm 多模态 Prefix-LM 支持 | 进行中 |
| [#60049](https://github.com/vllm-project/vllm/pull/60049) | MRV2 PIECEWISE dummy 填充计数修复 | 进行中 |
| [#59600](https://github.com/vllm-project/vllm/pull/59600) | ShortConv drafter 历史恢复与 Chunked Prefill 输入修复 | 进行中 |
| [#59550](https://github.com/vllm-project/vllm/pull/59550) | ROCm Attention 滑动窗口边界修复 | 已合并 |
| [#53283](https://github.com/vllm-project/vllm/pull/53283) | 单输出 GEMM 的 wvSplitK 分发优化 | 已合并 |
| [#52637](https://github.com/vllm-project/vllm/pull/52637) | 对称 GPTQ 加载修复与 RDNA INT4 重排优化 | 进行中 |
| [#58513](https://github.com/vllm-project/vllm/pull/58513) | Channelwise GPTQ MoE fallback 修复 | 进行中 |
| [#52619](https://github.com/vllm-project/vllm/pull/52619) | RDNA W4A16 medium-skinny 分发范围补齐 | 进行中 |

## 1. 你在 vLLM 社区主要做了哪些贡献？挑一条主线介绍一下

### 简要回答

我的贡献主要集中在 ROCm 推理路径，涉及量化权重加载、算子分发、Attention 语义和投机解码状态管理。几个问题的共同点是，上层配置或逻辑状态在传到底层执行时被错误解释。我会沿着配置、元数据、wrapper 和 GPU kernel 追踪同一个量，在发生语义偏差的接口处修复，再用回归测试、数值对照和实际模型运行验证。单输出 GEMM 优化与滑动窗口修复已经合并，其余几项仍在推进。

### 完整阐述

我主要围绕 AMD ROCm/RDNA 上的 vLLM 推理做贡献，可以从三个方向介绍。

第一个方向是量化模型的加载和执行。我修复了对称 GPTQ checkpoint 中冗余 `qzeros` 被错误交给 backend 的问题，优化了 RDNA INT4 权重布局重排，也修复了 Channelwise GPTQ MoE fallback 对 `group_size=-1` 的处理。此外，我核对了 Python 分发条件和已有 HIP kernel 的支持范围，补齐 medium-skinny W4A16 路径。

第二个方向是 Attention 正确性。我修复了 ROCM_ATTN 从窗口左跨度转换到完整窗口长度时漏加 1 的问题，并为普通 ROCM_ATTN 接入多模态 Prefix-LM，让同一图像范围内部支持双向注意力。这项工作需要贯通图像范围元数据、分页 KV 读取、kernel mask 与 graph 执行能力。

第三个方向是投机解码。我修复了 MRV2 自适应验证启动 profiling 中有效 token 与 PIECEWISE padding 被混淆的问题，也处理了 LFM2.5 独立 drafter 的 ShortConv 历史跨轮污染和 Chunked Prefill 下一输入错误。

这些工作都需要核对接口两端对同一个对象的定义。例如：`W-1` 表示左跨度，kernel 却需要总窗口长度 `W`；`group_size=-1` 表示整通道量化，不能直接参与尺寸计算；graph 的输入容量也不能自动成为有效 Query 数。我会先明确每个量的生产者和消费者，再决定改动应该落在哪里。

对于性能优化，我会区分算子收益和模型收益。例如 #53283 复用已有 wvSplitK 路径，在特定 BF16 单输出 shape 上获得约 37.8 倍算子加速，对应 Qwen3.5 MoE 配置的纯 Decode 吞吐提升约 12.79%。对于正确性修复，我更关注修复前能否稳定复现、修复后状态和数值是否满足原有契约。

如果面试官让我挑一项深入讲，我会优先选 Prefix-LM 或 ShortConv 状态恢复。它们能够展示我如何从模型计算语义一直追到框架元数据和 GPU 执行，而单输出 GEMM 更适合展示性能定位与小范围工程优化。

**可能追问：你自己的工作和已有框架能力分别是什么？**

我会按 PR 说明贡献边界。例如 #53283 和 #52619 主要补齐分发，复用社区已有 HIP kernel；#60040 复用已有多模态 mask helper，但新增了 ROCm 路径的元数据接入、遍历和读取保护；#59600 则新增了 ShortConv checkpoint 的管理与 kernel 恢复路径。这些贡献应按实际改动介绍。

## 2. 多模态 Prefix-LM 和普通 causal attention 有什么区别？为什么 Gemma3 需要它？

对应 [PR #60040](https://github.com/vllm-project/vllm/pull/60040)。

### 简要回答

普通 causal attention 只允许位置 q 读取此前及当前位置。Gemma3 将图像编码成一段 embedding 放进语言模型序列，同一图像范围内部需要双向 attention，因此前面的图像 token 也能看到后面的图像 token。实现时在 causal mask 上加入「q、k 属于同一图像范围」的条件；未来文本仍然不可见，不同图像也不会自动双向连接。Prefix-LM 决定可见性，prefix caching 决定 KV 复用，两者是独立机制。

### 完整阐述

普通自回归语言模型的 causal mask 是 `k <= q`。Prefill 虽然可以同时计算很多 Query，但每个位置仍然只能读取自己的前缀，以保持自回归依赖。

Gemma3 的多模态路径先用 SigLIP 编码图像，再通过 projector 压缩视觉 token 数、转换特征宽度，把结果合并到语言模型的图像占位位置。进入语言解码器后，图像与文本都表现为 token 行，但图像范围具有不同的 Attention 可见性：同一图像的 embedding 内部可以双向读取。

用六个位置说明：0、1 是前置文本，2、3、4 是同一张图的三个 embedding，5 是后续文本。这里把图像简化成三个 token，仅用于说明 mask。

| Query 位置 | 普通 causal 可见 Key | 加入图像双向规则后 |
|---|---|---|
| 1，前置文本 | 0、1 | 0、1 |
| 2，第一个图像 token | 0、1、2 | 0、1、2、3、4 |
| 3，第二个图像 token | 0、1、2、3 | 0、1、2、3、4 |
| 4，最后一个图像 token | 0～4 | 0～4 |
| 5，后续文本 | 0～5 | 0～5 |

令 `C(q,k)` 表示 causal 条件，`I(q,k)` 表示 q、k 落在同一有效图像范围。无滑动窗口时，可见性是：

$$
M(q,k)=C(q,k)\lor I(q,k).
$$

Query 2 额外读取的 Key 3、4 属于已经给定的图像条件，没有开放未来回答。若输入中有第二张图，则要使用它自己的范围；仅凭两个位置都属于 image modality，不能让它们互相双向读取。

Gemma3 还交错使用局部窗口层和全局层。令 `S(q,k)` 表示 `q-k<W`，普通多模态局部规则是 `(C AND S) OR I`，图像项可以补回窗口外的同图像 Key；开启 `mm_prefix_clamp_sliding_window` 时，规则变成 `(C OR I) AND S`。这里沿用共享 helper 的单侧窗口条件，不能自行换成 `abs(q-k)<W`。实际传入的范围还受到 runner 的范围筛选约束。

**可能追问：视觉 encoder 已经做了双向 attention，语言模型为什么还要做？**

视觉 encoder 混合的是 patch 特征；projector 后的图像 embedding 进入语言解码器，每层又有新的 Q/K/V 投影和 token 交互。两处 Attention 发生在不同的特征空间。视觉 encoder 的双向性不能替代语言模型规定的图像可见性。

**可能追问：它和 prefix caching 有什么关系？**

Prefix-LM 改变 Attention 的依赖关系，prefix caching 复用已有的 KV 计算结果。缓存复用需要满足相应依赖与缓存有效性，不能把一个图像内部双向区间随意切开，然后假设前半段 KV 与完整图像计算相同。

实现与模型背景见[多模态 Prefix-LM 笔记](../../vllm/开发实践/多模态Prefix-LM在ROCm-Attention中的实现.md)。

## 3. Prefix-LM 支持具体改了哪条执行链？只改 causal mask 够吗？

对应 [PR #60040](https://github.com/vllm-project/vllm/pull/60040)。

### 简要回答

只改 mask 不够。图像范围需要从 runner 的通用元数据传到 ROCm backend，再传入 kernel；kernel 还要扩大 Key 遍历范围，否则允许读取的未来图像 Key 根本不会被计算。同时要保护 block-table 和 KV 读取边界，处理不可见缓存中的 NaN，并判断哪些 Decode 批次可以安全复用 causal graph。我分别用 kernel 数值测试、HF hidden-state 对照、模型评测和实际 graph replay 验证。

### 完整阐述

这项支持有两条数据流。模型数值流产生图像 embedding 和各层 Q/K/V；元数据流描述图像在语言序列中占据哪些位置。Attention kernel 必须同时消费它们。

Runner 从 `MultiModalFeatureSpec.mm_position` 提取图像 embedding 的连续范围，写入 `CommonAttentionMetadata.mm_req_doc_ranges`。我在 ROCm metadata builder 中接收并筛选这些范围，用已有 helper 构造 `int32 [num_reqs, max_ranges, 2]` 的设备 tensor，再沿下面的路径传递：

```text
Runner 生成图像位置范围
  → RocmAttentionMetadataBuilder.build
  → RocmAttentionMetadata.mm_prefix_range_tensor
  → RocmAttentionImpl.forward
  → chunked_prefill_paged_decode
  → context_attention_fwd / prefix-prefill kernel
  → 共享多模态 mask helper
```

范围 tensor 只描述位置关系，实际 K/V 仍保存在当前层 tensor 和 GPU KV cache 中。`block_table` 把逻辑历史页映射到物理页，负责历史 KV 的读取；`slot_mapping` 指定新 K/V 的写入槽位。

进入 kernel 后，第一项改变是 Key 遍历范围。假设 Query 在图像位置 2，同一图像还包含位置 3、4。旧 causal 循环可能只遍历到位置 2，即使后面把 mask 放开，也没有读取 3、4 的 K/V。因此需要先让 tile bounds 覆盖可能可见的 Key，再对每个 Query 应用精确的二维 mask。

第二项改变是读取边界保护。扩展 tile 可能跨越真实 cached context 的末尾，必须在 block-table load 时就屏蔽越界位置。仅给后面的 K/V load 加 mask，无法撤销已经发生的非法页表读取。逻辑 q/k 位置还必须包含此前 context 的长度，不能用 chunk 内局部行号计算图像范围。

第三项改变是不可见数值的处理。概率为零并不保证乘上残留 NaN 后还是零。对整个 Query tile 都不可见的缓存列，我清理其 K/V 数值；如果其中任意一行真实 Query 需要这列，就保留原值，再按逐行 mask 处理分数。Padding Query 也不能参加「是否有真实行需要该列」的判断。

第四项改变是 graph 支持判断。范围 tensor 是否存在会影响编译分支和实际执行路径。生成位置已经越过图像、且图像扩展不再改变当前可见性时，普通 causal decode graph 才有复用依据。我据 runner、窗口和 clamp 条件报告 backend 的 graph 能力，并验证宽度 1 和固定投机验证宽度 4 的实际 FULL replay。

验证覆盖 kernel mask、缓存边界、不可见 NaN、多种窗口规则，以及已有 HF 图像 Prefill hidden-state 对照。PR 还报告了 Gemma3-4B IT 在 gfx1151、BF16 上的 AI2D 512 图评测；不同 backend 仍有少量输出差异，因此这些结果支持已测试路径的数值与任务表现，不能概括成所有模式完全一致。

**可能追问：Chunked Prefill 把图像切开，mask 能自动解决吗？**

不能。Mask 只能允许访问已经存在的 Key，不能生成尚未调度的图像 KV。调度和实际输入范围必须保证所需图像 Key 已经可用，不能仅根据双向 mask 推导任意切分都与整图 Prefill 等价。

源码链与边界例子见[多模态 Prefix-LM 笔记](../../vllm/开发实践/多模态Prefix-LM在ROCm-Attention中的实现.md)。

## 4. 输入 1032 超出编译范围 1024，为什么不直接调大上限？根因是什么？

对应 [PR #60049](https://github.com/vllm-project/vllm/pull/60049)。MRV2 指 v2 GPU model runner。

### 简要回答

根因是 dummy profiling 把 PIECEWISE padding 后的容量当成有效 Query 数。1024 的预算里，8 条请求各需要一个 correction slot，所以有效输入已预留到 1016；dispatch 补齐到 1024 后，旧代码又构造了 1024 个有效 Query，drafter 再加 8 就变成 1032。我让 dummy batch 分别保存有效数和 padded 容量，并保留正确的 Query 边界、positions 和尾部 padding 标记。调大上限会掩盖错误计数，无法修复预算语义。

### 完整阐述

自适应投机验证在服务启动时采集不同执行规模的成本，用来估计运行时应该验证多少 draft token。因此 dummy run 需要执行真实模型路径，同时构造一致的请求与 Query metadata。

复现配置的编译 token 上限是 1024，请求数是 8。独立 draft model 的 step 0 会为每条请求追加一个 correction 输入，因此 `_dummy_run` 已预留额外空间：

$$
T_{\mathrm{valid}}=1024-8=1016.
$$

每条 dummy 请求有 127 个有效 Query。PIECEWISE dispatcher 随后选择输入容量为 1024 的执行 descriptor。问题发生在下一步：旧 `execute_model` 用 descriptor 的 1024 调用 `InputBatch.make_dummy`，使 padding 被重新计入有效 Query。Drafter 再按请求数扩展，得到 `1024+8=1032`，超过编译范围。

正确流程要同时保留两种长度：

| 阶段 | 有效 token 数 | 执行输入容量 |
|---|---:|---:|
| 为 correction 预留后 | 1016 | 尚未选择 bucket |
| PIECEWISE dispatch 后的 target dummy batch | 1016 | 1024 |
| 独立 drafter 的 expanded 输入 | 1016 + 8 = 1024 | 满足 1024 上限 |

这里的 padding 和 correction 发生在不同输入布局中，不能理解成直接把 target 的八个 padding 槽原地改写成 correction。

修复从同时拥有两种长度的 `GPUModelRunner.execute_model` 开始。PIECEWISE 下，用原始 `num_toks` 建立有效 Query，再单独传入 `batch_desc.num_tokens` 作为 `num_tokens_after_padding`。`make_dummy` 用有效数建立 `num_scheduled_tokens`、`query_start_loc` 和 logits 索引，用 padded 容量切片 `input_ids`、`positions` 与 `is_padding`，并把尾部标成 padding。`set_dummy_context` 也只写有效 positions。

比如两条请求各有三个有效 Query，padding 后长度为 8，context 为 4：

```text
query_start_loc = [0, 3, 6]
positions       = [4, 5, 6, 4, 5, 6, 0, 0]
is_padding      = [F, F, F, F, F, F, T, T]
```

最后两个槽没有请求内 Query 身份。Profiling 可以让有效 dummy token 参与 MoE 路由，以测量实际成本，但尾部 padding 不能被当成专家工作量。

我没有把这个规则无条件推广到 FULL dummy。FULL 的请求和 Query 布局需要符合 capture descriptor，原有重塑行为继续保留，并用 FULL varlen 控制测试覆盖。有效工作量和执行容量还影响自适应成本表：同一个 graph bucket 内，多个有效长度可能对应相同执行 shape 和相近成本，不能把 padded 数直接解释为新增业务工作。

**可能追问：怎么证明这是共享 MRV2 问题，而不是 Prefix-LM PR 引入的？**

我换成纯文本 Qwen3-1.7B target 与 Qwen3-0.6B draft，在上游代码上分别使用 TRITON_ATTN 和 ROCM_ATTN，复现同样的 1032 越界。修复后两种 backend 都完成服务启动、smoke 请求和 128 条 GSM8K 请求，并观察到非零 draft acceptance。恢复原函数会让新增回归用例失败。这些证据支持启动与执行修复；PR 没有据此宣称普遍性能提升或所有模型精度完全一致。

详细计数与源码见[PIECEWISE 填充及启动 Profiling 笔记](../../vllm/开发实践/MRV2的PIECEWISE填充与投机解码启动Profiling.md)。

## 5. ShortConv 历史为什么会被投机解码污染？拒绝后缩回序列长度还不够吗？

对应 [PR #59600](https://github.com/vllm-project/vllm/pull/59600)。

### 简要回答

ShortConv 保存的是每层卷积输入 B*X 的固定长度历史。生成 draft 时这个窗口会被覆盖，target 拒绝候选后，缩回逻辑长度并不能恢复旧数值。我为每层增加按稳定 request slot 索引的 checkpoint，让 step 0 从可信历史重放有效输入；保存 correction 前的历史供下一轮恢复，同时留下 correction 后的 runtime state 供本轮继续 drafting。Chunked Prefill 尚未采样时，追加输入改用下一枚 prompt token，避免读取旧 last-sampled token。

### 完整阐述

LFM2.5 混合使用 Attention 和 ShortConv。ShortConv 将 hidden states 投影成 B、C、X，先计算逐元素卷积输入 `u=B*X`，再做因果 depthwise 卷积，最后由 C 调制并经过输出投影。宽度为 L 的卷积，每个 channel 需要保存此前 `L-1` 个 u 值。

所以 history 的数值来源是当前层的 `B*X`，不能直接用 token ID、最终 hidden states 或 Attention KV 替代。以宽度 3 为例，历史 `[2,5]`、当前输入 7、从旧到新的系数 `[1,2,3]`，输出为 `2+2×5+3×7=33`，新历史是 `[5,7]`。

投机路径的问题在于状态更新发生在 target 作出最终接受决定之前。Drafter 消费 a 预测 d0，消费 d0 预测 d1，消费 d1 预测 d2；target 验证 `[a,d0,d1,d2]`，可能只接受 d0，拒绝后续候选并给出 correction c。有效路径推进到 `[a,d0,c]`，但 drafter 临时 state 已可能包含后来被拒绝的输入，expanded 输入中的拒绝尾部也可能污染窗口。

这里还存在输入和预测结果的一位错开：预测出 d2，不表示 drafter 已经消费 d2。恢复边界必须根据实际输入位置确定，不能只按候选数量随意截断。若沿用错误历史 `[11,13]` 计算 correction 对应的输入 7，输出会变成 58；这类误差会继续传播到后续层和 draft logits。

我的方案是每层保存一份最小历史 checkpoint，step 0 读取 checkpoint 重放本轮有效输入，并在 correction 边界建立两个窗口。令 H(p) 表示消费位置 `[0,p)` 后的历史，correction 位于 p：

| 保存的对象 | 数值边界 | 用途 |
|---|---|---|
| `draft_checkpoint` | H(p)，未消费 correction | 下一轮 step 0 恢复并重放 |
| `conv_states` 有效窗口 | H(p+1)，已消费 correction | 本轮后续 draft steps |

这两份窗口在 kernel 内按同一组输入建立。Correction 位于 chunk 前部、所需历史位置为负时，要从旧历史补齐，不能读负 X 地址，也不能一律补零。

Checkpoint 按稳定 request slot 保存。当前 batch 从 `[A,B]` 变成 `[B,C]` 后，B 的 batch row 会变，但其 checkpoint 仍属于 B。Speculator 通过 `input_batch.idx_mapping` 将 batch row 映射到 request slot，通过逐层 metadata 把 checkpoint、请求索引和 correction 索引传给 `ShortConv.forward_cuda`，最终进入卷积 kernel。Runtime state 的物理 cache index 则继续沿原缓存元数据定位。

Chunked Prefill 还有一个相邻问题：prompt 尚未结束时，target 本轮可能没有采样回复，统一追加位置应放下一枚 prompt token。例如本轮处理 `[p0,p1,p2]`，后面还有 p3、p4，则下一输入应该是 p3。我按 `num_sampled` 选择 correction 的来源：已采样时用最后采样 token，否则用 `next_prefill_tokens`。

验证包括卷积数值与历史边界测试、连续投机轮次，以及请求重排、slot 复用和抢占恢复。相同 LFM2.5-350M 作为 target/drafter 的 E2E 回归，在 base 上接受率约 35.56%，修复后连续两批都超过 90%。同模型不能完全消除执行差异，但能显著减少模型能力差异这个变量，配合逐层状态检查验证恢复逻辑。

该实现要求独立 ShortConv draft 路径使用 `mamba_cache_mode='none'` 并关闭 prefix caching，新增 checkpoint kernel 支持卷积宽度 2～4。这里的 `none` 是特定缓存模式配置，ShortConv 仍保存运行时历史。已报告的本地验证是 ROCm、TP1。

**可能追问：有了接受率改善，就能证明目标模型输出分布正确吗？**

接受率主要反映 drafter 与 target 的匹配情况，不能单独证明分布正确。还要检查目标模型的验证采样规则、逐层状态、数值结果和输出评测。PR 中真实大小 target/draft 的 32 题 GSM8K 单次实验，两种配置均为 16/32，固定 drafter 相对 target-only 的输出吞吐增加约 33.66%；这个结果只描述该小样本工作负载。

模型结构、状态边界与源码见[ShortConv 历史恢复笔记](../../vllm/开发实践/MRV2投机解码中的ShortConv历史恢复.md)。

## 6. 滑动窗口修复只加了一个 1，为什么会改变模型输出？

对应 [PR #59550](https://github.com/vllm-project/vllm/pull/59550)。

### 简要回答

调用两端对参数的定义不同。Backend 保存 `(W-1,0)`，其中 W-1 是左侧跨度；底层 helper 使用 `q-k<W`，需要包含当前 token 的完整长度 W。旧代码直接传 W-1，漏掉窗口最早的有效 token；W=1 时传零还会错误关闭窗口。我在 backend 调用 helper 的边界补上 1，并用可手算的 Attention 输出覆盖窗口边界、W=1 和无窗口情况。

### 完整阐述

对包含当前 token 的 causal 窗口，长度 W 的可见条件是：

$$
k\le q,\qquad q-k<W.
$$

当 `q=7、W=4`，Key 4、5、6、7 都应可见。FlashAttention 风格的 tuple 使用左右跨度，因而这个窗口表示为 `(3,0)`。ROCM_ATTN 保存 tuple，而 `chunked_prefill_paged_decode` 和 prefix-prefill kernel 接受总窗口长度。

旧代码直接传 `self.sliding_window[0]`，把总长度变成了 3，只保留 Key 5、6、7。W=1 时 tuple 左跨度为零，helper 又把非正值视为禁用窗口，实际可能读取所有历史。

我追踪的路径是模型 Attention、`RocmAttentionImpl.forward`、`chunked_prefill_paged_decode`，最后到 decode 或 prefix-prefill kernel 的窗口 mask。修复在调用边界传入 `1 + self.sliding_window[0]`。无窗口 sentinel 的左跨度 -1 同时映射成 0，继续表达关闭窗口。

这个位置适合修复，是因为 backend 仍有消费者需要 tuple 表示，而 kernel 已经按总长度定义。修改 kernel 的比较运算会改变它与其他调用者的契约；统一修改字段含义也可能破坏其他 backend 路径。

为了直接验证最早位置有没有参与 Attention，我令 Q/K 全零、Value 为 `V[k]=k+1`。可见位置的分数相同，因此输出就是这些 Value 的均值：

| 情况 | 可见 Key | 输出 |
|---|---|---:|
| 正确 W=4 | 4、5、6、7 | (5+6+7+8)/4 = 6.5 |
| 旧代码传 3 | 5、6、7 | (6+7+8)/3 = 7 |
| 正确 W=1 | 7 | 8 |
| 旧代码传 0，关闭窗口 | 0～7 | (1+2+…+8)/8 = 4.5 |

这个例子直接证明窗口选择和 softmax 归一化集合发生变化，比仅观察模型生成的一段文本更容易判断根因。回归测试覆盖有窗口、无窗口以及 cached-KV 边界；记录中修复前四个 windowed case 失败、两个 no-window control 通过，修复后相关测试共 7 个通过。

Chunked Prefill 必须使用绝对逻辑位置。总序列长度 8、本轮 Query 长度 3 时，context 长度为 5，局部行 0、1、2 对应 q=5、6、7。用局部行号计算窗口会在每个 chunk 开头错误重置。

**可能追问：KV cache manager 要同步修改吗？**

这次不用。Mask 决定逻辑位置能否参与 Attention，manager 根据 block size、引用和生命周期保留或回收物理页。边界页可能包含不可见 token；仍存储在 GPU 中，不表示参与本轮归一化。PR 修复的是表示转换，两者原有责任保持成立。

模型级验证覆盖了 gfx1150 上 Gemma3-1B、窗口内外输入、Chunked Prefill 和 HIP graph Decode。长输入的参考误差随输入长度可能增加或减少，因此主要正确性依据是明确的窗口契约与回归数值，不能宣称每个样本误差都降低。

完整调用链与手算见[滑动窗口边界笔记](../../vllm/开发实践/ROCm-Attention滑动窗口的参数与边界.md)。

## 7. 单输出 GEMM 为什么会成为瓶颈？你新增了 HIP kernel 吗？

对应 [PR #53283](https://github.com/vllm-project/vllm/pull/53283)。

### 简要回答

我补齐的是已有 wvSplitK kernel 的 Python 分发。Qwen shared expert 的 gate 是一个输出维度为 1 的未量化 Linear，旧条件只接受输出通道数大于 8，导致它落到通用 fallback；底层已经有单输出模板。改后在 gfx1151 的 BF16 `[1,2048]` 算子上延迟降低约 37.8 倍，对应 Qwen3.5 MoE 配置的纯 Decode 吞吐提升约 12.79%。它每层反复执行，所以小算子也会积累成明显延迟。

### 完整阐述

这条路径来自 Qwen shared expert 的标量门控。输入 hidden states 为 `[T,H]`，gate 权重为 `[1,H]`，产生 `[T,1]` 的 logit，经 sigmoid 后沿 hidden 维广播，缩放共享专家输出：

$$
Y_{\mathrm{shared}}(X)
=\sigma(Xw_g^T)\odot\mathrm{MLP}_{\mathrm{shared}}(X).
$$

这个 gate 与产生多个专家分数、执行 Top-K 选择的 routing gate 是两个对象。在已核对的实现中，shared-expert gate 的 `quant_config=None`，所以即使主体权重使用 GPTQ INT4，这个 Linear 仍可以走 BF16/FP16 未量化 GEMM。

实际路径从 `ReplicatedLinear.forward` 到 `UnquantizedLinearMethod.apply`，进入平台选择的 ROCm wrapper，再到 `rocm_unquantized_gemm_impl`。旧分发中的局部变量定义需要特别注意：

```python
n = x.numel() // x.size(-1)  # 激活/token 行数
m = weight.shape[0]         # 输出通道数
k = weight.shape[1]         # 收缩维度
```

因此 `m==1` 是单输出，`n==1` 才是单激活行。旧条件 `m > 8 and 0 < n <= 5` 排除了单输出 gate。底层 wvSplitK 已支持单输出，并有 `YTILE=1` 模板，我把条件扩展为：

```python
if (m == 1 or m > 8) and 0 < n <= 5:
    out = ops.wvSplitK(weight, x_view, cu_count, bias)
```

原有的架构、环境开关、FP16/BF16、K 对齐和连续性要求仍在。分发还受 AITER 等前置分支影响，不能只看这一个条件就断言所有同形状调用都会进入 wvSplitK。激活行数也取决于完整 leading dimensions：例如 `[2,3,2048]` 展平后有六行，会超过五行限制。

对于 `[1,2048]` 激活和 `[1,2048]` 权重，这个计算本质上是一个点积。通用 GEMM 对宽输出矩阵的分块和调用成本，不一定适合一个标量输出。已核对的 gfx1151 路径通过 wave 内 K 方向计算和归约完成单输出；不能只凭 SplitK 名称就描述成多个 CU 使用 atomic 共同计算一个 gate。

性能实验在 Radeon 8060S/gfx1151 上进行，对应模型为 Qwen3.5-35B-A3B-GPTQ-Int4：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| BF16 `[1,2048]` 算子延迟 | 119.484 μs | 3.161 μs |
| 纯 Decode 吞吐 | 23.74 tok/s | 26.77 tok/s |
| 每 token Decode 延迟 | 42.125 ms | 37.350 ms |

Profiler 确认命中了 `YTILE=1` 专用路径，greedy 输出 token 一致。该 PR 的核心是让已支持的 shape 使用已有 kernel，因此改动可以落在公共 wrapper，避免把 ROCm 特判写到某个模型 gate 中。

**可能追问：算子快了 37.8 倍，为什么模型只提升约 13%？**

完整 Decode 还包含其他 Linear、Attention、专家计算和框架开销。若该算子原先占总时间的比例为 f，局部加速倍数为 s，在其余时间不变的假设下，总加速上限可写成 `1 / ((1-f) + f/s)`。重复执行让小 gate 值得优化，但算子倍数不能直接当成模型倍数。

模型门控与真实线程分工见[单输出 GEMM 笔记](../../vllm/开发实践/单输出GEMM与Shared-Expert门控的ROCm优化.md)。

## 8. 对称 GPTQ 为什么还带 qzeros？为什么删 tensor 却保留名字？INT4 重排怎么优化？

对应 [PR #52637](https://github.com/vllm-project/vllm/pull/52637)。

### 简要回答

这条 PR 有加载修复和布局优化两部分。对称 GPTQ 使用固定零点 8，但 checkpoint 仍可能含 qzeros；选中的 kernel 不需要显式零点，我在共享 post-load 边界移除冗余 tensor，保留参数名供其他 backend 创建运行时占位。RDNA 重排则复用 packed gptq_shuffle，避免将整层 INT4 展开成 INT32 再打包，减少中间内存与加载开销。通过最终权重 hash 和数值对照确认布局等价，收益主要体现在启动阶段。

### 完整阐述

首先是对称 GPTQ 的数值与存储契约。这里 `uint4b8` 的量化整数范围为 0～15，反量化使用固定偏移 8：

$$
\widehat W_{n,k}=s_{n,g(k)}(q_{n,k}-8).
$$

`zero_points=False` 表示 kernel 不需要额外显式零点 tensor；固定偏移仍然存在。Checkpoint 可能保存 packed `qzeros`，loader 也正常注册、加载它，但选中的 backend 后处理不应该消费这份冗余对象。实际故障表现为期望 `[320,16]` 却得到 `[16,320]` 的零点 shape。

我在 `AutoGPTQLinearMethod.process_weights_after_loading` 中，根据选中 kernel 的配置清理 tensor，再调用 backend 后处理。这是共享 loader 边界，已经掌握参数名称和 kernel 契约，能够在错误对象被消费之前处理。

删除的是 layer 上的 checkpoint tensor，`kernel.w_zp_name` 保留。其他 backend 可能通过该名称创建 fake zero point 或运行时 placeholder，例如 CPU W4A8 的相应处理；如果名字也清空，会破坏这部分后处理契约。需要显式零点的配置则继续保留原路径。

第二部分是 RDNA 权重准备。原流程将 packed INT4 全部展开成 INT32，再按 ExLlama nibble 顺序打包。对于 N×K 个量化值，packed 权重约为 `N×K/2` 字节，INT32 中间量约为 `4×N×K` 字节，后者是前者的八倍；这个比较只计算整数表示，不包含 scale 和其他 buffer。

我复用已有 `gptq_shuffle`，直接重排 packed word。一个 INT32 内普通顺序为 `q0,q1,…,q7`，shuffle 后为 `q0,q2,q4,q6,q1,q3,q5,q7`。例如低位到高位的值 0～7，word 从 `0x76543210` 变为 `0x75316420`。改变的是存放位置，解码后的逻辑权重应相同。

预处理先根据 `packed_dim` 将 packed K 轴规范到第 0 维，建立连续独立副本，调用原地 shuffle，再转置成连续的输出通道行布局，最后按 int8 解释字节。GPTQ 常见输入 `[K/8,N]` 与 compressed-tensors 的 `[N,K/8]` 因此可以收敛到同一 runtime buffer。Clone 的目的同时包括连续性和保护 checkpoint 共享存储，单独 transpose 通常只是 view，无法满足这两个要求。

验证包含冗余 qzeros 清理、名字保留及 backend 替换参数的回归，也检查非连续输入、不同 group size、tail 和 checkpoint 未被修改。Qwen3-4B GPTQ 与 Qwen3-VL-4B compressed-tensors 的对照中，各有 144 份 prepared weight hash 一致，并比较 greedy token、文本和 top-3 logprobs。

性能比较两边都采用相同 loader fix，仅改变重排实现。gfx1151 上已有记录中的 Qwen3-4B GPTQ 初始化时间从约 15.68～16.29 秒降到 12.95～13.28 秒，初始化下降约 17.5%～18.5%。实验使用已有 OS/JIT cache，并非所有进程启动阶段都计入，讲述收益时需要保留这个口径。

**可能追问：为什么 Decode 不一定变快？**

重排发生在一次性加载阶段，最终 inference kernel、dispatch 和权重布局用途没有改变。省掉加载中间量能够改善加载时间与该阶段的内存成本，不能直接推导每 token 计算更快。

零点兼容性与布局细节见[GPTQ 加载与 INT4 重排笔记](../../vllm/开发实践/GPTQ权重加载与RDNA-INT4布局重排.md)。

## 9. Channelwise GPTQ 的 group_size=-1 是什么意思？只修分配为什么不够？

对应 [PR #58513](https://github.com/vllm-project/vllm/pull/58513)。

### 简要回答

`group_size=-1` 表示每个输出通道沿整个 K 维共享一组量化参数。旧 MoE fallback 直接把 -1 用于尺寸计算，导致负维度；只把分配改为单组，kernel 仍可能按 -1 算出错误 scale offset。我在加载阶段分配一组参数，并在每次 GEMM 启动前把 -1 解析为当前输入的 K，同时传给 block 配置和 kernel。w13、w2 的 K 不同，所以不能在整个层初始化时统一替换成一个固定值。

### 完整阐述

Groupwise 量化沿逻辑 K 维分组，每组共享一个 scale，存在显式零点时也共享相应 zero point。Channelwise 的 `group_size=-1` 表示每个输出通道覆盖完整 K 维，只保留一组参数。它是配置含义，进入实际 tensor 分配或索引前需要解析。

在 ROCm 上，无法使用 GPTQMoeMarlin 的 AutoGPTQ MoE 层会进入 Triton WNA16 fallback。旧实现用 `-1` 直接参与 group 数量的计算，创建 scale/zero-point tensor 时出现 `Dimension size must be non-negative`。

第一项修复是在 `MoeWNA16Method.create_weights` 中，对 Channelwise 分配单组 scale 和 zero point。以某次 GEMM 的逻辑权重 `[N,K]` 为例，每个输出通道对应一个 scale，沿 K 的 group 数是 1。

第二项修复在公共 Triton wrapper `invoke_fused_moe_wna16_triton_kernel`。每次 launch 都能看到实际激活 A 的形状，因此把 `group_size=-1` 解析为 `A.size(1)`，并检查 scale/zero-point 的 group 轴为单组。解析后的正值既传给 block 配置选择，也传给 GPU kernel，保证配置和索引使用同一含义。

为什么要在每次 launch 解析？MoE 有两次 GEMM：w13 消费 hidden states，w2 消费中间激活。用简化例子，hidden size 为 4096、intermediate size 为 14336，则两次 GEMM 的有效 group size 分别是 4096 和 14336；两者都只有一个 scale group，但真实 K 不同。

如果初始化时把整个层的 -1 替换成 4096，w2 的后续 K 位置可能得到非零 group index，试图访问不存在的第二、第三组。反过来，只修分配而继续把 -1 送进 kernel，也会留下负的 group 索引。因此加载和执行必须同时修复。

这项修改落在共同 wrapper，能让 w13、w2 按各自当前 K 解析，也覆盖进入该 fallback 的不同模型。普通正 group size 继续按原有 groupwise 路径处理。

我使用 allocation 回归和 Channelwise Triton 数值测试检查两部分，并实际加载 Mixtral-8x7B GPTQ Channelwise 模型。记录中 targeted tests 为 18 passed、9 skipped；模型加载约 22.18 GiB 权重，选中 `TritonWNA16Experts`，执行 `fused_moe_kernel_gptq_awq` 并完成离线推理。实际运行支持这条加载与执行链可用，数值测试则进一步检查 scale 索引正确性。

**可能追问：模型能回答“巴黎”就说明修复正确了吗？**

自然语言 smoke 能证明端到端链路运行，但不足以检查所有量化索引。需要让多个 K 位置都参与计算，分别覆盖 w13/w2 的维度，并与解量化参考结果比较。否则部分位置 scale 读错，也可能恰好生成看似合理的文本。

分配与两次 GEMM 的调用链见[Channelwise GPTQ MoE 笔记](../../vllm/开发实践/Channelwise-GPTQ-MoE的加载与Triton执行.md)。

## 10. W4A16 分发上限从 32768 改到 39321，依据是什么？超过 LDS 容量安全吗？

对应 [PR #52619](https://github.com/vllm-project/vllm/pull/52619)。

### 简要回答

新上限来自已有 C++ kernel 的 medium 模式契约。Python 原来只允许 M≤5 且 K×M≤32768，底层却已经支持到 `int(32768×1.2)=39321` 个激活元素：前一部分放 LDS，尾部从 global memory 读取。我补齐 Python 分发，保留其余限制，并用边界与 FP16/BF16 数值测试覆盖新增范围。特定 Qwen3-8B batch 3 配置吞吐提升约 63.4%，另一配置约 6.3%，收益取决于 shape 与模型时间占比。

### 完整阐述

`RDNAHybridW4A16LinearKernel` 在小激活行数时可以使用 HIP `wvSplitK_int4_g`，其他 shape 走 Triton。这个 PR 的重点是 Python 允许范围与现有 C++ 算子能力不一致。

按常用 GEMM 记号，激活为 `[M,K]`、权重为 `[N,K]`、输出为 `[M,N]`。这里 Python 的 M 映射到 C++ 的 `N_in`，即激活行数；不要与单输出 GEMM wrapper 中表示输出通道数的局部 m 混淆。

旧 Python 条件是 `M<=5` 且 `K×M<=32768`。底层已经支持同样的最多五行，并允许激活元素数达到 `int(max_lds_len×1.2)`；`max_lds_len=32768`，得到 39321。

Medium 模式并不把全部超出部分继续写进 LDS。它把前一段激活保留在 LDS，尾部通过 global memory 读取。因此是否安全，要根据这条 kernel 的分配和寻址契约判断，不能仅凭总激活数大于常规 LDS 元素容量就断言越界，也不能凭经验把上限扩大到任意值。

两个已验证模型 shape 位于新增范围：

| 模型与 batch | down_proj shape | 激活元素数 K×M | 旧路径 | 新路径 |
|---|---|---:|---|---|
| Qwen3-8B GPTQ，batch 3 | M=3、K=12288、N=4096 | 36864 | Triton | HIP medium |
| Qwen3-4B GPTQ，batch 4 | M=4、K=9728、N=2560 | 38912 | Triton | HIP medium |

我修改 Python 分发门槛，并围绕常规边界、medium 上限和上限外的形状增加路径测试；同时用 FP16/BF16 对照 FP32 参考，覆盖实际新增范围的数值结果。39321 是浮点计算后转整数得到的上限，比较时包含这个整数边界。

性能实验在 Radeon 8060S/gfx1151 上，采用相同 ROCM_ATTN、关闭 AITER、eager 执行、关闭 graph/compile 与 prefix cache，固定 prompt、seed、输出长度，经过 warmup 后测量三次。结果为：

| 配置 | Triton 直接算子耗时 | HIP medium 耗时 | 模型聚合 Decode 吞吐 |
|---|---:|---:|---|
| Qwen3-8B，batch 3 | 0.643 ms | 0.388 ms | 41.70 → 68.16 tok/s，+63.43% |
| Qwen3-4B，batch 4 | 0.136 ms | 0.071 ms | 119.30 → 126.83 tok/s，+6.31% |

两组数值都说明新增路径有收益，但模型级幅度差别很大。直接算子计时是孤立调用，模型中还有大量其他工作、不同缓存与调度状态；不能把某一个直接算子的比值代入整模型，推导所有配置都会获得相同比例。

**可能追问：为什么不再把阈值调大一些？**

这次依据是已有 C++ operator 的明确支持上限。继续扩大需要重新检查 kernel 的地址、容量、数值和性能，也可能需要修改 kernel。安全支持和性能合适是两个判断：即使某个 shape 可以运行，仍要对照 fallback 测量才能决定是否值得分发。

**可能追问：32768 是 32 KiB 吗？**

这里比较的是激活元素数。FP16/BF16 每元素两字节，32768 个元素对应 65536 字节的数据载荷；实际 kernel 的 LDS 分配还需要按源码中的 buffer 与布局理解，不能把元素数直接写成字节数。

Python/C++ 形状映射与 LDS 路径见[RDNA W4A16 分发笔记](../../vllm/开发实践/RDNA-W4A16的形状分发与LDS边界.md)。
