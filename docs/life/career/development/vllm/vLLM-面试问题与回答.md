# vLLM 开源贡献：面试问题与回答

<div class="interview-notes" markdown="1">

先完整说明个人贡献的主线，再逐个展开 PR 的触发问题、改动与难点。基础知识跟随调用链出现，正文直接展示，便于沿问题连续阅读。完整实现分析与实验记录见 [vLLM PR 材料](vLLM-PR.md)及各节的分析入口；这里保留机制和形状示例，性能数据回到分析页复习。

## 面试内容概览 {#overview}

共 52 个问题，分为贡献总览和 8 个 PR 主题。每个 PR 都从触发问题与个人改动展开到相关基础知识；点击主题进入整组，或直接跳到具体追问。

| 复习主题 | 问题范围 | 主要内容与直接跳转 |
| --- | --- | --- |
| [贡献总览与框架位置](#contribution-overview) | 1–3 | 完整贡献介绍、[模块与请求调用链](#q2)、个人新增与组件复用边界 |
| [单输出 GEMM 分发](#pr-53283) · [#53283](https://github.com/vllm-project/vllm/pull/53283) | 4–9 | shared expert 与 router、未量化 Linear、[Linear 到 GEMM 的调用链](#q7)、M/N/K、[Split-K](#q9) |
| [W4A16 medium-skinny 分发](#pr-52619) · [#52619](https://github.com/vllm-project/vllm/pull/52619) | 10–14 | W4A16 存储与计算精度、shape 分发、[LDS 容量与代价](#q13)、custom op 与 fake implementation |
| [GPTQ 加载与 packed 重排](#pr-52637) · [#52637](https://github.com/vllm-project/vllm/pull/52637) | 15–20 | 对称/非对称零点、[权重加载生命周期](#q17)、冗余 qzeros、packed shuffle、view 与独立副本 |
| [Channelwise GPTQ MoE](#pr-58513) · [#58513](https://github.com/vllm-project/vllm/pull/58513) | 21–26 | 专家两次 GEMM、group size、[为什么 -1 要按每次 K 解析](#q24)、fallback 与 token-expert pair 索引 |
| [滑动窗口 Attention 边界](#pr-59550) · [#59550](https://github.com/vllm-project/vllm/pull/59550) | 27–31 | causal/window mask、逻辑位置与分页 KV、[softmax 前 mask](#q29)、online softmax、可见范围与缓存回收 |
| [多模态 Prefix-LM](#pr-60040) · [#60040](https://github.com/vllm-project/vllm/pull/60040) | 32–38 | 图像双向可见性、[Attention metadata](#q34)、[完整调用链](#q35)、安全 KV 读取与 NaN、窗口组合、Graph 与 prefix caching |
| [MRV2 dummy 与执行容量](#pr-60049) · [#60049](https://github.com/vllm-project/vllm/pull/60049) | 39–44 | 启动 profiling、compile/Graph、[有效 Query 与 bucket padding](#q42)、字段修复、自适应投机验证成本表 |
| [ShortConv drafter 历史恢复](#pr-59600) · [#59600](https://github.com/vllm-project/vllm/pull/59600) | 45–52 | [draft/verify/correction](#q46)、卷积历史、[correction 前后双窗口](#q48)、稳定请求槽位、Chunked Prefill 与采样语义 |

各节源码链接绑定对应 PR 分支的固定提交，文件名与调用链以该提交为准。不同 PR 的实现分别核对，不把多个分支视为同一个版本。

## 一、完整介绍：配置、执行和状态之间的接口 {#contribution-overview}

这些 PR 分布在不同模块，但共同关注一个接口问题：上层描述的工作或状态，传到底层后是否仍保持同一个含义。介绍之后先确定修改位置和复用边界，再从 GEMM 的具体案例展开。

### 1. 请介绍你的 vLLM 开源贡献，以及这些 PR 之间的联系。 {#q1}

**贡献主线：沿接口定位配置、shape 与状态问题**

我的 vLLM 开源贡献主要围绕 **ROCm 推理路径中的量化、GEMM 分发、Attention 和投机解码** 。这些 PR 看起来覆盖的模块不同，但共同的问题是：配置、shape 或状态在上层表达正确，传到底层后却被当成另一个含义。

我通常沿配置生产者、Python wrapper、backend 和 kernel 消费者追踪同一个对象，先确定语义在哪个边界发生偏差，再选择最小且完整的修改范围。

**量化加载：从 checkpoint 到运行时表示**

第一条主线是 **量化模型从 checkpoint 到执行 kernel 的连接** 。

对称 GPTQ 使用固定偏移零点，文件却可能带有冗余 qzeros，我在 backend 后处理前清理不应被消费的 tensor；RDNA 权重重排则复用 packed shuffle，避免为了换 nibble 顺序把整层展开成 INT32。

Channelwise GPTQ MoE 的 -1 代表整条 K 一组，我同时修参数分配与每次 GEMM launch，避免 gate/up 和 down 不同 K 被统一当成 hidden size。这些工作要求分清文件表示、运行时布局和实际量化数学。

**GEMM 分发：让已有 kernel 适配实际形状**

第二条主线是 **让已有 kernel 在适合的 shape 上真正被调用** 。单输出 shared-expert gate 原先漏掉 skinny GEMM 路径；W4A16 的 Python 条件又比已有 medium kernel 的支持范围更窄。

我核对实际的输出宽度、token 行数和 LDS 工作集后补齐分发，复用已有 HIP kernel。这里的贡献是执行选择与接口对齐，不是重新写了所有矩阵乘；小算子能否改善模型表现，还要看它在完整执行链中的位置。

**Attention：从模型可见性到 GPU 读取**

第三条主线是 **Attention 的可见性与执行 metadata** 。滑动窗口的 W 包含当前 token，左跨度却是 W-1，少一次转换就会少读一个 Key；Gemma3 Prefix-LM 更进一步，需要同图像内部双向可见。

我把图像范围接入 ROCm metadata 和 kernel，同时扩大 Key 遍历并保护页表、KV 读取与不可见 NaN。只改布尔 mask 不够，模型所需的 Key 必须真的进入计算，缓存地址也必须合法。

**跨轮状态：执行容量与 drafter 历史恢复**

最后一条是 **MRV2 和独立 drafter 的执行状态** 。在启动时构造模拟输入的路径中，PIECEWISE dummy 把填充后的容量重新当成有效 Query 数，后续追加 correction 输入便会越过预算。

另一个问题发生在投机解码：drafter 提前生成候选时，ShortConv 的固定历史窗口会被后来遭拒的输入覆盖，缩回序列长度无法恢复旧数值。我分别保留有效数与容量，并建立按稳定请求槽位索引的 correction 前 checkpoint，让下一轮重放可信历史。

这几组贡献覆盖了权重加载、算子选择、Attention 可见性和跨轮状态；它们共同要求沿调用链确认每个接口到底描述什么、由谁消费。

可以从[贡献边界](#q3)选一条深入：[Prefix-LM](#q32)贯穿模型语义到 GPU 读取，[ShortConv](#q45)贯穿投机时间顺序到状态恢复。

### 2. 这些 PR 主要修改了 vLLM 的哪些模块？能放到请求调用链里讲吗？ {#q2}

修改主要落在加载链、运行链和投机状态三条路径：

| 路径 | 主要调用 | 对应改动 |
| --- | --- | --- |
| 加载 | 模型 loader → 量化方法 `process_weights_after_loading` → backend 权重准备 | quantization、GPTQ 加载与重排 |
| 模型执行 | Scheduler → MRV2/InputBatch/metadata → 模型 → Linear 或 Attention backend | GEMM 分发、Attention backend/ops、runner/input batch |
| 投机执行 | speculator 生成候选 → drafter 状态更新与恢复 | speculator、ShortConv checkpoint 与 kernel |

Linear 通过 `quant_method.apply` 进入平台 GEMM 或量化 kernel；Attention 通过 backend 读取分页 KV。具体入口随对应 PR 的固定版本说明。

### 3. 哪些工作是你自己新增的，哪些复用了 vLLM 已有能力？ {#q3}

| 工作 | 复用的能力 | 我补齐的内容 |
| --- | --- | --- |
| 单输出与 W4A16 GEMM | 已有 HIP kernel | 核对支持范围，对齐 Python/C++ 分发条件 |
| GPTQ 重排 | `gptq_shuffle` | 加载表示转换与布局契约 |
| Prefix-LM | 多模态范围与 mask helper | ROCm metadata、Key 遍历和读取保护 |
| ShortConv | 既有卷积与 drafter 状态框架 | 按请求保存 checkpoint、传恢复信息、kernel 重建历史 |

我的贡献是将这些能力接到正确的生命周期和调用链上，复用部分与个人新增分别说明。

[返回面试内容概览](#overview)

## 二、单输出 GEMM：从 shared expert 到平台分发 {#pr-53283}

先定位模型里的门控计算，再看它为什么走未量化 Linear，以及输出宽度在分发层如何被解释。这条链明确后，才讨论小 GEMM 的并行组织。

深入分析见[单输出 GEMM 与 Shared-Expert 门控](单输出GEMM与Shared-Expert门控的ROCm优化.md)。
{.interview-source}

### 4. 单输出 GEMM 这个 PR 解决了什么问题？你具体改了什么？ {#q4}

**触发问题**

这项优化从 **MoE 的 shared-expert gate** 出发。该分支用一个输出通道的 Linear 为每个 token 生成门控标量，计算很小，却会在许多层和 Decode 步中重复。

我沿模型层到 ROCm GEMM 的调用确认，它属于未量化 FP16/BF16 计算；问题出在平台分发没有让这个单输出形状进入已有 skinny 路径。

**我的修改**

我核对 `wvSplitK` 的单输出支持后，在 `rocm_unquantized_gemm_impl` 中把条件扩展为 **输出宽度为 1，或原先支持的较宽输出，且 token 行数在支持范围内** ，保留架构、dtype、布局和 K 整除要求。

这里尤其要核对变量含义：局部 m 是输出通道数，n 才是 token 行数，不能把 `m==1` 当作 batch=1。

**设计难点**

我的贡献是 **把模型的真实 shape 接到已有 HIP 能力** ，门控公式、router 和专家选择不变，也没有新写整套 GEMM。后面的追问可以沿 shared expert、未量化 Linear、平台分发和小矩阵并行逐层展开，而不是只说“改了一个 if”。

修改见[ROCm 未量化 GEMM 分发](https://github.com/vllm-project/vllm/blob/922b4da6010bb2d82bb0efe669f45bf48b26ce3a/vllm/model_executor/layers/utils.py#L337-L360)。

先回答[这个 gate 与 router 的关系](#q5)，就能继续解释[为什么它走未量化 Linear](#q6)和[怎样进入平台 GEMM](#q7)。

### 5. shared expert 和 routed expert 有什么区别？这个 gate 是 MoE router 吗？ {#q5}

| 分支 | 如何选择或使用 | 输出如何合并 |
| --- | --- | --- |
| routed expert | router 为每个 token 选择 top-k 专家 | 按路由权重合并所选专家 |
| shared expert | 每个 token 都经过共享 MLP | 本 PR 的 gate 调节共享分支强度 |

这里的 gate 计算为：

`gate = sigmoid(Xw_gᵀ)`

每个 token 得到一个标量，再乘 shared MLP 的输出。**单输出 gate 的优化不改变 top-k 或专家选择** ，只替换同一投影的执行路径。

### 6. 模型已经量化了，为什么你还会优化未量化 GEMM？ {#q6}

量化配置通常会按层选择方法，**一个量化模型可以同时包含量化层和未量化层** 。

- 例如这个 shared-expert gate 的构造使用 `quant_config=None`，会走 FP16/BF16 的未量化路径；其他 Linear 则可能使用 INT4 权重。

- 某些层出于数值、支持范围或模型实现要求被保留为浮点。所以定位时要看这一层实际的 `quant_method` 和 tensor dtype，不能根据模型名称带 GPTQ 或 AWQ，就认定所有乘法都走量化 kernel。

### 7. 模型调用一个 Linear，怎么最终走到 ROCm 的 GEMM？ {#q7}

Linear 的执行路径分成模型表达和平台选择：

1. 模型构造 vLLM Linear，并选择 `quant_method`。
2. forward 通过统一的 `quant_method.apply` 执行。
3. 这个 gate 使用 `ReplicatedLinear` 和 `UnquantizedLinearMethod`，进入平台 GEMM wrapper。
4. `rocm_unquantized_gemm_impl` 按 shape、dtype 和设备能力选择 `wvSplitK` 或其他实现。

**模型层表达线性变换，平台/kernel 层选择执行方式** 。我修改的是 ROCm 分发条件。

### 8. 你说的单输出 GEMM 是 batch size 为 1 吗？GEMM 的 M、N、K 分别是什么？ {#q8}

单输出表示 **输出特征数为 1** 。矩阵形状统一写成：

`X[M,K] × W[N,K]ᵀ → Y[M,N]`

| 维度 | 含义 | 本 PR 的 Python 局部变量 |
| --- | --- | --- |
| M | 当前 forward 的 token 行数 | `n` |
| N | 输出通道数 | `m` |
| K | 输入特征数 | 收缩维 |

所以 `m==1` 指单输出，`0<n<=5` 才限制行数。

普通 Decode 的 M 常接近活跃请求数；Prefill 与投机验证中，一条请求可以贡献多行。**M 不能始终等同于并发数** 。

### 9. 为什么小 GEMM 值得单独优化？Split-K 的收益和代价是什么？ {#q9}

当 token 行数和输出宽度都很小，按 M/N 分块可能产生不了足够多的并行工作，通用实现的 **launch、调度和归约成本也可能占较大比例** 。

- Split-K 类思路把收缩维 K 分给更多执行单元，再合并部分和，以增加并行度；代价可能包括归约、额外中间量或同步，具体取决于实现。

- 不能仅凭 `wvSplitK` 的名字就断言它用了跨 workgroup 原子加。我的做法是看实际 kernel 分工和 trace，再限定分发范围；**单个算子改善能贡献多少模型收益，还取决于它原来占的时间比例** 。

[返回面试内容概览](#overview)

## 三、W4A16 分发：已有 kernel 能力为什么没有被用上 {#pr-52619}

上一项补齐单输出 shape，这一项补齐小 M 的激活工作集范围。先说明原生 kernel 的真实契约，再区分量化存储、GEMM shape 与 LDS 容量，最后走到 custom op 的 launch。

深入分析见[W4A16 形状分发与 LDS 边界](RDNA-W4A16的形状分发与LDS边界.md)。
{.interview-source}

### 10. medium-skinny 这个 PR 改了什么？为什么不是直接扩大 LDS？ {#q10}

**触发问题**

这个 PR 也是分发问题，但约束来自 **小 M 激活的工作集** 。旧 Python 条件要求 `M<=5` 且 `K*M<=32768`；已有 C++ medium kernel 却允许略大的总工作集，前缀留在 LDS、尾部从 global memory 读取。

因此一些 kernel 已支持的 shape 被上层提前排除，转到了另一条执行路径。

**我的修改**

我将 **物理 LDS 容量与 medium 的总支持上限分开** ，让 Python 对齐已有 C++ 契约，medium 上限为 39321 个 16 bit 激活元素，同时保留行数与其他条件。

以 `M=3、K=12288` 为例，36864 个元素超过常规 LDS 工作集，但仍在 medium 支持范围；这不意味着物理 LDS 增大，而是利用已有尾部 global load 的实现。

**设计难点**

难点在于核对上下两层对同一 shape 的解释，包括 Python 与 C++ 变量名、激活工作集和权重大小的区别。这个修改 **补齐已有实现的可达范围** ，不能据此宣称所有小 M 都该使用 HIP，或把其他量化、对齐检查跳过。后面的 W4A16、LDS 和 custom op 问题都从这条分发边界产生。

见[容量定义](https://github.com/vllm-project/vllm/blob/5c1345eaaab03d345c72eecff13da0b2de7b6cd1/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py#L60-L67)与[运行时分发](https://github.com/vllm-project/vllm/blob/5c1345eaaab03d345c72eecff13da0b2de7b6cd1/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py#L450-L490)。
{.interview-source}

这个上限约束的是激活，所以自然要区分[W4A16 存储与计算](#q11)、[运行时 shape 分发](#q12)和[LDS 的物理容量](#q13)。

### 11. W4A16 指什么？它一定使用 INT4 矩阵乘指令吗？ {#q11}

W4A16 描述 **权重按 4 bit 量化存储、激活使用 16 bit 浮点** ，它没有指定唯一的计算指令。

- 该 RDNA 路径从 packed 权重取出量化整数，结合 scale 和零点恢复计算值，再与 FP16/BF16 激活运算；已有 HIP 实现使用相应的浮点 dot 指令并用 FP32 累加。

- 因此要分别说明 **存储位宽、反量化规则、乘法输入精度和累加精度** 。权重以 `int8` 或 `int32` tensor 包装只是 packed buffer 的读取方式，也不意味着模型被改成 INT8 或 INT32 量化。

### 12. 怎么判断一个 GEMM shape 应该走 skinny 还是大矩阵路径？ {#q12}

先看 **实际 M、N、K 与资源工作集** ，再看 dtype、布局和设备能力。

- 这个路径中，小 M 且 `K*M` 在支持范围内进入 HIP skinny，C++ 再选 small 或 medium；其他形状进入 Triton。

- 例如 `M=3、K=12288` 有 36864 个激活元素，超过常规 LDS 工作集，却仍在 medium 范围内；M 超过支持行数则不能因为 K 很小就进入 skinny。

- 分发依据是执行张量，**Prefill、Decode 的业务名称只能帮助理解常见形状，不能替代条件判断** 。

### 13. LDS 是什么？为什么放进去不一定越多越快？ {#q13}

LDS 是 AMD GPU 中 **workgroup 内可共享的片上存储** ，可以让不同线程复用已读取的数据，降低重复 global load。

- 这里缓存的是小批次的激活 `K*M`，权重仍主要从 global memory 读取。LDS 容量、寄存器占用和 workgroup 大小都会限制同时驻留的工作量；tile 放得更多，也可能降低 occupancy 或增加同步成本。

- medium 路径用 LDS 前缀加 global 尾部，是在已有资源限制下扩展工作集，**不能把支持范围更大直接等同于所有形状都更快** 。

### 14. Python 调用 custom op 后谁启动 GPU kernel？fake implementation 又做什么？ {#q14}

| 层次 | 执行位置 | 做什么 |
| --- | --- | --- |
| Python wrapper | CPU | 经注册的 custom op 进入 host 函数 |
| C++ host | CPU | 读 shape、检查参数、分配输出，在当前 PyTorch stream 上 launch |
| `__global__` kernel | GPU | 执行真实乘法与内存访问 |
| fake implementation | tracing / fake tensor | 描述输出 shape、dtype、device，供编译系统推导 |

fake implementation 不执行真实乘法，但输出契约仍须一致。

返回 tensor 对象不代表 GPU 已执行完，前后依赖通常由 stream 顺序维持。核对时要贯穿 **Python 分发、host 契约、GPU 访问与 fake 描述** 。

[返回面试内容概览](#overview)

## 四、GPTQ 加载：先决定消费什么，再决定怎样存放 {#pr-52637}

执行分发正确还不够，kernel 读取的参数必须已经准备好。这个 PR 先修对称路径的零点消费，再优化 packed 布局重排；两项都应放回一次性的加载生命周期。

深入分析见[GPTQ 加载与 INT4 布局](GPTQ权重加载与RDNA-INT4布局重排.md)。
{.interview-source}

### 15. 这个 GPTQ PR 包含哪些工作？问题发生在加载还是推理阶段？ {#q15}

**触发问题**

这个 PR 包含两项工作，都发生在 **checkpoint 加载完成、backend 准备运行时权重** 的阶段。

第一项故障是对称 GPTQ 已用固定偏移表示零点，文件中的 qzeros 却仍被传入不应消费它的路径；第二项成本是 RDNA 为改变 packed 布局，先把整层 INT4 展开成 INT32，再重排打包。

**我的修改**

零点修复放在共享 `AutoGPTQLinearMethod.process_weights_after_loading`：在选定 kernel 处理前，按 `zero_points` 契约删除 layer 的冗余零点 tensor，保留参数名称供 backend 自己使用。

布局优化则先规范 packed 轴、建立连续独立副本，再调用 **已有 `gptq_shuffle` 直接重排 word 内的 nibble** ，最后准备 HIP/Triton 读取的行布局。

**设计难点**

两项修改共同说明 **文件表示不等于运行时表示** 。删除 qzeros 修正的是是否被消费，shuffle 修正的是同一逻辑权重怎样存放；两者都不改变量化数值。原地重排前保护 alias、转换后保持消费者布局，是实现难点。

这主要影响加载与启动，不能从一次性预处理减少就推导每 token Decode 同样加速。

两项修改分别沿[零点数值语义](#q16)、[加载生命周期](#q17)和[packed shuffle](#q19)展开，不能把它们合成一次“转置修复”。

### 16. GPTQ 是什么？对称和非对称量化怎样解释零点？ {#q16}

GPTQ 是一种后训练权重量化方法，利用校准输入估计量化对层输出的影响，并在逐步量化时补偿误差。

- 运行时仍要按 scale 和零点恢复权重，一般写作 `ŵ=s(q-z)`。这个 PR 的对称 unsigned INT4 用固定偏移 8，即 **`ŵ=s(q-8)`** ，所以 backend 不需要额外零点 tensor；非对称路径则可能需要保存分组零点。

- `zero_points=False` 表示 **无需消费显式零点参数** ，不表示数学上没有零点。`q=9、s=0.25` 在这里解码为 0.25。

### 17. vLLM 的权重加载链是什么？为什么 forward 里不做这次转换？ {#q17}

权重转换发生在加载生命周期：

1. **注册参数** ：量化配置为 Linear 选方法，`create_weights` 注册 qweight、scales、qzeros。
2. **写入 checkpoint** ：loader 按参数接口加载数据。
3. **准备运行表示** ：量化方法与选定 kernel 的 `process_weights_after_loading` 转换布局。
4. **执行 forward** ：`apply → apply_weights` 读取已准备好的权重。

权重推理时通常不变，一次性转换适合放在加载边界，避免每步重复。

`PackedvLLMParameter` 还描述输入轴、输出轴和打包轴。**TP 分片与转置不能只凭二维 shape 猜含义** 。

### 18. checkpoint 带了 qzeros，为什么删除它？直接转置修复 shape 不行吗？ {#q18}

文件保存了一个参数，不代表选中的 backend 必须使用它。

- 这个问题中 **对称配置已用固定偏移表达零点** ，继续传入文件 `qzeros` 会使 backend 误走显式零点处理；shape 不匹配只是暴露出来的症状。

- 我在 `AutoGPTQLinearMethod.process_weights_after_loading` 中先按 `kernel.config.zero_points` 判断，再删除 layer 上的冗余参数，随后调用 kernel 后处理。

- `w_zp_name` 仍保留，因为其他 backend 可能依赖它创建自己的 placeholder。这里应 **修正对象是否被消费，再讨论它的布局** 。

见[零点清理和调用顺序](https://github.com/vllm-project/vllm/blob/3e550f77be642335ea8d68262808baee951f764c/vllm/model_executor/layers/quantization/auto_gptq.py#L426-L451)。
{.interview-source}

### 19. packed shuffle 怎样避免解包？会改变权重的数值或量化精度吗？ {#q19}

一个 INT32 word 可以保存八个 INT4 值。

- 原流程把所有值展开为 INT32 后重排，再打包；新流程直接在 packed word 中把 nibble 顺序从 `q0,q1,…,q7` 转为 **`q0,q2,q4,q6,q1,q3,q5,q7`** ，复用已有 `gptq_shuffle`。

- 它改变物理存放顺序，scale 和逻辑权重含义不变，因此是布局转换，不能称为重新量化。关键是 **loader 产生的顺序与 HIP/Triton 的解码顺序一致** ，否则形状和位宽都对，也会算出错误的矩阵乘。

见[RDNA packed 后处理](https://github.com/vllm-project/vllm/blob/3e550f77be642335ea8d68262808baee951f764c/vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py#L576-L604)。
{.interview-source}

### 20. transpose、contiguous、clone 和 view 有什么区别？为什么 shuffle 前要 clone？ {#q20}

| 操作 | 对存储的影响 |
| --- | --- |
| `transpose` | 通常改变 shape/stride，仍共享原存储 |
| `contiguous` | 布局需要时创建连续副本 |
| `clone` | 创建独立副本 |
| dtype `view` | 重新解释同一批字节 |

这里先按 `packed_dim` 规范打包轴，再做 `clone(contiguous_format)`，同时满足原地 shuffle 的布局要求并保护共享存储。

重排后准备 backend 行布局，必要时保留行 stride/padding，HIP 与 Triton 读取同一 packed 权重。**int8/int32 的 view 既不是反量化，也不是复制** 。

[返回面试内容概览](#overview)

## 五、量化 MoE：配置 sentinel 要落到每次 GEMM {#pr-58513}

沿 GPTQ 加载继续看 MoE，一个模型级 group-size 标记会经过参数分配、backend fallback 和两次专家 GEMM。先看专家计算，再说明为何每次 launch 都要按各自 K 解析 -1。

深入分析见[Channelwise GPTQ MoE](Channelwise-GPTQ-MoE的加载与Triton执行.md)。
{.interview-source}

### 21. Channelwise GPTQ MoE 的 bug 是什么？你改了哪两个边界？ {#q21}

**触发问题**

问题由 **Channelwise GPTQ 的 `group_size=-1`** 触发。-1 表示一个输出通道沿整条 K 维共用一组量化参数，旧 MoE fallback 却把它当作普通尺寸或除数。它会跨两个边界：加载时如何分配 scale/zero，执行时每个 K 元素应该使用哪一组参数。

**我的修改**

我在 `MoeWNA16Method` 的参数创建中让 gate/up 和 down 都按一组分配，并在 Triton launch 前把 -1 解析为 **本次输入的实际 K** 。

不能在模型级统一替换成 hidden size，因为 gate/up 的 K 是 hidden H，down 的 K 是 intermediate I；它们分别要解析成 H 和 I。

**设计难点**

因此只修 shape 或只修 kernel 都不完整。加载和执行必须保持同一个量化契约，fallback 也不能因为可以启动就被视为等价。

沿这项 PR 深入时，专家的两次 GEMM、groupwise/channelwise 与 token-expert pair 的索引正好连接起来，体现的是 **模型配置如何落到算子级尺寸和参数选择** 。

见[量化参数分配](https://github.com/vllm-project/vllm/blob/a08f2cb3c54605c8306f42d4cf00b62322bf804f/vllm/model_executor/layers/quantization/moe_wna16.py#L277-L302)与[launch 前解析 group size](https://github.com/vllm-project/vllm/blob/a08f2cb3c54605c8306f42d4cf00b62322bf804f/vllm/model_executor/layers/fused_moe/fused_moe.py#L671-L705)。
{.interview-source}

理解这次修复先看[专家计算的两次投影](#q22)，再看[-1 的含义](#q23)和[每次 launch 的 K](#q24)。

### 22. 一个 MoE 专家里具体算什么？为什么总参数多不等于每 token 计算多？ {#q22}

常见 gated expert 先合并算 gate/up，做 **`SiLU(gate) × up`** ，再由 down 投影回 hidden size，最后按 router 权重合并多个专家输出。

- 模型保存所有专家权重，但每个 token 通常只激活 top-k routed experts，因此总参数规模和单 token 活跃计算量不同。

- 不过执行还取决于 **每个专家实际收到多少 pair、路由是否均衡及对齐 padding** 。把 pair 按专家组织成 grouped GEMM 能提高权重复用，小 batch 或严重倾斜时则可能形成很多很小的专家计算；shared expert 还要另外计入。

### 23. per-tensor、per-channel 和 groupwise 量化有什么区别？-1 为什么不能直接参与计算？ {#q23}

| 量化粒度 | scale/zero 覆盖的范围 |
| --- | --- |
| per-tensor | 整张权重共用一组 |
| per-channel | 每个输出通道各有一组 |
| groupwise | 每个输出通道沿 K 再分成多组 |

这里 `group_size=-1` 表示 **整条 K 一组** ，是配置标记，不是合法的负尺寸或除数。

若本次 GEMM 的 K 为 H，则解析为 G=H，使合法 k 的 `k//G` 全部为 0。更细分组通常提高表达灵活性，也增加 metadata 和执行处理；取舍还受模型/kernel 支持影响。

### 24. 为什么不能把 -1 在加载时一次性改成模型的 hidden size？ {#q24}

两次专家 GEMM 的收缩维不同：

| 投影 | 输入宽度 K | channelwise 的实际 group size |
| --- | --- | --- |
| gate/up | hidden size H | H |
| down | intermediate size I | I |

因此模型级统一把 -1 替成 H，会让 down 错误分组。

正确方式是保留配置语义，在每次 launch 按 `A.size(1)` 解析。**模型字段与算子实际尺寸要区分** ，TP 分片后也按当前 kernel 接口理解 K。

### 25. 量化 MoE 怎么选择 backend？fallback 只要能运行就可以了吗？ {#q25}

量化方法会结合 **权重类型、分组方式、shape、dtype 与设备支持** 选择专家实现。

- 该 PR 关注 AutoGPTQ MoE 在 Marlin 支持检查不满足时进入 WNA16 fallback，再由相应的 Triton experts 执行。

- WNA16 中 N 表示权重量化位数，不是 GEMM 的输出通道。Fallback 必须保留相同的零点、scale、打包轴和路由语义；**能启动只是必要条件** ，如果它把 channelwise 解释成普通负 group size，即使偶然绕过报错也不是等价执行。

### 26. MoE 为什么要把 token 展开为 token-expert pair？第二次 GEMM 为什么传 top_k=1？ {#q26}

一个 token 选到 top-k 专家，就产生 top-k 份 token-expert pair：

| 阶段 | 输入行代表什么 | 读取方式 |
| --- | --- | --- |
| 第一次 GEMM | 原始 token | `pair_id//top_k` 找到输入行 |
| 激活后 | token 在某个专家的中间结果 | tensor 已按 pair 展开 |
| 第二次 GEMM | 每行就是一份 pair | 索引参数传 1，直接读取 |

第二次传 `top_k=1` 改变的是 **输入行索引方式** ，router 的 top-k 没变。最终仍按路由权重合并多个专家输出。

两次调用见[Triton WNA16 experts](https://github.com/vllm-project/vllm/blob/a08f2cb3c54605c8306f42d4cf00b62322bf804f/vllm/model_executor/layers/fused_moe/experts/triton_moe.py#L742-L808)。
{.interview-source}

[返回面试内容概览](#overview)

## 六、滑动窗口：长度转换如何改变 Attention 结果 {#pr-59550}

量化 PR 关注数值如何被解释，Attention PR 关注位置如何被解释。从少一个可见 Key 的边界问题出发，依次讨论逻辑位置、softmax 归一化和缓存回收。

深入分析见[滑动窗口参数与边界](ROCm-Attention滑动窗口的参数与边界.md)。
{.interview-source}

### 27. 滑动窗口 PR 为什么只加了 1，却影响 Attention 结果？ {#q27}

**触发问题**

这个 bug 是 **同一个窗口字段在接口两端含义不同** 。模型配置的 W 包含当前 token，backend 保存的是左跨度 W-1，下游 prefix-prefill helper 需要的却是完整长度 W。旧调用直接传左跨度，使可见窗口少一个 Key，改变了 softmax 的参与集合。

**我的修改**

我在 ROCm wrapper 的调用边界传入 `1+self.sliding_window[0]`。Query 在 7、W=4 时应看 Key 4～7，误传 3 只看 5～7。还要检查 sentinel：无窗口 -1 转成 0，符合 helper 的关闭约定；W=1 转成 1，避免被误判为关闭窗口。

**普通值、最小窗口和关闭状态必须同时满足契约** 。

**设计难点**

改动代码很小，分析重点却在从模型配置追到实际 mask 的位置定义。它不改变 KVCacheManager 的回收，也不是给模型额外加历史；它恢复原本要求的可见集合。由此可以连续追问 causal 与窗口组合、逻辑位置与分页地址，以及 mask 为什么必须影响归一化。

见[ROCm wrapper 的窗口传参](https://github.com/vllm-project/vllm/blob/eb815579d5fab709aa0e2f9fc6be35ef2af59fd1/vllm/v1/attention/backends/rocm_attn.py#L468-L490)。
{.interview-source}

可见集合为什么影响输出，由[softmax 前 mask](#q29)解释；可见范围与实际存储是否一起缩小，则是[KV 回收](#q31)的问题。

### 28. causal mask 与 sliding-window mask 怎样组合？为什么要用逻辑位置？ {#q28}

普通 causal 条件是 `k<=q`，长度 W 的左窗口再要求 **`q-k<W`** ，两者相交得到 `max(0,q-W+1)<=k<=q`。

- q/k 必须是请求内的逻辑序列位置；Chunked Prefill 的局部行号要加上已计算 context，分页缓存的物理 block ID 更不能用来判断窗口。

- `block_table` 解决 **历史位置对应哪块 GPU 存储** ，mask 解决 **这个位置是否可见** 。物理页不连续、被复用，都不应改变模型的 Attention 依赖。

### 29. 为什么 mask 应该在 softmax 前加？先算 attention 再把不可见 Value 置零可以吗？ {#q29}

Softmax 要在 **允许参与 Attention 的 Key 集合** 上归一化。

- 通常把不可见位置的 score 置为负无穷，使其不贡献分母和概率；若先把所有 score 纳入 softmax，再只把 Value 置零，非法 Key 仍分走概率，合法输出被缩小。

- 例如两个 score 相同、只有一个 Key 合法，正确权重是 1；先对两个位置归一化则变成 1/2。需要区分 **数值读取保护** 与 **可见性 mask** ：清理某些不可见 K/V 可以避免 NaN 污染，但不能替代 softmax 前的 mask。

### 30. 分块 Attention 为什么不能把各块 softmax 输出直接取平均？online softmax 保存什么？ {#q30}

各 Key 块的 softmax 分母不同，**单块归一化输出不能直接平均成全局输出** 。

- Online softmax 保存已见 score 的最大值 m、按该最大值缩放的指数和 l，以及同尺度的加权 Value 和 o。

- 加入新块时取新的共同最大值，重缩放旧 l/o 和新块贡献，再合并；最后用 o/l 得到结果。这样可以逐块读取分页 KV，而不保存完整 Attention 矩阵。

- Mask 要在这个统计过程前生效，否则非法 Key 已经改变最大值和分母，后面仅清理输出无法恢复。

### 31. 滑动窗口只看最近 W 个 token，KV cache 就只保存 W 个吗？ {#q31}

需要分开 **Attention 可见范围、KV 已存储范围和物理 block 回收** 。

- Window mask 只规定某层当前 Query 能看哪些 Key，不会自动释放显存。缓存管理还要考虑全局层、其他 cache group、共享引用、请求生命周期和 backend 支持，才能判断哪些旧块可回收。

- 混合局部层与全局层的模型尤其不能给所有层统一删到 W。我的 PR 修改的是窗口传参，没有改 KVCacheManager 的回收策略，所以不能据此宣称缓存容量也被同步缩小。

[返回面试内容概览](#overview)

## 七、Prefix-LM：让模型可见性贯通 metadata 与 KV 读取 {#pr-60040}

窗口修复仍保留普通 causal 关系，Prefix-LM 则要表达同图像内部的双向依赖。先解释模型需要什么，再沿 metadata、遍历和读取保护落实，最后回到窗口、Graph 与缓存复用边界。

深入分析见[多模态 Prefix-LM](多模态Prefix-LM在ROCm-Attention中的实现.md)。
{.interview-source}

### 32. Prefix-LM 这个 PR 做了什么？为什么 Gemma3 不能只用普通 causal attention？ {#q32}

**触发问题**

这项支持针对 **Gemma3 的同图像内部双向 Attention** 。图像 embedding 被放进语言序列，但普通 causal mask 会让前面的图像位置看不到后面的同图像 Key；模型要求允许这类读取，同时未来文本和不同图像之间的关系仍受原规则约束。

**我的修改**

我为 ROCM_ATTN 接入已有 runner 产生的图像范围，由 metadata builder 准备设备 tensor，再经过 Attention forward、wrapper 和 prefix-prefill kernel。

底层复用共享多模态 mask helper，同时 **扩大 Key 遍历，保护 block-table 和 KV 读取，处理不可见缓存的 NaN** 。只改变 mask 不够，因为没有进入旧循环的 Key 根本不会参与计算。

**设计难点**

难点是模型的逻辑可见性必须同时满足实际存储与执行条件：范围使用请求内位置，页表 load 本身就要合法，清理不可见列不能误删任何真实 Query 需要的 K/V。窗口和 clamp 还会影响生成批次能否复用 causal graph。

因此这项贡献贯穿 **模型语义、metadata、循环边界和 GPU 数值读取** ，不是只增加一个布尔条件。

同图像双向规则需要先落成[位置 metadata](#q34)，再通过[调用链](#q35)进入真正的 Key 遍历；[读取边界](#q36)是随后产生的难点。

### 33. 能用几个 token 举例说明 Prefix-LM mask 吗？视觉 encoder 已双向，为什么 decoder 还需要它？ {#q33}

假设位置 0、1 是文本，2、3、4 是同图像 embedding，5 是后续文本。

- Query 2 在普通 causal 下只能看 0～2，Prefix-LM 还允许它看 3、4；Query 1 仍不能看图像或后续文本。

- 无窗口时是 **`causal OR same_image_range`** 。视觉 encoder 的双向 Attention 混合 patch 特征，而 projector 后的 embedding 在 decoder 每层又产生新的 Q/K/V 和文本交互，属于另一处计算。**前一处的双向性不能替代后一处的模型语义** ，实现应以 decoder 的依赖规则为准。

### 34. Attention metadata 里 query_start_loc、seq_lens、block_table 和 slot_mapping 各做什么？ {#q34}

| metadata | 表达什么 | 用途 |
| --- | --- | --- |
| `query_start_loc` | 本轮每条请求的 Query 起止边界 | `[0,2,5]` 表示两条请求分别贡献 2、3 行 |
| `seq_lens` | 本次 Attention 对应的序列长度 | 通常包含历史 context 与本轮有效输入 |
| `block_table` | 逻辑历史页到物理 KV 页的映射 | 读取历史 K/V |
| `slot_mapping` | 本次新 K/V 的写入槽位 | 定位新 KV |

这些对象是索引或长度，**真实 KV 在 GPU 的逐层缓存 tensor 中** 。图像范围另行描述位置可见性，不能替代地址映射或 Query 边界。

### 35. Prefix-LM 的调用链怎么讲？为什么只改 kernel mask 不够？ {#q35}

范围沿下面的路径进入实际 Attention：

1. **Runner** ：从多模态位置生成 `mm_req_doc_ranges`。
2. **Metadata builder** ：`RocmAttentionMetadataBuilder.build` 接收范围，创建设备 tensor。
3. **执行接口** ：经 `RocmAttentionImpl.forward → chunked_prefill_paged_decode → context_attention_fwd` 传入 kernel。
4. **Key 遍历与 mask** ：扩展旧 causal 循环可能遗漏的同图像 Key，再用共享 helper 按真实 Query 精确 mask。

范围是位置 metadata，K/V 仍来自模型 tensor 和 cache。**只放开 mask 无法让未进入循环的 Key 参与计算** ，范围传递、遍历边界和精确 mask 要一起修改。

见[metadata 构造](https://github.com/vllm-project/vllm/blob/309e622d88ff6606e5d71519c8a2326c1f102044/vllm/v1/attention/backends/rocm_attn.py#L239-L260)与[forward 传递](https://github.com/vllm-project/vllm/blob/309e622d88ff6606e5d71519c8a2326c1f102044/vllm/v1/attention/backends/rocm_attn.py#L564-L594)。
{.interview-source}

### 36. 扩大 Key 遍历会遇到哪些内存和数值问题？为什么概率为零还可能传播 NaN？ {#q36}

扩展 tile 可能跨过有效历史，首先要 **在 block-table load 时就屏蔽越界** ，不能等拿到非法页号后才保护 KV load。

- 已分配 cache 也可能残留 NaN，浮点中 `0*NaN` 仍是 NaN，所以 mask 后概率为零不保证 Value 乘法安全。

- 我对整个真实 Query tile 都不可见的缓存列清理 K/V；只要任意真实 Query 需要该列就必须保留，padding Query 也不能参与这个判断。

- 这里要分别保护 **页表地址、KV 读取和后续数值计算** 。

读取边界说明见[prefix-prefill 的分页加载](https://github.com/vllm-project/vllm/blob/309e622d88ff6606e5d71519c8a2326c1f102044/vllm/v1/attention/ops/prefix_prefill.py#L57-L102)。

### 37. Prefix-LM 和 sliding window 怎么组合？生成阶段能直接复用 causal graph 吗？ {#q37}

令 C 为 causal、I 为同图像范围、S 为 `q-k<W`，普通局部多模态规则是 **`(C AND S) OR I`** ，图像项可补回窗口外同图像 Key；开启 clamp 后是 **`(C OR I) AND S`** 。

- 这里 S 是该 helper 的单侧条件，不能擅自换成绝对距离。生成阶段复用 causal graph 的前提是当前范围筛选、窗口和 clamp 条件让两条路径可见性等价，metadata 与编译分支也要匹配。

- **不能只凭“Query 已过图像末尾”就无条件认为可复用** ，局部窗口边界仍可能影响范围判断。

### 38. Prefix-LM 和 prefix caching 有什么区别？Chunked Prefill 把图像切开能自动等价吗？ {#q38}

Prefix-LM 决定 **哪些位置可以互相 Attention** ，prefix caching 决定 **哪些已算 KV 可以复用** 。

- 缓存复用要满足原计算依赖；双向图像范围内前半段的状态可能依赖后半段，不能任意截断后仍假设它与整图计算相同。Chunked Prefill 的 mask 也只能访问已经存在的 Key，不能生成尚未调度的未来图像 KV。

- 因此切分策略和输入准备必须让需要的图像 Key 可用，不能从“mask 放开了”直接推导任意切分都等价。这是模型依赖、调度和缓存有效性共同决定的。

[返回面试内容概览](#overview)

## 八、MRV2 dummy：有效 Query 与执行容量不能混用 {#pr-60049}

Attention metadata 的逻辑长度同样贯穿启动执行。这个问题从真实的越界计数开始，再说明 dummy 为何运行模型、PIECEWISE 如何 padding，以及成本表为何依赖一致的预算语义。

深入分析见[MRV2 填充与启动 Profiling](MRV2的PIECEWISE填充与投机解码启动Profiling.md)。
{.interview-source}

### 39. 这个 PR 为什么在服务启动时出现输入超出编译上限？根因是什么？ {#q39}

**触发问题**

这个问题在 **MRV2 自适应投机验证的启动 dummy profiling** 中出现。

独立 drafter 的 step 0 会为每条请求追加 correction 输入，原逻辑已为它预留预算；但 PIECEWISE dispatch 选择更大的输入 bucket 后，dummy batch 把 buffer 容量重新当成有效 Query 数，预留空间被逻辑计数吞掉。

**我的修改**

复现中预算 1024、8 条请求，有效 target token 先降到 1016；PIECEWISE 填充容量为 1024。旧代码按 1024 重建有效 Query，drafter 再加 8 就成为 1032。

我让 execute_model 保留有效数，再独立传 padded 容量；make_dummy 用前者构造 Query 边界、调度数和 logits 索引，用后者切输入 buffer，并将尾部标为 padding。

**设计难点**

**扩大编译上限不能修复重复计入的语义错误** 。有效行与物理容量本来可以不同，所有 metadata 消费者都应尊重这条边界；FULL dummy 又有自己的 descriptor 布局，不能无条件套 PIECEWISE 规则。

这项工作也连接启动执行、graph bucket 与成本采样，说明构造输入仍需满足真实模型接口。

启动为何执行模型见[dummy run](#q40)，容量为何被填充见[bucket 与有效 Query](#q42)，最后再回到[具体字段修复](#q43)。

### 40. vLLM 为什么启动时要执行 dummy run？它是真实用户请求吗？ {#q40}

Dummy run 使用构造的输入和 metadata，**会执行真实模型路径，但没有真实用户请求** 。

- 它可以用于内存 profiling、算子 JIT/warmup、编译、graph capture，或自适应投机验证的成本采样；不同调用的目标和执行模式不同。

- 因而 dummy metadata 也必须内部一致，不能因输入是假的就随意把 padding 当有效 Query。MoE profiling 还要让有效 dummy 行参与专家执行，才能测到相关工作量，而尾部 padding 仍要正确标记。

### 41. torch.compile 和 CUDA Graph 分别解决什么问题？PIECEWISE 与 FULL 怎样理解？ {#q41}

| 机制 | 主要作用 | 执行约束 |
| --- | --- | --- |
| `torch.compile` | 图捕获、编译与算子融合 | 需要明确动态维和算子接口 |
| CUDA Graph | 复用 GPU 执行序列，减少反复提交 | 需要匹配执行 shape、地址和 metadata |
| PIECEWISE | 按可捕获片段组织执行 | 某些动态部分留在片段外 |
| FULL | 捕获更完整的模型执行 | 对 shape、地址和 metadata 的匹配更严格 |

compile 与 Graph 可以配合。PIECEWISE/FULL 也不是是否 padding 的同义词。

本 PR 的关键是 **descriptor 可以要求更大 buffer，但有效 token 数仍须独立保留** 。

### 42. 为什么要按 bucket padding？可变请求数、Query 长度和固定输入容量怎样共存？ {#q42}

Bucket 让一组实际输入规模复用有限的编译或 graph 执行形状，避免每个长度都准备一个版本。

- 框架把有效输入放在 buffer 中，再用 **Query 边界、序列长度和 padding 标记** 描述实际工作。比如两条请求贡献 2 和 3 个 Query，逻辑边界是 `[0,2,5]`，buffer 可以为某个 bucket 分配 8 行，尾部 3 行不属于请求。

- 某些 FULL descriptor 还约束请求槽位和最大 Query 长度，需要对应 metadata。**固定物理容量与可变逻辑工作量可以共存，前提是所有消费者尊重有效边界** 。

### 43. 你具体修改了 MRV2 哪些代码？哪些字段用有效数，哪些用 padded 数？ {#q43}

我在 `GPUModelRunner.execute_model` 的 dummy 分支保留 PIECEWISE 原始有效数，再独立传 descriptor 容量给 `InputBatch.make_dummy`：

| 使用的计数 | 对应字段/操作 |
| --- | --- |
| 有效 token 数 | `num_tokens`、`num_scheduled_tokens`、`query_start_loc`、logits 索引；dummy context 只填有效 positions |
| padded 容量 | 切 `input_ids`、`positions`、`is_padding` buffer，将尾部标为 padding |

例如有效 6 行、容量 8 行，两条请求各 3 行，逻辑边界仍为 `[0,3,6]`，后两行只是 padding。

FULL dummy 保留符合 capture descriptor 的布局，不能无条件套同一重建规则。

见[dummy 分支](https://github.com/vllm-project/vllm/blob/74adb439e41fe6c78ff223313c80d3d6e0dea43f/vllm/v1/worker/gpu/model_runner.py#L1821-L1864)与[make_dummy](https://github.com/vllm-project/vllm/blob/74adb439e41fe6c78ff223313c80d3d6e0dea43f/vllm/v1/worker/gpu/input_batch.py#L132-L191)。
{.interview-source}

### 44. 自适应投机验证为什么需要启动成本表？它和 target 的接受规则有什么关系？ {#q44}

多验证候选可能让一次 target forward 推进更多 token，也会增加 drafter 与 verification 成本，自适应策略据 **预计接受收益和执行成本** 选择验证宽度。

- 启动采样用来估计不同执行规模的成本，bucket 边界会让成本呈阶梯变化，不能只按有效 token 数线性外推。接受率或置信度估计服务于“这轮试多少”的决策；**候选是否被最终接受仍由 target 的验证与采样算法决定** 。

- 修复 dummy 计数是让成本采样和模型执行尊重同一预算语义，不是修改接受标准。

[返回面试内容概览](#overview)

## 九、ShortConv drafter：恢复的是数值历史，不只是长度 {#pr-59600}

上一项修复执行规模，这一项修复投机后状态。先建立 draft/verify 的时间顺序与 ShortConv 历史的含义，再解释双窗口 checkpoint、稳定请求索引和 chunk 左边界。源码按后续提交 `442f7d9afa`，独立 drafter 入口已命名为 `standalone_ar/speculator.py`；[深入分析](MRV2投机解码中的ShortConv历史恢复.md)保留早期版本路径。

### 45. ShortConv 这个 PR 解决了什么问题？为什么缩回序列长度还不够？ {#q45}

**触发问题**

这项修复针对 **LFM2.5 独立 drafter 的 ShortConv 历史** 。Drafter 提前生成候选时，会消费一些后来被 target 拒绝的输入，固定长度卷积窗口随之被覆盖。

逻辑 token 序列可以截断，缓存长度也可以缩回，但已经覆盖的 u=B*X 数值不会因此自动恢复，后续候选仍可能建立在错误历史上。

**我的修改**

我为每层 ShortConv 分配按稳定 request slot 索引的 checkpoint，在 step 0 metadata 中传入 checkpoint、请求索引和 correction 位置，卷积 kernel 从可信初始窗口重放有效输入。

它同时保存 **correction 前的 checkpoint** 供下一轮重新消费 correction，以及 **correction 后的 runtime state** 供本轮继续 drafting。两份历史处于不同消费边界，不能只留下一个“最新窗口”。

**设计难点**

实现还要处理 batch 重排、correction 靠近 chunk 开头时的 prior history，以及未完成 Chunked Prefill 尚未采样时应追加下一 prompt token。我改的是独立 drafter 的状态与输入恢复路径，不是 target 的接受算法。

核心判断是 **状态必须与实际消费的可信前缀一致** ，而不只是表面序列长度一致。

恢复窗口前先明确[draft/verify 的时间顺序](#q46)和[ShortConv 缓存的数值](#q47)，这两点决定[correction 前后双窗口](#q48)。

### 46. 投机解码的 draft、verify、correction 流程是什么？为什么拒绝后必须恢复内部状态？ {#q46}

投机解码的顺序是：

1. **Draft** ：drafter 自回归提出多个候选。
2. **Verify** ：target 用一次较宽 forward 验证，保留连续接受的前缀。
3. **Correction** ：首个拒绝点由 target 给出修正 token；全部接受时也可能继续产生下一 token。
4. **恢复并继续** ：下一轮沿最终保留序列执行，超前的 drafter KV/递归状态回到对应边界。

Drafter 为预测后续候选已消费早先候选，有些随后会被拒绝。**截断输出、回退长度、恢复数值状态是不同动作** 。

独立 draft model 还有自己的权重与状态，不能用 target hidden states 直接替代。

### 47. ShortConv 到底缓存了什么？它和 Attention KV、Mamba 状态一样吗？ {#q47}

ShortConv 将当前层输入投影为 B、C、X，先算 **`u=B*X`** ，再对各 channel 做 causal depthwise 卷积，用 C 调制结果并输出投影。

- 长度 L 的卷积缓存此前 `L-1` 个 u 值；L=3 就需要两行历史，它们不是 token ID，也不是 Attention K/V。

- Attention 缓存通常按 token 位置增长，ShortConv 窗口则不断覆盖旧值。该实现复用了 **Mamba 的状态存储基础设施** ，不代表 ShortConv 实现了完整 SSM。

- B/C 是学习得到的调制特征，也不应直接称为 sigmoid 概率。

### 48. 为什么要同时保存 correction 前的 checkpoint 和 correction 后的 runtime state？ {#q48}

两份窗口对应 correction 的两个消费边界：

| 状态 | 保存的历史 | 用途 |
| --- | --- | --- |
| checkpoint | 已消费 `[0,p)`，尚未消费位置 p 的 correction | 下一轮 step 0 重放 correction |
| runtime state | 已消费 correction | 本轮继续 drafting |

以卷积长度 3 为例，此前 u 历史为 `[2,5]`，correction 的 u 为 7：

- **checkpoint** 保留 `[2,5]`。
- **runtime** 更新为 `[5,7]`。

只留一个最新窗口，会导致下一轮重复消费 correction，或丢失 correction 前的可信历史。

见[kernel 的双窗口更新](https://github.com/vllm-project/vllm/blob/442f7d9afa5bbe6a2b72b49ec782db00f8744d73/vllm/model_executor/layers/mamba/ops/causal_conv1d.py#L223-L258)。
{.interview-source}

### 49. checkpoint 为什么按稳定 request slot 保存？直接按 batch 行号存不行吗？ {#q49}

Continuous batching 中，请求完成或新增后 batch 会重排。

- 例如本轮 `[A,B]`、下轮 `[B,C]`，B 从第 1 行变成第 0 行，**batch 行号不是请求的稳定身份** 。Checkpoint 按 request state slot 保存，`input_batch.idx_mapping` 把本轮行映射到稳定槽位，才能让 B 取到自己的历史。

- 这里还要区分该槽位与卷积 runtime cache 的物理 state index；它们用途不同，不能直接互换。请求结束、槽位复用时也必须让新请求按初始状态条件处理，避免继承旧历史。

### 50. 这个 PR 的恢复调用链是什么？correction 靠近 chunk 开头时历史从哪里来？ {#q50}

恢复信息沿以下调用链传递：

1. **Speculator** 为 draft ShortConv 层分配 checkpoint。
2. **step 0 metadata** 传 checkpoint、稳定请求索引和 correction 索引。
3. **`ShortConv.forward_cuda`** 将恢复信息交给 `causal_conv1d_fn`。
4. **卷积 kernel** 从可信窗口计算，分别写回 correction 前和后的历史。

若 correction 靠近 chunk 开头，所需历史可能位于 chunk 左侧，应从 prior checkpoint 取，不能读取负下标地址，也不能全部补零。

**只有请求确实没有历史时，才使用初始零状态** 。

见[checkpoint 分配](https://github.com/vllm-project/vllm/blob/442f7d9afa5bbe6a2b72b49ec782db00f8744d73/vllm/v1/worker/gpu/spec_decode/standalone_ar/speculator.py#L98-L117)与[卷积调用的恢复参数](https://github.com/vllm-project/vllm/blob/442f7d9afa5bbe6a2b72b49ec782db00f8744d73/vllm/model_executor/layers/mamba/short_conv.py#L284-L306)。
{.interview-source}

### 51. Chunked Prefill 尚未结束时，为什么不能继续用 last_sampled token？这条恢复路径有哪些执行前提？ {#q51}

未完成的 Chunked Prefill 可能 **还没有产生 target 的新采样结果** ，此时 `last_sampled` 可能是旧值，不能把它当本轮追加输入。

- 该版本按 `num_sampled` 判断：有采样时读本轮 correction，否则读下一枚 prompt token，使 drafter 继续沿真实 prompt 处理。

- 状态恢复还要遵守所选实现的支持条件：该固定提交要求 ShortConv 独立 drafter 使用 **`mamba_cache_mode='none'` 且关闭 prefix caching** ，standalone AR 的 drafter 自身也尚未支持 graph capture。不能把已修复的路径推广成所有缓存和 graph 模式都兼容。

见[追加 token 的选择](https://github.com/vllm-project/vllm/blob/442f7d9afa5bbe6a2b72b49ec782db00f8744d73/vllm/v1/worker/gpu/spec_decode/standalone_ar/speculator.py#L474-L486)。
{.interview-source}

### 52. 投机解码怎样保持 target 的输出语义？随机采样能简单比较两个 token 吗？ {#q52}

Greedy 场景可以按 target 的确定性选择检查候选是否匹配，随机采样则要考虑 **target 与 draft 的条件分布** 。

- 经典 rejection sampling 对候选使用相应接受概率，拒绝时从修正后的剩余分布采样，不能简单地把两个独立随机采样 token 是否相等当作通用接受规则。

- 具体要以所用 speculator 和 sampler 为准。我的 ShortConv PR 处理的是 **状态重放与恢复** ，没有据此改变接受算法；状态错误会让 drafter 的后续分布失去预期，但修复状态也不能代替对采样规则的理解。

[返回面试内容概览](#overview)

</div>
