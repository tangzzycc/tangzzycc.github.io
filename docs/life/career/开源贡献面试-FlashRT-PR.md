---
title: "开源贡献面试：FlashRT PR"
---

# 开源贡献面试：FlashRT PR

围绕简历中的 FlashRT 开源贡献，按 AI Infra 与 GPU Kernel 技术面组织问题。每题的「简要回答」用于约 30～60 秒的初步口述；「完整阐述」展开模型语义、数据流、实现选择和验证依据，并附可能追问。

本文整理于 2026 年 10 月 9 日，依据仓库已有源码笔记和 PR 验证记录。以下两个 PR 在已有记录中均已合并，性能数字分别采用各自报告的配置。

| PR | 贡献内容 | 源码笔记版本 |
|---|---|---|
| [#213](https://github.com/flashrt-project/FlashRT/pull/213) | Pi0.5 的 AMD RDNA 3.5 BF16 后端、共享 Pipeline 与 HIP 算子优化 | `e2cf5d1` |
| [#215](https://github.com/flashrt-project/FlashRT/pull/215) | Temporal K/V 跨帧复用、decoder-only 路径与缓存生命周期 | `839b159` |

## 1. 你在 FlashRT 做了什么？和直接用 OpenPI 或 torch.compile 相比，价值在哪里？

### 简要回答

我为 Pi0.5 增加了 gfx1151 上的 BF16 推理后端，把输入和 checkpoint 适配、共享模型 Pipeline、硬件算子 provider 分开，实现了归一化、激活、RoPE、Attention 和 small-M WMMA GEMM 等 HIP 路径。随后增加 Temporal KV，让中间帧复用最近一次 Prefix K/V，只重跑动作 Decoder。两相机、10 步去噪的记录中，完整后端相对 torch.compile 为约 1.62 倍加速；Temporal KV 在另一组配置中约为 1.51 倍，但它允许视觉上下文滞后。

### 完整阐述

第一条 PR #213 解决的是 Pi0.5 在 AMD RDNA 3.5 上的完整 BF16 推理与性能问题。我建立了输入 schema 和 checkpoint 接入路径，预分配模型工作区，再把计算映射到适合该设备的算子。原生实现覆盖 activation、residual、normalization、QKV/RoPE、Prefix Encoder Attention、Action Decoder GQA 和 small-M GEMM，通用矩阵乘仍可使用 hipBLASLt。

架构上，Frontend 负责输入格式、prompt/state、权重转换与输出反归一化；`Pi05TorchPipeline` 持有唯一的 Vision、Prefix Encoder、Action Decoder 遍历与去噪循环；RDNA providers 决定具体操作用 HIP、hipBLASLt 还是 PyTorch 执行。这让硬件优化围绕稳定的模型语义进行。

优化包括提前合并 QKV 与 gate/up 权重、折叠静态 RMSNorm 参数、固定宽度归一化、QKV Split 与 RoPE 融合、适合短 Query 的 Decoder Attention，以及小形状 GEMM 的 WMMA 实现。相应的数值舍入边界、mask 和布局都需要与参考路径对齐。

第二条 PR #215 在完整后端基础上增加跨帧 Prefix K/V 复用。每次完整推理生成 18 层 Prefix K/V，随后若干帧跳过 Vision 与 Prefix Encoder，但仍以新的 noise 执行完整动作去噪。这改变了视觉条件刷新频率，默认 `cache_frames=1` 维持每帧完整计算。

两条 PR 的性能基线需要分别介绍：

| 实验 | 配置 | 记录中的中位延迟 |
|---|---|---|
| #213 完整推理 | Pi0.5、两相机、10 步去噪、gfx1151 | PyTorch eager 288.85 ms；torch.compile 212.03 ms；FlashRT BF16 130.52 ms |
| #215 Temporal KV | 两相机、47-token prompt、horizon 15、10 步去噪、BF16、gfx1151 | cache_frames=1 为 132.27 ms；cache_frames=2 为 87.84 ms |

第一项相对 torch.compile 约快 1.62 倍，第二项相对每帧刷新约快 1.51 倍。它们来自不同实验口径，不能直接把倍数相乘当成一个新实测结论。

**可能追问：这是否说明手写 HIP 总比 torch.compile 快？**

结果只支持已测设备和 Pi0.5 配置。收益来自固定 shape、提前准备权重、模型生命周期优化和特定算子的组合；其他模型、shape 或硬件要重新测量。通用大 GEMM 仍然适合库实现，手写 kernel 应由具体瓶颈决定。

背景见[FlashRT RDNA 3.5 笔记索引](../../gpu-kernel/flashrt-rdna35/index.md)。

## 2. Pi0.5 怎么从图像和指令产生动作？它和语言模型逐 token Decode 有什么区别？

### 简要回答

Pi0.5 先用视觉编码器处理多相机图像，把图像特征和文本 embedding 组成 Prefix，Prefix Encoder 为每层生成 K/V。Action Expert 从连续随机 noise 出发，结合时间条件和 Prefix K/V，多步预测速度并更新整个动作 horizon。动作 token 在同一个去噪块内可以互相读取，所有动作位置联合更新，因此不能把这里的 Decoder 等同于语言模型一次生成一个 token。

### 完整阐述

以已有源码笔记中的两相机、47-token prompt、horizon 10、10 步去噪配置为例，两张 `[224,224,3]` 图像经 patch size 14 的视觉编码，分别产生 256 个视觉 token。SigLIP 的 residual width 为 1152，经 projector 映射到 Prefix 的 2048 维；文本从 embedding table 查出后按模型定义缩放，再与图像特征拼接。

```text
两相机图像 → Vision → 512 个图像 embedding
文本及格式化 state → 47 个文本 embedding
                 ↓
559-token Prefix → 18 层 Prefix Encoder → 每层的 Prefix K/V
                 ↓
新 noise [10,32] + 时间条件 → 18 层 Action Expert
                 ↓
速度预测 → Euler 更新 → 下一去噪步 → 最终动作
```

Prefix Encoder 的 hidden width 是 2048，Action Expert 是 1024，但二者的 Q/K head dimension 均为 256，且采用 8 个 Query head、1 个 KV head。Encoder 把 Prefix hidden 投影成 `[559,1,256]` K/V，Decoder 把动作 hidden 投影成 `[10,8,256]` Q 与 `[10,1,256]` Suffix K/V。Attention 在共同的 256 维空间交互，再通过 Decoder O projection 回到 1024 维。

所以 Decoder 读取的是对应层的 Prefix K/V，并不是把 Encoder 最终 hidden 简单拷贝进 Action Expert。源码中的最后一层 Encoder 写完 K/V 后就停止后续 Attention/FFN，是因为已经没有下一层 Prefix hidden 的消费者，而对应层的动作 Decoder 仍需要这份 K/V。

动作生成采用 flow matching 的迭代推理。从 noise 开始，模型在时间 t 预测速度 v，使用负步长更新：

$$
x_{t+\Delta t}=x_t+\Delta t\,v(x_t,t),\qquad
\Delta t=-1/\mathrm{num\_steps}.
$$

两步时应使用 t=1、0.5，而不能截取十步时间表的前两项 1、0.9。时间 embedding 和 AdaRMS modulation 必须匹配步数。

Prefix 可以内部双向交互，Action Suffix 的所有 horizon 位置属于同一个 Attention block，可以读取 Prefix 和当前 Suffix。它与逐 token 自回归生成的 causal mask 不同。每个去噪步会重写 Suffix K/V，Prefix K/V 在同一次完整请求的去噪循环中保持复用。

**可能追问：robot state 是额外的动作输入 token 吗？**

在这里核对的 Pi0.5 路径中，state 被格式化到语言 prompt，时间和 noise 则进入动作路径。因此 state 变化可能改变 Prefix 内容，需要更新 prompt 并重新生成相应 K/V，不能只检查用户指令字符串是否相同。

模型结构与尺寸见[Pi0.5 推理流程笔记](../../gpu-kernel/flashrt-rdna35/02-Pi05模型结构与推理流程.md)。

## 3. 为什么要共享一份 Pipeline？checkpoint 转换只是权重改名吗？

### 简要回答

共享 Pipeline 让所有 backend 使用同一套模型遍历、mask 和状态更新，硬件 provider 只替换具体算子，减少模型语义分叉。Checkpoint 转换也会提前决定运行时计算：合并 QKV 和 gate/up、规范矩阵布局、重排 Q/K 以适应 RoPE，以及把静态 RMSNorm 的 1+w 折叠到后续 Linear。折叠要在 FP32 中做，Decoder 的动态 AdaRMS 参数则不能按静态权重折叠。

### 完整阐述

如果为每个硬件复制整套 forward，去噪步数、最后一层 Encoder 的提前停止、Prefix/Suffix mask、modulation 或数值修复都可能只更新其中一份。共享 `Pi05TorchPipeline` 将这些模型责任集中起来，RDNA binding 只注入 ops、gemm、attn providers。

从公开接口到执行的路径是 `load_model`、`VLAModel.predict`、RDNA Frontend `infer`，再进入 Pipeline。Frontend 锁定受支持的设备、精度和输入 profile；Pipeline 负责工作区与计算生命周期；providers 检查单个算子的 shape/layout，选择原生实现或 fallback。

Checkpoint 转换包含提前合并、布局准备和数学参数折叠。例如 Vision 的 Q/K/V 权重合成一份，运行时一次 GEMM 产生 fused QKV；gate/up 也可以合成一份 merged workspace。Q/K 的权重按 RoPE pair 布局重排，使 kernel 可以连续读取配对元素。不同格式中的矩阵轴则被规范为 provider 消费的 row-major 布局。

Encoder 的 Gemma RMSNorm 带静态参数：

$$
y=\mathrm{rms}(x)\odot(1+w).
$$

若后面紧接 row-vector Linear `yW`，可以构造 `W' = diag(1+w)W`，运行时只执行无权 RMSNorm 再乘 W'。折叠乘的是 Linear 的输入轴，不能凭矩阵外形随意选轴。

这一转换先读取 FP32 权重、完成 `1+w` 与乘法，再转成 BF16。如果先把接近 -1 的 w 舍入成 BF16 的 -1，再计算 1+w，该通道可能被错误清零。代数等价也要考虑低精度转换顺序。

Action Decoder 使用随去噪时间变化的 AdaRMS scale/shift/gate，不能当静态参数折叠到一份权重里。对于已知离散时间表，可以提前计算每步 modulation，但每一步仍需要使用自己的条件值。Frontend 还预乘 action output projection 的步长，以把速度投影与 Euler residual 接起来。

**可能追问：fallback 会破坏共享模型语义吗？**

Fallback 发生在单个算子的 provider 内，例如不满足 small-M 条件时使用 hipBLASLt、不满足原生 Attention 条件时使用 SDPA。Pipeline 继续执行同一层和同一去噪步骤。如果 fused Linear-residual 不接受某个 shape，可以先 Linear 再 residual add，无需另写 Decoder loop。

转换与职责划分见[Frontend 与 Pipeline 笔记](../../gpu-kernel/flashrt-rdna35/01-Pi05-Frontend与Pipeline.md)。

## 4. 归一化和 Residual 融合为什么能省时间？为什么不能随意去掉 BF16 舍入？

### 简要回答

固定宽度 Norm 可以把输入留在寄存器，避免归约后重新读 global memory；Residual 与 Norm 融合还减少一次 launch 和中间结果重读。归约采用 FP32，每个 wave 先得局部结果，再经 LDS 合并。但原路径若先把 residual sum 写成 BF16，Norm 应使用这份已舍入值，融合也要保留这个数值边界。优化的是执行和访存，不能仅凭实数公式相同就改变参考计算。

### 完整阐述

Norm 的并行单位是一行。一个 workgroup 拥有整行，每个线程处理部分 channel，先产生 sum 或 square-sum。LayerNorm 需要均值和中心化方差，RMSNorm 只需要平方和，AdaRMS 则在 RMSNorm 后加入动态 scale/shift。

归约分两级：wave32 内用 shuffle 合并线程 partial；各 wave 发布 partial 到 LDS，通过 block barrier 让消费者看见完整结果，再合成整行统计量。跨 wave 的数据不能只靠 wave shuffle 传播。

Generic Norm 的输入宽度不固定，通常在求统计量后重新读输入。针对 Pi0.5 的热点宽度，fixed-width specialization 把每个线程负责的 BF16 片段保存在寄存器数组中，归约后直接生成输出。这样少读 global memory，代价是 VGPR 压力和可能的 occupancy 变化，因此只给已测宽度使用专门路径。

Residual-RMSNorm 的融合产生两个输出：更新后的 residual stream 和归一化结果。原路径先得到 BF16 sum，再由下一算子读取它，因此融合要先执行：

$$
s=\mathrm{BF16}(\mathrm{FP32}(r)+\mathrm{FP32}(u)),
$$

再用 `FP32(s)` 计算平方和和 Norm。Fixed-width kernel 将同一个已舍入 s 写入 residual 输出，并保留在寄存器供当前 Norm 使用，减少单独 residual launch 和中间 tensor 的重读。

同样的原则也出现在 small-M Linear-residual 的 epilogue：如果普通 Linear 先输出 BF16，再与 residual 相加，融合应先把 projection 舍入到 BF16，再做相加；QKV/RoPE 中也有参考路径要求的乘法舍入边界。某个 GeGLU helper 在 FP32 中保留乘法前结果，则需要按它自身的参考契约判断。不存在适用于所有融合算子的统一「只在最后转 BF16」规则。

**可能追问：寄存器保留整行是否总比重读快？**

不一定。更多寄存器可能减少并发 wave，甚至发生 spill。需要同时观察寄存器占用、访存和 kernel 时间。固定 width 使布局和寄存器数组可编译期展开，但是否采用仍要由测量决定。

实现见[Normalization 笔记](../../gpu-kernel/flashrt-rdna35/03-Normalization-Kernel.md)与[Activation/Residual 笔记](../../gpu-kernel/flashrt-rdna35/02-Activation与Residual-Kernel.md)。

## 5. Encoder Attention 和 Decoder Attention 为什么要用不同的并行组织？

### 简要回答

Encoder 有数百个 Query，适合按 Query tile 并行并流式读取 KV；一个 wave pair 负责 16 行 Query，leader 做 QK/softmax，两波分别生成输出的前后 128 维。Decoder Query 只有动作 horizon 那么多，但要读长 Prefix，所以每个 workgroup 负责一个 Query/head，八个 wave 分摊 Key 序列，再合并各自 online-softmax 的 max、sum 和加权分子。不能把各分区已经归一化的输出直接平均。

### 完整阐述

Prefix Encoder 的 Q 与 KV 长度相近，shape 为 `[S,Hq,256]` Q 和 `[S,1,256]` K/V。基本协作单位是两条 wave，负责同一 Query head 的 16 行 Query：leader wave 用 WMMA 计算 QK 并更新 online-softmax，两个 wave 分别负责 P×V 输出的维度 `[0,128)` 和 `[128,256)`。

Workgroup 根据 template 放置多个 Query tile 或多个 head，共享 V 的 staging。LDS 在 producer 写入与 consumer 读取之间有 barrier，下一轮覆盖前也要保证当前消费者已用完。整个路径不把完整 S×S score matrix 写到 global memory。

Action Decoder 则是短 Query、长 KV。典型 horizon 为 10 或 15，K/V 包括数百个 Prefix token 和当前 Suffix。每个 workgroup 输出一个 `(Query row, Query head)` 的 256 维结果，八条 wave 读取同一 Q，但各自遍历不相交的 Key。

每个 lane 持有连续八维 Q/V，32 lane 合成完整 256 维 dot product。对一个 Key，wave 归约出统一 score，每个 lane 用相同概率更新自己负责的八维输出。因此可以沿 KV 并行，而无需为整个 score vector 分配 global workspace。

每个 wave 保存本地最大分数 m、相对该最大值的指数和 l，以及 Value 加权分子 a。分区合并时先统一指数基准：

$$
m=\max_i m_i,
$$

$$
l=\sum_i l_i\exp(m_i-m),\qquad
a=\sum_i a_i\exp(m_i-m),\qquad O=a/l.
$$

本实现使用 exp2 与相应 log2 缩放。Partial 状态经 LDS 和 barrier 发布，由 leader wave 合并；不需要跨 workgroup atomic。

举一个所有 score 为零的例子：分区 A 有两个 Value 1、3，分区 B 只有一个 Value 10。正确输出是 `(1+3+10)/3=14/3`。两个分区分别归一化得到 2 和 10，再平均是 6，丢失了分区之间的概率质量。必须合并分母和未归一化分子。

**可能追问：8 个 Query head 共享 KV，能否共享同一份 softmax？**

不能。GQA 共享的是 K/V 表示，每个 Query head 有不同 Q，其 score、概率和输出仍然独立。部分 KV 搬运可以复用，但数学上不能把不同 head 的 attention 分布合并。

波间职责与实现见[Encoder Attention 笔记](../../gpu-kernel/flashrt-rdna35/05-Encoder-FlashAttention-Kernel.md)和[Decoder GQA 笔记](../../gpu-kernel/flashrt-rdna35/06-Decoder-GQA-Kernel.md)。

## 6. Small-M WMMA GEMM 怎么优化动作投影？Split-K 是否把整个 GPU 填满了？

### 简要回答

动作投影典型 shape 是 M=10、K=1024、N=32，输出 tile 很少。我用已有准备好的转置权重，让每个 workgroup 的多个 wave 计算同一输出 tile 的不同 K 段，再在 LDS 合并 FP32 partial。四波时每波处理 K=256，两个输出 tile 共八条有效 wave。它增加的是组内 K 并行，不能说已经填满所有 CU；更宽的 N 则可以采用共享 A tile、不同 wave 负责不同输出列的路径。

### 完整阐述

Small-M 路径计算 `D[M,N]=A[M,K]W_nt[N,K]^T`。持久 `W_nt` 使每个输出列的 K 方向数据连续，便于 BF16 WMMA fragment load；直接使用 `[K,N]` 权重，同一列的 K 元素会隔着 stride N。

Wave32 WMMA 原子是 `16×16×16`，输入 BF16、累加 FP32。Direct Split-K 路径一个 workgroup 拥有一个 `16×16` 输出 tile，多条 wave 各自处理 K 的不同区间。

对 M=10、N=32、K=1024、四波配置，grid 只有两个输出列 tile。每个 tile 内的四条 wave 分别处理 K 区间 `[0,256)`、`[256,512)`、`[512,768)`、`[768,1024)`，各执行 16 次 WMMA。它们把 FP32 accumulator 写到 LDS，barrier 后 wave 0 合并，再完成 bias/residual epilogue。

四波 partial 的 LDS 载荷是 `4×32×8×4=4096` 字节。归约在同一 workgroup 内完成，不需要 global split-K workspace 或 atomic。相较每个 tile 一波，有效 wave 数从二增加到八，但 grid 仍只有两个 workgroup，不能把这描述成在全设备 CU 之间均匀分配一个投影。

对于更宽的 N，另一套 tiled kernel 让每个 wave 负责不同的 16 列，workgroup 协作将 A/W 的 K tile 搬入 LDS。A 在多个输出列 tile 之间共享，减少重复 load；LDS 行 padding 改变 bank 映射，producer/consumer 两个 barrier 保护 tile 的发布与重用。

这两套组织由 shape policy 选择。Small-M 路径只接受已准备转置权重、M≤48、N/K 满足对齐等契约，具体 template 还受 K 与 N 范围约束。常规大 GEMM 使用 hipBLASLt。所谓「宽 N 走 tiled」也要看分发顺序，前面的 direct 条件可能先接受该 shape。

**可能追问：M=10，小于 WMMA 的 16，越界 lane 能提前退出吗？**

不能让需要参与 WMMA 的 lane 缺席。Direct 路径用合法行 clamp 保证读取，tiled 路径为越界行填零，最后由真实 row predicate 限制写回。读取地址保护与输出有效性是两个判断。

Fragment、LDS 和 dispatch 见[Small-M GEMM 笔记](../../gpu-kernel/flashrt-rdna35/07-SmallM-WMMA-GEMM-Kernel.md)。

## 7. valid_prefix 和 prefix_capacity 为什么都要保留？支持 Graph 就等于默认用 Graph 吗？

### 简要回答

valid_prefix 表示有效上下文长度，prefix_capacity 表示物理预留容量，两者在 padded 布局中不同。两视角、47-token prompt 的有效 Prefix 是 559，但最大 prompt 200 时容量是 712，Suffix 可写在 712 后面，中间空洞必须在 softmax 前屏蔽。Graph 则记录固定操作和地址，数据可通过原地址更新，prompt 长度变化可能要求重新捕获。实现支持 Graph，不代表公开 infer 默认一定启用它。

### 完整阐述

Pipeline 在构造时预分配图像、Encoder、Decoder、noise 和逐层 K/V。假设两视角各 256 token，最大 prompt 长度 200，动作 horizon 10，则 Prefix capacity 是 712，K/V buffer 的序列容量是 722。

实际 prompt 为 47 token 时，`valid_prefix=559`。Compact 模式只计算 559 个 Encoder 行，Suffix 写入 `[559,569)`；padded 模式保持 712 行，Suffix 写入 `[712,722)`，中间 `[559,712)` 为无效空洞。

```text
compact: [0,559) Prefix | [559,569) Suffix
padded : [0,559) Prefix | [559,712) masked gap | [712,722) Suffix
```

Decoder mask 使用有效 Prefix 边界与 Suffix 物理起点，在真实 KV 长度范围内判断 `j<valid_prefix OR j>=suffix_start`。无效空洞必须在 softmax 前排除；算完包括空洞的分母后再清零输出并不能修复概率分布。

这些 buffer 也是 graph 的稳定地址。图像内容和 noise 可以在 replay 前复制进原地址，不需要因数值变化重新捕获。Prompt length 影响 Encoder row count、Suffix 起点或 launch 参数，因此 graph plan 按它建立；变化时重新 warmup/capture。

Graph 捕获的是执行操作和指针关系，通常不是把输入 tensor 的旧数值冻结。缓存有效性与 graph plan 有效性也不同：Prefix 内容更新后，只要形状和地址契约相同，原 plan 可以继续执行并读取新值。

#213 的固定版本实现了 `use_graph=True`，但普通 frontend `infer` 没有据此默认走 full graph。#215 的 decoder-only graph 也是可选路径，记录中没有测到明显性能改善，因此默认关闭。性能描述必须说明实际启用方式，不能把支持能力当成默认执行证据。

**可能追问：PyTorch 里为什么还有 torch.cuda API？**

ROCm PyTorch 继续使用 `torch.cuda` 的 API 命名，但实际执行后端是 HIP。设备、build target、wave size 和扩展 backend 仍应通过真实属性核对，不能从 Python 命名推断它使用 NVIDIA CUDA。

工作区与 graph 行为见[Pipeline 笔记](../../gpu-kernel/flashrt-rdna35/01-Pi05-Frontend与Pipeline.md)。

## 8. Temporal KV 到底缓存了什么？跨帧复用是否与每帧重新计算完全等价？

### 简要回答

缓存的是最近一次完整帧生成的 18 层 Prefix K/V，包括图像和 prompt 条件。复用帧跳过 Vision/Encoder，不消费当前图像，但仍从新 noise 运行完整动作去噪，并重写每层 Suffix K/V。因此相同条件和 noise 可以验证 decoder-only 等价；图像变化时会使用旧视觉上下文，属于显式近似。cache_frames=1 每帧刷新，cache_frames=3 的成功调用序列是 Full、Decode-only、Decode-only，再 Full。

### 完整阐述

普通 full forward 先通过图像和 prompt 生成逐层 Prefix K/V，再用该条件运行动作 Decoder。Temporal reuse 帧直接读取最近一次 full 的 Prefix K/V，省掉 Vision 与 Prefix Encoder。

保留的对象是每层的 K/V，而不是上一帧动作。Noise 每次准备新的值，Action Expert 的全部层与所有去噪步骤重新运行；Suffix K/V 也随本轮动作 hidden 在每个步骤重新写入。

Frontend 用刷新周期、成功帧计数和 cache valid 状态决定路径：

```text
周期关闭或到刷新点或 cache 无效 → Full
其余成功周期位置且 cache 有效   → Decode-only
```

以周期 3 为例，调用前成功计数 0、1、2、3、4、5 对应 F、D、D、F、D、D。Prompt/state 更新会重置周期并失效缓存，下一帧强制 full。State 通过语言 prompt 进入 Prefix，因此应用若每帧更新 state，也会每帧 full。

这项近似的边界需要具体讲：如果相机画面变化而本帧仍处于 D，当前图像不会进入条件。变化到下一次 F 才影响动作。相同 cache 与 fixed noise 下，复用帧应重现最近一次 full 的对应 Decoder 结果；这验证的是缓存路径的执行语义，不能证明真实轨迹中的动作质量没有变化。

记录中的 Temporal 性能实验为 gfx1151、两相机、47-token prompt、horizon 15、10 步去噪、BF16。每帧刷新中位延迟为 132.27 ms，周期 2 为 87.84 ms，报告约 1.51 倍加速。

做一个理想化成本估计，若完整帧耗时 F，decoder-only 耗时 D，周期 c 稳态平均成本约为 `[F+(c-1)D]/c`。这是模型估计，不能直接把已报告的中位延迟代入后反推实测 D，也不能忽略刷新帧与复用帧不同的尾延迟。c 增大时成本趋向 D，视觉上下文也可能更旧。

**可能追问：怎么选择刷新周期？**

已有实现使用明确的固定周期。部署时可以按轨迹评测可接受的上下文滞后与延迟，必要时设计基于变化量的刷新策略，但后者属于进一步工作，不能说本 PR 已实现。默认周期 1 保持原完整推理。

缓存对象与周期见[Temporal KV 笔记](../../gpu-kernel/flashrt-rdna35/06-Temporal-KV跨帧复用.md)。

## 9. 完整刷新失败后，为什么必须失效 KV？为什么保留已经捕获的 Graph？

### 简要回答

Full 刷新可能只覆盖部分层的 K/V，不能因为 buffer 中还有数据就认为缓存有效。Pipeline 在开始刷新时撤销有效标记，完整执行成功后才提交；frontend 还覆盖同步、CPU copy 和反归一化失败，只在成功返回后推进帧计数。异常后下次强制 full。Graph plan 描述执行结构，缓存 valid 描述当前内容能否使用，失效内容不必同时销毁形状仍匹配的 graph。

### 完整阐述

K/V 是持久 buffer，刷新过程会逐层覆盖它。若 Encoder 中途失败，可能出现部分层新值、部分层旧值；如果 Decoder、GPU synchronize 或输出后处理失败，也不能把整个调用当成已成功完成的缓存周期。

因此 cache valid 是一个提交状态。Pipeline 使用保存的 prompt length 作为有效性标记：None 表示无可复用缓存，成功 full 后的长度表示当前 Prefix 内容已经建立。流程是先 invalidate，检查输入并运行，再提交；异常则 invalidate 并继续抛出。

Frontend 还包住 Pipeline 之外的失败点，包括输入准备、同步、CPU 拷贝、动作反归一化和结果构造。它只在整个 infer 成功之后增加 frame count；任一失败都撤销缓存并重置周期。

一个具体故障是：切换 prompt 后 full 刷新失败，但计数提前推进。重试时若恰好落到复用位置，就可能读取旧 prompt 的 K/V。入口失效、成功提交与计数延后共同阻止了这条错误路径。

Invalidation 本身只撤销标记，不必清零全部 GPU 字节。公开 frontend 有当前 images/prompt 可以重建，因此下次选 full；低层 `forward_decode_only` 没有这些输入，cache 无效时应明确报错，不能静默猜测条件。

Graph plan 则可以保留。如果新 context 的 prompt 长度不变，full 把新 K/V 写入原地址，旧 decoder-only graph replay 会读取新值。若长度变化，则需要重新 capture。这里的失败处理没有承诺恢复一份旧 cache，只是阻止继续使用未提交内容。

**可能追问：怎么测试 graph 没有偷偷读取旧上下文？**

覆盖两种变化：state 改变但 token 长度相同，full refresh 后复用同一 graph；prompt 长度改变，刷新后重新 capture。两种情况下均用 fixed noise，将 decoder-only graph 与新的 full forward 比较。缓存 invalidation 还单独检查 plan 保留，并测试 partial writes、周期刷新失败及后处理失败后的 full retry。最终 review 回归记录为 gfx1151 上 169 passed、151 skipped。

生命周期与异常路径见[Temporal KV 笔记](../../gpu-kernel/flashrt-rdna35/06-Temporal-KV跨帧复用.md)。

## 10. 你如何证明完整后端与 OpenPI 对齐？优化路径对比自己的 fallback 足够吗？

### 简要回答

我分三层验证：native kernel 对数值参考，共享 Pipeline 内优化 provider 对 ATen/SDPA fallback，再用官方 OpenPI 独立生成的固定输入 NPZ fixture 对比完整输出。后两者的作用不同，共享 fallback 可能与优化路径一起带着模型遍历错误，所以还要外部参考。对照固定 checkpoint、图像、prompt/state、noise 和去噪步数，同时检查 raw actions 与反归一化后的公开 actions，并记录容差、skip 条件和性能配置。

### 完整阐述

Kernel 测试首先检查局部问题：激活和归一化的数值、矩阵布局、RoPE 配对、Attention mask、小 M/tail 和原生实现的 shape 边界。随后，在共享 Pipeline 中切换 optimized provider 与 ATen/SDPA，能定位某个硬件实现偏离。

但两种 provider 共享同一个 Prefix/Decoder loop、时间表和状态结构。如果 shared loop 本身有错误，两边仍然可能相同，因此不能只依赖这一层回归。

独立参考使用官方 OpenPI 生成 NPZ fixture。记录固定了 producer revision、checkpoint SHA-256、两幅确定性相机输入、prompt、归一化八维 DROID state、BF16 可表示的初始 noise 和十步去噪。相同初始 noise 是必要的，否则随机动作差异会掩盖实现差异。

对比包含两份输出：归一化 raw model actions `[15,32]` 和经过 OpenPI quantile 反归一化后的公开 DROID actions `[15,8]`。后者能够检查动作 schema、有效维度和后处理，不会把模型内部 32 通道都当成真实机器人动作。

已报告的独立 fixture 使用 cosine≥0.999、atol=0.1、rtol=0.03，并通过相应对照。这说明该确定性案例在报告容差内对齐，并非 BF16 路径 bit-exact，也不能代替真实机器人轨迹的任务成功率评测。

完整 backend 的适用测试结果为 153 passed、151 skipped，skip 包括其他硬件、可选 backend 或 checkpoint gate。Temporal follow-up 的最终 AMD 回归为 169 passed、151 skipped。需要同时说明执行了什么与哪些路径未适用，不能把 skip 写成已经验证。

性能测量使用相同模型与输入配置，分别比较 eager、torch.compile 和 FlashRT，明确是否包括预处理、拷贝或 graph。记录中的 130.52 ms 是完整 BF16 后端的测量；Temporal KV 的 87.84 ms 属于另一条周期复用实验，解释它时还需要说明上下文近似。

**可能追问：为什么不直接从统计量猜 action_dim，省去配置？**

某个恒定通道可能是真实关节，也可能是 padding，仅凭 quantile 无法确认。Frontend 根据显式 action schema 或 checkpoint 配置解析输出维度，并用 raw/public 两层输出验证，避免把格式猜测混入数值后处理。

参考入口见[Frontend 与验证笔记](../../gpu-kernel/flashrt-rdna35/01-Pi05-Frontend与Pipeline.md)。
