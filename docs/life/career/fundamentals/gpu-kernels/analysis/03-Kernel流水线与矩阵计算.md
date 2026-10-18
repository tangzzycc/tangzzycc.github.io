---
title: "Kernel 优化：双缓冲、Async Copy、Split-K 与 MMA"
---

# Kernel 优化：双缓冲、Async Copy、Split-K 与 MMA

<div class="study-notes study-analysis" markdown="1">

来源：提供的「小鹏汽车二面」「腾讯 AI Infra 部门二面」「英特尔校招面经」「阿里平头哥一面」；对应模块总文件[第 3～4、8～12 题](../index.md#q16)。

## 1. 双缓冲的执行顺序 {#double-buffer}

- <strong class="study-key">填充与轮换。</strong> 先加载 tile 0；计算 tile i 时预取 tile i+1 到另一槽位，再交换读写角色。物理地址固定，交换的是角色。
- **两种依赖。** 异步写入完成后才能读取；所有消费者读完后才能覆盖。复制完成等待与线程间交接需按协作范围建立。
- **资源成本。** 更多 Shared Memory 或寄存器可能减少驻留；尾块还需掩码、补零和对齐保护。

两个槽位用 `slot=i mod 2` 轮换。Tile 0 装入槽 0 后，先提交 tile 1 到槽 1，再计算槽 0；准备覆盖槽 0 为 tile 2 前，必须确认 tile 0 的所有消费者已结束。槽位的简单状态可写为 `空闲→搬运中→可读→使用中→空闲`，每次轮换还要区分当前是哪一代 tile，避免把上一次完成事件当作本次完成。

!!! tip "双缓冲要守住两条依赖"
    **数据写完才能读取；所有读取结束才能覆盖。** Async copy 的完成等待负责前一条，消费者完成与槽位复用协议负责后一条。不同 CTA、warp 或异步矩阵引擎之间，所需同步方式并不相同。

返回面试题：[08](../index.md#q11)
{.study-source}

## 2. 隐藏延迟的上界 {#pipeline-latency}

- **串行。** n 个 tile，每块搬运时间 T、计算时间 C，忽略其他成本时总时间约为 `n(T+C)`。
- **理想流水。** 双缓冲约为 `T + (n-1)max(T,C) + C`；首块填充、最后排空无法被后续工作覆盖。
- **实际条件。** 需有独立计算、硬件重叠能力与足够 tile；同步、带宽竞争和资源占用会缩小收益。缓冲不会创造额外物理带宽。

取 n=4、T=3、C=5，串行是 32 个时间单位；理想流水线是 `3+3×5+5=23`。第一块必须先加载，随后每块的搬运都被上一块的计算覆盖；最后一块计算后没有后续工作可继续隐藏。若 T=8、C=2，流水线仍要 `8+3×8+2=34`，访存主导没有被双缓冲消除。

流水线隐藏的是部分等待，不减少必须搬运的字节。增加 stage 可让预取更早，但当带宽已经饱和，更多 stage 不能突破带宽下界，还可能因占用 shared 降低驻留。用时间线和总时间确认重叠，而不是用缓冲数量判断优化成立。

返回面试题：[09](../index.md#q12)
{.study-source}

## 3. Async Copy {#async-copy}

- **设备侧搬运。** 支持的 Ampere `cp.async` 路径可将 global 数据复制到 shared，避免传统的显式寄存器中转；它不是主机到 GPU 的 `cudaMemcpyAsync`。
- **延迟等待。** 提交后推进独立工作，消费前再等待对应复制完成，并建立所需线程同步；立即等待通常失去重叠机会。
- **架构约束。** 复制宽度、对齐、完成协议和具体指令依架构而变；`cuda::memcpy_async` 是否生成预期路径也要确认。
- **后续架构。** TMA、WGMMA、TMEM 和 `tcgen05` 的角色与异步协议见 [Ampere、Hopper 与 Blackwell](06-Ampere-Hopper与Blackwell.md)；数据加载、矩阵发射和结果消费的协作范围不同。

传统 global→shared 的一种路径是 global load 到线程寄存器，再 shared store；`cp.async` 可直接启动指定地址间搬运，让线程在完成前继续做独立工作。Commit/wait 是跟踪异步搬运组的协议，是否需要额外线程间 barrier，还取决于谁提交、谁读取和参与范围。

边界 tile 需正确处理越界和补零；向量宽度与对齐也必须符合指令约束。如果线程 A 完成等待，而线程 B 的数据仍在途中，A 的等待不能自动成为整个 block 的完成证明。Hopper TMA 与异步 proxy 使用不同协议，不能直接复制 Ampere 的等待语句。

### commit_group 与 wait_group N 到底在等待什么？

Ampere PTX 的普通 `cp.async.commit_group` 把 **当前线程** 尚未 commit 的 copy 归为一组；它不表示整 CTA 的所有 copy 已经完成。`cp.async.wait_group N` 等到较早的组完成，只允许最新的至多 N 个组仍在途；N 是保留的未完成组数，不是要等待的组编号。`wait_group 0` 才等待当前线程先前已提交的全部组。

例：按顺序提交 G0、G1、G2，`wait_group 1` 保证 G0/G1 已完成，G2 可继续在途；此时消费 G0/G1 对应 tile，同时让 G2 搬运。若三个组被错误地写到同一槽位，等待不能修复覆盖冲突；提交顺序、缓冲槽号和消费者完成必须一致。

**每线程 copy 完成协议与 CTA 线程间交接是两层同步。普通 barrier 不能替代异步完成等待，单个线程的等待也不能证明其他线程的 copy 已就绪。**
{.study-caution}

### 双缓冲怎样落到一次实际 K 循环？

下面是协作范围一致时的概念时序，省略架构相关的指令封装与边界处理：

```text
提交 tile 0 到槽 0；commit
等待 tile 0 完成；建立 CTA 可见性
for i = 0 .. num_tiles-1:
    若存在下一块：提交 tile i+1 到另一槽；commit
    计算当前槽 tile i，累加到当前输出
    确认所有消费者不再读当前槽
    若存在下一块：等待对应 copy 完成；建立消费者可见性
```

循环头的预取与本轮计算有独立输入，因此可重叠；循环尾的等待只暴露未被计算覆盖的部分。最后排空时没有下一块，不能按稳态保留一个尚未完成组的规则盲目读取。普通同步 MMA 与异步 WGMMA 对“计算已结束”的判据不同，后者还需等待矩阵引擎不再消费槽位。

资料：[PTX cp.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)、[PTX wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group)。

返回面试题：[10](../index.md#q13)
{.study-source}

## 4. Split-K 的并行来源 {#split-k}

- **统一 shape。** `A[M,K] × B[K,N] → C[M,N]`；将 K 拆成 S 段，形成 S 份同一输出 tile 的部分和。
- **适用条件。** 若 M/N 输出 tile 数不足，跨 block 切 K 可扩展 grid；例如 4 个输出 tile 分 8 段，最多形成 32 个部分和任务。
- **范围区别。** block 内线程或 warp 分担 K，只增加组内并行；跨 block 才增加 grid。K 已很短或输出任务充足时，切分可能只增加成本。

### Split-K 与 Sliced-K 怎样改变线程工作？

跨 CTA 的 parallel Split-K 让多个 CTA 计算同一输出区域，需要 workspace 或 atomic 汇总；CTA 内 Sliced-K 让不同 warp 计算不同 K 段，在 block 内归约，grid 的输出 tile 数不变。两者都沿归约维划工作，但解决的并行不足范围不同。

选 split 数时可以先估计未切分输出任务 C、目标 SM 数 G 以及每 SM 适合的驻留 CTA r；若 C 远小于 Gr，可把 split 作为候选。但 `C×split≥Gr` 只是任务供给条件，每段 K 变短后 prologue/epilogue 占比增长，矩阵指令对齐与归约又增加成本，不能把满足该不等式当成最优配置。

### Split-KV 与 Linear 的 Split-K 为什么不能混用？

Linear 的 K 部分积是可直接相加的矩阵部分和；Attention 沿历史序列切分后，每段有独立 max、分母及加权输出，要按共同最大值重缩放再合并。把局部归一化 Attention 输出直接求和会改变语义。它们都增加工作份数，但合并状态与元数据不同，见[FlashAttention 的状态合并](10-FlashAttention与GroupedGEMM.md#online-softmax)。

返回面试题：[11](../index.md#q14)
{.study-source}

## 5. Split-K 的合并成本与语义 {#split-k-epilogue}

- **合并方式。** workspace 保存部分和后归约，或用适合的 atomic；前者增加存储、读写和 launch，后者可能有竞争与顺序不确定性。
- **Epilogue。** `C = αΣ partial + βC_old` 中，旧输出和 bias 不能在各分段重复加入；非线性激活通常要在完整归约后执行。
- **数值与效率。** 分段改变浮点求和顺序；过多分段还降低单任务 K 工作量，需同时对照容差与总时间。

若 split 数为 S，全部部分和以 FP32 存在 workspace，理想空间为 `4SMN` bytes；部分和写入一次、归约读一次，新增主流量约 `8SMN` bytes，再加最终输出和其他访问。M=N=128、S=8 时仅部分和空间为 512 KiB，写读约 1 MiB。K 计算得到更多并行的收益，必须能覆盖这份归约成本。

例如完整输出是 `ReLU(P₁+P₂+b)`，不能改成 `ReLU(P₁+b)+ReLU(P₂+b)`；取 P₁=−2、P₂=3、b=0，前者为 1，后者为 3。线性缩放可以按数学条件分配，非线性 epilogue 必须等完整部分和到齐。
{.study-caution}

返回面试题：[12](../index.md#q15)
{.study-source}

## 6. Tensor Core 与 MMA {#mma-fragments}

- **数学操作。** MMA 执行 tile 级 `D=A×B+C`；完整 GEMM 仍沿 M/N 分 tile、沿 K 反复累加。
- **线程协作。** 常见 warp 级指令把输入与输出 fragment 分布到各 lane，映射由指令约定；WMMA 是一种编程接口，不能当作每个 lane 独立的矩阵运算。
- **供数与精度。** 核对指令 shape、输入/累加 dtype、布局和参与线程；通过复用与流水线持续供数，不能只替换标量 FMA 调用。

`m16n8k16` 这样的逻辑指令形状表示计算一个 16×8 输出片段，归约 K=16。它对应 `16×8×16=2048` 次 FMA；按一次乘法加一次加法计 2 FLOPs 的口径，为 4096 FLOPs。完整 GEMM 需要多个 warp tile 和多个 K 步才能覆盖输出。具体哪些 A/B/C 元素落到哪一个 lane 的哪些寄存器，由指令布局约定决定。

不能用普通 row-major 指针直接代替 fragment。供数路径必须完成 global 的合并搬运、shared 布局与矩阵加载，epilogue 还要把分布式累加器重新映射到合适的输出地址。矩阵引擎速度很高时，布局转换、寄存器依赖和供数常成为新的瓶颈。

### 一个 lane 究竟保存了输出矩阵的哪几个值？

以 PTX `mma.sync` 的 `m16n8k16`、FP32 累加器布局为具体例子，32 个 lane 共同保存 16×8=128 个输出，每 lane 持有 4 个 FP32。设 `g=lane>>2`、`t=lane mod 4`，该 lane 的 c₀、c₁ 位于行 g，c₂、c₃ 位于行 g+8；列分别是 2t、2t+1。

例如 lane 0 保存 `(0,0),(0,1),(8,0),(8,1)`，lane 5 保存 `(1,2),(1,3),(9,2),(9,3)`。一个 lane 的四个值并不是矩阵内连续的四个元素，直接按线程 ID 顺序写出会打乱结果。A/B 的 fragment 另有布局，FP16 operand 还可能以两个半精度打包在一个 32-bit 寄存器中。

这份映射只适用于指定指令与类型；WMMA 抽象接口的 fragment 内部映射不应通过这个例子假定。`ldmatrix` 帮助协作加载适合矩阵指令的 shared 数据，但它不是任意 row-major 张量都能自动转换的万能加载。

资料：[PTX mma.m16n8k16 fragment](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float)。

返回面试题：[03](../index.md#q16)
{.study-source}

## 7. 从输出 Tile 到 CTA、Warp 与 SM {#cta-warp-sm}

以普通 dense warp-MMA GEMM 为例，一个 CTA（thread block）计算 `BM×BN` 输出 tile，warp 分担其中的子 tile，各 lane 按矩阵指令提供 fragment 并保存分布式累加器。沿 K 的循环不断加载 `A[BM,BK]`、`B[BK,BN]`，累计同一输出，最后执行 epilogue；不是把每个 K tile 当成独立完整输出。

例如 BM=BN=64、BK=32、K=4096，共有 128 次 K tile 迭代。FP16 双 stage 的理想 A/B shared 载荷为 `2×(64×32+32×64)×2=16384 bytes`，还未计 padding 与其他状态；`64×64` FP32 输出累加器逻辑载荷也为 16 KiB，但实际每线程寄存器分配要看 fragment 和编译结果。

Grid 输出任务数约为 `ceil(M/BM)×ceil(N/BN)`，寄存器、shared、线程和硬件上限决定每个 SM 可驻留多少 CTA。普通 launch 无法保证某个 block ID 固定分配到某 SM，正确性不能依赖 block 执行先后；任务 swizzle 是索引策略，不是 SM 绑定。

Prefill 的 token 行数较大时通常已有较多输出任务；小 batch Decode 的 M 小可能更值得评估 Split-K，但不能按阶段直接决定开关，仍需看 N/K、本地 TP shape 与归约成本。Decode Attention 的 Split-KV 与这里 Linear GEMM 的 Split-K 应分别说明。

相关笔记：[CUDA GEMM 的双缓冲与异步搬运](../../../../../ai-infra/CUDA矩阵乘法优化：从Naive到cuBLAS.md)。矩阵流水线参考：[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)。

返回面试题：[04](../index.md#q47)
{.study-source}

</div>
