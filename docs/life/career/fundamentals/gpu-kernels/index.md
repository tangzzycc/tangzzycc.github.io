---
title: "GPU与算子优化：面试问题与示例回答"
study_guide: true
---

# GPU与算子优化

<div class="study-notes" markdown="1">

先掌握 GEMM、低精度执行与访存布局，再用 Profiler 解释优化为什么有效。这条顺序对应简历中的 FlashRT/FlyDSL、LoRA 算子和 GPTQ/AWQ 工作。

64 道题 · 8 个主题 · 答案展开 · 每题附原理分析。
{.study-meta}

前五组是第一轮复习重点；Attention 与归约接着练，稀疏和架构细节按岗位补充。
{.study-intro}

## 一、GEMM：分块、供数与计算映射 {#gemm}

### 01. GEMM 优化为什么要沿 Roofline、Tiling、数据搬运和计算映射逐步分析？ {#q42 .study-question}

先固定 shape、dtype 和目标计算路径，用 <strong class="study-key">Roofline 与基线 profile</strong> 判断瓶颈；再选 M/N/K tile 和复用方式，安排合并搬运、shared 布局及流水线，最后匹配线程/warp 与矩阵指令 fragment。Tile、寄存器、Shared Memory 和并行度互相制约，需迭代调整。

每步用 **正确性、实际流量与计时** 验证，不能只换一个更大 tile 或指令就推断整体加速。

[原理分析：GEMM 优化是相互制约的迭代](analysis/11-GEMM缓存与Reduction.md#gemm-workflow)
{.study-analysis-link}

### 02. Register、Shared Memory、L2、HBM 分别有什么作用？ {#q6 .study-question}

Register 保存线程的局部值、操作数和累加器；Shared Memory 是 **block 内显式管理的片上共享空间**，用于数据复用和线程通信。L2 是设备侧共享的硬件缓存，自动缓存经过相应路径的数据；HBM 是承载权重、输入输出等数据的片外内存。它们的容量、延迟和管理方式不同，优化时要 **减少片外流量并平衡片上资源占用**。

设备内存也可能使用 GDDR，不是所有 CUDA GPU 都有 HBM。

[原理分析：四类存储的职责](analysis/02-CUDA存储与访存.md#memory-hierarchy)
{.study-analysis-link}

### 03. Tensor Core / MMA 是怎样完成矩阵乘的？ {#q16 .study-question}

Tensor Core 是专门的矩阵计算硬件，MMA 指令对小矩阵 tile 执行 <strong class="study-key">D = A × B + C</strong>。线程协作提供分布在寄存器中的 operand fragment，并保存各自的输出片段；常见 warp 级 MMA 由整个 warp 配合完成，不是每个线程独立算完整矩阵。

优化要匹配 **指令形状、dtype、fragment 布局和供数流水线**；部分新架构还提供不同协作范围的矩阵指令。

[原理分析：Tensor Core 与 MMA](analysis/03-Kernel流水线与矩阵计算.md#mma-fragments)
{.study-analysis-link}

### 04. GEMM 中 CTA、Warp 和 K 维遍历怎样对应，Block 如何分配到 SM？ {#q47 .study-question}

一个 CTA 通常负责 <strong class="study-key">一个输出 tile</strong>，多个 warp 再分担其子 tile，各线程持有 operand 和累加器片段，沿 K 分块加载并累计同一输出。CTA 就是 thread block，硬件按资源条件将 block 调度到 SM，一个 SM 可驻留多个 block，但普通 kernel **不能保证 block ID 对应固定 SM 或执行顺序**。

Tile 和 stage 增大可能提升复用，也会消耗寄存器/shared、降低驻留或 grid 并行度。

[原理分析：从输出 Tile 到 CTA、Warp 与 SM](analysis/03-Kernel流水线与矩阵计算.md#cta-warp-sm)
{.study-analysis-link}

### 05. 什么是 Memory Coalescing？ {#q7 .study-question}

Memory Coalescing 指硬件将同一 warp 中线程的内存访问按地址合并为尽量少的事务。典型做法是让 **相邻 lane 访问相邻且适当对齐的地址**，提高每次事务中有效数据的比例。

它看的是 warp 的整体地址分布，不只是单线程是否连续；向量化可以减少指令数，但 **不能自动修复离散的跨线程访问**。

[原理分析：合并访存看 warp 的地址](analysis/02-CUDA存储与访存.md#coalescing)
{.study-analysis-link}

### 06. 非连续访存为什么会降低带宽利用率？ {#q8 .study-question}

如果同一 warp 的访问分散在许多内存区段，就要发出更多事务，每次搬来的数据却只使用一小部分，导致 **实际搬运字节远多于有效字节**。带宽被无用数据和额外事务消耗，访问延迟也可能增加。关键是看地址分布、对齐和 cache 复用；非连续访问若仍落在少量事务内，未必损失很大。

[原理分析：为什么跨步访问浪费带宽？](analysis/02-CUDA存储与访存.md#strided-access)
{.study-analysis-link}

### 07. GEMM 的 BM、BN、BK 和 Stage 数应该怎样权衡？ {#q54 .study-question}

BM/BN 增大通常提高 <strong class="study-key">输入复用</strong>，但输出累加器约随 BM×BN 增长，并减少输出任务数；BK 增大减少 K 循环次数，也增大每 stage 输入和尾部开销。更多 stage 可能改善重叠，却增加 shared、寄存器和同步需求。

小 tile 未必更慢，大 tile 也未必更快，应联合比较 **供数、驻留、并行任务和尾部效率**，按真实 shape 选参数。

[原理分析：BM/BN/BK 与 Stage 是联合参数](analysis/14-GEMM切分与Persistent调度.md#tile-resources)
{.study-analysis-link}

继续深入：[Kernel 优化](analysis/03-Kernel流水线与矩阵计算.md) · [GEMM、缓存与 Reduction](analysis/11-GEMM缓存与Reduction.md) · [GEMM 切分与 Persistent 调度](analysis/14-GEMM切分与Persistent调度.md)
{.study-source}

## 二、流水线与工作划分 {#pipeline}

### 08. Double Buffering 是怎么做的？ {#q11 .study-question}

准备两组 tile buffer，计算当前 buffer 时，把下一块数据预取到另一组，完成交接后交换读写角色。核心是 <strong class="study-key">计算当前块、搬运下一块</strong>，而不是在两个 buffer 之间复制数据。

必须保证 **写完才能读、读完才能覆盖**，用异步完成等待与线程同步维护依赖；代价是更多 Shared Memory、寄存器和同步开销。

[原理分析：双缓冲的执行顺序](analysis/03-Kernel流水线与矩阵计算.md#double-buffer)
{.study-analysis-link}

### 09. Double Buffering 为什么能隐藏访存延迟？ {#q12 .study-question}

它让下一块数据搬运提前发生，使当前计算覆盖一部分搬运等待。流水线稳定后，理想每块时间可从 **搬运时间加计算时间** 接近 **两者的较大值**，但首块加载和最后计算仍无法完全隐藏。

收益要求存在独立工作且硬件能重叠执行；两组 buffer 本身不保证并行，也不会提高物理内存带宽。

[原理分析：隐藏延迟的上界](analysis/03-Kernel流水线与矩阵计算.md#pipeline-latency)
{.study-analysis-link}

### 10. Async Copy 有什么作用？ {#q13 .study-question}

在支持的 CUDA 路径上，Async Copy 可以异步提交 global 到 shared 的搬运，例如 Ampere 的 `cp.async` 能避免传统的显式寄存器中转。提交后先做独立计算，消费数据前再等待，有助于 **重叠搬运与计算、减少寄存器压力**。

但 **复制完成等待与线程间同步仍要正确建立**；提交后立刻等待可能仍是串行执行，具体能力取决于架构和指令。

[原理分析：Async Copy](analysis/03-Kernel流水线与矩阵计算.md#async-copy)
{.study-analysis-link}

### 11. Split-K 是什么，什么时候有效？ {#q14 .study-question}

Split-K 将 GEMM 的 K 维分成多段，各自计算同一输出 tile 的部分和，再合并，是 <strong class="study-key">可选的并行策略</strong>。跨 block Split-K 能在 M、N 较小、K 较大且输出任务不足时增加并行度，但每段仍需足够工作摊薄成本；普通 GEMM 沿 K 循环并不等于 Split-K。

组内切 K 不增加 grid，是否有效要比较 <strong class="study-risk">新增并行度与归约成本</strong>，不能只看 K 大就启用。

[原理分析：Split-K 的并行来源](analysis/03-Kernel流水线与矩阵计算.md#split-k)
{.study-analysis-link}

### 12. Split-K 会引入哪些额外开销？ {#q15 .study-question}

多段部分和需要 workspace 和额外读写、归约 kernel，或通过 atomic 合并，可能增加 **launch、内存流量、原子竞争和同步成本**。切得过细还会降低每段计算效率，改变浮点求和顺序。

Bias、残差和激活等 epilogue 必须放在正确的合并阶段，**非线性激活不能对每个部分和分别执行后再相加**。

[原理分析：Split-K 的合并成本与语义](analysis/03-Kernel流水线与矩阵计算.md#split-k-epilogue)
{.study-analysis-link}

### 13. Split-M、Split-N 与 Split-K 有什么区别，通用 GEMM 怎样选？ {#q52 .study-question}

M/N 分块负责 **互不重叠的输出区域**，一般无需跨块归约；Split-K 则协作同一输出，必须合并部分和。选择时先算输出 tile 数、每 tile 资源和尾部浪费：任务不足可调整 M/N tile，K 足够长时再评估 Split-K。

矩阵小也可能受并行不足或 launch 限制，**不能固定套用‘K 大切 K、M/N 大切 M/N、小矩阵不拆’**，应按 shape 和 backend 做性能分发。

[原理分析：普通 K 循环与 Split-K 不同](analysis/14-GEMM切分与Persistent调度.md#split-dimensions)
{.study-analysis-link}

### 14. Persistent Kernel 是什么，怎样用于 GEMM？ {#q53 .study-question}

Persistent GEMM 用 **数量受控的 CTA 循环处理多个输出 tile**，通过静态分配或设备任务调度摊薄 block 初始化和流水线成本，也可帮助 grouped 工作分配。普通 tiled GEMM 本来就能一次 launch 完成许多 tile，因此 persistent 不自动减少 launch 次数；长期服务任务的 persistent 模式才可能合并多次提交。

代价包括 **调度、资源占用、尾部和并发竞争**，不保证固定 CTA 绑定某个 SM。

[原理分析：Persistent GEMM 是循环分配输出任务](analysis/14-GEMM切分与Persistent调度.md#persistent-gemm)
{.study-analysis-link}

继续深入：[Kernel 优化](analysis/03-Kernel流水线与矩阵计算.md) · [GEMM 切分与 Persistent 调度](analysis/14-GEMM切分与Persistent调度.md)
{.study-source}

## 三、量化与低精度执行 {#quantization}

### 15. INT8、INT4 量化主要解决什么问题？ {#q1 .study-question}

量化用更少的 bit 表示权重或激活，主要降低 <strong class="study-key">存储容量和数据搬运成本</strong>；有合适硬件与 kernel 时，还能利用低精度计算加速。比如权重量化通常对小 batch Decode 更有吸引力。代价是量化误差，以及 scale 读取、解包或反量化开销。

**低 bit 不等于一定更快**，要分别验证模型质量、实际执行路径和端到端性能。

[原理分析：量化主要节省什么？](analysis/01-量化与稀疏.md#quantization-benefit)
{.study-analysis-link}

### 16. 权重量化与激活量化有什么区别？ {#q3 .study-question}

权重量化处理模型参数，权重通常能离线校准、量化和打包，主要减少 <strong class="study-key">权重容量与读取成本</strong>。激活随输入变化，激活量化需要处理运行时范围和离群值，并控制转换与 scale 计算开销。

像 **W4A16 是权重 4 bit、激活 16 bit**，常在计算 tile 内反量化后使用浮点矩阵运算；W8A8 则可能使用低精度整数乘法，具体取决于 backend。

[原理分析：权重量化与激活量化](analysis/01-量化与稀疏.md#weight-activation)
{.study-analysis-link}

### 17. INT4 相比 INT8 有什么收益与代价？ {#q2 .study-question}

相同参数数量下，INT4 的 **纯权重载荷约为 INT8 的一半**，更省内存，也可能减少权重读取时间；实际还要加 scale、零点和对齐开销。代价是可表示级别更少，**量化误差通常更难控制**，还可能增加解包、反量化和布局处理成本。

因此应在质量要求下比较，不能直接认为 INT4 会比 INT8 快两倍。

[原理分析：INT4 相比 INT8 的取舍](analysis/01-量化与稀疏.md#int4-tradeoff)
{.study-analysis-link}

### 18. Scale 和 Group Size 怎样影响量化精度与性能？ {#q4 .study-question}

Scale 决定量化值到实际数值的映射，选取不合适会导致 <strong class="study-key">截断或分辨率不足</strong>。Group Size 决定多少元素共用一组量化参数：较小分组通常更能适应局部范围，但会增加 **scale 元数据、读取与索引成本**；较大分组更省元数据，却更容易受离群值影响。

精度和性能还取决于分组轴、校准方法及 kernel 支持，不能只按分组大小判断。

[原理分析：Scale 与 Group Size](analysis/01-量化与稀疏.md#scale-group)
{.study-analysis-link}

### 19. 反量化 Kernel 应该独立执行还是融合进 GEMM？ {#q50 .study-question}

独立反量化会增加 <strong class="study-key">中间高精度张量写回和再读取</strong>，但可以复用结果或调用成熟 dense GEMM；融合可在 tile 内解包、缩放后直接计算，节省流量和 launch，却增加寄存器、指令与供数复杂度。沿 K 的 group scale、zero-point 修正和稀疏解码都要放在正确数学位置。

应按 **形状、复用次数、实际计算路径和总成本** 比较，融合并非必然更快。

[原理分析：反量化独立执行与融合的取舍](analysis/07-W4A4与稀疏元数据.md#fused-dequantization)
{.study-analysis-link}

### 20. W2A4、W4A4、W8A8 在不同端侧芯片上应该怎样验证？ {#q49 .study-question}

先明确 **编码、scale/zero-point、累加精度与稀疏格式**，检查目标芯片和 backend 的真实指令路径；W2 存储可能先解包到更高精度，并不证明有原生 W2 乘法。对照原始模型与量化参考分别验证 kernel 正确性和任务质量，再测代表性 Prefill/Decode shape、总流量、TTFT/TPOT、功耗和持续温度。

核心是 **区分压缩收益、执行加速与误差来源**，不能由 Orin/Thor 名称直接推断支持或速度。

[原理分析：跨端侧平台验证 W2A4/W4A4/W8A8](analysis/07-W4A4与稀疏元数据.md#platform-validation)
{.study-analysis-link}

### 21. W4A4 Kernel 需要明确哪些量化与 MMA 约定，M16N8K64 表示什么？ {#q31 .study-question}

W4A4 只说明 **权重和激活都以 4 bit 表示**，还要说明整数或浮点编码、signedness、scale/zero-point、打包与累加精度。`m16n8k64` 表示一次 MMA 的逻辑 M=16、N=8、K=64，不是线程数；需看完整指令判断稠密或稀疏。

比如 `mma.sp...s32.s4.s4.s32` 带稀疏元数据并用 INT32 累加，**shape 或 WMMA 名字本身不能确定整个计算路径**。

[原理分析：M16N8K64 只给出逻辑运算形状](analysis/07-W4A4与稀疏元数据.md#mma-shape) · [W4A4 的计算约定](analysis/07-W4A4与稀疏元数据.md#quantization-contract)
{.study-analysis-link}

### 22. FP4、FP6 与 INT4 有什么区别，低 bit 是否一定更快？ {#q29 .study-question}

INT4 使用整数编码，FP4/FP6 将 bit 分配给 **符号、指数和尾数**，可表示集合与动态范围不同；微缩放格式还结合共享的 block scale。它们都要处理量化误差，但不能互换解码，也不能只按 bit 数判断精度。

是否加速取决于 **原生指令、scale 开销、打包布局和矩阵形状**；权重存成 4 bit 不代表乘法已经走 FP4 或 INT4 Tensor Core。

[原理分析：Scale Factor 如何参与计算？](analysis/06-Ampere-Hopper与Blackwell.md#block-scaling)
{.study-analysis-link}

### 23. 如何计算量化 GEMM 的 Sustained TOPS，为什么不能给出固定值？ {#q55 .study-question}

稠密 GEMM 常按乘加算两次操作，用 **`2MNK/(耗时秒×10¹²)`** 算 TOPS，需说明计时是否包含解包、scale、归约和 launch。稀疏可报告原稠密工作量的等效 TOPS或实际非零操作率，两者不能混用。

持续性能还受卡型号、功耗频率、shape、编码和实际 MMA 路径影响，**W4A4 或 Ampere 名称不足以确定数值**；图片里的约 7 TOPS 只属于原测试口径。

[原理分析：Sustained TOPS 的计量口径](analysis/14-GEMM切分与Persistent调度.md#sustained-tops)
{.study-analysis-link}

继续深入：[量化与稀疏](analysis/01-量化与稀疏.md) · [W4A4 与稀疏元数据](analysis/07-W4A4与稀疏元数据.md)
{.study-source}

## 四、访存布局与片上资源 {#memory-layout}

### 24. 什么是 Shared Memory Bank Conflict？ {#q9 .study-question}

Shared Memory 被划分成多个 bank。一个 warp 的访问在同一服务事务中，若多个线程请求 **同一 bank 的不同地址**，访问可能被拆成多次服务，降低吞吐。这与 global memory 的合并访存是两个问题。

多个线程 **读取同一个地址通常可以广播**，不能把所有“同 bank”都当作冲突；具体还要看访问宽度和架构。

[原理分析：Bank Conflict 的条件](analysis/02-CUDA存储与访存.md#bank-conflict)
{.study-analysis-link}

### 25. 如何避免 Shared Memory Bank Conflict？ {#q10 .study-question}

先根据每个 lane 的地址推导 bank 映射，再用 <strong class="study-key">padding、swizzle 或线程映射调整</strong>，把同时访问的数据分散到不同 bank。比如常见 32-bank、4-byte 模型下，FP32 的 `tile[32][32]` 按列读取容易冲突，改成 `tile[32][33]` 能改变行跨度。

布局也要兼顾 global 搬运、矩阵指令和资源占用，最后用 **实际指令与 profiler 指标** 验证。

[原理分析：Padding 与 Swizzle](analysis/02-CUDA存储与访存.md#padding-swizzle)
{.study-analysis-link}

### 26. XOR Swizzle 怎样避免 Bank Conflict，读取时是否要恢复原布局？ {#q51 .study-question}

在简化的 FP32 `32×32` tile 上，把逻辑 `[r,c]` 存到物理 `[r,c xor r]`，固定 c 的列读取可分散到 **不同的 32 个 bank**。消费者按同一映射读取，就得到逻辑元素，通常不需把整个 tile 再搬回原布局；XOR 映射可逆不代表任何指令都自动兼容。

要按 **真实访问宽度、事务与 MMA 布局** 推导，padding 或 XOR 都不是通用消冲突公式。

[原理分析：XOR Swizzle 的完整映射例子](analysis/02-CUDA存储与访存.md#xor-layout)
{.study-analysis-link}

### 27. Occupancy 是什么？ {#q18 .study-question}

Occupancy 是 **SM 上驻留 warp 数相对于硬件最大驻留 warp 数的比例**。它受线程数、寄存器、Shared Memory 和硬件 block/warp 上限约束。

驻留 warp 多通常能提供更多隐藏延迟的机会，但驻留不代表当前都能发射指令；**理论 occupancy 与实际运行中的 achieved occupancy** 也要区分。

[原理分析：Occupancy 的含义](analysis/04-Kernel性能分析.md#occupancy)
{.study-analysis-link}

### 28. Occupancy 越高是否一定越快？ {#q19 .study-question}

不一定。更高 occupancy 可能增加可调度 warp，帮助隐藏延迟；但为此减少 tile、寄存器或缓存复用，也可能增加访存和指令。一个 kernel 已有足够独立工作时，继续提高 occupancy 未必有收益。

应该平衡 **驻留并行度、数据复用和指令级并行**，最终看 **总耗时与有效指令发射**。

[原理分析：为什么更高 Occupancy 未必更快？](analysis/04-Kernel性能分析.md#latency-hiding)
{.study-analysis-link}

### 29. Register 使用过多会造成什么后果？ {#q20 .study-question}

寄存器按 SM 预算分配，每线程使用增加可能减少可驻留的 block 和 warp，降低 occupancy；如果分配受限，编译器还可能 **spill 到 local memory**，增加经过缓存和设备内存的读写。但寄存器也能保存更多复用数据和独立累加器，所以不能一味压低用量，要结合 **驻留数量、spill 流量与实际时间** 判断。

[原理分析：寄存器压力](analysis/04-Kernel性能分析.md#register-pressure)
{.study-analysis-link}

### 30. Shared Memory 使用过多会造成什么问题？ {#q21 .study-question}

每 block 占用更多 Shared Memory，会减少同一 SM 可同时驻留的 block 数，可能降低延迟隐藏能力；超过单 block 可用上限，还可能无法 launch，或需要支持的容量配置。部分架构的 L1 与 Shared Memory 共用片上容量，分配也会影响缓存。

应平衡 **片上复用收益与驻留并行度**，不能把“放得更多”直接当作更快。

[原理分析：Shared Memory 压力](analysis/04-Kernel性能分析.md#shared-pressure)
{.study-analysis-link}

### 31. 如何提升 GPU L1/L2 Cache 命中率，命中率越高是否一定更快？ {#q41 .study-question}

先找可复用数据，再用 **tiling、布局与任务顺序** 缩短复用距离、控制工作集，减少一次性流量对热数据的挤占；支持的设备还可评估缓存策略。L1 通常更局部，L2 可供跨 SM 复用，但 block 的实际调度不能随意假定。

命中率提高可能来自多做了重复读取，因此要同时看 **请求数、实际显存字节与耗时**，不能把 cache hit rate 当成唯一目标。

[原理分析：如何改善 L1/L2 复用？](analysis/11-GEMM缓存与Reduction.md#cache-reuse)
{.study-analysis-link}

### 32. Cache Hit 率低到多少算异常，连续对齐加载一定能提高命中率吗？ {#q61 .study-question}

**没有通用正常阈值**，一次性流式读取可以低命中且高效；应明确 cache 层级、请求/sector 口径、冷热条件及是否有预期复用。连续对齐主要提高事务效率，命中率还取决于工作集与复用距离。

布局、tile、block 顺序可能改善复用，却也可能多做读取，应同时比较 **实际 miss 字节、指令和耗时**，6～7% 或 40% 不是跨任务标准。

[原理分析：Cache Hit 没有统一异常阈值](analysis/04-Kernel性能分析.md#cache-hit)
{.study-analysis-link}

### 33. Orin 的 Shared Memory 与哪个 Cache 共用片上容量，怎样调整？ {#q60 .study-question}

GPU 中相关 carveout 是 **SM 局部 L1/Texture 与 Shared Memory 的容量划分**，不是与设备共享 L2 划分。支持时可用 `cudaFuncSetAttribute` 的 preferred shared-memory carveout 请求偏好，较大动态 shared 另需满足 opt-in 上限；具体选项按设备与版本核对。

它是偏好而非任意精确分区，应同时看 **shared 用量、L1 行为与驻留**，不能把 SoC 共享 DRAM 与这个片上池混为一谈。

[原理分析：L1/Shared Carveout 不涉及 L2](analysis/04-Kernel性能分析.md#carveout)
{.study-analysis-link}

继续深入：[CUDA 存储与访存](analysis/02-CUDA存储与访存.md) · [Kernel 性能分析](analysis/04-Kernel性能分析.md)
{.study-source}

## 五、性能分析与优化证据 {#profiling}

### 34. 怎样判断 Kernel 是计算受限还是带宽受限？ {#q17 .study-question}

先估算运算量与实际数据流量，用 <strong class="study-key">算术强度和 Roofline</strong> 判断可能的上界，再结合 profiler 看计算管线、各层带宽和等待原因。接近带宽上限且数据搬运主导，优先优化复用与流量；计算管线饱和则关注指令与矩阵效率。

**两者都低也可能是延迟、并行度或提交不足**，不能只凭算力利用率低就认定带宽受限。

[原理分析：计算受限与带宽受限](analysis/04-Kernel性能分析.md#roofline)
{.study-analysis-link}

### 35. 如何通过 Profile 证明优化真正有效？ {#q22 .study-question}

先固定 shape、dtype、布局、硬件与计量条件，做正确性对照和多轮设备计时；再用 profiler 确认执行了预期 kernel，并检查 <strong class="study-key">与优化目标对应的指标变化</strong>，例如流量、bank conflict、等待或资源占用。用 Nsight Compute 看单 kernel，Nsight Systems 看时间线与提交空隙。

最终要证明 **耗时下降且机制吻合**；profiler 的采集开销需要与正式性能计时区分。

[原理分析：怎样用 Profile 验证优化？](analysis/04-Kernel性能分析.md#profile-evidence)
{.study-analysis-link}

### 36. 用 NCU 判断 Kernel 性能问题应看哪些指标，计算与访存利用率必须均衡吗？ {#q57 .study-question}

先有正确性和公平计时，再看 **SpeedOfLight、计算指令路径、MemoryWorkload、Scheduler、Occupancy 和源码/SASS**，与同 shape 基线比较。计算与内存吞吐百分比来自不同资源上界，不要求数值相等；一项高说明可能受该资源限制，两项低要查依赖、指令发射和任务不足。

**NCU 性能指标不能单独证明正确性或无越界**，最终要有耗时改善与机制证据。

[原理分析：NCU 从概览追到具体等待](analysis/04-Kernel性能分析.md#ncu-workflow)
{.study-analysis-link}

### 37. Memory Throughput 不高时，应该怎样排查？ {#q58 .study-question}

先明确指的是 **DRAM、L2、L1/shared 还是内存指令管线**，低 DRAM 吞吐可能只是缓存命中或工作量少。若数据等待主导，检查合并事务、有效字节、复用距离、shared bank、依赖链、eligible warp 和 grid 任务量；内存指令发射饱和也未必打满字节带宽。

优化应减少 **瓶颈上的实际成本**，而不是无条件提高某个吞吐百分比。

[原理分析：低 Memory Throughput 可能意味着什么？](analysis/04-Kernel性能分析.md#low-throughput)
{.study-analysis-link}

### 38. Long Scoreboard Stall 与 Warp 数量怎样解读，Stall 高是否就有问题？ {#q59 .study-question}

Long Scoreboard 通常表示等待 **L1TEX 路径相关的 global/local/texture/surface 等操作依赖**，不直接等于 HBM 带宽饱和。驻留 warp 多但 eligible warp 少，可能都在等相似依赖；其他 warp 能持续发射时，高 stall 也未必限制总吞吐。

结合源码位置、每 scheduler 可发射 warp、issue、缓存/流量和基线判断，**Long Stall 不是‘某比例 warp 一直停住’的通用量**。

[原理分析：Long Scoreboard 与高 Stall 的判断](analysis/04-Kernel性能分析.md#scoreboard)
{.study-analysis-link}

### 39. Sin/Cos 等特殊函数管线利用率较高时，怎样排查？ {#q62 .study-question}

先核对 **NCU 指标及实际 SASS**，确定是否真的反复计算特殊函数，还是编译辅助指令、索引/类型转换走了相关管线；较高百分比本身不证明瓶颈。可检查重复角度是否能复用、sin/cos 是否共享计算，或预计算表能否摊薄成本，但会增加内存访问。

Approximate math 会改变误差，必须验证 **范围、精度与端到端耗时**，不能单凭 70% 判异常。

[原理分析：特殊函数管线的计算与供数](analysis/04-Kernel性能分析.md#special-functions)
{.study-analysis-link}

### 40. Ampere 上优化的 Kernel 换到 Blackwell 后，为什么可能效率较低？ {#q56 .study-question}

正确运行不代表 **沿用旧 tile、供数和 warp 分工仍然最优**。新设备的计算/带宽比、SM 数量和指令路径变化，原方案可能任务不足、标量解码或同步主导，也可能没有使用适合该架构的新矩阵与搬运机制。先核对目标编译、SASS、dtype 支持及同卡基线，再 profile。

**不要假定旧 INT4 指令在新卡全部原生高效，也不能将 FP4 与 INT4 互换**。

[原理分析：从 Ampere 移到 Blackwell 要重做假设](analysis/14-GEMM切分与Persistent调度.md#architecture-retuning)
{.study-analysis-link}

继续深入：[Kernel 性能分析](analysis/04-Kernel性能分析.md)
{.study-source}

## 六、Attention 与归约算子 {#attention-reduction}

### 41. FlashAttention 的原理和主要优势是什么？ {#q38 .study-question}

FlashAttention 将 <strong class="study-key">QK、稳定 Softmax 和 PV 按 tile 融合</strong>，用 online softmax 状态逐块累积，避免完整分数与概率矩阵在显存中反复写读。它保持既定 dense Attention 的数学目标，浮点顺序可能有容差内差异，主要节省 **中间数据流量和工作空间**。

完整可见 KV 的依赖仍存在，dense Prefill 的二次运算量没有因此变成线性；Decode 的收益也需按其短 Query 形状判断。

[原理分析：FlashAttention 避免完整中间矩阵](analysis/10-FlashAttention与GroupedGEMM.md#online-softmax)
{.study-analysis-link}

### 42. FlashAttention 的融合边界是什么，QKV 投影是否也包含在内？ {#q39 .study-question}

核心融合通常是 **打分、缩放/支持的 mask、Softmax 与 Value 加权**，局部矩阵保留在片上，靠在线状态跨 tile 合并。QKV 的 Linear、RoPE、KV 写入和输出投影是否进一步融合取决于 backend，不能默认都属于同一个 kernel。

融合需兼顾 **中间布局、数据依赖、寄存器与 Shared Memory 预算**；资源压力或形状不匹配可能抵消减少访存和 launch 的收益。

[原理分析：融合有什么收益和边界？](analysis/10-FlashAttention与GroupedGEMM.md#fusion-boundary)
{.study-analysis-link}

### 43. Safe Softmax 如何实现，为什么先减最大值？ {#q33 .study-question}

对一行先求 **`m=max(x)`**，再算 `e=exp(x−m)`、`l=sum(e)`，输出 `e/l`。减去同一常数不改变 Softmax，而有限输入的指数不超过 1，可避免大正数导致指数溢出；通常用 FP32 做归约。

Mask 位置可以排除或设为负无穷，但 **全 mask、NaN 和正无穷输入要有明确处理约定**，不能只套公式。

[原理分析：减最大值保持结果，同时避免溢出](analysis/08-SafeSoftmax与Warp归约.md#stable-softmax)
{.study-analysis-link}

### 44. 如何用 Warp Shuffle 完成 Softmax 的 Max / Sum 归约？ {#q34 .study-question}

每个 lane 先对自己的元素求局部 max/sum，再用 **Shuffle 树交换寄存器值并归约**，例如完整 warp 用 XOR 距离 16、8、4、2、1 做 all-reduce。它减少 shared-memory 通信和同步，但必须保证 **参与 mask、活跃 lane 与数据来源正确**；不足一行的 lane 可提供 max 的负无穷和 sum 的零。

跨 warp 的行还要增加共享归约与同步。

[原理分析：一行由一个 Warp 处理](analysis/08-SafeSoftmax与Warp归约.md#warp-reduction)
{.study-analysis-link}

### 45. Softmax 如何通过寄存器暂存减少访存，有什么代价？ {#q35 .study-question}

一行能合理放入协作线程的寄存器时，可 **加载输入一次**，暂存数据求 max，再原地生成并保留指数，求 sum 后直接写归一化结果，避免重复读取或写出中间张量。行变长会增加寄存器压力、降低驻留，甚至 spill，把收益抵消。

应按行长度选择 warp/block 或多阶段算法，比较 **实际字节流量、归约与指数开销、总耗时**，不能认为所有 Softmax 都必然带宽受限。

[原理分析：寄存器暂存怎样降低流量？](analysis/08-SafeSoftmax与Warp归约.md#register-staging)
{.study-analysis-link}

### 46. 通用 Reduction Kernel 如何从 Warp 扩展到 Block 和多 Block？ {#q43 .study-question}

每线程先归约自己的输入，warp 内用 shuffle 合并；多个 warp 时把各 warp 的部分值放入 shared，<strong class="study-key">同步后再完成 block 归约</strong>。跨 block 可写 workspace 后启动第二阶段，或在满足语义时用 atomic，但普通 block barrier 不能同步整个 grid。

还需处理尾部、初始值、溢出和 **浮点归约顺序**；操作应有可组合的归约规则，不能将任意顺序敏感运算照搬。

[原理分析：Reduction 的协作范围](analysis/11-GEMM缓存与Reduction.md#reduction-scope)
{.study-analysis-link}

### 47. FP16/BF16 累加、Atomic Add 与 All-Reduce 为什么会影响数值和确定性？ {#q48 .study-question}

浮点加法 **不满足精确结合律**，改变归约树、分片或 atomic 到达顺序会改变舍入；BF16 范围更大但尾数精度较低，低精度乘法也不代表一定用低精度累加，应核对 FP32/INT32 等实际路径。Atomic 保证单次更新的原子性，不保证固定求和次序；All-Reduce 也受算法和拓扑影响。

固定归约树可改善 **同环境复现性**，仍需误差容差、质量检查和成本评估。

[原理分析：精度、原子性与确定性分别是什么？](analysis/13-直方图选择与数值归约.md#numerical-order)
{.study-analysis-link}

### 48. MoE Grouped GEMM 如何执行，主要瓶颈是什么？ {#q40 .study-question}

将 token—expert 对按专家聚集，为每个专家构造 **不同 M 的 GEMM**，用 grouped 调度让线程块处理这批矩阵，再经过激活和收缩投影，最后恢复 token 顺序与路由组合。Grouped GEMM 可减少多个小 GEMM 的提交开销，但小专家 batch、tile padding、权重读取、打包和任务偏斜仍可能主导。

**它不会自动均衡 EP 跨卡负载**，必须区分本地计算与 dispatch/combine 通信。

[原理分析：MoE 怎样形成一组 GEMM？](analysis/10-FlashAttention与GroupedGEMM.md#expert-gemm) · [Grouped GEMM 瓶颈](analysis/10-FlashAttention与GroupedGEMM.md#grouped-bottleneck)
{.study-analysis-link}

继续深入：[FlashAttention 与 Grouped GEMM](analysis/10-FlashAttention与GroupedGEMM.md) · [Safe Softmax](analysis/08-SafeSoftmax与Warp归约.md) · [直方图选择与数值归约](analysis/13-直方图选择与数值归约.md)
{.study-source}

## 七、稀疏格式与执行代价 {#sparsity}

### 49. 稀疏化与量化如何结合？ {#q5 .study-question}

稀疏化减少需要保存或计算的元素，量化减少每个保留值的位宽，两者可共同降低存储和计算成本。关键是 **执行路径能真正跳过零值，并直接消费低 bit 数据**；只把稠密张量中的值置零不会自动加速。结构化稀疏通常更容易匹配硬件，非结构化稀疏需要索引与专用 kernel。

两种误差会叠加，应联合验证质量，**收益也不能直接相乘**。

[原理分析：稀疏化如何与量化结合？](analysis/01-量化与稀疏.md#joint-compression)
{.study-analysis-link}

### 50. 2:4 结构化稀疏与非结构化稀疏有什么区别，何时能使用硬件加速？ {#q63 .study-question}

2:4 通常要求沿规定维度 **每四个元素至多两个非零**，按支持的编码压缩值和位置元数据；非结构化稀疏的位置更自由，通常依赖软件索引和专用执行。NVIDIA 相关机制是支持该模式的 Sparse Tensor Core/MMA，不是所有 GPU 都有通用 Sparse Core。

需要匹配 **架构、dtype、模式、布局与 kernel**；硬件理想峰值不代表端到端翻倍，剪枝后质量也要另验证。

[原理分析：2:4 结构与硬件约束](analysis/01-量化与稀疏.md#structured-2of4)
{.study-analysis-link}

### 51. CSR、COO、块稀疏与 Bitmap 格式如何取舍？ {#q37 .study-question}

CSR 按行保存非零值、列索引和行边界，减少重复行索引，但长短行会造成负载不均；COO 显式存行列，格式直接却有较高元数据和归并成本。块稀疏用 **一个索引描述一块数据**，更容易复用和匹配矩阵计算，但可能计算块内零值；Bitmap 用 bit 描述存在性，还需紧凑偏移与解码。

选择要平衡 **压缩率、访存、均衡和计算单元适配**，最省字节不一定最快。

[原理分析：格式取舍](analysis/09-SpMM与稀疏格式.md#sparse-formats)
{.study-analysis-link}

### 52. 稀疏 Bitmap 的作用是什么，如何判断它能否带来加速？ {#q32 .study-question}

Bitmap 用 bit 描述 **哪些元素或 tile 需要保存、加载或计算**，必须先定义每个 bit 与数据的映射。收益来自实际跳过工作，代价是 bitmap 读取、解码、压缩地址计算和不规则负载。

自定义 tile bitmap 不等于硬件 2:4 MMA 的位置元数据；可用非零率、模式、tile 大小与解码成本判断候选场景，再验证 **相同精度下的总耗时和质量**。

[原理分析：Bitmap 与硬件稀疏元数据分开理解](analysis/07-W4A4与稀疏元数据.md#bitmap-layout)
{.study-analysis-link}

### 53. 剪枝量化后的 SpMM 为什么可能更偏 Memory Bound？ {#q36 .study-question}

剪枝减少非零乘加，但 **索引、scale、输出和部分激活访问未按同比例减少**，不规则读取还可能降低复用与事务利用率，使单位字节的有效计算变少。量化减少数值载荷后，元数据占比也可能上升，计算加速又会暴露供数成本。

不过量化同样能降低流量，**SpMM 不必然更带宽受限**；应按稀疏格式、非零分布和实际流量计算算术强度，再用 profile 验证。

[原理分析：剪枝、量化怎样改变瓶颈？](analysis/09-SpMM与稀疏格式.md#sparsity-bottleneck)
{.study-analysis-link}

### 54. SparseGPT 等剪枝与量化后，怎样验证质量，PPL 能否证明精度无损？ {#q64 .study-question}

SparseGPT 利用校准激活和二阶近似，在剪枝时补偿剩余权重以降低 **层输出重构误差**，仍不能保证所有任务无损。应分别比较原始、仅剪枝、仅量化和组合模型，用独立评估测 PPL及目标任务质量，并核对真实低精度 kernel。

**PPL 是 token 平均负对数似然的指数**，越低越好；单个 WikiText-2 PPL≈10 或 group size=128 没有通用结论，需同口径基线与容差。

[原理分析：SparseGPT、PPL 与“无损”结论](analysis/01-量化与稀疏.md#quality-ppl)
{.study-analysis-link}

继续深入：[SpMM 与稀疏格式](analysis/09-SpMM与稀疏格式.md) · [量化与稀疏](analysis/01-量化与稀疏.md)
{.study-source}

## 八、架构进阶与编程补充 {#architecture}

### 55. 部署大模型时，怎样比较显存容量、算力和带宽指标？ {#q23 .study-question}

先用 **权重、KV Cache 和运行缓冲** 判断模型与目标并发能否放下，再按 Prefill/Decode 的实际 shape 判断计算与访存需求。峰值算力要匹配 dtype、矩阵指令和稀疏条件；**设备显存带宽与卡间互联带宽是不同指标**。

最终还要比较框架支持、持续功耗和满足 TTFT/TPOT 目标时的成本，不能只按产品档次或峰值 FLOPS 选卡。

[原理分析：三种资源分别限制什么？](analysis/05-硬件指标与互联.md#capacity-compute-bandwidth)
{.study-analysis-link}

### 56. 没有 NVLink、主要通过 PCIe 通信，会怎样影响多卡推理？ {#q24 .study-question}

TP 的层内归约、EP 的 token 交换和 P/D 的 KV 交接都可能更难摊薄 **互联延迟与传输成本**，因此少量模型并行加更多副本有时更合适。PCIe 的代际、通道宽度、交换机和跨 CPU 路径会影响实际通信；还要检查 P2P 是否可用。

**没有 NVLink 不等于不能多卡推理**，应测实际 collective 和端到端性能，不能把 PCIe 标称带宽直接当成每对 GPU 的有效带宽。

[原理分析：PCIe 拓扑为什么重要？](analysis/05-硬件指标与互联.md#pcie-topology)
{.study-analysis-link}

### 57. Ampere、Hopper、Blackwell 的矩阵计算与搬运机制有哪些重要变化？ {#q25 .study-question}

Ampere 常用 **warp 级 MMA 与 `cp.async`** 组织 shared-memory 供数；Hopper 新增 TMA 和 warpgroup 级异步 WGMMA，更便于搬运、矩阵计算与标量计算重叠。Blackwell 的第五代 Tensor Core 路径使用 **`tcgen05.mma` 与 Tensor Memory**，并增加 FP4/FP6 等低精度格式支持。

新机制没有替代所有旧路径，具体 shape、格式和指令支持必须按目标架构核对。

[原理分析：先区分搬运、存储与运算](analysis/06-Ampere-Hopper与Blackwell.md#architecture-paths)
{.study-analysis-link}

### 58. TMA 与 Tensor Memory 有什么区别？ {#q26 .study-question}

TMA 是 **张量数据的异步搬运机制**，支持按 tensor map 描述的多维布局等路径在 global/shared 之间搬运；Tensor Memory（TMEM）是 Blackwell Tensor Core 使用的 **专用片上存储**，可承载累加器、部分操作数和 block scale。一个解决数据搬运，一个承载矩阵运算状态；TMEM 需要显式分配和专用访问指令，不能把它当成 TMA、普通寄存器或通用 Shared Memory。

[原理分析：TMA 与 TMEM 是不同对象](analysis/06-Ampere-Hopper与Blackwell.md#tma-tmem)
{.study-analysis-link}

### 59. Hopper WGMMA 为什么需要 Warpgroup，Blackwell UMMA 的发射范围有什么不同？ {#q27 .study-question}

WGMMA 定义为 **4 个 warp、128 个线程协同提交**，操作数描述和寄存器 fragment 都要满足该集体指令的约束。常称 UMMA 的 Blackwell `tcgen05.mma` 使用 TMEM 与矩阵描述符，PTX 定义为 **单线程提交即可启动完整矩阵操作**，不需要 128 个线程共同发射。

TMEM 分配、读写和 epilogue 仍有各自的协作范围，因此不能把单线程发射等同于单线程完成整个 GEMM。

[原理分析：发射范围不是数据参与范围](analysis/06-Ampere-Hopper与Blackwell.md#issue-granularity)
{.study-analysis-link}

### 60. Blackwell Block-scaled MMA 如何使用 TMEM 中的 Scale Factor？ {#q28 .study-question}

支持的 `tcgen05.mma` block-scale 路径从 **TMEM 读取 A/B 的 scale 矩阵**，按规定布局给对应低精度元素组缩放，再进行矩阵乘加。需要匹配 scale 的格式、分组大小、布局和加载就绪状态。

若 scale 沿 K 分组变化，不同部分积要用不同 scale，**不能把它们先全部累加，再任意乘一个统一 scale**；普通 epilogue 缩放与输入 block scaling 是两件事。

[原理分析：Scale Factor 如何参与计算？](analysis/06-Ampere-Hopper与Blackwell.md#block-scaling)
{.study-analysis-link}

### 61. Warp Specialization 怎样组织异步流水线，为什么仍需要同步？ {#q30 .study-question}

让不同 warp 承担 **生产数据、提交矩阵运算、消费结果或执行 epilogue** 等角色，重叠可独立推进的工作。消费者必须等待输入完成，生产者必须等旧数据不再被使用才能复用槽位；异步矩阵结果也需完成协议后才能读取。

TMA、WGMMA 与 `tcgen05` 的等待和可见性规则不同，**普通 block barrier 不能概括所有异步依赖**，还要权衡角色分配和缓冲资源。

[原理分析：异步流水线要建立两类依赖](analysis/06-Ampere-Hopper与Blackwell.md#async-dependencies)
{.study-analysis-link}

### 62. Intel GPU 与 NVIDIA GPU 的架构差异怎样影响 Kernel 移植？ {#q44 .study-question}

要比较具体产品代际，主要核对 **执行组宽度、矩阵指令、寄存器/共享存储和访存布局**。NVIDIA 常以 32-thread warp、SM 和 Tensor Core 描述；Intel Xe 产品有各自的执行单元组织，SYCL subgroup 宽度需按设备支持选择，支持的产品可用 XMX/DPAS 路径。

CUDA warp、fragment 和 bank 映射不能逐项照搬，移植要重做 **线程映射、同步、供数与性能验证**，不能只改 API 名称。

[原理分析：移植必须重新检查哪些机制？](analysis/12-Intel与NVIDIA架构.md#porting-contract) · [执行组与矩阵硬件对照](analysis/12-Intel与NVIDIA架构.md#architecture-comparison)
{.study-analysis-link}

### 63. 怎样用直方图与前缀和查找数组第 K 大，适用条件是什么？ {#q45 .study-question}

若元素来自 **可直接计数的有限离散值域**，先统计每个值的频数，再从大值方向累计，找到累计数量首次达到 K 的值；升序前缀和则定位第 `N−K+1` 小，重复值按出现次数计数。值域很大时可分桶后继续选择，普通浮点粗分桶只能确定候选区间，不能直接给出精确结果。

关键是 **明确值域、K 边界与精确性要求**，而不是默认所有数组都适合完整直方图。

[原理分析：先明确第 K 大的定义与值域](analysis/13-直方图选择与数值归约.md#histogram-selection)
{.study-analysis-link}

### 64. Shared Memory 直方图和线程批量处理分别优化什么？ {#q46 .study-question}

Block 先在 shared 中统计局部直方图，再归并到全局，可减少 **全局 atomic 次数与竞争**，但热点 bin 仍可能在 shared 中冲突，也有清零、同步和合并成本。每线程循环处理多个元素能摊薄开销，合并加载、寄存器暂存或局部计数可减少冗余工作；若每个输入只读一次，**批量处理不自动增加 L2 复用或减少必要读字节**，要用实际事务、atomic 与耗时验证。

[原理分析：Shared Histogram 降低全局原子竞争](analysis/13-直方图选择与数值归约.md#shared-histogram) · [线程批量处理与缓存复用](analysis/13-直方图选择与数值归约.md#grid-stride)
{.study-analysis-link}

继续深入：[硬件指标与互联](analysis/05-硬件指标与互联.md) · [Ampere、Hopper 与 Blackwell](analysis/06-Ampere-Hopper与Blackwell.md) · [Intel 与 NVIDIA GPU](analysis/12-Intel与NVIDIA架构.md) · [直方图选择与数值归约](analysis/13-直方图选择与数值归约.md)
{.study-source}

</div>
