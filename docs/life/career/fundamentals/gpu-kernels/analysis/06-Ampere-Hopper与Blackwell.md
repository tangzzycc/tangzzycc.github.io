---
title: "Ampere、Hopper 与 Blackwell：异步矩阵计算和 TMEM"
---

# Ampere、Hopper 与 Blackwell：异步矩阵计算和 TMEM

<div class="study-notes study-analysis" markdown="1">

来源：提供的「腾讯 AI Infra 部门二面」记录；对应 GPU与算子优化[第 22、57～61 题](../index.md#q29)。指令语义根据 NVIDIA PTX 与 CUTLASS 文档核对，查阅日期 2026-10-13。

## 1. 先区分搬运、存储与运算 {#architecture-paths}

| 代表路径 | 数据搬运 | 矩阵计算 | 累加结果与协作 |
| --- | --- | --- | --- |
| Ampere | `cp.async` 等 global→shared 路径 | `mma.sync` 等 warp 级 MMA | 常见累加 fragment 分布在线程寄存器中 |
| Hopper | TMA 多维异步搬运，可使用 tensor map | `wgmma.mma_async`，warpgroup 级异步矩阵指令 | 4 warp 集体提交，累加结果为寄存器 fragment |
| Blackwell 第五代 Tensor Core 路径 | TMA 供数及 shared→TMEM 等专用搬运 | `tcgen05.mma`，常称 UMMA | 单线程提交，累加器位于 TMEM |

这是代表性路径，不是每代只能执行这些指令。目标 SM、dtype、shape、编译选项和具体内核决定是否可用；新机制也不保证自动加速小矩阵。

返回面试题：[57](../index.md#q25)
{.study-source}

## 2. TMA 与 TMEM 是不同对象 {#tma-tmem}

- <strong class="study-key">TMA，Tensor Memory Accelerator。</strong> 搬运机制，支持 global/shared 多维张量等路径；tensor map 描述布局、步长和相关搬运属性，可减少逐线程地址计算与发射负担。
- **TMEM，Tensor Memory。** Tensor Core 专用片上存储，支持累加器、部分 A 操作数、scale 等；需要显式分配、回收与专用 load/store/copy。
- **容量形状。** PTX 给出的 `sm_100a/sm_100f` 组织为 128 lane×512 column、每 cell 32 bit，表示专用结构容量，不是所有 CTA 都能无约束地占满；分配和并发受资源规则限制。

### TMA 为什么比“每个线程手算一个地址”更适合多维 tile？

传统协作 copy 由线程计算行列、stride、越界和源目标地址，再各自发起小段搬运。TMA 使用 tensor map 描述全局维度、步长与规定的 tile/布局属性，由少量提交线程发起多维块搬运，减少逐线程地址计算和发射负担。它还能与受支持的 shared 布局、cluster multicast 等路径配合；并非任意 tensor view 都能直接使用。

Descriptor 是解释存储地址的元数据，不含整个张量数值；global tensor 地址、stride 或排列变化时，descriptor/调用必须仍匹配实际数据。搬运到 shared 之后，矩阵消费者也要按对应布局读取，TMA 不自动修复错误的 MMA fragment 映射。

TMEM 则改变累加器/operand 的存放和访问方式。结果位于 TMEM 时，epilogue 需通过指定访问指令取到线程可处理的值，再转换/写出；不能把 TMEM 地址当普通 global/shared 指针使用。将累加器从线程寄存器转移到专用存储会改变资源预算，但输入、临时值和 epilogue 仍需寄存器。

返回面试题：[58](../index.md#q26)
{.study-source}

## 3. 发射范围不是数据参与范围 {#issue-granularity}

| 操作 | PTX 所述主要发射要求 |
| --- | --- |
| Hopper `wgmma.mma_async` | 同一 warpgroup 的 128 线程满足集体提交约束 |
| Blackwell `tcgen05.mma` / `cp` / `commit` | 单线程即可启动对应基操作，CTA-pair 路径另有同伴存活要求 |
| TMEM 分配/释放，`cta_group::1` | 一个完整 warp 协作 |
| TMEM 分配/释放，`cta_group::2` | 两个 peer CTA 各一个 warp 协作 |
| TMEM `ld/st` | 按规定 warp 与 lane 映射访问；单 warp 只能覆盖对应部分 |

WGMMA 的编程模型由分布式 fragment 和集体提交约束组成；`tcgen05` 将矩阵描述与 TMEM 状态交给专用矩阵引擎，因此提交线程数与矩阵规模解耦。单线程提交之后，输入准备、结果读取和 epilogue 仍由相应线程协作。

返回面试题：[59](../index.md#q27)
{.study-source}

## 4. Scale Factor 如何参与计算？ {#block-scaling}

支持的 block-scaled 指令接收 A/B 的 TMEM scale 基地址，并按指令定义的布局及分组应用缩放。用沿 K 分组的简化数学式说明：

$$
D_{mn}=\sum_g s^A_{m,g}s^B_{n,g}
\sum_{k\in g}\widetilde A_{mk}\widetilde B_{kn}+C_{mn}.
$$

不同 K 组的 scale 产品不同，不能直接提到整个求和外面。真实 scale 格式、索引和向量大小由 MMA kind/descriptor 决定；MX 与 NVFP4 等路径的分组和编码需要分别核对。数据和 scale 都需在消费前就绪，不能只等低 bit 主数据。

FP4 的 E2M1、FP6 的 E2M3/E3M2 等浮点编码与 INT4 不同；“4 bit 存储”“带 block scale”“原生低精度 MMA”需要分别确认。

### FP4 与 INT4 的解码差异在哪里？

INT4 用整数 code 加外部 scale/zero-point 表示均匀数值网格；FP4 的 E2M1 按符号、指数和尾数表示非均匀浮点集合。两者即使都占 4 bit，同一个 bit pattern 的数值也可能不同，不能直接替换 checkpoint dtype 或复用整数解包规则。

Block scaling 是外部缩放维度：一组低精度值共用 scale，矩阵指令按规定沿 K 分组应用 A/B scale 产品。它与低精度值内部的浮点指数是两件事。Scale 本身的编码、向量大小、布局和 TMEM 落点属于指令契约；只知道“FP4”无法推断支持 MX、NVFP4 或某种整数 MMA。

两个 K 组的原始点积相同为 8，scale 产品分别为 0.25 和 0.5，最终贡献为 2+4=6。主数据载荷即使很小，scale 加载、格式转换和每组处理依然必须完成；更低 bit 的峰值矩阵吞吐也会让供数与 epilogue 更容易暴露为瓶颈。

返回面试题：[22](../index.md#q29) · [60](../index.md#q28)
{.study-source}

## 5. 异步流水线要建立两类依赖 {#async-dependencies}

1. **输入就绪。** TMA 或其他生产路径完成且对消费者可见，才能执行读取该 tile 的 MMA。
2. **结果与槽位安全。** 异步 MMA 完成后才能读取结果；它不再消费输入时，才能覆盖相应输入槽位。

Warp specialization 可分配搬运、矩阵提交和 epilogue 角色，双缓冲或多 stage 可覆盖部分等待。Hopper WGMMA 的 fence/commit/wait 与 Blackwell `tcgen05` 的 commit/mbarrier 协议不能机械互换；异步 proxy 的可见性也不能只用普通 `__syncthreads()` 推断。

### mbarrier 为什么需要 phase 和完成字节计数？

对于受支持的 TMA 完成跟踪，barrier 不只记录“线程到达了几次”，还可记录异步 transaction 的预期完成量。一个 phase 需要对应的 arrival 和事务完成条件满足后才能进入完成状态；槽位复用时下一代 phase 必须和旧完成区分，否则消费者可能把旧信号当成新 tile 已就绪。

概念流水线为：生产者获得空槽→登记本代搬运完成条件→提交 TMA→消费者等待本代完成→提交矩阵计算→等待计算不再使用输入→释放空槽。普通 `__syncthreads()` 只能表达 CTA 线程同步，不能凭线程到达就宣布 TMA/WGMMA/`tcgen05` 工作完成；不同异步 proxy 之间的可见性也需按 PTX 规定建立。

Warp specialization 把加载、矩阵提交与 epilogue 分给不同协作组，使各角色寄存器需求和等待位置更清楚，但生产者过慢会饿死消费者，生产者过快会占满槽位。角色数、stage 和 barrier 协议共同决定稳态效率，不能只看到“一组 warp 搬数据”就认为延迟已隐藏。

资料：[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions)、[PTX Issue Granularity](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-issue-granularity)、[PTX Block Scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-block-scaling)、[CUTLASS Blackwell Functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。已有分析：[FlashAttention-3 异步流水线](<../../../../../vllm/进阶篇/FlashAttention-3：异步搬运与计算流水线.md>)。

返回面试题：[61](../index.md#q30)
{.study-source}

</div>
