---
title: "Kernel 性能分析：Roofline、Occupancy 与验证"
---

# Kernel 性能分析：Roofline、Occupancy 与验证

<div class="study-notes study-analysis" markdown="1">

来源：提供的「小鹏汽车二面」「B站 AI Infra 二面」图片与记录；对应模块总文件[第 27～30、32～39 题](../index.md#q18)。

## 1. 计算受限与带宽受限 {#roofline}

- <strong class="study-key">先估算。</strong> 算术强度 `I=运算量/字节流量`，Roofline 上界为 `P≤min(P_peak,BW×I)`；计算峰值要与实际 dtype、指令路径匹配。
- **区分层级。** L2、设备内存、Shared Memory 都可能成为限制；用相应层级的实际流量，不把逻辑张量大小当作全部流量。
- **再确认。** 看计算管线、带宽、eligible warp 和 stall。计算与带宽都未饱和时，还需检查依赖延迟、并行任务量及 CPU 提交。

设 F 为实际运算量，Q 为某一存储层实际搬运字节，峰值计算速度 P 和带宽 BW 使用同一目标 dtype/指令路径，则计算时间下界为 F/P，搬运下界为 Q/BW；忽略其他成本时，总时间不小于两者较大值。由此得到算术强度 `I=F/Q` 和转折点 `I*=P/BW`：I 低于转折点时，带宽上界更紧。

例如 P=100 TFLOP/s、BW=2 TB/s，转折点为 50 FLOP/byte。若 kernel 的实际强度只有 5，则带宽侧上界为 10 TFLOP/s，峰值再高也无法把这个读取成本消除。数字是教学例子，真实 GPU 的 dtype、dense/sparse 峰值以及持续带宽需要独立核对。

**Roofline 给出候选上界；低算力与低带宽同时出现时，应继续查并行不足、依赖、指令与提交空隙。**
{.study-core}

### 为什么同一个 GEMM 在 HBM Roofline 下不受限，却仍被供数卡住？

同一 kernel 可以分别计算 HBM、L2、shared 的算术强度。假设多个 CTA 的权重读取命中 L2，HBM 字节很少，但 L2→SM 仍持续搬运；CTA 内又可能反复从 shared 读取 operand。优化减少了片外流量，并没有消除内部供数与矩阵加载需求。

分层判断应保持同一个 F，而每层使用对应实际 Q 和带宽上界。Shared bank conflict 会增加服务轮次，反量化可能消耗普通 ALU，矩阵 operand 的准备也可能形成依赖。HBM 吞吐低且 Tensor Core 吞吐低，并不说明两者“应该平衡”，而说明还需查中间路径。

返回面试题：[34](../index.md#q17)
{.study-source}

## 2. Occupancy 的含义 {#occupancy}

- **定义。** SM 驻留 warp 数除以硬件允许的最大驻留 warp 数，受寄存器、Shared Memory、线程和 block 上限共同约束。
- **理论与实际。** 理论值按 launch 配置和资源计算；实际 achieved occupancy 会受工作分布、尾部与运行过程影响。
- **驻留与发射。** 驻留 warp 可能在等数据或依赖；occupancy 不等于活跃 lane 比例、SM 利用率或每周期发射率。

设每 CTA 有 256 线程，即 8 个 warp。只为说明资源计算，假设 SM 有 65536 个 32-bit 寄存器、64 KiB shared、最多 64 warp；每线程用 64 个寄存器，每 CTA 用 32 KiB shared。寄存器最多容纳 4 个 CTA，shared 最多容纳 2 个 CTA，所以先忽略其他限制时只有 16 个驻留 warp，理论 occupancy 为 25%。真实资源分配还按硬件粒度向上取整，并受线程/CTA 上限约束。

这 16 个 warp 不一定每周期都能发射。Warp 在等内存、矩阵结果或 barrier 时仍可以驻留，eligible warp 才表示当前有可发射指令；同一 warp 的独立指令也可以提供 ILP。因此 occupancy 是资源条件，不能代替发射效率。

返回面试题：[27](../index.md#q18)
{.study-source}

## 3. 为什么更高 Occupancy 未必更快？ {#latency-hiding}

- **有益情况。** 可运行 warp 太少时，提高驻留能给调度器更多隐藏延迟的机会。
- **收益边界。** 已有足够 warp 或独立累加链时，更高驻留可能不再改善发射；为提高驻留而缩小 tile，还可能损失复用。
- **决策。** 比较耗时、eligible warp、指令发射与流量，保留复用和并行度综合更好的配置。

### 高 occupancy、低 eligible warp 常见于什么情况？

驻留 warp 多说明资源允许它们存在；eligible 少说明它们大多无法发射下一条指令。若全部刚执行同一类加载，再一起等结果，增加 warp 只是让更多工作同时等待。若存在独立加载与计算，可错开阶段、增加 ILP 或调整 producer/consumer 分工；若带宽本来已满，则更多请求只加深队列。

Barrier 也会让提前到达的 warp 等慢 warp；长 K 依赖链会让同一个累加器的后续更新必须等待前一步。诊断应把等待对应到具体消费者指令，再追踪产生数据的 load/MMA，区分执行依赖、内存依赖与组内工作偏斜。

返回面试题：[28](../index.md#q19)
{.study-source}

## 4. 寄存器压力 {#register-pressure}

- **驻留限制。** 每线程寄存器增加会提高 block 总需求；受分配粒度影响，驻留 block 数可能阶跃下降。
- **Spill。** 编译器受资源或显式上限约束时，部分值可能落入线程私有 local memory，访问通常经过缓存和设备内存。
- **取舍。** 更多寄存器也能减少重读并提高指令级并行；查看编译资源报告与实际 local 读写，不能只按寄存器数调优。

寄存器增长有两个不同后果。第一种是驻留 CTA/warp 数在资源阈值处下降，即使没有 spill，也可能减少可切换工作。第二种是编译器把局部值放到 local memory，出现额外 load/store；这些访问可能命中 cache，也可能增加片外流量。主动降低寄存器上限可能触发第二种代价。

展开更多 K 迭代、保存更多 Softmax 元素、增加量化解包临时值都会扩大活跃变量的生命周期。先缩短不必要的生命周期或改变 tile，再比较编译资源报告、local 访问、eligible warp 与耗时，避免只追求更少的寄存器数字。

返回面试题：[29](../index.md#q20)
{.study-source}

## 5. Shared Memory 压力 {#shared-pressure}

- **驻留限制。** 每 block 用量过大，SM 可容纳的 block 变少；双缓冲和更多 stage 尤其需要重新估算。
- **容量边界。** 单 block 预算与每 SM 总预算不同；超过默认上限可能需要设备支持的 opt-in 配置，超过可用上限则无法执行。
- **复用收益。** 部分 NVIDIA 架构还需平衡 L1/shared 容量；实际收益同时受复用、布局冲突和同步影响。

Shared 的预算是每个 CTA 申请量乘可同时驻留 CTA 数，而不是整个 grid 的所有 tile 同时分配。若每 CTA 从 24 KiB 增到 40 KiB，在可用 64 KiB 的简化 SM 中，可驻留 CTA 从 2 变为 1；增加 stage 带来的重叠可能被更少的并发抵消。

另一方面，减少 shared tile 会降低复用、增加 global 读取或同步次数。应联合优化 tile、stage、线程数与寄存器，不能把“shared 占用降低”直接当作性能收益。动态 shared 超过默认 per-block 限额时还需目标设备支持和对应 launch 配置。

返回面试题：[30](../index.md#q21)
{.study-source}

## 6. 怎样用 Profile 验证优化？ {#profile-evidence}

- **公平计时。** 固定输入、精度、布局和设备条件，先检查正确性，再 warmup、多轮设备计时并记录统计。模型阶段还要固定请求负载。
- **机制证据。** 合并访存看事务与有效流量；布局看 shared 冲突；流水线看等待、发射和异步指令；资源调整看驻留与 spill。
- **工具职责。** Nsight Compute 检查 kernel 指标，Nsight Systems 检查时间线、CPU 提交和依赖。Profiler 可能重放或扰动执行，正式性能计时需另测。
- **最终结论。** 既要耗时下降，也要指标支持预期原因；影响模型收益时，再验证端到端变化，不能直接把单 kernel 加速比当作整体收益。

证据应能连成因果链：例如优化前一行 Softmax 反复读取输入，优化后寄存器暂存；预期 global 输入流量下降，实测流量确实下降，且未产生严重 spill，最后正常设备计时变短。单个 cache hit 百分比变化只能是辅助观察，无法证明因果。

如果被优化算子原占端到端时间 20%，算子快 2 倍，其他成本固定时整体加速仅为 `1/(0.8+0.2/2)≈1.11` 倍。在线服务还会改变组批、队列和竞争，需要完整请求负载验证。Profiler 的 replay、序列化和缓存控制也可能扰动时间，因此正式计时与诊断采集要分开解释。

### 怎样排除计时中的异步、缓存和 replay 假象？

主机调用结束只说明工作已经提交，直接用 CPU 时间差可能漏掉 GPU 执行。单设备算子常用同 stream 的设备事件计时，结束后等待完成；完整 serving 则还需包含 CPU 调度、传输、客户端返回，不能只报事件间的 kernel 时间。

预热与重复调用可以形成热缓存，真实服务中的其他模型层会改变复用距离；NCU replay 还可能控制缓存状态、重复 kernel 或串行采集。报告需要说明正常计时与诊断采集的不同条件。固定输入/shape 后，先核对结果和实际指令，再看流量、资源和分布，最后确认完整路径的收益。

若声称“减少访存”，要对应绝对字节或 sector 数；若声称“隐藏等待”，要对应关键路径上的未覆盖等待；若声称“提升缓存命中”，还要排除访问总数增加。百分比单独变化不能解释加速。

返回面试题：[35](../index.md#q22)
{.study-source}

## 7. NCU 从概览追到具体等待 {#ncu-workflow}

| 观察项/Section | 要回答的问题 | 解读边界 |
| --- | --- | --- |
| SpeedOfLight | 哪些计算或内存资源接近各自上界？ | 百分比基于不同资源，不要求计算与内存相等 |
| ComputeWorkloadAnalysis / 指令 | 预期 Tensor/标量路径是否执行，哪个管线繁忙？ | 高 Tensor 利用不等于所有工作都有效 |
| MemoryWorkloadAnalysis | DRAM/L2/L1/shared 的流量、事务和管线压力怎样？ | Mem Busy、字节带宽与内存指令发射是不同限制 |
| SchedulerStats | Active、Eligible 与 Issued warp 是否足够？ | 驻留不表示当前可以发射 |
| Occupancy / LaunchStats | 资源、grid 与尾部是否限制工作供给？ | 高 occupancy 不是性能目标本身 |
| WarpStateStats / SourceCounters | 等待发生在什么依赖和源码/SASS 位置？ | 采样项、归一化口径与具体指令需核对 |

Section 和指标可用性随 NCU、设备和 kernel 路径变化。先与同 dtype/shape/layout 的正确基线比较耗时与实际工作，再解释指标；NCU 不替代数值对照和越界/同步检查。

Compute Throughput 与 Memory Throughput 不需要“均衡到一样”。真实瓶颈可在单个管线、内存层级或依赖链上；两个百分比都低时要进一步看任务数、指令发射、barrier 和供数等待。若只有少数 CTA，单 kernel NCU 反映设备工作不足；连续 kernel 之间的 Host 间隙则用 NSYS 观察。

返回面试题：[36](../index.md#q57)
{.study-source}

## 8. 低 Memory Throughput 可能意味着什么？ {#low-throughput}

先明确 DRAM、L2、shared 或内存指令吞吐。数据在 cache 命中时 DRAM 流量本来就小；工作量少、计算主导时也不需要打满内存带宽。若存在严重数据等待，按以下证据定位：

- 地址是否合并对齐，有效字节占事务字节的比例是否低。
- 本来应复用的数据是否跨越较长距离或工作集超出 cache。
- Shared 访问是否出现额外 bank 事务，宽度与布局是否兼容消费者。
- 寄存器 spill 是否引入 local 读写，内存指令队列是否繁忙。
- Eligible warp 是否过少，长依赖链是否缺少独立加载与计算。
- Grid、尾 tile 和 stage 是否让设备任务供给不足。

同样低 DRAM 字节率，可能是“已用 cache 节省读写”或“未能提供足够在途访问”，优化方向相反。不能为了提高带宽图表数值，重新增加本来已经省掉的访问。

返回面试题：[37](../index.md#q58)
{.study-source}

## 9. Long Scoreboard 与高 Stall 的判断 {#scoreboard}

NCU 的 `smsp__pcsamp_warps_issue_stalled_long_scoreboard` 关联等待 L1TEX 的 local/global/surface/texture 等操作依赖；采样通常落在等待结果的消费者指令上，需再找产生该数据的加载。它不直接表示 HBM 带宽饱和，也不是通用“长时间 stalled warp 占比”。

若 active warp 较多却 eligible/issue 都低，可能很多 warp 等相同的供数、依赖或同步。可在未饱和时通过独立加载、ILP、适当展开、预取或额外驻留隐藏延迟；但增加 warp 也可能加剧带宽/资源竞争。

若其他 warp 仍能持续发射，高 stall 采样并不必然拖慢总执行。应结合采样分母、issue、指令位置、流量、基线与正式计时。Short scoreboard、barrier、math-pipe throttle 又对应不同路径，不能把所有等待都解释成同一种内存问题。

返回面试题：[38](../index.md#q59)
{.study-source}

## 10. L1/Shared Carveout 不涉及 L2 {#carveout}

图片中的“Orin Shared Memory 与 L2 共用空间”应修正为：相关 NVIDIA GPU 的 **SM 局部 L1/Texture 与 Shared Memory 共用片上容量池**，设备共享 L2 是另一层。SoC 的 CPU/GPU 共享物理 DRAM 又是独立概念，不决定该片上 carveout。

支持时，`cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, preference)` 请求 shared 容量偏好；不是任意精确分割或硬保证。较大动态 shared 还要满足设备支持的单 block/SM 上限，并使用相应 opt-in 配置；不将 A100 的数值照搬到 Orin。

比较实际 shared 用量、驻留、L1 行为与耗时。使用 `cp.async` 的某些路径可能绕过 L1，也不意味着所有访问都无需 cache；Cache Hit 与内存类型行为按 Tegra/目标设备实际路径核对。

返回面试题：[33](../index.md#q60)
{.study-source}

## 11. Cache Hit 没有统一异常阈值 {#cache-hit}

图片中的 6～7%→40%+ 是个人测试描述。流式读取低命中可正常，循环重读却低命中需要检查复用距离；连续对齐主要改善事务效率，不保证命中率上升。缓存工作集、指标层级与冷热条件必须一致，命中率还需配合绝对请求/miss 字节和耗时。

### 相同命中率为什么可能对应相反优化方向？

单遍流式读取没有时间复用，低命中是合理现象；反复访问同一小工作集却低命中，才应检查布局、复用距离、冲突或采集缓存控制。假设原来 100 次访问命中 50 次，后来 200 次命中 140 次，命中率从 50% 升到 70%，miss 却从 50 增到 60。还要比较每次请求的 bytes/sector，不能只看次数。

合并访问减少无用 sector，可能同时减少“访问次数”这个分母；融合减少中间张量流量，也可能降低某个 cache 命中百分比但更快。正确目标是减少实际成本，辅助指标须和绝对流量、指令与正式计时连起来。缓存复用的具体 tile 关系见[GEMM 的跨 CTA 复用](11-GEMM缓存与Reduction.md#cache-reuse)。

返回面试题：[32](../index.md#q61)
{.study-source}

## 12. 特殊函数管线的计算与供数 {#special-functions}

特殊函数管线包含 sin/cos 等，也可能负责其他数学或数据类型转换。先检查实际 SASS、相关源码、管线指标与发射限制，不能看到 70% 就断定代码反复调用三角函数。

如果同一角度多次计算，可评估复用、共享 sin/cos 计算或预计算表；表格增加 global/cache 流量，近似数学改变范围与误差。浮点范围缩减、数据转换和编译策略都要查证，用正确性及总耗时确认收益。

相关笔记：[GEMM 性能分析](../../../../../ai-infra/CUDA矩阵乘法优化：从Naive到cuBLAS.md)、[GEMM 任务与 Tile](14-GEMM切分与Persistent调度.md)、[Cache 与 Layout](11-GEMM缓存与Reduction.md)、[Prefill 与 Decode](../../inference/analysis/02-Prefill与Decode.md)。资料：[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)、[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)、[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)，查阅日期 2026-10-13。

返回面试题：[39](../index.md#q62)
{.study-source}

</div>
