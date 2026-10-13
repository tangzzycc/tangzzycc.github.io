---
title: "GEMM、缓存与 Reduction：复用、映射和归约范围"
---

# GEMM、缓存与 Reduction：复用、映射和归约范围

<div class="study-notes study-analysis" markdown="1">

来源：提供的「英特尔校招面经」「B站 AI Infra 二面」图片与记录；对应 GPU与算子优化[第 1、31～32、46 题](../index.md#q42)，复用[第 3、5～6、8～12、24～25、27～30、34～35 题](../index.md#q16)的访存、Split-K、双缓冲与性能分析。

## 1. GEMM 优化是相互制约的迭代 {#gemm-workflow}

| 步骤 | 要做的决定 | 验证内容 |
| --- | --- | --- |
| Roofline 与基线 | 实际 dtype/指令、FLOPs、各存储层流量、瓶颈 | 耗时、计算利用、带宽及等待 |
| Tiling | M/N/K tile、block/warp 分工、复用 | 并行任务数、尾部浪费、资源预算 |
| 数据搬运 | 合并加载、shared 布局、寄存器暂存、多 stage | 有效事务、bank conflict、异步依赖 |
| 计算映射 | Scalar/SIMD 或矩阵指令、fragment 与 epilogue | 指令路径、供数、结果布局 |

以 `A[M,K]B[K,N]` 为例，一个 `BM×BN` 输出 tile 每 K tile 约做 `2BM×BN×BK` 次运算，理想加载 `BM×BK+BK×BN` 个输入元素。扩大 BM/BN 可提高局部复用，却增加累加器和片上资源，并减少 grid 并行度；BK 和 stage 数也要共同调整。

一个输出 tile 沿 K 的局部输入强度，在输入元素均为 s bytes、忽略输出流量时，约为 `2BM·BN/[s(BM+BN)]`。若 BM=BN=64、s=2，强度为 32 FLOP/byte；增为 128×128 时为 64。BK 在这个理想比值中约掉了，所以增大 BK 不直接提高这里的输入复用比，它改变的是循环、搬运粒度、空间与尾部效率。

<strong class="study-key">选 tile 要同时看复用比、累加器量和输出任务数。</strong> 只看这一个强度公式会遗漏 grid 并行和 shared/register 的阈值，实际 L2 复用也会使 HBM 口径的强度不同。
{.study-core}

返回面试题：[01](../index.md#q42)
{.study-source}

## 2. 如何改善 L1/L2 复用？ {#cache-reuse}

- **复用距离。** 让重复使用的数据在被大量其他数据挤出前再次访问，例如分块、融合或调节任务分组顺序。
- **工作集。** 控制 tile 和并发数据规模，减少一次性流量污染；L1 通常按 SM 更局部，L2 可供跨 SM 访问复用。
- **布局。** 合并加载提高 sector 中的有效字节，但它与缓存命中是两个指标；离散访问即使命中也可能有较多指令与事务。
- **策略。** 支持的访问路径、设备可评估缓存 hint 或持久化 L2 策略；这是偏好和资源管理，不能保证数据一直驻留。
- **调度。** 程序可设计逻辑 tile 顺序，实际 block→SM 映射和完成顺序不能随意假定；正确性不应依赖缓存驻留。

如果请求数从 100 增到 200、命中率从 50% 增到 70%，理想等大请求的 miss 仍从 50 增到 60。命中率提高并不保证片外流量下降，更不保证耗时下降。应同时看请求/sector、实际设备内存字节与正式计时，区分冷缓存和热缓存测量。

### 为什么 tile 顺序影响 L2，不能保证 L1？

输出 C 的 `(m,n)` tile 读取 A 的 m 行块和 B 的 n 列块。先处理同一 m 的多个 n tile，可缩短 A 的跨 CTA 复用距离；先处理同一 n 的多个 m tile，则更利于 B。哪个更有价值取决于 A/B 大小、tile、并发和实际调度。逻辑任务 grouping 给硬件更有局部性的工作，不保证它们按固定顺序或落在同一个 SM。

L1 更局部，同一 SM 的访存路径和工作集决定复用；L2 可跨 SM 复用，但容量、替换和其他流量仍会驱逐数据。连续加载提高有效 sector 比例，缓存复用缩短读取距离，它们可以互补，不能混为一项。Cache hint 是偏好，不能作为正确性或永久驻留保证。

返回面试题：[31](../index.md#q41)
{.study-source}

## 3. Reduction 的协作范围 {#reduction-scope}

1. **线程局部。** 多元素累积，访问映射尽量合并，减小后续部分和数。
2. **Warp。** Shuffle 合并局部值，遵守参与 mask，尾部使用正确单位元。
3. **Block。** 各 warp 输出部分值到 shared，block 同步，再由一个 warp 完成最终合并。
4. **Grid。** 各 block 写 workspace，第二个 kernel 归约；或选择适合的 atomic/cooperative 路径，并满足对应约束。

Sum 的单位元为 0，max 为负无穷。整数要考虑溢出与实际累加类型，浮点要考虑顺序和容差；跨 block 无普通 `__syncthreads()` 同步，不能自行做可能死锁的全 grid 等待。

### 两阶段 grid 归约为什么比“所有 block 等一个 flag”安全？

普通 launch 不保证全部 CTA 同时驻留。先驻留 CTA 若自旋等待其他尚未被调度的 CTA，会占住全部资源，使后者永远无法执行，形成死锁。两阶段方案让第一 kernel 写所有部分和并结束，在同 stream 依赖下第二 kernel 再读取，kernel 边界提供清楚的全 grid 执行顺序。

Atomic 可以汇总结果但需要初始化、数值范围与完成边界；某些协作式 launch 支持 grid 同步，则必须满足该机制的驻留/启动约束。块内 `__syncthreads()` 不能跨 block，warp shuffle 更不能跨 warp。最终归约顺序还影响浮点精度和可复现性，见[数值顺序](13-直方图选择与数值归约.md#numerical-order)。

返回面试题：[46](../index.md#q43)
{.study-source}

## 4. Swizzle 的两种语境 {#swizzle-types}

Shared-memory swizzle 改变局部坐标到地址的映射，用于 bank 布局；tile/threadblock swizzle 改变任务索引顺序，可能改善缓存局部性。二者不能混同，布局必须同时满足生产者、消费者和矩阵指令要求。

## 5. Layout 要覆盖前后算子与 Launch {#operator-layout}

生产者可以直接写出下游合适的 packed/tiled 格式，减少转换，并改善合并加载或 tile 复用；但要计算生产者写入、下游读入、额外重排和保存空间的总成本。对齐 Load Size、线程映射、quantization group 和 MMA fragment 需共同匹配。

多次 launch 在同一设备访问相同数据时，L2 可能保留数据，但正确性不依赖驻留，也不能保证跨 kernel 同 SM 的 L1 行为。单 kernel 的冷缓存 benchmark 与真实算子链的热缓存条件可能不同；NCU 的 replay/cache control 也会影响测试，需与正常执行对照。

优化要解释是在减少实际数据读写、缩短复用距离，还是只提高有效事务比例；同时比较完整算子链耗时，避免让前一个 kernel 变慢以换取后一个孤立测量变快。

相关分析：[存储、合并访问与 Bank Conflict](02-CUDA存储与访存.md)、[流水线与 Split-K](03-Kernel流水线与矩阵计算.md)、[Safe Softmax Warp 归约](08-SafeSoftmax与Warp归约.md)、[NCU 指标解读](04-Kernel性能分析.md)。

</div>
