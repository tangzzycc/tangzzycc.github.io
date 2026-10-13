---
title: "GEMM 切分与 Persistent 调度：任务量、Tile 和持续性能"
---

# GEMM 切分与 Persistent 调度：任务量、Tile 和持续性能

<div class="study-notes study-analysis" markdown="1">

来源：提供的「B站 AI Infra 二面」图片；对应 GPU与算子优化[第 7、13～14、23、40 题](../index.md#q54)，复用[第 1、3～4、8～12 题](../index.md#q42)。图片中的选型口诀和个人 TOPS 数字不作为通用性能规律。

## 1. 普通 K 循环与 Split-K 不同 {#split-dimensions}

普通 tiled GEMM 将 `C[M,N]=A[M,K]B[K,N]` 的输出按 M/N 分块。每个 CTA 负责一个输出 tile，并沿 K 循环累加；这是 GEMM 的普通分块，不需要另加 Split-K 才能执行。

| 切分维度 | 每份工作 | 结果关系 | 主要取舍 |
| --- | --- | --- | --- |
| M | 不同输出行/行块 | 输出区域互不重叠 | 输出任务与输入 B 的跨 tile 复用 |
| N | 不同输出列/列块 | 输出区域互不重叠 | 输出任务与输入 A 的跨 tile 复用 |
| K | 同一输出的不同 K 范围 | 部分和必须归约 | 增加并行，新增归约与流量 |

单卡 Split-M/N 常指调整输出 tiling 或任务划分，不一定是额外的独立多 kernel 算法。跨 GPU 切分又涉及 TP 等布局和通信，不能将单卡 tiling 与分布式划分混用。

返回面试题：[13](../index.md#q52)
{.study-source}

## 2. 先算任务量，再选切分 {#task-count}

输出 tile 数为 `ceil(M/BM)×ceil(N/BN)`，还需结合 SM 数、每 SM 可驻留 CTA 与任务耗时，判断是否能提供足够工作。

例如 M=128、N=256、K=8192，BM=BN=128，仅有 2 个输出 CTA；在 SM 数量很多的设备上，K 长但输出任务少。减小 BM/BN 可增加任务但降低复用；Split-K=8 可增加到 16 份部分和任务，仍要计归约和小分段效率，未必已经充足。

若 M=N=4096、同样 128×128 tile，则有 1024 个输出 CTA，此时 K 大本身不构成使用 Split-K 的理由。小矩阵也可能需要专门的小 M/GEMV、分片或 persistent grouped 路径，并非只要“小”就使用一个固定通用 kernel。

可按 shape、dtype、架构和真实性能建立分发策略，先保留最简单正确基线，再比较不同 tile、Split-K、batched/grouped 与调度方式。Bias、残差和非线性 epilogue 要放在完整部分和合并后，见 [Split-K 的语义](03-Kernel流水线与矩阵计算.md)。

## 3. BM/BN/BK 与 Stage 是联合参数 {#tile-resources}

理想每个 K tile 的运算量为 `2BM×BN×BK`，输入载荷为 `(BM×BK+BK×BN)×输入字节数`。设 stage 数为 s，则 shared 输入预算粗估为 `s×BK×(BM+BN)×输入字节数`，另计 padding、元数据和状态。

BM/BN 放大可提高输入复用，但累加器随 BM×BN 增长，输出任务变少。BK 增大通常减少循环次数，也增大 stage 空间和末尾浪费。Stage 增加有助于隐藏供数等待，却可能降低驻留，生产消费同步也更复杂。

例如 FP16、BM=BN=128、BK=32、s=2，理想 shared 输入为 32 KiB；BK=64 时变为 64 KiB，而逻辑输出 FP32 累加器仍为 64 KiB，实际寄存器布局依指令和线程分工。不能只按 BK 变大、循环减半就预测加速。

### Tile 边界和 wave 数如何形成突然的性能变化？

输出 grid 为 `ceil(M/BM)×ceil(N/BN)`，若 M=129、BM=128，则第二个行 tile 只有 1/128 行有效，但仍可能承担近似完整 CTA 的调度与部分计算。Padding 和 predicate 可以保护地址正确，却不必然跳过矩阵引擎全部无效运算。

设设备有 G 个 SM、每 SM 适合同时驻留 r 个 CTA，grid 共 C 个 CTA，粗略执行 wave 数为 `ceil(C/(Gr))`。C 刚好越过一个 wave 边界时，最后少量 CTA 可能形成明显尾部；这是均匀任务教学模型，实际 CTA 动态调度与变长任务会改变时序。

更大 BM/BN 既减少 C，又增加寄存器/每块 shared，可能让 r 从 2 变成 1；tile 复用提升与 wave 效率下降会同时发生。调参应围绕实际本地 shape、量化组边界和资源报告，不能靠某组固定 tile 对所有 LoRA/GEMM 通用。

返回面试题：[07](../index.md#q54)
{.study-source}

## 4. Persistent GEMM 是循环分配输出任务 {#persistent-gemm}

普通 kernel 常让一个 CTA 处理一个输出 tile，后续 CTA 再由硬件调度；persistent 则 launch 数量受控的 CTA，各 CTA 在同次 kernel 生命周期中取多个 tile。

例如输出任务编号 0～99，使用 8 个 CTA：静态方案可让 CTA b 处理 `b,b+8,b+16,…`；动态队列可以领取下一任务，需增加队列/atomic 或调度开销。每个任务必须被恰好处理一次，累加器在新输出前重置，旧 tile 的搬运/计算完成后才复用槽。

潜在收益是摊薄 CTA prologue、复用流水线状态、减少任务尾部或改善 grouped 工作分配；但静态顺序不保证负载均衡，也不保证数据驻留 cache。受控 CTA 不意味着“只有少量线程”就能满足矩阵供数，CTA 内仍需足够协作。

普通 GEMM 本来就一次 launch 处理整个矩阵。因此 persistent tiled GEMM 的收益常在 block 初始化与任务组织，而非必然减少 kernel launch；另有长期驻留处理多个外部任务的服务模式，才可能合并多次 launch，需要独立的队列、可见性和退出协议。

Persistent 并不赋予普通 kernel 固定 block→SM 绑定；占用设备资源也可能影响其他 kernel 的并发与在线延迟。是否采用要在实际 workload 下验证。

返回面试题：[14](../index.md#q53)
{.study-source}

## 5. Sustained TOPS 的计量口径 {#sustained-tops}

稠密 GEMM 常将一个乘加计为两次操作：

$$
\mathrm{TOPS}_{dense}=\frac{2MNK}{t_{sec}\times10^{12}}.
$$

例如 M=N=K=1024、耗时 0.3 ms，则约 7.16 TOPS；这是教学算例，不是某 Ampere 卡的规格或图片数字的还原。

必须说明是设备主计算 kernel，还是包含解包、scale、归约、epilogue、launch 的全部算子时间。若 50% 稀疏仍用原稠密操作量作分子，是 dense-equivalent TOPS；实际非零操作数另有口径，不能用高等效值证明完成了相同数量的真实乘法。

Sustained 还要记录预热、持续负载、功耗模式、频率和温度；整模型 TOPS 也不能由单 shape 峰值代替。与规格峰值比较时，dtype、稀疏条件和操作计数必须匹配。

返回面试题：[23](../index.md#q55)
{.study-source}

## 6. 从 Ampere 移到 Blackwell 要重做假设 {#architecture-retuning}

先区分运行失败、兼容但慢、绝对速度提高却占峰值比例降低这三类现象。确认目标编译、SASS、实际矩阵格式和同设备成熟基线；不同 Blackwell 产品和 PTX 路径的低精度支持需逐项核对，不假设原 INT4 路径全部原生高效。

旧 tile 可能无法给更多 SM 足够任务，新计算/带宽比会让原供数、解码或同步暴露为瓶颈。适合新架构的 TMA、矩阵机制、TMEM 和角色划分可能提供更好路径，但不是只改指令名字即可；FP4 与 INT4 编码也不能互换。

以同 shape、同精度和同计时边界比较，结合供数、算术、任务量与资源 profile 调整，见 [架构机制](06-Ampere-Hopper与Blackwell.md)、[W4A4 执行契约](07-W4A4与稀疏元数据.md)。

资料：[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)，查阅日期 2026-10-13。

返回面试题：[40](../index.md#q56)
{.study-source}

</div>
