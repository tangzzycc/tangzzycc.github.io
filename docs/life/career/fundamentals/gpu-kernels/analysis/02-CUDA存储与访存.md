---
title: "CUDA 存储与访存：层级、合并访问和 Bank Conflict"
---

# CUDA 存储与访存：层级、合并访问和 Bank Conflict

<div class="study-notes study-analysis" markdown="1">

来源：提供的「小鹏汽车二面」「英特尔校招面经」「阿里平头哥一面」；对应模块总文件[第 2、5～6、24～26 题](../index.md#q6)。

## 1. 四类存储的职责 {#memory-hierarchy}

| 层次 | 主要用途 | 管理与限制 |
| --- | --- | --- |
| Register | 线程操作数、局部值、累加器 | 编译器分配；使用量影响 SM 驻留，受限时可能 spill |
| Shared Memory | block 内复用与通信 | 程序安排加载、布局和同步；消耗每 block/SM 的片上预算 |
| L2 | 缓存设备内存访问，支持跨 SM 的数据复用 | 硬件管理；命中取决于访问路径和工作集，不能代替正确同步 |
| HBM / GDDR | 权重、输入输出、KV 等容量较大的数据 | 片外设备内存；容量大，带宽与延迟通常更需要摊薄 |

这是职责对应表，实际访问还可能经过 L1 等路径。Shared Memory 通常只供同一 block 协作；CUDA local memory 是线程私有地址空间，不能与 Shared Memory 混淆。

理解层级时，可以沿一个 GEMM tile 跟踪数据：A/B 的完整张量放在设备内存；CTA 把需要的子块加载到 shared；warp 再按 fragment 布局把操作数送入寄存器或相应矩阵路径；累加器跨 K 循环保留，最后输出回设备内存。L1/L2 可能减少片外读取，属于硬件缓存行为；shared tile 的内容、复用顺序和同步则由程序明确安排。

寄存器既不由线程自由按地址向邻居读取，也不是一个按线程平均分配即可用满的池。局部数组若不能静态寻址或寄存器预算不足，编译器可能将其放入 local memory。<strong class="study-key">CUDA 的 local memory 表示线程私有地址空间，物理访问可经过设备内存及缓存。</strong>
{.study-caution}

### SM 如何在访存等待期间执行其他工作？

Warp scheduler 从驻留 warp 中选择依赖已满足的指令发射。一个 warp 发起 global load 后，结果尚未返回时，后续依赖该值的指令受 scoreboard 限制；调度器可以选择其他 ready warp，也可以发射当前 warp 不依赖该加载的独立指令。前者是线程级并行 TLP，后者是指令级并行 ILP。

隐藏等待需要足够的独立在途工作。减少寄存器可能提高驻留 warp，却破坏局部复用或触发 spill；增加循环展开可提高 ILP，又扩大活跃寄存器。应看实际 eligible warp、发射和流量，而不是把寄存器或 occupancy 单独最大化。

### 高带宽为什么仍可能被访存延迟卡住？

带宽是单位时间能够搬运的数据总量，延迟是一个访问从提交到可消费所需时间。用简化稳态关系 `在途字节≈带宽×延迟`，若有效带宽为 1 TB/s、访问延迟为 400 ns，要维持这个速率约需 400 KB 在途数据。这是说明并发需求的教学估算，不是某卡固定规格。

指针追踪式加载中，下一地址依赖上一次返回，一个 warp 很难主动提供多个独立请求；即使总流量小，也可能慢。更多独立 warp、批量地址预计算或把离散索引先整理再加载，才可能提高 memory-level parallelism。已经饱和的带宽则不能通过更多在途请求继续提升。

返回面试题：[02](../index.md#q6)
{.study-source}

## 2. 合并访存看 warp 的地址 {#coalescing}

- **典型访问。** 对常见 32-byte sector 模型，32 lane 各读一个 FP32，相邻地址覆盖 128 bytes；对齐时可由四个 sector 覆盖，具体事务还取决于架构和访问路径。
- **判断方法。** 写出每个 lane 的最终字节地址，检查跨度、对齐和边界；不能只看 block 是一维还是二维。
- **向量化。** 一次加载多个元素可减少单线程指令，但跨 lane 地址若仍分散，事务利用率依然可能很低。

对 row-major FP32 矩阵 `A[M,K]`，元素地址为 `base+4(rK+c)`。固定 r，lane ℓ 访问 c=ℓ 时地址间隔为 4 bytes；固定 c，lane ℓ 访问 r=ℓ 时间隔为 4K bytes。当 K 很大时，后一种映射会让一个 warp 分散到许多 sector。这是线程到数据坐标的映射问题，矩阵本身连续存储也无法自动保证合并。

优化转置类访问时，常让生产者先按连续行搬进 shared，再由消费者按列读取。这样把片外离散访问转为片上布局问题；下一步必须解决 shared 的 bank conflict，不能把 global coalescing 和 shared 无冲突当成同一个条件。

### 只把 float 改成 float4，为什么不一定更快？

向量化影响单线程的指令宽度，coalescing 看同一 warp 的地址集合。若 lane ℓ 读 `base+16ℓ` 开始的四个 FP32，整个 warp 覆盖连续 512 bytes，满足对齐时向量指令可减少发射；若 lane ℓ 读 `base+4096ℓ`，每线程仍只使用一个离散区域，向量化没有消除区域之间的空洞。

向量指令还要求 base 和偏移对齐、尾部掩码正确；动态索引、别名或不确定对齐可能让编译器拆成标量 load。宽指令增加一次携带的数据与寄存器，不代表每个真实 shape 都更快，需检查编译结果与有效 sector 流量。

返回面试题：[05](../index.md#q7)
{.study-source}

## 3. 为什么跨步访问浪费带宽？ {#strided-access}

- **有效字节与搬运字节。** 在无命中复用的简化例子中，若每 lane 读取不同 32-byte sector 的一个 FP32，128 bytes 的有效数据对应 1024 bytes 的 sector 流量，仅使用 1/8。
- **缓存边界。** 相邻轮次若能复用剩余数据，L2 或设备内存的实际流量可能减少；以上比例不能直接当作所有层级的实测带宽效率。
- **优化方向。** 重排线程访问、先合并搬运再在片上转换布局，或调整数据存储；要计入重排和同步成本。

返回面试题：[06](../index.md#q8)
{.study-source}

## 4. Bank Conflict 的条件 {#bank-conflict}

- **简化映射。** 常见 32-bank、4-byte 模型下，FP32 元素的 bank 可写为 `floor(byte_address/4) mod 32`。
- **冲突与广播。** 同一服务事务内，同 bank 的不同地址可能需要分次服务；读同一个地址通常可广播。多个线程写同一地址还存在正确性问题。
- **宽度限制。** FP16、向量访问和矩阵加载需按真实指令拆分的事务判断；指令本就需要多个事务，不一定表示有额外冲突。

Bank 是并行服务 shared 请求的存储分区。冲突不是多个线程“访问 shared”就出现，而是同一指令被划分的同一事务中，多个地址落在相同 bank，且不能通过广播一次满足。一个 warp 访问 `tile[0][lane]`，在上述 FP32 模型下 bank 分别为 0～31；访问 `tile[lane][0]` 则因行跨度为 32，全部落在 bank 0。

实际 `ldmatrix`、128-bit 向量读写等会有专门的 lane/地址组织，并分成多个服务事务。应先确定真实指令怎样拆事务，再计算每个事务内的 bank，而不是把一个 warp 总共读了 512 bytes 就直接称为 4 倍冲突。

返回面试题：[24](../index.md#q9)
{.study-source}

## 5. Padding 与 Swizzle {#padding-swizzle}

- **闭合例子。** `tile[32][32]` 的 FP32 列访问，lane i 读 `[i][0]`，bank 为 `32i mod 32 = 0`；stride 改为 33 后，bank 为 `33i mod 32 = i`。
- **布局取舍。** Padding 增加存储和行跨度；swizzle 重新映射坐标。生产者的合并搬运和消费者的 Shared/MMA 读取都要匹配。
- **验证。** 核对生成的访问指令，比较 shared 事务、冲突指标及耗时；不能对所有 dtype 和 tile 固定加一列。
- **术语边界。** Shared-memory swizzle 改变局部地址布局；threadblock/tile swizzle 改变任务索引顺序，可能改善 L2 复用。两者解决的问题不同，见 [GEMM、缓存与 Reduction](11-GEMM缓存与Reduction.md)。

返回面试题：[25](../index.md#q10)
{.study-source}

## 6. XOR Swizzle 的完整映射例子 {#xor-layout}

对简化的 FP32 `32×32` tile，定义逻辑坐标 `[r,c]` 对应物理 `[r,c xor r]`，其中 `0≤r,c<32`。以 32-bank、4-byte 模型计算，物理 bank 为 `(32r+(c xor r)) mod 32=c xor r`。

固定 c、lane r 读取一列时，bank 随 r 遍历 0～31，所以该标量事务无冲突。固定 r 读一行时，c 的 XOR 映射也是排列。生产者写和消费者读都用同一映射即可返回正确逻辑值；没有必要先复制一份恢复为普通 row-major tile。

逆映射为 `c=physical_col xor r`，这是坐标关系，不意味着向量加载或 MMA 自动执行逆映射。真实矩阵指令的 lane/fragment、访问宽度、对齐和事务拆分可能让此简化布局不适用。Padding 对常见转置列读有效，也不能保证所有消费者无冲突；需核对生成指令与 profiler。

### Swizzle 为什么要同时检查写入与读取两条路径？

XOR 只是坐标置换；它可以改善一种消费者映射，也可能破坏生产者向量写的连续性。设计时先列出 global→shared 的线程地址，再列出 shared→fragment 的 lane 地址，分别检查 global sector、shared bank、向量对齐和矩阵指令要求。前面 `[r,c xor r]` 的例子只证明特定 FP32 标量访问，不能直接照搬到 `ldmatrix` 或低 bit packed operand。

对常见 4-byte bank 模型，两个 FP16 元素共享一个 bank-word；同一个 bank 的相同 word、不同半字，与两个完全不同 word 的服务方式也要按指令核对。128-bit shared load 会拆成多个事务，应逐事务判断冲突。所谓“32 bank 就必然需要 stride 33”只适用于某些访问，padding 还增加 stage 空间，可能跨过驻留阈值。

相关笔记：[CUDA GEMM 的访存与布局](../../../../../gpu-kernel/GEMM-CUDA-Core优化.md)。

返回面试题：[26](../index.md#q51)
{.study-source}

</div>
