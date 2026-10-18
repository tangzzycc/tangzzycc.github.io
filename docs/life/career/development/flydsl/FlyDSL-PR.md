---
title: "开源贡献面试：FlyDSL PR"
---

# 开源贡献面试：FlyDSL PR

围绕简历中的 FlyDSL 开源贡献，按 GPU Kernel 与编译执行链路的技术面组织问题。每题分「简要回答」和「完整阐述」，分别用于首次口述与深入追问，回答采用第一人称。

本文整理于 2026 年 10 月 9 日，依据已有 PR 与固定源码笔记。以下两个 PR 在仓库 2026 年 10 月 5～6 日的记录中仍在进行。

| PR | 贡献内容 | 源码笔记版本 |
|---|---|---|
| [#1057](https://github.com/ROCm/FlyDSL/pull/1057) | gfx1151 GEMM tile policy、设备架构传递与 JIT/cache 适配 | `ddd2040` |
| [#1113](https://github.com/ROCm/FlyDSL/pull/1113) | Fused RoPE 的 head dimension 96 向量宽度与 lane 配对修复 | `f4fa09f` |

## 1. 你在 FlyDSL 的贡献是什么？它和你做 HIP kernel 的工作有什么联系？

### 简要回答

我主要做了两项 RDNA 适配：为 gfx1151 增加按矩阵 shape 选择 GEMM tile 的策略，并让真实设备架构贯穿 JIT target 和缓存；另外修复 wave32 下 D=96 fused RoPE 的非法三元素向量及 12-lane half 的错误 XOR 配对。FlyDSL 提供 Python 的布局和编译抽象，但仍需要理解线程数据分配、位宽、WMMA、资源和最终 ISA，这些判断与 HIP kernel 开发相通。

### 完整阐述

FlyDSL 允许用 Python 描述 GPU tile、线程和值的映射、global/LDS/register copy 以及矩阵操作，再通过 MLIR 和 LLVM 生成 AMDGPU code object。它提高了表达和组合能力，但具体硬件的 wave size、向量搬运与资源约束仍然存在。

PR #1057 的问题是 RDNA GEMM 默认使用为 gfx1100 调整的策略，其中部分阈值建立在 96 CU 上，而这次 gfx1151 测试设备只有 40 CU。我增加了范围收敛的 shape-based policy，并把输入 tensor 所在设备的架构一直传递到 build、launcher signature 和 JIT target，避免选对 tile 却编译错架构，或跨架构错误复用 launcher。

PR #1113 的问题是 fused RoPE 在 wave32、head dimension 96 时计算出 width 3，但底层向量 copy 和 32-bit shuffle 重建不支持这条路径；改成原生 width 4 后，两半各占 12 lane，旧 XOR 配对仍然错误。因此修复同时约束 vector width 与 NeoX 两半配对。

这两项分别体现性能和正确性工作。Tile policy 在十个发生选择变化的 shape 上，报告 BF16 几何平均约 1.199 倍、FP16 约 1.164 倍加速。D=96 原本无法正常运行，所以新路径报告的是正确性、资源与绝对延迟，而没有可运行旧版本的相对加速。

从 HIP 转到 DSL，问题仍然是「谁持有哪些值、值如何移动、哪条指令消费它们」。此外需要核对 Python tracing、IR 类型与 target/cache 的一致性，而不是只看高层数值表达式。

**可能追问：#1057 是新写了一套 GEMM 算法吗？**

这项改动主要选择已有 RDNA3 WMMA GEMM 的 tile 配置，并完善架构传递。已有 kernel 的数学与主体流水线继续复用，贡献边界应说清是 policy、编译目标与缓存适配。

背景见[FlyDSL 编程与编译流程](notes/01-FlyDSL编程模型与编译流程.md)。

## 2. GEMM 的 tile 五元组表示什么？它为什么会影响性能？

### 简要回答

五元组是 reg_m、reg_n、reg_k、waves_m、waves_n。WMMA 原子为 16×16×16，因此 block tile 的 M/N 分别是 16 乘寄存器 tile 数和 wave 网格，K 是 16×reg_k。它同时改变输出 workgroup 数、wave 数、寄存器 accumulator、LDS 和 copy 几何。大 tile 提高局部复用，但可能减少全设备并行或增加资源压力，所以要先过滤合法配置，再按硬件与 shape 测量选择。

### 完整阐述

PR 的 selector 返回 `(reg_m, reg_n, reg_k, waves_m, waves_n)`。底层 WMMA 原子尺寸固定为 16，workgroup 的逻辑 tile 是：

$$
B_M=16\,reg_m\,waves_m,\qquad
B_N=16\,reg_n\,waves_n,\qquad
B_K=16\,reg_k.
$$

以 `(4,4,2,2,2)` 为例，得到 `128×128×32`，wave 网格为 2×2，共四条 wave，wave32 下为 128 个线程。

这组参数既描述逻辑工作范围，也决定线程组织和数据分布。M/N tile 更大，A/B 可被更多输出复用，但输出网格可能更小；reg_m/reg_n 更大还可能增加每个 wave 的 accumulator，压缩可同时驻留的 wave。K tile 增大可能改变预取与 LDS buffer 负担，因此不能只按单次 WMMA 数量判断快慢。

当前实现先过滤 feasible tiles，检查 M/N/K 整除、K 至少有两个 block-K tile、A/B 元素能按完整八元素 vector 分配给线程，以及双 buffer 的 LDS 不超过 64 KiB。这些是该 kernel 的构建契约，其他实现可能用 tail predicate 支持非整除 shape，不能推广成所有 GEMM 的限制。

通过过滤只说明能够运行，随后 policy 才根据工作量、网格深度和架构选择候选。默认布局下一个 `256×256×32` BF16 A/B tile 的数据载荷已经很大；其特定 kblock LDS layout 才能在当前预算内组织资源。因此也不能只用 BM/BN/BK 乘积替代真实 LDS 分配。

**可能追问：能否仅靠公式选出最快 tile？**

公式可以排除不合法配置并解释权衡，但缓存、指令调度、寄存器和每个 workgroup 的执行时间需要测量。Policy 根据已测 shape 家族选择，不能把可行性检查当成性能结论。

定义与合法性见[GEMM tile policy 笔记](notes/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)。

## 3. gfx1151 为什么需要单独的 tile policy？把 96 CU 改成 40 CU 不就够了吗？

### 简要回答

原阈值描述的是网格相对 gfx1100 的并行深度，不能原封不动套给 40-CU gfx1151。简单按 40/96 缩放也无法反映 tile 的资源、复用和执行时间变化。我用 benchmark 选择有限 shape 家族：小矩阵用更小 tile 增加网格，接近方形且大 tile 产生 32～40 个 workgroup 时采用 256 tile，深网格等配置使用其他已测选择，范围外继续 fallback。

### 完整阐述

原 gfx1100 heuristic 中，一些候选门槛是若干倍的 96 CU，例如要求特定 tile 产生至少几轮 workgroup。它反映的是一台设备上的并行深度，不能把 96 当作矩阵数学的一部分。

测试的 Radeon 8060S/gfx1151 为 40 CU。相同 shape、相同 tile 在这台设备上对应不同的网格深度，且 tile 的执行时间、LDS、wave 数和复用不随 CU 数线性改变。因此我针对该设备的已测 shape 增加独立策略。

例如 1536×1536 的输出，用 256×256 tile 产生 `6×6=36` 个 workgroup，接近设备的一轮分配；它同时满足近方形和策略的 32～40 workgroup 区间，采用 256×256×32。这个统计只是 grid 相对 CU 的规模，不能承诺运行时恰好一个 block 驻留每个 CU。

6144×6144 用同一 tile 会产生 `24×24=576` 个 workgroup，已不在该窄区间，已测策略选择 128×128×32。256³ 的总量又较小，特判为 32×64×64，通过较小 M tile 提供更多 workgroup。

策略还对 128 tile 的网格定义区间，并对部分长 K shape 保留 64×64×64。名为 `one_wave` 的区间是接近一轮 workgroup 分配的规模判断，不是硬件 wavefront。硬件 wave size 仍是 32。

我让这些规则只覆盖经过测量的 shape family，其他配置退回原 heuristic。这样既利用了设备专属证据，也避免把少量方形测试推广到所有高度矩形 shape。

**可能追问：为什么不用每次运行都遍历所有 tile 的 autotuning？**

本 PR 交付的是确定性的 shape policy。在线遍历会引入候选构建、warmup 和计时成本，还涉及结果缓存与服务启动行为。离线测量可以支持轻量选择；若后续增加动态 tuning，需要单独设计生命周期和缓存，而不是把当前 selector 描述成在线 exhaustive search。

具体区间与例子见[gfx1151 tile 选择笔记](notes/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)。

## 4. 为什么 tile policy 的 PR 还要修改 JIT target 和 cache？

### 简要回答

设备专属选择必须贯穿编译与执行。Policy 按 gfx1151 选 tile，但 JIT 若从另一默认设备解析 target，就可能生成错误架构的 code object；上层 build/launcher cache 若不含 arch，也可能先返回另一架构的对象。我从 A.device 读取真实 gcnArchName，传入 policy、build 参数和 launcher signature，再通过可选 jit arch 解析 target，利用 JIT key 的 target 区分产物。

### 完整阐述

进程的当前 device 与输入 tensor 所在 device 可能不同，尤其在多设备环境中。如果只查询全局默认设备，选择与编译的依据就可能与真正执行设备不一致。

该路径从 `torch.cuda.get_device_properties(A.device.index).gcnArchName` 得到架构，规范 feature suffix 后用于 selector。Arch 随后进入 `_build` 的缓存参数、resolved launcher signature，并作为 `@flyc.jit(arch=gpu_arch)` 的显式 target 输入。

执行链可以概括为：

```text
输入 A 所在设备
  → 架构解析
  → tile policy
  → arch-specific build / launcher cache
  → JIT backend target
  → target-specific compiled artifact
  → 提交到实际设备的 stream
```

JIT cache key 原本已经包含 target；新增 arch 参数让 resolved target 根据调用选择的架构确定。但只修底层 key 不足以保证所有层正确：上层缓存可能在到达 JIT 之前就返回已建 launcher，所以 build 与 launcher 的身份也需要携带 arch。

反过来，只把 arch 加入上层 key、JIT 仍按默认设备编译，也只能得到「分别缓存的错误产物」。选择、缓存和 codegen 的架构必须一致。

这是内容身份的问题：相同 dtype、shape、tile 和 kernel 源码，在不同 GPU target 上不一定对应同一二进制。持有不同对象本身不够，还要验证它们实际解析和编译的 target。

**可能追问：加 arch 参数会不会让每次调用都重新编译？**

Arch 是 specialization 身份的一部分。同一架构与相同编译签名继续复用产物，只有不同架构需要区分。当前 tensor adapter 还有动态 shape/layout 的规则，不能概括成每组 tensor shape 都必定编译一次；tile 等结构常量与 target 才是需要明确区分的配置。

**可能追问：单卡测试能证明多 GPU 全部正确吗？**

设备解析和 arch-specific cache 测试可以检查这条信息流；本地 gfx1151 benchmark证明该设备路径。它们不能替代所有异构多卡的真实运行，需要按已有验证范围介绍。

JIT/cache 细节见[tile policy 笔记](notes/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)与[编译流程笔记](notes/01-FlyDSL编程模型与编译流程.md)。

## 5. 你怎么证明新 policy 更好？为什么用几何平均，又有什么边界？

### 简要回答

我保持 kernel 与 per-tile options 相同，对照新旧 policy 的选择，并分别测试合法性、数值、策略边界和架构缓存。gfx1151 的十个发生 tile 变化的 shape 上，BF16 几何平均约 1.199 倍，FP16 约 1.164 倍。几何平均适合汇总乘法加速比，但这个集合是变化 shape 的子集，不能写成所有 GEMM 或模型整体提升，更不能推广到全部 RDNA 设备。

### 完整阐述

首先要让实验回答明确的问题：在同一设备、同一 kernel 主体、同一 per-tile 配置规则下，policy 选出的 tile 是否优于旧 heuristic。这样才不会把 kernel 改写或其他参数变化混进 tile 选择收益。

已有 PR 在 Radeon 8060S/gfx1151、40 CU、记录中的 ROCm 7.14 环境测试，报告十个选择发生变化的 shape。代表性结果如下：

| M×N×K | 旧 tile | 新 tile | BF16 加速比 | FP16 加速比 |
|---|---|---|---:|---:|
| 256³ | 64×64×64 | 32×64×64 | 1.158× | 1.048× |
| 896³ | 128×64×32 | 128×128×32 | 1.545× | 1.423× |
| 1536³ | 64×64×64 | 256×256×32 | 1.115× | 1.121× |
| 6144³ | 256×256×32 | 128×128×32 | 1.216× | 1.210× |

每个加速比为基线耗时除以新耗时。多个比值用几何平均 `exp(mean(log(r_i)))` 汇总，适合反映乘法比例；它不是所有 shape 总耗时的加权收益，也没有考虑真实模型中各 shape 出现频率。

因此「BF16 几何平均约 1.199×」需要保留十个 changed shapes 的口径。不能解释为模型吞吐提升 19.9%，也不能保证范围外 shape 同样改善。若用于模型部署，还要在实际 Linear 频率与 shape 分布上验证。

正确性还包括每个候选是否能构建和数值是否正确。Policy 边界、held-out shapes、设备架构解析与 build cache 测试分别检查：规则没有选择非法 tile、范围外行为合理、arch 不发生错配。性能、数值、编译目标和缓存身份是不同验证对象。

**可能追问：如果某个新 tile 更快但数值不对，你会接受吗？**

不会把它纳入正常分发。先检查 layout、同步、tail 或实际编译 target，确认满足数学和执行契约后再讨论速度；性能测量必须建立在正确输出上。

数据与测试范围见[性能证据笔记](notes/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)。

## 6. D=96 为什么不能让 32 个 lane 每人处理三个元素？只改 vector width 就够了吗？

### 简要回答

数量能分完不代表底层指令能搬完。旧公式在 wave32 上得到 width=3，三个 BF16 共 48 bit，但 copy 与 32-bit bpermute 重建不支持这条向量路径，出现两元素和三元素 shape 不匹配。我从原生 1、2、4、8 中选择同时整除 head 与 NeoX half、且一波能容纳的最小宽度；D=96 选 4，用 24 个 lane。两半各 12 lane 后，还要修复非二次幂跨度的配对，所以有两个独立改动。

### 完整阐述

Fused kernel 一条 wave 处理一个 token/head，对 Q/K 做 NeoX RoPE，同时将 K/V 写入 cache。旧宽度公式是 `ceil(head_dim/wave_size)`，D=96、wave32 时得到 3，表面上 32 lane 正好覆盖 96 个值。

但 FP16/BF16 三元素向量是 48 bit。跨 lane helper 以 32-bit word 搬运，旧重建逻辑按 `VEC_WIDTH//2` 只能得到一个 i32，即两个 16-bit 元素，导致 `(2,)` 与 `(3,)` 无法广播。原生 copy 也有向量位宽约束，不能根据总元素数整除就认定 width 3 可用。

新选择限定候选 w 为 1、2、4、8，并检查：

```text
D % w == 0           # head 完整切片
(D/2) % w == 0       # NeoX half 边界不切穿 vector
D / w <= wave_size   # 一条 wave 能分配全部片段
```

D=96、wave32 的 w=1、2 无法在一波容纳，w=4 合法，得到 24 个 active lane，每半 12 个 lane，另外八个 lane 不读写有效元素。这个分配宁可有 inactive lane，也要满足 load/store 与 shuffle 的完整数据表示。

跨 lane helper 将 width 4 的 BF16 vector bitcast 成两个 i32，分别用相同 peer address 交换，再拼回四元素向量。Width 1 的 16 bit 值扩展成一个 word；width 2、4、8 分别需要 1、2、4 个 32-bit 交换。

这里的 `ds_bpermute_b32` 按 byte address 指定 peer lane，搬的是它的寄存器值；虽然属于 LDS cross-lane 指令族，不能据此说 kernel 使用了一块 LDS scratch buffer。

改宽度解决向量重建后，NeoX 前后两半的跨度变为 12 lane，旧 XOR 又出现配对问题。因此不能只确认 kernel 能编译，还必须检查旋转伙伴是谁。

**可能追问：为什么要额外检查 D/2 也能被 w 整除？**

NeoX 的前半元素需要与后半同位置元素组成向量对应关系。如果一个 vector 跨过 half 边界，简单的整 vector peer exchange 不能再代表同一组配对，必须改变算法。当前方案用整除约束让两半各由完整 vector 构成。

向量约束见[Fused RoPE D=96 笔记](notes/FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)。

## 7. NeoX RoPE 的 12-lane 配对为什么不能用 XOR？给一个具体索引例子

### 简要回答

NeoX 前后半区配对相隔 D/2 个元素。D=96、width 4 时，每半占 12 lane，前半 lane 应加 12，后半应减 12。XOR 只有在当前布局的 half span 是二次幂时才能通过翻转一位实现这种对称交换。比如 lane 4 的伙伴应为 16，4 XOR 12 却是 8，仍在前半区。我对二次幂继续用 XOR，其余跨度显式加减。

### 完整阐述

NeoX 风格把 head 向量分为等长两半，前半元素 i 与后半元素 `i+D/2` 配对：

$$
y_i=x_i\cos\theta_i-x_{i+D/2}\sin\theta_i,
$$

$$
y_{i+D/2}=x_{i+D/2}\cos\theta_i+x_i\sin\theta_i.
$$

在当前连续 vector 分配中，每个 lane 负责 w 个元素。D=128、w=4 时，每半 16 lane，前半为 0～15，后半为 16～31，`lane XOR 16` 翻转 half 位，得到正确伙伴。

D=96、w=4 时，前半 lane 为 0～11，后半为 12～23。12 的二进制不是单个置位，XOR 12 不是「移动 12 个位置」。例如 lane 4 XOR 12 得到 8，所交换片段仍属于同一个 half；正确伙伴是 lane 16。

新映射在 half span 为二次幂时沿用 XOR，其余用第一半加 span、第二半减 span。这在当前两段连续 lane 布局中实现一一对应。

以 lane 3 为例，它负责前半元素 `[12,13,14,15]`，peer lane `3+12=15` 负责后半 `[60,61,62,63]`，恰好每个位置相差 D/2=48。两边交换相同的四元素片段，前半使用负 sine 项，后半使用正 sine 项。

Q 和 K 均执行同一旋转关系，V 不旋转，只进入 cache write。Cos/sin 对应相同 pair 的位置，不能因为 peer lane 改变就读取不匹配的角度。

**可能追问：能不能对所有跨度都改成加减？**

数学上可以在当前布局构造等价配对，但原二次幂 specialization 已有高效 XOR 路径。我保留它，让 D=64、128、256 的常见路径继续使用原分支，并用生成 ISA 对照确认旧 specialization 没有变化。这个选择兼顾新维度支持和已有路径兼容性。

Lane 到元素的完整对应见[RoPE 配对笔记](notes/FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)。

## 8. 你怎样验证 Fused RoPE 修复？为什么只比较 Q 不够，又为何不能报告 D=96 加速比？

### 简要回答

Kernel 同时输出旋转后的 Q、K 和更新后的 key/value cache，所以要逐项对照 PyTorch reference，覆盖 BF16/FP16 和两种 cache layout。完整记录为 74 passed、248 skipped；已有 D=64、128、256 在 gfx1151 上的 ISA 与基线一致。D=96 旧实现无法运行，没有可比耗时，只能报告新路径绝对延迟与资源。原生 wave32 验证也不能自动替代所有 wave64 硬件测试。

### 完整阐述

Fused kernel 有四类可观察结果：Q_out、K_out、key cache 和 value cache。Q 正确不保证 K 的配对、slot mapping 或 cache 布局正确；V 虽然不旋转，写地址仍可能有问题。

因此 D=96 新用例固定 32 个 Q/KV head，覆盖 BF16 与 FP16、flash 与非 flash cache layout，检查所有输出和缓存。正常 slot 写入旋转后的 K 与原 V，负 slot 跳过 cache write，但 Q/K 输出仍然需要正确。

执行边界是 NeoX-style、full rotary dimension、FP16/BF16，以及当前已支持 cache layout。Partial RoPE、GPT-J 相邻配对或任意 dtype 不能从这组测试推导为已经支持。记录的 74 passed、248 skipped 也需要保留 gate 范围，skip 不代表验证通过。

对既有 D=64、128、256，half lane span 仍为二次幂，编译期继续选择原 XOR 分支。PR 对比 gfx1151 的最终 ISA，报告 byte-for-byte 一致，这说明新增静态条件在旧 specialization 中没有改变指令流。它对兼容性提供了比单次近似 benchmark 更直接的证据，但不扩大到其他所有架构。

新 D=96 路径在 Radeon 8060S 原生 wave32、BF16、QH=KH=32 的 graph replay 延迟如下：

| Tokens | 记录中的延迟 |
|---:|---:|
| 1 | 4.025 μs |
| 128 | 10.637 μs |
| 512 | 35.252 μs |
| 2048 | 456.657 μs |
| 4096 | 909.765 μs |

记录中 kernel 使用 35 VGPR、26 SGPR，没有 LDS 分配、scratch 或 register spill。它描述资源形态和已测绝对耗时，不能在没有旧 D=96 可运行基线时声称加速百分比。

**可能追问：无 spill、VGPR 较少就证明 occupancy 足够或性能最优吗？**

这些属性排除了一些资源问题，但实际并发还受硬件配额、分配粒度、grid、调度和访存影响。需要完整 occupancy 与时间分析，不能只根据一个寄存器数宣称达到最优。

**可能追问：如何区分编译错误和数值错误？**

旧 width 3 首先表现为向量类型和广播错误，需要检查 tracing/IR 中的 copy/shuffle 类型；修复到可编译之后，错误 peer lane 会表现为数值配对不对，需要回到元素与 lane 映射。两种问题分层定位，最后再核对最终 ISA 和实际 GPU 结果。

测试与资源证据见[Fused RoPE 验证笔记](notes/FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)。
