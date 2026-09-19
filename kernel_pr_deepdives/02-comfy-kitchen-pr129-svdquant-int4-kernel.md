# Comfy-Kitchen PR #129：SVDQuant INT4 Quantizer 与 W4A4 GEMM 深读

## PR 动机

SVDQuant 把线性层拆成 INT4 主路径和低秩补偿，但 HIP activation preprocessing 的旧实现
是一线程串行处理一个 64-element group：先读取 64 个 `x/smooth` 求 absmax，再重新读取
一遍完成量化和 packing。线程内依赖链很长，group 内没有并行，输入还被逻辑读取两遍。

PR #129 的最终有效优化，是让 16 lanes 合作处理一个 64-element group：每 lane 一次保留
4 个平滑后的值，用 shuffle 求 group absmax，再把 4 个 INT4 nibble 合成一次 16-bit store。
FP16/BF16 代表 shape 上已有 `17.4–21.5×` 的 quantizer 加速记录。

这项 PR 初版还写过自定义 LoRA-down scalar/WMMA kernel，但最终提交把它们全部删除，改为
`torch.mm`。因此阅读最终 PR 时必须把两件事分开：保留的是 INT4 activation quantizer；
LoRA-down 最终采用成熟 GEMM library。

本文基于最终 PR head：

- PR：<https://github.com/Comfy-Org/comfy-kitchen/pull/129>
- Head：`71fd9edb197eda150fd288d6a45c12a6944b0b97`
- 真实提交范围：`2bf7b4b^..71fd9ed`
- 核心文件：`comfy_kitchen/backends/hip/ops/svdquant_w4a4.hip`
- 公共 staging：`comfy_kitchen/backends/hip/gemm_wmma.h`
- Python 最终选路：`comfy_kitchen/backends/hip/__init__.py`

固定读取方式：

```bash
git -C ~/comfy-kitchen show \
  refs/remotes/pr/129:comfy_kitchen/backends/hip/ops/svdquant_w4a4.hip
```

## 1. SVDQuant 整体计算是什么

SVDQuant 的线性层可以按实现理解成：

```text
主路径：
  x_smooth = x / smooth
  A_int4   = groupwise_quantize_int4(x_smooth)
  Y_main   = dequant(A_int4 @ W_int4^T, activation_scales, weight_scales)

低秩路径：
  L        = x @ proj_down
  Y_lora   = L @ proj_up^T

最终：
  Y = Y_main + Y_lora + bias
```

PR 涉及三段容易混淆的代码：

1. `svdquant_quant_kernel`：浮点 activation → packed INT4 + activation scales；这是主要优化点。
2. LoRA-down：最终由 `torch.mm(x, lora_down)` 完成，没有保留自定义 HIP kernel。
3. `svdquant_gemm_kernel`：消费 packed activation/weight，按 group 反标定，并在 writeback 融合
   LoRA-up 和 bias。理解整个 feature 时仍应看懂它，但 PR 的大倍率 quantizer benchmark 不是
   这段 GEMM 的加速数字。

## 2. Activation INT4 的数学与存储布局

每行沿 K 每 64 个元素分成一组。对 group `g` 中的元素：

```text
v_i    = x_i / smooth_i
absmax = max_i abs(v_i)
scale  = max(absmax, 1e-10) / qmax
q_i    = clamp(rint(v_i * qmax / absmax), qmin, qmax)
```

范围分两种模板实例：

| 模式 | `qmin` | `qmax` | 说明 |
|---|---:|---:|---|
| signed | -7 | 7 | 对称 INT4，nibble 使用二进制补码低 4 bit |
| unsigned | 0 | 15 | caller 定义的非负 activation 路径 |

输出布局非常重要：

```text
q        [M_pad, K/2]   int8 storage，每 byte 两个 nibble
ascales  [K/64, M_pad]  与 x 相同的浮点 dtype
lora_act [M_pad, R]     FP32
```

`ascales` 不是常见的 `[M, K/64]`，而是 group-major：

```cpp
ascales[g * M_pad + m]
```

后续 GEMM 沿 K group 推进时，同一个 group 的所有 row scale 连续，正好符合消费方式。

## 3. 旧 quantizer：一个线程承担整个 group

旧 scalar fallback 仍保留在最终源码中，因此可以直接对比：

```cpp
for (int g = threadIdx.x; g < ngroups; g += 256) {
    float absmax = 0.0f;
    for (int i = 0; i < 64; ++i) {
        float v = load_in(x, base + i, x_code)
                / load_in(smooth, k, s_code);
        absmax = max(absmax, abs(v));
    }

    store scale;

    for (int i = 0; i < 64; i += 2) {
        reload x and smooth;
        quantize q0 and q1;
        qrow[i / 2] = q0 | q1 << 4;
    }
}
```

问题并不是 block 线程数少，而是一个 group 的 64 个元素全在单线程中串行：

- 64 次除法/abs/max 形成长依赖链；
- 量化阶段再次读取 `x` 与 `smooth`；
- 一个线程每次只写 1 byte；
- dtype 通过运行时 code 解释；
- 当一行 group 数小于 256 时，一部分线程根本没有工作。

`K=4096` 时每行只有 64 groups；`K=14336` 时有 224 groups。两者都没有充分利用
256-thread block，更没有利用 group 内天然的并行性。

## 4. 新映射：16 lanes × 4 elements = 一个 group

新 kernel 的编译期常量是：

```cpp
constexpr int kThreadsPerGroup = 16;
constexpr int kElemsPerThread  = 64 / 16;  // 4
constexpr int kGroupsPerBlock  = 256 / 16; // 16
```

线程索引拆解为：

```cpp
lane       = threadIdx.x % 16;
group_slot = threadIdx.x / 16;
```

映射图：

```text
thread 0..15    -> group_slot 0，lane 0..15
thread 16..31   -> group_slot 1，lane 0..15
...
thread 240..255 -> group_slot 15，lane 0..15

group g 内：
lane 0  -> elements 0..3
lane 1  -> elements 4..7
...
lane 15 -> elements 60..63
```

若一行超过 16 groups，每个 slot 继续处理：

```cpp
for (int g = group_slot; g < ngroups; g += 16)
```

gfx11 wave size 是 32，所以一个 wave 中同时装有两个独立的 16-lane logical subgroup。

## 5. 单遍加载：为什么这里适合缓存 `values[4]`

每个 lane 只处理四个连续元素：

```cpp
float values[4];
float absmax = 0.0f;

#pragma unroll
for (int i = 0; i < 4; ++i) {
    values[i] = float(x[xbase + i]) / float(smooth[kbase + i]);
    absmax = fmaxf(absmax, fabsf(values[i]));
}
```

四个 `values` 会跨 shuffle reduction 保持活跃，得到完整 group scale 后直接复用。与 PR #127
缓存整行的设想不同，这里每 lane 的缓存规模固定为 4 个 FP32，和 K 无关，寄存器压力更容易
控制。

因此新 kernel 在算法层面将 `x` 和 `smooth` 从两遍逻辑读取变成一遍。是否完全对应 DRAM
流量减半仍取决于 cache，但源码的 necessary loads 确实减少了。

连续性也很好：lane 0 读 0..3，lane 1 读 4..7，相邻 lane 覆盖相邻元素。这里没有显式
`uint4` load，不能把它描述成手写 128-bit vector global load；主要变化是 group 内并行、
寄存器复用和 typed access。

## 6. 16-lane shuffle reduction 到底发生了什么

每 lane 先得到四个数的 local max，然后：

```cpp
for (int offset = 8; offset > 0; offset >>= 1) {
    absmax = fmaxf(absmax, __shfl_xor(absmax, offset, 16));
}
```

offset 顺序是 `8,4,2,1`：

```text
offset 8：lane 0 与 8，1 与 9，... 交换
offset 4：合并 4-lane 间隔的结果
offset 2：继续合并
offset 1：完成 16 lanes 全归约
```

关键参数是 `width=16`。它把一个 wave32 切成 `[0..15]` 和 `[16..31]` 两组；lane 0 不会
与 lane 16 交换。因此一个物理 wave 同时完成两个互不干扰的 group reduction。

与 block shared-memory tree 相比，这段 reduction：

- 不分配 group reduction LDS；
- 不需要 `__syncthreads()`；
- 只有四轮 wave shuffle；
- 最终每个 lane 都拿到相同 absmax，便于各自量化四个值。

只有 logical lane 0 写 scale：

```cpp
if (lane == 0) {
    ascales[g * M_pad + m] = Elem(scale);
}
```

每个 16-lane subgroup 都有自己的 `lane==0`，即物理 wave 的 lane 0 和 lane 16 都会分别写
各自 group 的 scale。

## 7. 四个 nibble 合成一次 16-bit store

得到 inverse scale 后，每 lane 量化四个保留值：

```cpp
int v = int(rintf(values[i] * inv));
quantized[i] = clamp(v, qmin, qmax);
```

然后按低位到高位 packing：

```cpp
uint16_t packed =
    (q0 & 0xF)       |
    (q1 & 0xF) << 4  |
    (q2 & 0xF) << 8  |
    (q3 & 0xF) << 12;

reinterpret_cast<uint16_t*>(qrow)[lane] = packed;
```

完整 group 的守恒关系是：

```text
16 lanes × 4 INT4 values = 64 values
16 lanes × 2 output bytes = 32 bytes = 64 × 4 bit
```

例如 signed `-1` 的低 4 bit 是 `0xF`，packing 只保留 nibble；后续 signed INT4 WMMA 按
对应语义解释它。不能在 packing 阶段把 signed 值饱和到 unsigned 范围。

## 8. Typed fast path 与 scalar fallback

launcher 只在 `x_code == s_code` 时实例化 typed kernel：

```text
float  + signed/unsigned
half   + signed/unsigned
bf16   + signed/unsigned
```

不同 dtype 的 `x` 与 `smooth/ascales` 回到 scalar fallback。高层 Python 会把 `smooth`
转换为 `x.dtype`，所以正常入口通常命中 fast path；native binding 仍必须保护低层调用。

另外：

- `K` 必须是 64 的倍数，否则最后一个 group 无完整 scale/packing contract；
- `M_pad >= M`；
- q、ascales、lora_act 预先清零，padded rows 保持零；
- `M==0` 直接返回，避免非法 zero-block launch。

## 9. 最终 LoRA-down 为什么是 `torch.mm`

最终 Python 代码是：

```python
lora_act_rows = lora_act[:m]
if lora_act_rows.dtype == lora_src.dtype and lora_act_rows.is_contiguous():
    torch.mm(lora_src, lora_down, out=lora_act_rows)
else:
    lora_act_rows.copy_(lora_src @ lora_down, non_blocking=True)
```

PR 初版曾加入专用 FP16/BF16 WMMA LoRA-down，相比旧的一线程一输出 scalar kernel 很快；
但 reviewer 用真正合理的 baseline——`torch.mm`/hipBLASLt——比较后，专用实现仍慢约 2–4 倍。

这是重要的 kernel 工程判断：

- “比旧 kernel 快很多”不等于值得保留；
- GEMM 是高度成熟的库算子，应优先与 library baseline 比；
- 一个 PR 可以通过删除自己写的 kernel 变得更好；
- 最终代码事实优先于初始 commit 或 PR 标题。

## 10. W4A4 GEMM：输入、tile 与 wave 映射

`svdquant_gemm_kernel<OutT, UNSIGNED>` 计算：

```text
C = groupwise_dequant(A_int4 @ B_int4^T)
  + lora_act @ lora_up^T
  + bias
```

主 tile：

```text
BM  = 128 rows
BN  = 128 columns
BKB = 64 packed bytes = 128 INT4 K elements
TM  = 2 WMMA fragments in M per wave
TN  = 4 WMMA fragments in N per wave
threads = 256 = 8 wave32
```

wave 映射：

```cpp
warp = threadIdx.x / 32; // 0..7
wm   = warp / 2;         // 0..3
wn   = warp % 2;         // 0..1
```

一个 wave 覆盖：

```text
M：TM × 16 = 32 rows
N：TN × 16 = 64 columns
```

所以 `4 × 2` 的 wave 网格正好覆盖 `128 × 128` block tile。

每个 lane 对每个 WMMA accumulator fragment 持有 8 个 FP32 结果。总的 per-thread 浮点输出
暂存是：

```text
TM × TN × 8 = 2 × 4 × 8 = 64 FP32 values
```

这说明该 kernel 的 VGPR 压力不会小；理解大 tile 时必须同时看数据复用和寄存器占用。

## 11. `TileStager`：global → register → LDS

LDS 定义为 raw byte arrays：

```cpp
constexpr int kStride = BKB + 8; // 72 bytes
__shared__ uint8_t As[128 * 72];
__shared__ uint8_t Bs[128 * 72];
```

每个 A/B buffer 是 `128 × 72 = 9216 B`，合计 18 KiB。

公共 `TileStager<ROWS,BKB,THREADS>` 以 16-byte chunk 搬数据：

```cpp
uint4 regs[kPerThread];

load():
  global memory -> per-thread uint4 registers

store():
  uint4 registers -> LDS
```

对 `ROWS=128, BKB=64, THREADS=256`：

```text
每行 chunks = 64 / 16 = 4
总 chunks    = 128 × 4 = 512
每线程       = 512 / 256 = 2 个 uint4
```

即每个 stager 每线程持有 32 bytes，A/B 两个 stager 共 64 bytes 的待写数据。`load()` 与
`store()` 分开是为了软件流水：计算当前 LDS tile 时，下一 tile 可以先从 global 发到寄存器。

LDS row padding 是 8 bytes，用于改变相邻 row 的 bank 映射。因为 stride 72 不保持每行
16-byte 对齐，store 拆成两个 8-byte `uint2`，而不是直接写一个 `uint4`。

## 12. K pipeline：预取下一 tile，但只使用一套 LDS

初始化：

```cpp
sa.load(A, ..., 0);
sb.load(B, ..., 0);
sa.store(As);
sb.store(Bs);
__syncthreads();
```

每轮：

```text
若有下一 tile：global -> stager registers
计算当前 As/Bs
barrier，确保所有线程不再读当前 LDS
stager registers -> 同一套 As/Bs
barrier，确保新 tile 写完
下一轮
```

这不是双份 LDS ping-pong；它通过寄存器 prefetch 隐藏一部分 global latency，再覆盖同一套
LDS。越界 row 和 K tail 在 `TileStager::load()` 中写零，保证边界 tile 的 WMMA 输入合法。

## 13. 为什么 INT32 accumulator 必须每 64 元素 flush

一轮 `BKB=64 packed bytes` 表示 128 个 INT4 元素，也就是两个 64-element quant group：

```text
一个 group = 64 INT4 = 32 packed bytes
BKB        = 64 bytes = 两个 group
```

gfx11 与 gfx12 的 INT4 WMMA K-step 不同：

```text
gfx11: kStepBytes = 8  -> 16 INT4 K elements -> 每 group 4 次 MMA
gfx12: kStepBytes = 16 -> 32 INT4 K elements -> 每 group 2 次 MMA
```

代码为每个 quant group 单独清零 `int32 acc`，完成该 group 的若干 MMA 后立即：

```cpp
fout += int32_acc * activation_scale[g,row] * weight_scale[g,col];
```

原因不是防溢出，而是数学语义：不同 group 的 activation scale 和 weight scale 不同。
如果把多个 group 的整数点积先累加成一个 INT32，最后只乘一次 scale，就无法恢复正确结果：

```text
正确：sum_g (int_dot_g × ascale_g × wscale_g)
错误：(sum_g int_dot_g) × 某一个 scale
```

所以 `fout[TM][TN][8]` 是跨 group 的 FP32 accumulator，而临时 `MmaInt4::Acc` 只活一个 group。

## 14. Epilogue 为什么融合 LoRA-up

完成主 K loop 后，每个 lane 已拥有对应输出坐标的 `fout`。writeback 前：

```cpp
float v = fout;
for (int t = 0; t < R; ++t) {
    v += lora_act[r,t] * lora_up[col,t];
}
if (bias) v += bias[col];
C[r,col] = OutT(v);
```

R 是小 rank，因此每个输出额外做一个短 dot product。这样避免先写 `Y_main[M,N]`，再启动
另一 kernel 读写完整输出以加上 LoRA-up 和 bias。

这是一种典型 epilogue fusion：增加少量每输出计算和输入读取，换掉一次完整输出 tensor pass
以及额外 launch。它与 LoRA-down 是否使用自定义 kernel 是两回事。

## 15. Launcher 与边界 contract

GEMM launcher：

```text
M==0 或 N==0 -> return
K % 64 != 0  -> error
grid = (ceil(N/128), ceil(M/128))
block = 256
OutT = BF16 / FP16 / FP32
UNSIGNED = true / false
```

binding 还验证：

- packed A 至少有 `M × K/2` bytes；
- packed B 至少有 `N × K/2` bytes；
- ascales 为 `[K/64, M]`；
- wscales 为 `[K/64, N]`；
- lora_act 为 `[M,R]`，lora_up 为 `[N,R]`；
- 输出 dtype code 必须与真实 allocation 一致。

这些检查不是“测试附属品”。kernel 接收裸指针，一旦 shape/dtype code 错了，结果通常不是
友好异常，而是错误解码或越界设备访问。

## 16. 如果从零实现这项优化

建议按下面的推导过程：

1. 先把 group 固定为 64，写清 signed/unsigned 范围和 nibble 顺序。
2. 选择 logical subgroup 大小 16，使每 lane 固定处理 4 个值。
3. 确认一个 wave32 可以容纳两个 subgroup，并显式用 shuffle `width=16` 隔离。
4. 把四个平滑值放在小固定数组中，验证无 spill，再跨 reduction 复用。
5. 让每 lane 恰好输出 16 bit，获得自然、连续的 group packing。
6. 用 dtype 模板保留常用 fast path，用 scalar kernel承接 mixed dtype。
7. 阅读消费者 GEMM，确认 scale 布局必须是 group-major，而不是凭习惯设计输出。
8. 对任何新 LoRA-down kernel，必须与 `torch.mm` 比，而不只与旧 scalar 比。

## 17. 性能证据与不能夸大的地方

已有 Radeon 8060S（`gfx1151`）记录：

```text
FP16/BF16，两个指定大 shape：17.4–21.5×
FP32：                     7.4–9.1×
```

这些倍率属于 activation quantizer，不代表完整 SVDQuant linear、W4A4 GEMM 或模型端到端。
本文未重新运行硬件 benchmark。源码可直接证明的是：64-element group 从单线程串行改成
16-lane 协作，输入/smooth 从两遍逻辑读取变一遍，LDS reduction 变为 subgroup shuffle，
packing 从每线程 1 byte 变为 2 bytes。具体哪一项贡献多少，需要消融和 profiler 才能回答。

## 18. 最终应记住的设计结论

> PR #129 最终保留的是 16-lane groupwise INT4 quantizer：每 lane 缓存四个平滑值，
> 用 width-16 shuffle 求 scale并一次写四个 nibble；LoRA-down 自定义 kernel 被删除并回归
> `torch.mm`。后续 W4A4 GEMM按 quant group flush INT32 accumulator，再在 writeback 融合
> LoRA-up 与 bias。
