# Comfy-Kitchen PR #134：AWQ W4A16 的 GEMV、WMMA 与三级 Dispatch

## PR 动机

Comfy-Kitchen 已经有 AWQ W4A16 的 HIP wave GEMV：一个 wave 计算一个输出元素，边读 packed
INT4 weight 边反量化，再做 wave reduction。它非常适合 `M=1` 一类 decode/small-batch
workload，但 M 增大以后，同一份 weight 会为每个 activation row 重复读取、解包和反量化，
性能近似随 M 线性恶化。

PR #134 没有“新写一个 GEMV”；它保留既有 small-M GEMV，为 medium-M 新增 fused
`16×128×64` WMMA kernel，使一份反量化后的 weight group 被 16 行 activation 复用；当 M
非常大时，再切换到“完整反量化 weight + `torch.matmul`”。真正的 feature 是三层 dispatch，
而不是某一条 kernel 永远取代其他路径。

本文基于最终 PR：

- PR：<https://github.com/Comfy-Org/comfy-kitchen/pull/134>
- Head：`a50e24175609410477fcd99f8f8d1099e029b625`
- 真实提交范围：`031236d^..a50e241`
- 核心 HIP：`comfy_kitchen/backends/hip/ops/gemv_awq.hip`
- Python dispatch：`comfy_kitchen/backends/hip/__init__.py`

固定读取：

```bash
git -C ~/comfy-kitchen show \
  refs/remotes/pr/134:comfy_kitchen/backends/hip/ops/gemv_awq.hip
```

## 1. AWQ W4A16 的数学和真实 layout

W4A16 表示 weight 是 4-bit，activation 是 FP16/BF16。每个 weight group 使用浮点 scale 和
zero correction：

```text
W[n,k] = (q[n,k] - 8) * scale[k/G,n] + zero[k/G,n]
Y[m,n] = sum_k X[m,k] * W[n,k] + bias[n]
```

实现中的 tensor layout：

```text
x        [M,K]       FP16/BF16
qweight  [N,K/2]     int8 storage，每 byte 两个 unsigned INT4
wscales  [K/G,N]     FP16/BF16
wzeros   [K/G,N]     与 wscales 同 dtype
bias     [N]         optional
out      [M,N]
```

一个 byte 的 low nibble 对应偶数 K，high nibble 对应奇数 K：

```text
q[k]   = packed & 0xF
q[k+1] = packed >> 4
```

再减 8，得到以 0 为中心的 `[-8,7]` 整数值。

## 2. 整个执行路径先看全景

最终逻辑可以画成：

```text
Python：统一 x、scale、zero 的 compute dtype
  |
  |-- M > dtype-specific limit
  |     -> 解包完整 W -> dequant -> torch.matmul
  |
  `-- M <= limit
        |
        |-- has_wmma + x 16B aligned
        |     -> native use_wmma=true
        |          |
        |          |-- M>8, G=64, dtype contract 满足
        |          |     -> fused WMMA
        |          `-- 否则 -> scalar wave GEMV
        |
        `-- 不满足 -> scalar wave GEMV
```

大 M 阈值是：

```python
BF16: M > 1280 -> dequant + matmul
FP16: M > 2048 -> dequant + matmul
```

因此三条路径的目标区间是：

| M 区间 | 主要路径 | 核心理由 |
|---|---|---|
| 小，尤其 `M<=8` | wave GEMV | 固定开销低，不需要 LDS/WMMA staging |
| 中等 | fused WMMA | weight dequant 后可跨 16 rows 复用 |
| 很大 | dequant + library GEMM | 完整反量化成本可被大量矩阵计算摊薄 |

## 3. Small-M wave GEMV 的 launch 与线程映射

旧 kernel 仍是最终代码的重要组成：

```cpp
dim3 block(kWave, 8); // (32, 8) = 256 threads
dim3 grid(ceil(N / 8), M);
```

HIP 二维 thread block 中：

```cpp
lane = threadIdx.x;                      // 0..31
n    = blockIdx.x * 8 + threadIdx.y;     // 一个 block 的 8 个 output columns
m    = blockIdx.y;                       // 一个 activation row
```

也就是：

```text
threadIdx.y = 0..7 -> 8 个独立 wave
每个 wave32        -> 一个 (m,n) 输出元素
每个 lane          -> 这个点积的一部分 K
```

`__launch_bounds__(kWave * 8)` 写成 256，与实际 block 总线程数一致。

## 4. GEMV 内每个 lane 怎么读 packed weight

K loop 是：

```cpp
for (int k = lane * 8; k < K; k += 32 * 8) {
    int g = k / group_size;
    float s = wscales[g,n];
    float z = wzeros[g,n];

    uchar4 packed = load 4 bytes from qweight[n, k/2];
    ... unpack 8 weights ...
}
```

一笔 `uchar4` 是 4 bytes，对应 8 个 4-bit weights。lane `l` 第一轮处理：

```text
k = l × 8 ... l × 8 + 7
```

整个 wave 第一轮覆盖 256 个连续 K elements；下一轮每 lane 增加 `32×8=256`。

内层每个 byte：

```cpp
lo = packed & 0x0F;
hi = packed >> 4;

sum += x[k]   * ((lo - 8) * s + z);
sum += x[k+1] * ((hi - 8) * s + z);
```

因为 launcher 要求 `group_size % 8 == 0`，一个 lane 的 8-weight chunk 不会跨 quant group，
一次读取的 scale/zero 可安全用于整个 chunk。

## 5. GEMV 的 reduction 与为什么适合小 M

每 lane 得到一个 FP32 partial sum，随后：

```cpp
sum = wave_reduce_sum(sum);
if (lane == 0) {
    if (bias) sum += bias[n];
    out[m,n] = OutT(sum);
}
```

这种设计的优点：

- 没有 LDS tile；
- 不需要 block-wide barrier；
- 一个输出的 reduction 完全在 wave 内；
- M 很小时启动和 staging 成本小。

它的根本缺点是没有跨 M 复用。同一个 `qweight[n,k]` 在 `m=0,1,2,...` 的 wave 中重复被：

```text
global load -> nibble unpack -> subtract 8 -> scale/zero dequant
```

当 M 从 1 增长到 256，这部分工作也大约重复 256 次。这就是新增 WMMA path 的切入点。

## 6. Medium-M fused WMMA 的 tile

新 kernel：

```cpp
gemm_awq_w4a16_kernel<Mma, OutT>
```

固定配置：

```text
BM = 16
BN = 128
BK = 64
threads = 128 = 4 wave32
kNTiles = 2
```

一个 block 计算 `16×128` 的输出 tile，并沿 K 每次推进 64。`BK=64` 特意等于 fast path
支持的 AWQ group size，因此一个 K tile 中每个 output column 只使用一组 scale 和 zero。

grid：

```text
grid.x = ceil(N / 128)
grid.y = ceil(M / 16)
```

## 7. Activation staging：128 threads 恰好搬完 `16×64`

LDS activation buffer：

```cpp
Elem xs[16 * (64 + 8)];
```

每线程搬一个 `uint4`：

```cpp
xv = tid;
xm = xv / (64 / 8);       // tid / 8 -> row 0..15
xk = (xv % 8) * 8;        // K offset 0,8,...,56

*reinterpret_cast<uint4*>(xs + xm * 72 + xk) =
    *reinterpret_cast<const uint4*>(x + global_row * K + k0 + xk);
```

FP16/BF16 每元素 2 bytes，`uint4` 的 16 bytes 正好是 8 个元素：

```text
128 threads × 8 elements = 1024 elements
16 rows × 64 K           = 1024 elements
```

映射非常整齐：tid 0..7 搬 row 0 的八个 8-element chunks，tid 8..15 搬 row 1，以此类推。
若最后一个 M tile 的 row 越界，则对应线程向 LDS 写零。

fast path 要求 `x.data_ptr() % 16 == 0`，并且 K 是 64 的倍数，所以每行 stride 也是 16-byte
整数倍，所有 `uint4` load 保持对齐。

## 8. Weight staging：一线程负责一个 output column

LDS weight buffer：

```cpp
Elem ws[128 * (64 + 8)];
```

thread `tid` 对应全局 column：

```cpp
n = n0 + tid;
```

如果 `n<N`，该线程：

1. 读取当前 K group 的一个 scale 和 zero；
2. 定位 `qweight[n, k0/2]`；
3. 循环读取 32 packed bytes；
4. 每 byte 解出 low/high 两个 nibble；
5. 执行 `(q-8)*scale+zero`；
6. 转为 `Elem` 并写入 `ws[tid,0..63]`。

一线程串行解包 64 个 weight 看似很重，但它只做一次，之后同一个 block 的 16 个 activation
rows 都从 LDS 复用这 64 个值。相对 GEMV，理想的跨行反量化复用因子是 16。

最后一个 N tile 中越界的 column 会把整行 LDS weight 写零，后续 WMMA 可以无条件读取。

## 9. 为什么 LDS stride 是 72

两个 buffer 都使用：

```text
kStride = BK + 8 = 72 elements
```

LDS 占用，FP16/BF16 都是 2 bytes：

```text
xs = 16  × 72 × 2 =  2,304 bytes
ws = 128 × 72 × 2 = 18,432 bytes
总计                  20,736 bytes
```

额外 8 elements，即 16 bytes padding，使相邻 row 的 bank 映射发生偏移，降低 WMMA fragment
读取时的系统性 bank conflict。padding 不是为了越界保护，也不属于 global tensor layout。

## 10. WMMA wave 到输出 tile 的映射

四个 waves 分别负责 N 方向的一段：

```cpp
warp = tid / 32; // 0..3
```

每 wave 有两个 accumulator fragment：

```cpp
for (j = 0; j < 2; ++j) {
    wn = warp * (2 * 16) + j * 16;
}
```

于是：

```text
wave 0 -> N 0..31
wave 1 -> N 32..63
wave 2 -> N 64..95
wave 3 -> N 96..127
```

每个 fragment 是一个 `16×16` 输出块，M 方向固定覆盖当前 block 的 16 rows。四 waves × 每
wave 2 fragments × 16 columns = 128 columns。

BK=64 被拆成四个 K=16 WMMA step：

```cpp
for (kk = 0; kk < 64; kk += 16) {
    af = load 16x16 activation fragment from xs;
    bf = load 16x16 weight fragment from ws;
    acc[j] = Mma::mma(af, bf, acc[j]);
}
```

accumulator 是 FP32；K group 完成后进入下一 group，继续累计相同输出 tile。

## 11. Lane 最后拥有哪几个输出

gfx11/12 的 MMA wrapper 用 `frag_row(lane)`、`acc_row(lane,e)` 和 `acc_col(lane)` 封装
硬件 fragment layout。写回代码是：

```cpp
col = n0 + warp * 32 + j * 16 + acc_col(lane);

for (e = 0; e < 8; ++e) {
    m = m0 + acc_row(lane, e);
    out[m,col] = OutT(Mma::get(acc[j],e) + bias[col]);
}
```

每 lane 对每个 accumulator fragment 持有 8 个 FP32 output values；`acc_row` 把这 8 个值
映射回 16 output rows，`acc_col` 决定 lane 对应的 column。不要按“lane 0 就只算 row 0”
的普通 SIMT 直觉理解 WMMA fragment。

## 12. K loop 为什么每轮有两个 barrier

每个 group 的执行顺序：

```text
所有线程写 xs/ws
__syncthreads()
四次 WMMA K-step 读取 xs/ws
__syncthreads()
下一 group 覆盖 xs/ws
```

第一个 barrier 防止 compute wave 在其他线程尚未完成 weight dequant/LDS store 时读取半成品；
第二个 barrier 防止下一轮 staging 在仍有 wave 读取当前 tile 时覆盖 LDS。

该 kernel 没做双缓冲或 register prefetch；这使实现简单，但也可能让 weight dequant、LDS 和
WMMA 阶段较串行，是后续优化空间。

## 13. Native fast-path guard 为什么这么严格

native launcher 只有在以下条件同时成立时进入 WMMA：

```cpp
use_wmma
M > 8
group_size == 64
x_code == s_code
out_code == s_code
out dtype 是 BF16 或 FP16
```

Python 的 `use_wmma` 还要求：

```text
has_wmma()
x2d.data_ptr() % 16 == 0
```

逐个解释：

- `M>8`：M 太小时跨行复用不足，WMMA/LDS 固定成本不划算；
- `group_size=64`：kernel 的 BK=64 和一次 scale/zero staging 建立在这个 contract 上；
- same dtype：weight 在 scale dtype 中反量化，MMA fragment 也按同 dtype 解释；
- output same dtype：写回模板和高层 compute-dtype 语义一致；
- 16B aligned：activation 使用无 tail 的 `uint4` load；
- `has_wmma`：不能在无 matrix core 的设备上执行 intrinsic。

任何条件失败都回到原 scalar GEMV，而不是报错；只有违反基本 shape contract 才报错。

## 14. Python 为什么先统一 dtype

Python 最终在选路前执行：

```python
x2d = x2d.to(wscales.dtype)
wzeros = wzeros.to(wscales.dtype)
```

这是 review 后的重要修正。否则同一个 public operation 可能因为 M 不同走不同路径，并在
mixed-dtype 输入下使用不同 compute semantics。先统一到 scale dtype，三条路径再比较，
才能使 dispatch 主要改变执行方法，而不是悄悄改变数值定义。

native 仍保留 `x_code == s_code` guard，因为低层 `_C` binding 可以被绕过 Python 直接调用。

## 15. Large-M 为什么反而物化完整 weight

大 M 路径：

```python
values = empty([N,K], compute_dtype)
values[:,0::2] = qweight & 0xF
values[:,1::2] = (qweight >> 4) & 0xF

weight = values.view(N, K/G, G)
weight.sub_(8)
weight.mul_(wscales.T[...,None])
weight.add_(wzeros.T[...,None])

out = x.matmul(values.T)
```

它牺牲了一个完整 `[N,K]` 浮点 workspace，并先付出一次 weight dequant，但大 M 时：

- 反量化一次可被很多 activation rows 摊薄；
- hipBLAS/hipBLASLt 的成熟 GEMM tile、pipeline 和调度通常强于专用的简单 WMMA kernel；
- fused kernel 避免 workspace 的优势不再足以抵消 library GEMM 的计算效率。

BF16/FP16 阈值不同，说明 crossover 与 dtype、硬件实现和库性能相关，而不是纯数学常量。
当前阈值来自 gfx1151/gfx1200 的工程测量，不应外推为所有 GPU、N/K 和库版本的绝对最优。

## 16. 为什么一个 kernel 不可能覆盖所有 M

三条路径的资源模型不同：

| 路径 | 权重处理 | 跨 M 复用 | 固定成本 | workspace |
|---|---|---:|---:|---:|
| wave GEMV | 每输出边读边解包 | 无 | 最低 | 无 |
| fused WMMA | 每 block/group 解包到 LDS | 16 rows | 中等 | 约 20 KiB LDS/block |
| dequant + GEMM | 全 weight 一次反量化 | 全 M | 最高 | `[N,K]` 浮点 weight |

kernel writer 的工作不只是优化一段 device code，还要找到不同资源模型的 crossover，并把
shape、dtype、alignment、architecture 纳入 dispatch。

## 17. 从零设计这个 feature 的步骤

1. 先写清 packed weight layout、group scale layout 和 `(q-8)*s+z` 公式。
2. 保留小 M GEMV 作为低固定开销 baseline，不要一开始就删除。
3. 选 `BK=group_size`，让一个 K tile 的 scale/zero contract 最简单。
4. 选 `BM=16` 对齐 WMMA M dimension，使 weight 可跨 16 rows 复用。
5. 让 128 threads 一一负责 128 output columns 的 weight dequant。
6. 验证 activation tile 能被线程数和 16-byte copy 整除。
7. 计算 LDS：`(BM+BN)×(BK+pad)×sizeof(Elem)`，先确保资源合法。
8. 写 native guard，再在 Python 层处理 architecture、alignment 和 large-M crossover。
9. 对每条路径单独做 correctness，并验证边界输入实际命中了预期 kernel。
10. 大 M 必须与 library GEMM 比，而不是只与 GEMV 比。

## 18. 性能证据与边界

已有 Radeon 8060S（`gfx1151`）、BF16、`N=K=4096` 记录：

```text
M=16..256：fused WMMA 相比原 wave GEMV，9.8–19.6×
M=1、8：仍使用原 GEMV，性能基本不变
```

这些数字只证明指定 medium-M shape 下相对原 HIP GEMV 的 kernel-level 改善，不代表：

- 所有 group size；当前 WMMA 只支持 64；
- 所有 dtype；公开 fast path 是 FP16/BF16；
- 所有 M/N/K；大 M 已主动切换 library path；
- 模型端到端同倍率。

本文未重新运行硬件 benchmark。源码能证明的机制是跨 16 rows 复用反量化 weight、使用
WMMA FP32 accumulate，以及按 M/dtype 分层 dispatch；cache hit、实际 occupancy 和瓶颈比例
仍需要 profiler/benchmark 证据。

## 19. 最终应记住的一句话

> PR #134 在既有 small-M wave GEMV 之上增加 `16×128×64` fused WMMA，让每个 AWQ
> weight group 反量化一次后服务 16 行 activation；当 M 再增大时，主动切到完整反量化加
> `torch.matmul`。它最重要的设计不是单个 WMMA intrinsic，而是为三个计算区间选择不同
> 资源模型的 dispatch。
