# Comfy-Kitchen PR #127：Rowwise INT8 向量化 Kernel 逐行解读

## PR 动机

这项 PR 解决的不是 INT8 GEMM 本身，而是 GEMM 前面的 activation quantization 已经成为瓶颈。
在 Anima05 的代表 shape `M=8192, K=2048, BF16` 中，旧 rowwise quantization
约为 `2.3 ms`，后续 INT8 GEMM 约为 `0.74 ms`。如果只优化 GEMM，端到端收益会被前处理吞掉。

旧 HIP kernel 的数学没有问题，问题在执行方式：每次循环只加载一个标量，而且
`load_in(..., in_code)` 在热循环里根据运行时 dtype code 解释 FP32、FP16 或 BF16。
PR #127 把 dtype 变成模板参数，并让每个线程一次读取 16 bytes。它是一项
“减少 load/loop/address-control 指令”的优化，不是量化算法变更，也不是把必要数据量
减少了四倍或八倍。

本文基于公开 PR 的最终提交，而不是当前 main：

- PR：<https://github.com/Comfy-Org/comfy-kitchen/pull/127>
- Head：`6b38b04d8c3e6bb9c6b6a7462a7afd5eef5fbc28`
- 真实提交范围：`6b38b04^..6b38b04`
- 核心文件：`comfy_kitchen/backends/hip/ops/quantize_int8.hip`

可以在本地固定读取这份源码：

```bash
git -C ~/comfy-kitchen show \
  refs/remotes/pr/127:comfy_kitchen/backends/hip/ops/quantize_int8.hip
```

## 1. 先建立数学 contract

输入是连续二维矩阵 `x[M,K]`；更高维 Python 输入会先把前面的维度展平成 M。
对第 `m` 行：

```text
absmax[m] = max_k abs(x[m,k])
scale[m]  = max(absmax[m] / 127, 1e-30)
q[m,k]    = clamp(rint(x[m,k] / scale[m]), -128, 127)
```

代码中的 `rintf` 是 round-half-to-even，与 `torch.round` 对齐。scale 固定写成
FP32，量化结果写成 INT8。

这里有一个决定 kernel 结构的依赖：只有整行 absmax 已知以后才能得到 scale，只有
scale 已知以后才能量化这一行。因此直接实现通常是：

```text
第一遍读 x：求每线程 local max
      ↓
block reduction：得到整行 absmax
      ↓
第二遍读 x：乘 inverse scale、round、clamp、store
```

PR 没有试图消除这个两遍结构。

## 2. 旧 kernel 到底怎么跑

旧实现的 launch 是：

```text
grid.x  = M
block.x = 256
一个 block 处理一行
```

核心代码可缩写为：

```cpp
const int row = blockIdx.x;
const int t = threadIdx.x;

float lmax = 0.0f;
for (int j = t; j < K; j += 256) {
    lmax = fmaxf(lmax, fabsf(load_in(x, row * K + j, in_code)));
}

red[t] = lmax;
__syncthreads();
for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
}

const float scale = fmaxf(red[0] / 127.0f, 1e-30f);

for (int j = t; j < K; j += 256) {
    q[row * K + j] = quant_round(load_in(x, row * K + j, in_code), 1.0f / scale);
}
```

线程 `t` 处理 `t, t+256, t+512, ...`。相邻线程第一次迭代访问相邻元素，所以
基本 coalescing 没问题；慢点不应粗暴归因于“完全不合并的访存”。真正明显的源码成本是：

- 每个循环迭代只显式读取 2 或 4 bytes；
- 每次 `load_in` 都携带通用 dtype 分派；
- K 较大时会产生较多循环、比较、地址计算和分支；
- 两遍扫描让上述控制开销重复出现；
- block reduction 使用 `red[256]`，包含初始 barrier 和 8 轮树形 reduction barrier。

最后一项没有被这个 PR 优化，看到新 kernel 后不要误以为整个 reduction 也被改写了。

## 3. `uint4` 是 16-byte vector，不是四个 INT4

新代码先用模板计算一个 vector 能装多少个输入元素：

```cpp
template <typename InputT>
struct RowwiseVectorTraits {
    static constexpr int kElements = sizeof(uint4) / sizeof(InputT);
};
```

HIP 的 `uint4` 是四个 32-bit unsigned integer，总大小 128 bit，也就是 16 bytes：

| `InputT` | 单元素大小 | `kElements` | 每次 load |
|---|---:|---:|---:|
| `float` | 4 B | 4 | 4 个 FP32 |
| `__half` | 2 B | 8 | 8 个 FP16 |
| `__bf16` | 2 B | 8 | 8 个 BF16 |

这里的 “4” 是 vector 中的四个 32-bit word，与 4-bit quantization 没有任何关系。

## 4. 新 kernel 的线程与 chunk 映射

launch 仍然是一个 256-thread block 处理一行，但迭代单位从 element 变成 16-byte chunk：

```cpp
constexpr int kElements = RowwiseVectorTraits<InputT>::kElements;
const int row = blockIdx.x;
const int t = threadIdx.x;
const int chunks = K / kElements;

for (int chunk_idx = t; chunk_idx < chunks; chunk_idx += 256) {
    ...
}
```

以 BF16、`K=4096` 为例：

```text
kElements = 8
chunks    = 4096 / 8 = 512

thread 0   -> chunk 0,   256
thread 1   -> chunk 1,   257
...
thread 255 -> chunk 255, 511
```

第一轮一个 block 的 256 个线程覆盖 `256 × 16 B = 4096 B` 连续输入；第二轮继续覆盖
下一段 4096 B。对同一个 wave32，相邻 lane 读取相邻的 16-byte chunk，访问规律非常整齐。

## 5. 第一遍：global memory 到寄存器，再在寄存器内展开

热路径是：

```cpp
const uint4 raw = *reinterpret_cast<const uint4*>(
    x + row_offset + static_cast<int64_t>(chunk_idx) * kElements);
const InputT* values = reinterpret_cast<const InputT*>(&raw);

#pragma unroll
for (int i = 0; i < kElements; ++i) {
    lmax = fmaxf(lmax, fabsf(load_row_value<InputT>(values[i])));
}
```

理解这段代码要分三层：

1. `reinterpret_cast<const uint4*>` 要求地址满足 16-byte 对齐，发出一笔 128-bit 读取。
2. `raw` 是线程私有的局部值，编译器通常会把它拆在 VGPR 中，而不是再次访问 global。
3. `values[i]` 只是用真实 `InputT` 重新解释这 16 bytes；循环完全展开后逐个转成 FP32 求绝对值。

模板参数已经确定 dtype，所以 `load_row_value<InputT>` 可以在编译期选择正确转换，不再在
每个元素上检查 `in_code`。

因此收益来自两个叠加部分：

- typed specialization：删除热循环里的通用 dtype 解释；
- 16-byte load：减少显式 load 指令数量、循环次数和地址控制工作。

不能从源码直接声称 DRAM 字节数减少。旧、新 kernel 都至少要读取相同数量的输入元素，
而且都扫描两遍；第一遍的数据是否在第二遍命中 cache，要由 profiler 或受控实验判断。

## 6. reduction 完全没变

新 kernel 仍然使用：

```cpp
__shared__ float red[256];
red[t] = lmax;
__syncthreads();
for (int s = 128; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
}
```

执行序列是 `s=128,64,32,16,8,4,2,1`。每轮活跃线程减半，最后 `red[0]` 是整行
最大绝对值。源码中每轮后都有 block barrier，因此即使最后几轮已经落在同一个 wave，
也没有使用 wave shuffle 特化。

这是一个很好的 kernel 阅读习惯：看到 2× 加速，不要自动假设所有显眼的低效都被修了。
本 PR 只替换了两遍输入扫描，reduction 是明确保留的未来优化空间。

## 7. 第二遍：相同的 16-byte load，逐元素 INT8 store

得到 scale 后：

```cpp
const float scale = fmaxf(red[0] / kInt8Max, 1e-30f);
if (t == 0) scales[row] = scale;
const float inv = 1.0f / scale;

for (int chunk_idx = t; chunk_idx < chunks; chunk_idx += 256) {
    const int64_t base = row_offset + int64_t(chunk_idx) * kElements;
    const uint4 raw = *reinterpret_cast<const uint4*>(x + base);
    const InputT* values = reinterpret_cast<const InputT*>(&raw);
    #pragma unroll
    for (int i = 0; i < kElements; ++i) {
        q[base + i] = quant_round(load_row_value<InputT>(values[i]), inv);
    }
}
```

第二遍仍重新读取输入。输出在源码层面是逐元素 `int8_t` store，并没有手写 `uint2`、
`uint4` 或 packed output store。编译器可能合并一部分相邻 store，但文档和面试中只能说
“输入显式向量化”，不能说“输入输出都手写成 128-bit vector”。

`quant_round` 的顺序是：

```text
float input
 -> multiply by inverse scale
 -> rintf, round-to-nearest-even
 -> convert to int
 -> clamp [-128, 127]
 -> cast int8
```

## 8. 为什么 launcher 同时检查首地址与 K

fast path 条件是：

```cpp
const bool vector_aligned = uintptr_t(x) % sizeof(uint4) == 0;

FP32: vector_aligned && K % 4 == 0
FP16: vector_aligned && K % 8 == 0
BF16: vector_aligned && K % 8 == 0
```

两个条件缺一不可：

- 首地址对齐保证第 0 行的第一次 16-byte load 合法；
- `K % kElements == 0` 保证没有需要单独处理的 tail；
- 同时也保证每一行占整倍数的 16 bytes，所以第 1、2、… 行首地址继续保持 16-byte 对齐。

`tensor.is_contiguous()` 不足以证明首地址对齐。一个带 storage offset 的 view 仍可 contiguous，
但 `data_ptr()` 可能偏移 2 或 4 bytes。PR 保留 scalar fallback 正是为了处理这种输入和
不能整除 vector width 的 K。

完整 dispatch 是：

```text
M == 0
  -> return，避免 zero-block launch

支持的 dtype + x 首地址 16B 对齐 + K 可被 vector width 整除
  -> quantize_int8_rowwise_vectorized_kernel<InputT>

其他输入
  -> 原 quantize_int8_rowwise_kernel
```

## 9. 为什么不缓存第一遍读到的整行数据

看完两遍扫描后，很自然会问：第一遍把值留在寄存器里，reduction 后直接量化，不就少读
一遍了吗？问题是活跃值的规模。

以 `K=14336`、256 threads、BF16 为例，一个线程约负责：

```text
14336 / 256 = 56 elements
```

若 56 个值跨整个 block reduction 一直保持活跃，可能显著增加 VGPR 数和 live range，降低
occupancy，严重时还会 spill 到 private memory。省掉一次源码 load 不保证更快，尤其第一遍
数据可能已经在 cache 中。

这也是 #127 与 PR #129 的关键区别：#129 每 lane 固定只缓存 4 个值，寄存器预算可控；
#127 每线程负责的元素数随 K 增长。

## 10. 如何从零写出这类 kernel

可以按下面的顺序，而不是先写 `uint4`：

1. 固定数学 contract：scale 下界、rounding mode、clamp 范围和输出 dtype。
2. 先写正确的 scalar 两遍版本，明确一 block 一 row 的映射。
3. 找到热循环中的动态信息。本例是 `in_code`，可移到模板实例化阶段。
4. 选 vector width。16 bytes 与常见全局访存粒度和 FP16/BF16 的 8-element chunk 匹配。
5. 证明每一笔 vector load 都不越界：检查 base alignment、row stride 和 tail。
6. 保留 scalar fallback，不让 fast path 限制公开 API。
7. 先做 bitwise/容差正确性，再比较 aligned fast path 与 misaligned/ragged control。
8. 用生成 ISA 或 profiler 验证是否真的出现宽 load；只看 C++ 类型名不够。

## 11. 这个 PR 能教你的通用 kernel 原则

- 向量化首先是“每个线程每次搬多少连续字节”，不是“让更多线程工作”。
- compile-time specialization 可以与 vector load 同样重要。
- alignment guard 必须覆盖每一行，而不是只看 tensor 连续性。
- 源码 load 次数、cache transaction 和 DRAM bytes 是三个不同概念。
- 只优化最热、最确定的部分，保留成熟 fallback，往往比一次重写 reduction 更容易安全落地。

## 12. 已有性能证据与边界

已有记录的环境为 Radeon 8060S（`gfx1151`）、ROCm 7.2、BF16、六个代表 shape：

```text
几何平均：1.96×
最大值：  2.39×
```

这些是 rowwise quantizer 的 microbenchmark，不是包含 INT8 GEMM、模型调度和端到端生成的
总收益。本文没有重新运行硬件 benchmark；性能数字沿用 PR/已有实验记录。源码可以支持的
机制判断是：typed fast path、16-byte input loads、减少循环和地址控制；它不能单独证明
DRAM 带宽、cache hit rate 或 occupancy 是最终主因。

## 13. 继续优化时先看什么

合理的下一步不是盲目加大 vector width，而是逐个验证：

- 用 wave shuffle 先做 wave-local max，再用少量 LDS 合并 wave 结果，减少 block barrier；
- 尝试 packed INT8 store，验证编译器当前是否已经自动合并；
- 对短 K 考虑一个 wave 处理一行，降低 block 级同步固定成本；
- 对“缓存第一遍值”做严格的 VGPR、scratch 和 occupancy 对照；
- 按 K/shape 分段 dispatch，而不是假定同一个 reduction 结构适合所有行宽。

读完源码后，最准确的一句话总结是：

> PR #127 保留一 block 一 row、两遍扫描和 LDS tree reduction，只把两遍热循环改成
> compile-time typed 的 16-byte chunk 读取，并用严格 alignment/K guard 保留 scalar fallback。
