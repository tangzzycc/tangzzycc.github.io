# Comfy-Kitchen PR #157：Triton Stochastic FP8 Rounding Kernel 深读

## PR 动机

Comfy-Kitchen 的 `stochastic_rounding_fp8` 原本已有 eager、CUDA 和 HIP 实现，但 Triton
backend 没有对应 kernel。正常情况下 NVIDIA 用户会优先走原生 CUDA extension；问题出现在
CUDA extension 因 PyTorch/CUDA 版本组合而不可用时，registry 会直接退回 eager。

ComfyUI issue #15311 记录了一个真实后果：CUDA 12.8、RTX 4090、接近显存上限的
`fp8_scaled` 模型叠加多个 LoRA 时，整个大 weight tensor 会交给 eager 随机舍入。eager
实现同时物化 sign、absolute value、exponent、mantissa 等多个完整中间 tensor，再加上调用方
预先生成的 `uint8` RNG buffer，导致显存压力和大量逐算子 launch/synchronization；模型 partial
load 表现为长时间无法完成。

PR #157 增加一个单 pass Triton kernel，在 native CUDA backend 不可用但 Triton 可用时接住
这条路径。它逐元素读取输入和一个随机 byte，直接把结果写回 RNG 的 storage，并将同一块
storage 作为 FP8 tensor 返回。核心价值不是发明新的随机舍入公式，而是把 eager 的整张量
表达式融合成一个 kernel，同时复用已有的 1-byte-per-element RNG buffer。

本文基于公开 PR 的最终 head：

- PR：<https://github.com/Comfy-Org/comfy-kitchen/pull/157>
- Head：`73ed759ac02d07ed2a091d834e86f8e8aa8eb89d`
- Parent：`b678fdf63378409676aa5596721445d33794d0ea`
- 真实提交范围：`b678fdf..73ed759`
- 核心 kernel：`comfy_kitchen/backends/triton/quantization.py`
- 注册与 capability：`comfy_kitchen/backends/triton/__init__.py`
- 正确性测试：`tests/test_qdq.py`

固定读取：

```bash
git -C ~/comfy-kitchen show \
  refs/remotes/pr/157:comfy_kitchen/backends/triton/quantization.py
```

## 1. Public operation 的 contract

公开接口是：

```python
stochastic_rounding_fp8(
    x: Tensor,
    rng: Tensor,
    output_type: dtype = torch.float8_e4m3fn,
) -> Tensor
```

输入与输出 contract：

```text
x           任意 shape，FP32 / FP16 / BF16
rng         与 x 同 shape、同 device，uint8
output      与 x 同 shape，E4M3FN 或 E5M2
```

这里的 `rng` 不是 seed，也不是 generator state，而是已经生成好的“一元素一随机 byte”张量。
kernel 不在内部生成随机数；相同的 `x` 和 `rng` 会产生相同输出。

对于 contiguous `rng`：

```text
output.data_ptr() == rng.data_ptr()
```

因为 FP8 与 `uint8` 都是一 byte，wrapper 可以把 RNG storage 直接 view 成目标 FP8 dtype。调用后
原随机 byte 被 FP8 encoding 覆盖，所以 `rng` 是输入兼输出，不应在调用后继续当随机源使用。

## 2. 为什么要 stochastic rounding

设某个输入位于相邻两个可表示 FP8 数 `L` 和 `U` 之间：

```text
x = L + p × (U-L),  0 <= p < 1
```

随机舍入希望：

```text
P(output = U) = p
P(output = L) = 1-p
```

理想的连续概率模型在未饱和区间内满足：

```text
E[output] = (1-p)L + pU = x
```

普通 round-to-nearest 每次都会把同一个值舍到同一侧；大量权重反复量化时，误差可能形成稳定
bias。stochastic rounding 用概率保留低位信息。这里先写的是理想定义；本实现只有 256 个
随机等级，实际向上概率以 `1/256` 为粒度。这里的目标值也不是未经转换的原始 FP32/BF16，
而是先转换成 FP16 后的值。

## 3. 两种 FP8 format 的常量

wrapper 根据 `output_type` 选择编译期参数：

| 格式 | exponent bits | mantissa bits | bias | finite max | 最小 normal | 最小 subnormal |
|---|---:|---:|---:|---:|---:|---:|
| E4M3FN | 4 | 3 | 7 | 448 | `2^-6` | `2^-9` |
| E5M2 | 5 | 2 | 15 | 57344 | `2^-14` | `2^-16` |

传给 Triton kernel 的是：

```text
fp8_max
exponent_max = 2^exponent_bits - 1
exponent_bias
mantissa_bits
```

这些参数都是 `tl.constexpr`。E4M3FN 与 E5M2、不同 block size 会生成不同 specialization，
热循环里没有运行时格式分支。

## 4. 先经过 FP16 不是偶然精度损失

kernel 第一条数值路径是：

```python
x = tl.load(...).to(tl.float16).to(tl.float32)
```

也就是：

```text
FP32 / FP16 / BF16 input
  -> 先舍入到 FP16
  -> 再提升到 FP32 做 exponent/mantissa 算术
```

这与已有 CUDA kernel 的：

```cpp
__half2float(__float2half(static_cast<float>(src[idx])))
```

保持一致。它不是追求“直接从原 dtype 得到数学上最精确的 FP8 随机舍入”，而是保证已有 public
operation 在不同 backend 间采用同一数值 contract。

这也意味着：

- FP32 输入先失去 FP16 之外的低位；
- BF16 的 exponent range 大于 FP16，极大 BF16 值可能先变成 FP16 infinity；
- 随机舍入的期望对应 FP16-rounded value；
- 改成直接 FP32 算术即使看似更精确，也会改变与 CUDA backend 的逐元素结果。

## 5. 一维 program mapping

kernel 使用最简单的一维映射：

```python
pid = tl.program_id(0).to(tl.int64)
block_start = pid * block_size
offsets = block_start + tl.arange(0, block_size)
mask = offsets < n_elements
```

grid：

```text
ceil(n_elements / block_size)
```

每个 Triton program 负责一个连续的 element tile。这里的 `block_size` 是一个 program 的逻辑
vector extent，不等于 CUDA/HIP 的 `blockDim.x`；PR 归档的编译结果中各 specialization 使用
4 warps，但一个 program 可以通过每 lane 多元素映射覆盖 128–1024 个逻辑元素。

## 6. Adaptive block size 在做什么

wrapper 沿用同文件 FP8 quant/dequant 的分段规则：

| 元素数 | block size |
|---:|---:|
| `< 32K` | 128 |
| `32K .. <128K` | 256 |
| `128K .. <512K` | 512 |
| `>=512K` | 1024 |

小 tensor 用较小 tile，避免单 program 固定工作过多；大 tensor 用更宽 tile，减少 program
数量和调度开销。PR 测试专门选择：

```text
17, 32769, 131073, 524289
```

它们同时覆盖非整除 tail 和每个阈值刚过界的 dispatch，而不是只测容易整除的 power-of-two
shape。

这组阈值来自已有 Triton elementwise policy。源码可以确认它改变 program decomposition；
不能只凭 if/else 断言每个区间已经是所有 GPU 上的全局最优值。

## 7. 为什么地址计算显式提升到 64 bit

PR 写成：

```python
pid = tl.program_id(0).to(tl.int64)
block_start = pid * block_size
```

如果 `pid * block_size` 保持 32-bit，超过 `2^31` elements 的扁平 tensor 可能在地址形成阶段
溢出，即使 Python 的 `numel()` 本身是 64-bit。先把 `pid` 转为 `tl.int64`，后续 `offsets`
随算术提升为 64-bit。

这不代表 PR 实际 benchmark 了多十亿元素 tensor；它是从 pointer arithmetic contract 上消除
明显的规模上限。

## 8. sign、zero 与 exponent 的处理

每个元素先计算：

```python
abs_x = tl.abs(x)
sign = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
exponent = tl.floor(tl.log2(abs_x)) + exponent_bias
exponent = clamp(exponent, 0, exponent_max)
normal = exponent != 0
```

`sign` 对 `+0` 和 `-0` 都选择 0，因此输出按既有 eager contract 归一成零，不承诺保留 negative
zero bit。

对 `abs_x == 0`，`log2(0)` 是 `-inf`，clamp 后 exponent 为 0，进入 subnormal 公式；最终又乘
zero sign，结果仍是零。Triton 的表达式是无分支向量计算，不需要专门 return。

## 9. Normal 路径如何定位相邻 FP8 数

对 normal value，先求当前 binade 的基准：

```text
e              = exponent - bias
exponent_scale = 2^e
mantissa_levels= 2^mantissa_bits
```

再把真实尾数换算成“FP8 mantissa 格点坐标”：

```text
normal_mantissa
  = (abs_x / exponent_scale - 1) × mantissa_levels
```

例如 E4M3 的 `mantissa_levels=8`。若：

```text
abs_x / exponent_scale = 1.3125
normal_mantissa         = 2.5
```

说明输入正好位于 mantissa code 2 与 3 中间。随机舍入只需决定把 `2.5` floor 到 2，还是越过
下一个整数成为 3。

## 10. Subnormal 路径为什么使用另一套 scale

subnormal 没有隐含 leading 1。它的量化步长是：

```text
subnormal_step = 2^(-bias + 1 - mantissa_bits)
```

因此：

```python
subnormal_mantissa = abs_x / subnormal_step
```

E4M3FN 的 step 是 `2^-9`，E5M2 的 step 是 `2^-16`。随机 floor 得到整数 mantissa 后，重建：

```text
subnormal_value
  = 2^(-bias+1) × (mantissa / 2^mantissa_bits)
  = subnormal_step × integer_mantissa
```

normal 与 subnormal 最终共享同一段随机化代码，但“格点坐标”的定义不同。

## 11. 一个 random byte 如何决定向上舍入

随机数来自：

```python
random = uint8 / 256.0
```

所以 `random` 离散均匀分布在：

```text
0, 1/256, 2/256, ..., 255/256
```

然后：

```python
mantissa = floor(mantissa_coordinate + random)
```

若坐标为 `k+f`，只有 `random >= 1-f` 时结果才变为 `k+1`。因此向上概率由 fractional
mantissa 决定，并被量化为 `floor(256f)/256`。random 永远小于 1，所以输入已精确落在 FP8
格点时不会被无故推到下一格。

在 normal 区间，先转 FP16 后，被丢弃 mantissa bits 与这套 8-bit RNG 很好地对齐：

- FP16 有 10 个显式 mantissa bits；E4M3 丢 7 bits；
- E5M2 丢 8 bits；
- E4M3 normal 的 fractional coordinate 是 `1/128` 的整数倍，E5M2 normal 是 `1/256`
  的整数倍。

E4M3 的极小 subnormal 输入不一定落在 `1/256` 概率网格上，因此更严格的说法是：实现遵循
既有 CUDA 的 byte-RNG contract，并在该离散分辨率下近似理想随机舍入；不能仅凭公式宣称所有
subnormal 输入都严格无偏。

## 12. Mantissa carry 不需要单独分支

如果 mantissa 位于当前 binade 顶端，随机加法可能使整数 mantissa 等于
`mantissa_levels`。代码不手动把 mantissa 清零并增加 exponent，而是直接重建：

```text
2^e × (1 + mantissa_levels/mantissa_levels)
= 2^(e+1)
```

它自然得到下一 binade 的最低 normal 值。最高 exponent 处若越过有限范围，则后面的 saturation
统一截到 `fp8_max`。

## 13. Saturation、infinity 与 NaN

重建后的值：

```python
rounded = sign * selected_value
rounded = max(min(rounded, fp8_max), -fp8_max)
rounded = tl.where(x != x, x, rounded)
```

行为是：

- 超过范围的有限值饱和到 `±448` 或 `±57344`；
- `+inf/-inf` 也最终饱和到对应有限极值；
- `x != x` 识别 NaN，并在 min/max 可能把 NaN 替换成有限 operand 后重新放回 NaN。

PR 报告的逐 bit CUDA 对照明确限定为 finite values。Triton 对 NaN 的目标是遵循本 PR 的
reference/test contract；没有额外证据时，不应把“所有 finite FP16 输入逐 bit 相同”扩大成
“包括所有非有限 encoding 都与原生 CUDA 逐 bit 相同”。

## 14. 为什么可以把 RNG buffer 同时当 output

kernel 参数虽然有 `rng_ptr` 和 `output_ptr`，launcher 实际传入的是同一 storage 的两个 dtype
view：

```python
rng_flat = rng.flatten()
output = rng_flat.view(output_type)

kernel(x_flat, rng_flat, output, ...)
```

每个 offset 的数据依赖是：

```text
load rng[offset]
  -> 转成 random
  -> 完成本元素 FP8 计算
  -> store output[offset]
```

不同 program/element 不读取彼此的 RNG byte，因此没有跨元素 read-after-write hazard。对于同一
元素，store 依赖已加载的 random SSA value，必须发生在读取以后。

这使 contiguous fast path 的额外 PyTorch allocation 为零。它仍然会产生正常的 global
memory traffic：读取 `x`、读取 RNG byte、写一个 FP8 byte；“零额外 allocation”不等于“零
内存访问”。

## 15. Contiguity 语义有一个容易忽略的细节

wrapper 执行：

```python
if not x.is_contiguous():
    x = x.contiguous()
if not rng.is_contiguous():
    rng = rng.contiguous()
```

因此：

- contiguous RNG：返回 tensor 与调用者 RNG 共用原 storage；
- non-contiguous RNG：先物化一份 contiguous copy，返回值复用的是这份 copy；调用者原 view
  不会被原地覆盖；
- non-contiguous x：也会增加一份输入 copy。

PR 的“0 MiB peak PyTorch-allocated memory”结论针对 contiguous benchmark 输入。不能把它
无条件推广到任意 strided input。

这种 non-contiguous 行为与当时 CUDA wrapper 一致；HIP wrapper采用更严格 contract，会直接
拒绝 non-contiguous RNG。跨 backend 讨论 alias semantics 时需要区分。

## 16. Empty tensor 为什么在 wrapper 提前返回

代码在构造 output view 后检查：

```python
if n_elements == 0:
    return output.view(orig_shape)
```

这样同时保证：

- 返回 shape/dtype 正确；
- output 仍复用 RNG storage；
- 不构造 `grid=(0,)` 的非法或 backend-dependent launch。

这类 zero-size guard 是 kernel wrapper 的一部分，不是测试中的边角装饰。

## 17. Registry 为什么限制到 CUDA SM89+

Triton backend 新增 capability：

```python
default_devices = {"cuda"}
min_compute_capability = (8, 9)
```

输入 dtype 限定 FP32/FP16/BF16，RNG 限定 uint8，输出限定 E4M3FN/E5M2。SM89 是 Ada，也是
NVIDIA native FP8 conversion 的起点。kernel 最后通过向 FP8 typed pointer store 完成 encoding；
旧架构不能只因为 exponent/mantissa 算术是 FP32 就被视为支持。

这层约束的意义是让自动 backend dispatch 在调用 kernel 前拒绝不支持的硬件，而不是等 Triton
编译或 PTX lowering 失败。第二个 commit 也让测试在选择 Triton backend 前显式跳过 SM89 以下
设备，避免强制 backend 的测试路径绕过预期 eligibility。

## 18. Eager、CUDA 与 Triton 三条路径的资源模型

### Eager

大致执行：

```text
x -> half
 -> sign / abs
 -> log2 / exponent / normal mask
 -> normal/subnormal mantissa
 -> random add + floor
 -> reconstruct
 -> clamp
 -> cast FP8
```

这些 tensor expressions 会物化多个 full-size intermediates。issue #15311 对大权重记录的是
接近显存上限时严重退化，而不是数学错误。

### Native CUDA

一个 CUDA thread 处理一个元素，128 threads/block；读取 RNG byte 后把 FP8 encoding 写回同一
位置。固定开销低，小到中等 tensor 上是强 baseline。

### Triton

一个 program 处理 128–1024 个连续元素，所有公式融合在同一 kernel 中，同样复用 RNG storage。
它填补的是“原生 CUDA extension 不可用，但 Triton backend 可用”的空档，而不是删除原生
CUDA 实现。

## 19. 测试为什么具有针对性

PR 新增的自动测试覆盖：

- 三种输入 dtype：FP32、FP16、BF16；
- 两种输出：E4M3FN、E5M2；
- 四个 block-size 边界与非整除 tail；
- NaN、±Inf、极大值、格式最大值、±0、很小 subnormal 候选；
- non-contiguous x/rng；
- contiguous RNG 的 pointer alias；
- empty tensor；
- registry 的 SM89 capability。

reference 也显式执行 `x.half().float()`，防止测试误把“更高精度但不兼容 CUDA”的实现当作
正确。

PR 描述还报告了两类额外验证：

```text
65,536 个 FP16 bit patterns × 256 RNG bytes × 两种 FP8 format
有限值范围内 Triton/CUDA mismatch = 0
```

以及：

```text
SM89 / SM90 / SM100 / SM120
× 3 input dtypes
× 2 output formats
× 4 block sizes
= 96 个 offline compile variants
```

所有归档 variant 为 4 warps、0 shared memory、0 global scratch。自动测试结果记录为
`118 passed, 4 skipped`。

## 20. 性能数据应如何理解

RTX 5090，BF16 → E4M3FN，warmup 后 median，rotating working set 至少 256 MiB：

| Elements | CUDA | Triton | Eager | Triton vs CUDA | Triton vs Eager |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 0.0136 ms | 0.0266 ms | 0.3538 ms | 1.95× slower | 13.3× faster |
| 8,388,608 | 0.0329 ms | 0.0263 ms | 0.3696 ms | 1.25× faster | 14.1× faster |
| 16,777,216 | 0.0628 ms | 0.0415 ms | 0.8942 ms | 1.51× faster | 21.5× faster |

结论应限制为：

- 小一些的 1M shape，原生 CUDA latency 更低；
- 大 shape 上归档测量中 Triton 超过该 CUDA kernel；
- 相对 eager 的收益在所有三组数据上都很大；
- 16,777,216 contiguous BF16 elements 时，CUDA/Triton 都没有额外 PyTorch allocation，
  eager 增加约 240 MiB。

不能仅从源码断言 Triton 在所有 Ada/Blackwell GPU、所有 dtype 和所有 size 上都比 CUDA 快。
究竟来自 program width、指令调度、memory coalescing 还是 CUDA baseline 的一线程粒度，需要
生成代码和 profiler 才能分解。

## 21. 为什么这不是普通 FP8 quantize kernel

普通 `quantize_per_tensor_fp8` 是：

```text
round_to_nearest(clamp(x / scale))
```

本 PR operation：

```text
先把 x 变为 FP16 contract
按 FP8 exponent binade 求相邻 mantissa 格点
用外部 random byte 概率选择上下格点
饱和并编码 FP8
```

它没有 scale 参数，也不是简单地给 `x` 加一点随机噪声后 `.to(fp8)`。随机量必须按当前
binade 的 FP8 ULP 缩放，否则不同 exponent 区间的向上概率会错误。

## 22. 常见误读

### 误读一：Triton kernel 在 GPU 上生成随机数

没有。随机 byte 由调用方提供，kernel 只消费它。

### 误读二：返回值总与调用者传入的 RNG tensor 共用 storage

只有 RNG contiguous 时成立。non-contiguous RNG 会先 copy，返回值复用 copy 的 storage。

### 误读三：算法直接对原 FP32/BF16 值做无偏 FP8 rounding

不是。它先对输入做 FP16 conversion，以匹配已有 CUDA contract。

### 误读四：block size 1024 表示 1024 CUDA threads

不是。它是 Triton program 的逻辑 element tile；归档编译配置仍是 4 warps。

### 误读五：0 MiB allocation 意味着没有内存成本

不是。它仍读取输入与 RNG、写回 FP8，只是不再物化 eager 的多个完整中间 tensor和独立输出。

### 误读六：PR 替代了 native CUDA kernel

没有。registry 优先级仍让 CUDA backend 在可用时先被选择；Triton 主要填补 CUDA extension
不可用的场景。

## 23. 如果从零实现这项优化

建议按以下顺序：

1. 固定跨 backend 数值 contract，特别是是否先转 FP16、NaN/Inf 和 saturation 行为。
2. 用相邻 FP8 格点的概率定义推导 normal 与 subnormal 两套 mantissa 坐标。
3. 证明一个 8-bit RNG 足够表达从 FP16 到目标 FP8 被丢弃的低位概率。
4. 让每个元素只依赖自己的 RNG byte，才能安全做原地 RNG→FP8 覆盖。
5. 把 format 参数做成 compile-time specialization，避免热路径动态分支。
6. 对 flat index 使用 64-bit，tail 使用 mask，zero-size 在 wrapper 退出。
7. 把 shape、device、RNG dtype 和 output dtype contract 放在 registry/wrapper 两层保护。
8. 测试必须跨 block-size boundary，而不只是大而整齐的 tensor。
9. 用 exhaustive finite-value comparison 验证逐 bit 兼容，再单独定义 NaN/Inf contract。
10. 性能同时报告 latency 和 peak allocated memory，因为本 PR 的首要故障是内存压力。

## 24. 当前边界与后续方向

- Triton capability 仅注册在 CUDA SM89+；AMD 已有独立 HIP backend，不走这条实现。
- 只支持 E4M3FN 与 E5M2。
- RNG 必须预先生成并占一 byte/element；PR 没有消除调用方 RNG allocation。
- non-contiguous x/rng 仍可能产生 materialization。
- format 常量与 block size 会增加 JIT specialization 数量，但离线编译已覆盖 96 种目标组合。
- 当前性能表只有 RTX 5090；Ada、Hopper、其他 Blackwell 和不同 Triton 版本需要独立测量。
- 可继续对比显式 `num_warps`、更宽 load/store、persistent decomposition 或在上游按 chunk
  生成 RNG，但任何变化都必须保持随机 byte 消费顺序和 backend reproducibility contract。

## 25. 最终应记住的一句话

> PR #157 把原本由 eager 多个整张量算子完成的 FP8 stochastic rounding 融合成一个
> SM89+ Triton elementwise kernel：输入先按 CUDA contract 收窄到 FP16，再在 FP32 中计算
> normal/subnormal mantissa 格点，用每元素一个随机 byte 决定向上舍入，并把 FP8 结果原地写回
> RNG storage。它的主要工程收益是消除大权重上的中间 tensor 和独立输出 allocation，同时在
> native CUDA extension 不可用时保留高效 GPU 路径。
