# FlyDSL PR #1057：RDNA3 WMMA GEMM Kernel 与 gfx1151 Tile Policy

## PR 动机

FlyDSL 已有一套 RDNA3 wave32 WMMA GEMM kernel，但默认 tile heuristic 主要按 gfx1100 的
96 CU 调优。gfx1151 只有 40 个物理 CU；同一个 `128×128` 或 `256×256` block tile 在两种
GPU 上产生完全不同的全设备 grid depth。直接复用 gfx1100 阈值，会为了“增加并行度”选择
过小 tile，牺牲不必要的数据复用；也可能在大 tile 下让 workgroup 数太少，铺不满设备。

PR #1057 没有重写 WMMA 主循环。它做的是：

1. 为 gfx1151 增加基于合法 tile 和 workgroup 数的选择规则；
2. 最新提交把已测的 `256×256` occupancy band 扩展到 near-square rectangular GEMM；
3. 从输入 tensor 所在 device 获取 architecture，而不是依赖不相关的全局当前设备；
4. 让 arch 进入 dispatch/autotune/build cache，并真正传入 `@flyc.jit(arch=...)`，避免异构
   gfx11 设备复用错误 code object。

要真正理解这个 PR，不能只看几条 Python `if`：必须先看懂一个 tile tuple 如何改变 wave
数量、LDS layout、global copy、WMMA accumulator 和 grid depth。

本文基于最新公开 head：

- PR：<https://github.com/ROCm/FlyDSL/pull/1057>
- Head：`ddd20408baba52462c488d50c2a048754458d518`
- 真实提交范围：`3e4e2abc^..ddd20408`
- GEMM kernel：`kernels/gemm/rdna3_f16_gemm.py`
- tile policy：`kernels/gemm/rdna3_f16_gemm_autotune.py`
- JIT arch：`python/flydsl/compiler/jit_function.py`

固定读取：

```bash
git -C ~/projects/FlyDSL show \
  refs/remotes/pr/1057:kernels/gemm/rdna3_f16_gemm.py
git -C ~/projects/FlyDSL show \
  refs/remotes/pr/1057:kernels/gemm/rdna3_f16_gemm_autotune.py
```

## 1. Kernel 的数学与输入布局

接口计算：

```text
C[M,N] = A[M,K] @ B_T[N,K]^T
```

`B_T` 已按 `[N,K]` row-major 存储，所以 A row 和 B_T row 都沿 K 连续。kernel 不需要在
global memory 中真的转置 B。

输入支持 BF16/F16，WMMA 累加到 FP32；输出可为 BF16/F16/FP32。BF16 还可选择普通
round-to-nearest 或 stochastic rounding。

目标是 gfx11 wave32，硬件 atom 是：

```text
16 × 16 × 16 BF16/F16 -> FP32
```

常量：

```text
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WAVE_SIZE = 32
LOAD_VEC = 8 elements = 16 bytes
```

## 2. Tile tuple 如何变成真实 block

policy 中一个 tile 是：

```text
(reg_m, reg_n, reg_k, waves_m, waves_n)
```

展开公式：

```text
BLOCK_M = 16 × reg_m × waves_m
BLOCK_N = 16 × reg_n × waves_n
BLOCK_K = 16 × reg_k
NUM_WAVES = waves_m × waves_n
THREADS = 32 × NUM_WAVES
```

例如默认 `TILE_128x128x32 = (4,4,2,2,2)`：

```text
BLOCK_M = 16×4×2 = 128
BLOCK_N = 16×4×2 = 128
BLOCK_K = 16×2   = 32
waves   = 2×2    = 4
threads = 4×32   = 128
```

一个 wave 在 M 方向维护 `reg_m` 个 16-row fragment，在 N 方向维护 `reg_n` 个 16-column
fragment。每线程 FP32 accumulator 数是：

```text
8 × reg_m × reg_n
```

`(4,4,...)` 时就是 128 个 FP32 accumulator values/thread。这是大 tile 的关键成本之一。

## 3. 候选 tile 的资源含义

| Block tile | Tuple | Threads | 直觉 |
|---|---|---:|---|
| `256×256×32` | `(4,4,2,4,4)` | 512 | 最大 M/N 复用，网格最浅，LDS/线程数最重 |
| `128×128×32` | `(4,4,2,2,2)` | 128 | 复用和 grid depth 的折中 |
| `128×64×32` | `(4,2,2,2,2)` | 128 | N 较窄，更多 workgroups |
| `64×64×64` | `(2,2,4,2,2)` | 128 | K tile 更深，M/N tile 更小 |
| `32×64×64` | `(2,2,4,1,2)` | 64 | 小矩阵增加 grid 粒度 |
| `32×32×64` | `(2,2,4,1,1)` | 32 | 最细粒度，只有一个 wave |

大 tile 通常减少 A/B 重复加载、提高 reuse，但会：

- 减少 workgroup 总数；
- 增加每 workgroup 的线程、LDS 或 accumulator；
- 可能降低同驻留 workgroups；
- 对 shape 整除提出更严格要求。

所以 tile selection 不是“矩阵越大就选越大”。真正要平衡的是 reuse、grid depth 和每 WG
资源。

## 4. 合法性过滤不是性能 heuristic

`_tile_workgroups(M,N,K,cfg)` 先判断 tile 能不能运行：

```python
if M % block_m or N % block_n or K % block_k:
    return None

if K // block_k < 2:
    return None

if A_tile_elems % (threads * 8):
    return None
if B_tile_elems % (threads * 8):
    return None

if double_buffered_lds_bytes > 64 * 1024:
    return None
```

逐条含义：

1. 当前 kernel 无 M/N/K tail predication，三维都必须整除 block tile。
2. software pipeline 至少需要两个 K tiles，才能在计算当前 tile 时预取下一个。
3. global→LDS copy 以每线程 8 个 16-bit 元素，即 128 bit 为原子；tile 必须均匀分配。
4. A/B 的双缓冲 LDS 不能超过 64 KiB。

合法后才计算：

```text
workgroups = (M / BLOCK_M) × (N / BLOCK_N)
```

“非法”不等于“慢”，而是违反 kernel 构造、vector copy 或 pipeline contract。

## 5. Global→register→LDS 的线程几何

copy geometry：

```python
THRS_K = BLOCK_K / 8
THRS_M = THREADS / THRS_K
```

线程可理解成二维 `(tm, tk)`：

```text
tk 决定 K 方向哪个 8-element/16-byte chunk
tm 决定当前搬哪一行
tiled copy 再沿 M/N 重复，覆盖整个 A/B tile
```

以 `128×128×32`、128 threads 为例：

```text
THRS_K = 32/8 = 4
THRS_M = 128/4 = 32
```

一次线程布局覆盖 32 rows × 每行 4 chunks = 128 个 16-byte chunks。A tile 有
128 rows，因此 copy 会重复四次；B tile 同理。

源码的数据路径是：

```text
global A/B
  -> BufferCopy128b
  -> per-thread register fragments frag_copy_A/B
  -> UniversalCopy128b
  -> LDS current/next buffer
```

global load 与 LDS store 分开，使下一 K tile 可以先进入寄存器，再与当前 tile 的 WMMA
计算重叠。

## 6. 两种 LDS layout：`pad` 与 `kblock`

普通 tile 使用 row-major padded layout：

```text
row stride = BLOCK_K + K_PAD
```

padding 改变相邻 row 的 LDS bank 映射。双缓冲总元素数：

```text
LDS_TOTAL = 2 × [(BLOCK_M × row_stride_A)
               + (BLOCK_N × row_stride_B)]
```

`256×256×32` 若继续 padding 会超过 64 KiB，因此使用 `kblock`：

```python
layout = (rows, (LOAD_VEC, BLOCK_K/LOAD_VEC))
stride = (LOAD_VEC, (1, rows*LOAD_VEC))
```

直观上它把同一个 K 的 8-element block 作为内层，再跨 rows 排列，取消 padding：

```text
double buffer bytes
= 2 × (256+256) × 32 × 2
= 65536 bytes
```

正好卡在 64 KiB。`kblock` 不是普适更快的布局，而是让最大 tile 在资源上可构建，并改变
bank/访问模式；最终仍需实测。

## 7. Wave 与 register tile 映射

线程映射：

```python
wave_id = tid // 32
lane    = tid % 32
wave_m  = wave_id // waves_n
wave_n  = wave_id % waves_n
```

对 `waves_m=2, waves_n=2`：

```text
wave 0 -> (wave_m=0, wave_n=0)
wave 1 -> (0,1)
wave 2 -> (1,0)
wave 3 -> (1,1)
```

每 wave 的输出区域：

```text
M extent = 16 × reg_m
N extent = 16 × reg_n
```

默认 `reg_m=reg_n=4`，每 wave 覆盖 `64×64`；2×2 waves 合并成 `128×128` block。

## 8. gfx11 WMMA operand ABI：为什么 lane 16..31 看起来重复

源码明确使用：

```python
lane16 = lane % 16
```

gfx11 的 v16 operand ABI 中，lane 0..15 与 lane 16..31 对应相同的 M/N row fragment；每个
lane operand 携带连续 16 个 K elements。源码从 LDS 读取两个 v8，再拼成 v16：

```python
v_lo = load 8 elements at K + 0
v_hi = load 8 elements at K + 8
operand = concat(v_lo, v_hi)
```

A row：

```text
wave_m × (reg_m×16) + rm×16 + lane16
```

B row：

```text
wave_n × (reg_n×16) + rn×16 + lane16
```

因此同一个 wave 的 lane 0 和 lane 16 会读取相同 row fragment。这是 intrinsic ABI 的要求，
不是线程映射 bug。源码中也留下了可探索方向：用 LDS swizzle/broadcast 减少重复读取，但在
没有生成代码与测量前不能断言一定更快。

## 9. 单个 LDS K tile 如何计算

每个 `rk` 表示一个 `WMMA_K=16` step：

```python
for rk in range(reg_k):
    b_vecs = load reg_n B fragments
    for rm in range(reg_m):
        a_vec = load one A fragment
        for rn in range(reg_n):
            acc[rm,rn] = wmma(a_vec, b_vecs[rn], acc[rm,rn])
```

一个 A fragment 会与当前 wave 的所有 `reg_n` 个 B fragments 组合；一个 B fragment 也会
被所有 `reg_m` 个 A fragments 使用。这就是 register tiling 带来的计算复用。

`reg_m=4, reg_n=4` 时，一个 wave 每个 rk 发出 16 次 WMMA。`reg_k=2` 时，一个
`BLOCK_K=32` tile 发出 32 次 WMMA/wave。

## 10. 双缓冲 pipeline 的精确时序

LDS 分为 buffer 0 和 buffer 1。初始化：

```text
global load K tile 0 -> registers
registers -> LDS buffer 0
barrier
```

循环的 `_one_k_tile`：

```text
1. global load next K tile -> register fragments
2. compute current LDS buffer -> FP32 accumulators
3. register fragments -> alternate LDS buffer
4. optional scheduling hints
5. barrier
```

循环结束后，再计算最后剩余的 LDS tile。

关键是第 1 步不会立刻覆盖 LDS，而是先进入线程私有 registers，所以能与第 2 步的 LDS read
和 WMMA 形成 software pipeline。barrier 同时承担：当前 LDS 已不再被读取，且下一 LDS tile
已由所有线程写完。

## 11. Grid swizzle 为什么沿 M 分组

普通线性 block id 会变换成 `(bid_m,bid_n)`：

```python
num_pid_in_group = group_width * grid_n
group_id = pid // num_pid_in_group
pid_in_group = pid % num_pid_in_group

bid_m = group_id * group_width + pid_in_group % group_width
bid_n = pid_in_group // group_width
```

效果是连续 workgroups 先在一个 M group 内移动，再推进 N。这样一批并发 workgroups 更可能
共享相同的 B_T tile，改善 L2 locality。`group_width` 选为不超过 `group_m` 且整除 grid_m
的最大值，保证映射覆盖完整、无不规则尾部。

这属于 block traversal 优化，与 WMMA 数学无关；同一个 block tile 改 traversal，算术量不变，
主要影响 cache 行为。

## 12. 输出 fragment 如何写回

硬件 accumulator 是每个 atom 一个 v8 FP32。FlyDSL 的 `tiled_mma` 编码了：

- gfx11 lane 到 16×16 accumulator 的映射；
- wave 在 block M/N 方向的 band；
- `reg_m × reg_n` 多个 accumulator fragments 的顺序。

writeback 流程：

```text
FP32 accumulator fragments
 -> 按 tiled_mma 布局放入 frag_C
 -> 可选 FP32->BF16 stochastic rounding
 -> retile 成 global copy layout
 -> tiled copy 写 C
```

普通 BF16/F16 使用类型转换；stochastic BF16 为每 8 个 accumulator values 使用 Philox
随机位完成无偏期望的舍入。PR #1057 没有修改这段 epilogue，但 tile 改大时 accumulator
数量会随 `reg_m×reg_n` 改变，仍是资源模型的一部分。

## 13. gfx1151 为什么不能直接用 96-CU 规则

以 `M=N=768`、`128×128` tile 为例：

```text
grid = 6 × 6 = 36 workgroups

gfx1151：36 / 40 = 0.90 WG per physical CU
gfx1100：36 / 96 = 0.375 WG per physical CU
```

对 gfx1151，36 WG 已接近一层设备覆盖；对 gfx1100，网格很浅。同一个 workgroup count 在
两种设备上表示的并行度完全不同。

但把常量 96 机械改成 40 仍不够，因为 tile 选择还依赖：

- shape 是否整除；
- K pipeline 深度；
- 128-bit copy 是否能均分；
- LDS layout 与上限；
- accumulator/VGPR；
- L2 reuse 和 traversal；
- 方阵、near-square 和强偏斜矩阵的复用差异。

## 14. 最新 gfx1151 heuristic 逐行解释

先构造所有合法 tile 到 WG 数的映射。特殊小方阵：

```python
if M == N == 256 and TILE_32x64x64 in feasible:
    return TILE_32x64x64
```

大 tile在 256³ 上只有极少 workgroups，细粒度 `32×64` 可生成更多并行工作。

再看 `128×128` 的 WG 数：

```python
one_wave = 0.9*40 <= wgs_128 <= 1.6*40
deep_grid = wgs_128 >= 2.5*40
```

变量名 `one_wave` 不准确，它描述的是约一层 workgroup grid，不是 GPU wavefront。文档或
面试中最好称它为 one-layer grid band：`36..64 WG`。

若进入 one-layer 或 deep-grid 区域：

```python
wgs_256 = feasible.get(TILE_256x256x32, 0)
near_square = max(M,N) <= 2 * min(M,N)

if near_square and 32 <= wgs_256 <= 40:
    return TILE_256x256x32

if M == N:
    if M == 1024 and K >= 1024:
        return TILE_64x64x64
    return TILE_128x128x32
```

最后其余输入回到 gfx1100 policy。

最新 head 的关键变化是 `near_square`：只有长宽比不超过 2:1，且 256 tile 仍产生 32..40
个 WG 时，才使用已测的 256 tile occupancy band。强偏斜矩阵不套用这条规则。

## 15. 为什么 `256×256` 只在 32..40 WG 使用

`256×256` tile 的优势是每个 block 内 A/B reuse 高；代价是 512 threads、64 KiB LDS 和更浅
grid。对 40-CU gfx1151：

```text
32..40 workgroups
```

大致能覆盖大部分或全部 CUs，同时保留大 tile reuse。若只有 4 或 8 个 WG，设备明显
starved；若 WG 数非常深，128 tile 可能以更低单 WG 资源和更灵活调度取得更稳结果。

这个区间是测量驱动的有限规则，不是从理论峰值直接推导出的定律。

## 16. 为什么 near-square 才扩展 256 tile

矩阵接近方形时，`256×256` 同时扩大 M/N reuse，block 输出面积与输入 tile 复用较均衡。
强偏斜矩阵可能出现：

- 某一维 tile 太宽，合法 workgroups 很少；
- 增大的另一维并没有同等复用价值；
- 512-thread/64-KiB 资源成本仍然存在；
- fallback policy 中更窄的 `128×64` 或 `64×64` 反而适合。

所以规则只把 `max(M,N) <= 2*min(M,N)` 纳入已测区域，其余 rectangular shape 保持旧策略。
这是 conservative heuristic 的典型写法：只扩展有证据的 shape family。

## 17. Architecture 为什么必须来自输入 device

最终 runtime 获取：

```python
arch = _device_arch(A.device.index)
stream = torch.cuda.current_stream(A.device)
```

在异构多 GPU 进程中，全局 current device 或枚举到的第一个 GPU 不一定是 A 所在设备。错误
的 arch 会同时导致：

- 选错 gfx1151/gfx1100 tile policy；
- 在错误 device stream 上 launch/capture；
- 更严重时为错误 ISA target 构建 code object。

因此 device、stream 和 architecture 必须围绕同一个输入 tensor 解析。

## 18. Arch 必须进入每一层 cache key

最终 `_signature` 是：

```text
(M,N,K,arch,in_dtype,out_dtype,rounding,lda,ldb,ldc)
```

autotune key 也包含 arch，`_build` 的 `lru_cache` 参数包含 arch。row stride 进入 key，是因为
leading dimensions 是编译期参数；同 shape 的 tight tensor 与 padded slice 不能共享错误 pitch
的 launcher。

只在 Python dispatch key 中加 arch 仍不完整。如果底层 JIT 自动检测了另一个 device，仍可
生成错误 code object。因此最新提交继续传递：

```python
create_wmma_gemm_module(..., arch=arch)

@flyc.jit(arch=gpu_arch)
def launch_gemm(...):
    ...
```

JIT 保存 `self._arch`，并使用：

```python
get_backend(arch=self._arch)
```

这样选择策略、launcher cache 和 code generation target 才闭环一致。

## 19. Dispatch 热路径

第一次调用一个新 signature：

```text
从 A.device 获取 arch
 -> 计算/选择 tile
 -> 用 arch + tile + stride 构建 launcher
 -> 可选 autotune 合法候选
 -> 缓存 resolved launcher
 -> 在 A.device current stream launch
```

后续相同 signature：

```text
_resolved lookup
 -> 直接 launch
```

`_device_arch(device_index)` 自身用 `lru_cache`，避免每次热路径重复读取 device properties。

## 20. 如何从零设计一个 tile policy

不要先写 shape 特例；建议顺序是：

1. 从 WMMA atom 推导 `BLOCK_M/N/K` 和 threads。
2. 对每个候选计算 accumulator 数、LDS bytes 和 vector-copy divisibility。
3. 把不能构建的 tile 先过滤，不让 benchmark 混入失败点。
4. 计算每 shape 的 WG 数与 `WG/CU`，只把它当第一层解释变量。
5. 强制每个合法 tile，先验证正确性，再做预编译后的 GPU timing。
6. 小 kernel 用 graph replay 或足够批量，排除 Python launch 噪声。
7. baseline/candidate 随机交错，保留控制 shape 和反例。
8. 规则只覆盖稳定获益的 shape family，其余回退旧 policy。
9. 把 arch、dtype、rounding、stride 和 tile 全部纳入相应 cache key。
10. 最后检查 JIT target，不要只修 host-side if/else。

## 21. 性能数据应如何理解

较早 head `cca1d61a` 的归档测量，在 10 个 tile 发生变化的 shape 上记录：

```text
BF16 geometric mean = 1.199×
FP16 geometric mean = 1.164×
最大 896³：BF16 1.545×，FP16 1.423×
```

环境为 Radeon 8060S / gfx1151，测量使用预先 JIT 的 launcher、captured graph 和随机交错
A/B。它们是 kernel microbenchmark，不是模型端到端。

最新 `9794b02f` 又增加 near-square rectangular 规则；当前资料中没有一张可与上述十个旧
shape 合并的最新正式性能表。因此本文只把 near-square 解释为最新代码行为，不伪造或外推
它的统一几何平均收益。本文也没有重新运行硬件 benchmark。

从源码可以确认的机制是 tile 改变 block reuse、WG 数、LDS layout、线程数和 accumulator
规模；“为什么某个 shape 最终快了多少”仍应由固定 tile A/B、profiler 和生成代码回答。

## 22. 常见误读

### 误读一：PR 重写了 GEMM kernel

没有。WMMA、双缓冲 LDS、swizzle 和 epilogue 主体原本就存在；PR 主要改变选 tile 和
arch-aware build/dispatch。

### 误读二：40 CU 所以所有阈值乘 `40/96`

不成立。LDS、VGPR、shape 整除、K 深度、cache 和 tile reuse 都不是线性 CU 比例函数。

### 误读三：最大 tile一定最快

不成立。`256×256` 有 512 threads、64 KiB LDS，而且会把 WG 数缩小四倍；grid 太浅时会
严重 starve。

### 误读四：cache key 加 arch 就完成异构支持

不完整。arch 还必须传入 JIT/backend code generation target，否则只是在 host 侧隔离了
launcher 名称，code object 仍可能面向错误 ISA。

### 误读五：`one_wave` 是一个 wavefront

不是。它是 `wgs_128` 接近 40 CU 的一层 workgroup grid 区间，变量名不够准确。

## 23. 后续 kernel 优化可从哪里继续

- 针对 gfx11 v16 operand 的 lane 重复读取，实验 LDS broadcast/swizzle；
- 检查大 `reg_m×reg_n` 是否带来 VGPR spill 或 occupancy 限制；
- 对 `pad`/`kblock` 做 LDS bank-conflict 和吞吐消融；
- 为不整除 shape 增加 tail predication，但要评估分支和边界 load 成本；
- 将 near-square 规则扩展前，补齐 rectangular shape 的稳定 A/B 与反例；
- 研究 architecture、CU 数、resource metadata 驱动的更通用模型，但保留可解释 fallback。

## 24. 最终应记住的一句话

> PR #1057 的本质是让同一套 RDNA3 WMMA kernel在 gfx1151 上选择更合适的资源形态，并
> 保证选择使用的 architecture 与最终 JIT code object 的 architecture 是同一个。tile tuple
> 同时控制 block M/N/K、wave 网格、线程数、LDS layout、accumulator 和全设备 WG 深度；
> heuristic 只有在理解这些 kernel 机制后才不是几条孤立的 magic numbers。
