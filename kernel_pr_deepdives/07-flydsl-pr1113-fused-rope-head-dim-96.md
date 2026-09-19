# FlyDSL PR #1113：Fused RoPE + KV Cache 的 Head-Dim 96 修复

## PR 动机

FlyDSL 的 fused RoPE kernel 把三项工作放在一次 launch 中：旋转 Q、旋转 K，并把旋转后的 K
和原始 V 写入 KV cache。它用一个 wave 处理一个 `(token, head)`，每个 lane 持有一段连续
head elements，再用 AMD `ds_bpermute_b32` 从配对 lane 取得 NeoX RoPE 的另一半。

原实现根据：

```text
VEC_WIDTH = ceil(head_dim / wave_size)
```

决定每线程处理多少元素。这个公式只保证“一个 head 的 vector 数不超过一个 wave”，却不保证
vector width 是 kernel copy/shuffle helper 真正支持的原生宽度。对 RDNA wave32 和
`head_dim=96`，它得到 `VEC_WIDTH=3`。一个 lane 因而持有三个 BF16/FP16，共 48 bits；但
`ds_bpermute_pair()` 按 32-bit chunks 重建 peer vector 时只得到两个元素，随后与三元素 vector
相乘，编译器报：

```text
ValueError: cannot broadcast shapes (2,) and (3,)
```

PR #1113 将 width 改为从 `{1,2,4,8}` 中选择最小合法值。D=96/wave32 因而使用
`VEC_WIDTH=4`，但这又把每半 head 的 lane span 从 16 改成 12；原来的 XOR 配对只对 2 次幂
span 正确，所以 PR 同时增加非 2 次幂的显式 `+/- half-span` 配对。

这两个修改是一个完整修复：只改 vector width 会从“编译失败”变成“能编译但配错 RoPE
partner”；只改 partner mapping 又无法解决三元素 vector 的重建失败。

本文基于公开 PR head：

- PR：<https://github.com/ROCm/FlyDSL/pull/1113>
- Head：`f4fa09f694871a815084ddbb56116afd4843a118`
- Parent：`728f6b220518af94a8d641a0512ef87167c90125`
- 真实提交范围：`728f6b2..f4fa09f`
- 核心文件：`kernels/attention/fused_rope_cache_kernel.py`
- 测试文件：`tests/kernels/test_fused_rope_cache.py`

固定读取：

```bash
git -C ~/projects/FlyDSL show \
  refs/remotes/pr/1113:kernels/attention/fused_rope_cache_kernel.py
```

## 1. Kernel 实际融合了什么

输入：

```text
Q [T,QH,D]
K [T,KH,D]
V [T,KH,D]
positions    [T]
slot_mapping[T]
cos/sin cache
```

一次 kernel launch 完成：

```text
Q -> NeoX RoPE -> Q_out
K -> NeoX RoPE -> K_out -> key cache
V -----------------------> value cache
```

它没有把 attention 本身融合进来；输出仍是旋转后的 Q/K 和已经按 paged-cache layout 写好的
K/V。融合收益来自避免三次独立 launch、复用 position/cos/sin 读取，并让 K 的旋转结果直接
进入 cache write path。

## 2. NeoX RoPE 的数学 contract

把长度 D 的 head 拆成前后两半：

```text
x = [x1, x2],  x1/x2 length = D/2
```

NeoX-style rotation：

```text
out_first  = x1 * cos - x2 * sin
out_second = x2 * cos + x1 * sin
```

因此每个第一半元素需要读取对应第二半元素，每个第二半元素也需要读取对应第一半元素。若每
lane 处理 `VEC_WIDTH` 个连续元素，则配对发生在 lane 粒度：

```text
vecs_per_half = (D/2) / VEC_WIDTH
```

第一半 lane `t` 的 partner 是 `t + vecs_per_half`；第二半 lane 的 partner 是
`t - vecs_per_half`。

## 3. Grid 与 block：一个 block 就是一个 wave

launch：

```text
grid  = (max(QH,KH), T, 1)
block = (WARP_SIZE, 1, 1)
```

映射：

```text
blockIdx.x -> head_idx
blockIdx.y -> token_idx
threadIdx.x-> lane/thread within one wave
```

一个 block 在 `head_idx < QH` 时处理 Q；在 `head_idx < KH` 时处理 K、V 和 KV cache。典型
GQA 中 `QH > KH`：前 KH 个 head blocks 同时做 Q/K/V，后面的 blocks 只做 Q。

这种结构要求一个完整 head 的所有 vector chunks 必须落在同一个 wave 内，因为
`ds_bpermute` 只能从本 wave 的 peer lane 读取 VGPR 值。

## 4. 旧 width 公式只满足了一个条件

旧代码：

```python
VEC_WIDTH = max(1, ceil(head_dim / WARP_SIZE))
```

它保证：

```text
ceil(head_dim / VEC_WIDTH) <= WARP_SIZE
```

但 kernel 还隐含依赖：

1. `VEC_WIDTH` 应能整除完整 head；
2. 它应能整除 NeoX half；
3. copy width 应是实现支持的原生 16/32/64/128-bit 形式；
4. peer vector 必须能完整拆成若干 32-bit `ds_bpermute` chunks。

对 D=96、wave32：

```text
old VEC_WIDTH  = ceil(96/32) = 3 elements
copy_bits      = 3 × 16 = 48 bits
vecs_per_head  = 32
vecs_per_half  = 16
```

表面上正好用满 32 lanes，half span 还是 16；真正失败的是三元素/48-bit vector 不符合后续
32-bit chunk helper 的构造。

## 5. 编译错误是怎样产生的

旧 `ds_bpermute_pair()` 对 `VEC_WIDTH>=2` 执行：

```python
n_i32 = VEC_WIDTH // 2
v_i32 = bitcast(vec_val, Int32)
for ci in range(n_i32):
    peer_chunks.append(ds_bpermute(..., v_i32[ci]))
return peer_chunks.bitcast(elem_dtype)
```

当 `VEC_WIDTH=3`：

```text
n_i32 = 3 // 2 = 1
```

一轮 `ds_bpermute_b32` 只能搬回一个 32-bit chunk，也就是两个 BF16/FP16 元素；第三个 16-bit
元素没有被重建。结果：

```text
q_e       shape = (3,)
q_pair_e  shape = (2,)
```

在：

```python
q_pair_sin = q_pair_e * sin_e
```

处触发 `(2,)` 与 `(3,)` 无法 broadcast。这个错误发生在 kernel build/compile 阶段，不是运行
时数值偏差。

## 6. 新 width policy

PR 改为：

```python
VEC_WIDTH = first width in (1,2,4,8) satisfying:
    head_dim % width == 0
    half_dim % width == 0
    head_dim / width <= WARP_SIZE
```

如果没有候选，直接：

```text
ValueError: unsupported head_dim=... for wave...
```

这里“选最小合法 width”有两个效果：

- 让一个 head 放进单 wave；
- 在满足约束时尽可能保留更多 active lanes，而不是无条件使用最宽 vector。

候选 `{1,2,4,8}` 对 BF16/FP16 分别对应 16/32/64/128-bit copy，且除 width 1 的特殊处理外，
都能完整拆成 `VEC_WIDTH/2` 个 32-bit chunks。

## 7. 常见 head dimension 的新映射

### RDNA wave32

| D | VEC_WIDTH | copy bits | active lanes | lanes/half |
|---:|---:|---:|---:|---:|
| 64 | 2 | 32 | 32 | 16 |
| 96 | 4 | 64 | 24 | 12 |
| 128 | 4 | 64 | 32 | 16 |
| 256 | 8 | 128 | 32 | 16 |

### CDNA wave64

| D | VEC_WIDTH | copy bits | active lanes | lanes/half |
|---:|---:|---:|---:|---:|
| 64 | 1 | 16 | 64 | 32 |
| 96 | 2 | 32 | 48 | 24 |
| 128 | 2 | 32 | 64 | 32 |
| 256 | 4 | 64 | 64 | 32 |

PR 的硬件验证在 gfx1151/wave32 上。wave64 表是最终代码的静态映射结果，不应写成已经在 CDNA
完成同等 runtime benchmark。

## 8. D=96 为什么只激活 24 lanes

新配置：

```text
D              = 96
VEC_WIDTH      = 4
vecs_per_head  = 96/4 = 24
vecs_per_half  = 48/4 = 12
```

lane mapping：

```text
lane 0..11  -> head elements 0..47
lane 12..23 -> head elements 48..95
lane 24..31 -> inactive
```

每个 active lane 读取/计算/写回四个连续 BF16/FP16。虽然没有用满 wave32 的全部 lanes，但它
换来了原生 64-bit vector 以及完整的 32-bit shuffle chunks。试图用三元素 vector“刚好用满
32 lanes”反而违反底层表示约束。

## 9. 为什么 XOR 只适用于 2 次幂 half span

旧配对：

```python
pair_lane = tid ^ vecs_per_half
```

若 `vecs_per_half=16`，二进制是单 bit：

```text
0..15  XOR 16 -> 16..31
16..31 XOR 16 -> 0..15
```

它等价于切换表示“前半/后半”的那一位，所以既便宜又对称。

但 D=96/VEC4 时 span 是 12，即二进制 `01100`。XOR 12 会同时翻转多个 bit，不等价于加减
12：

```text
0  XOR 12 = 12   正确
1  XOR 12 = 13   正确
4  XOR 12 = 8    错误，仍落在第一半
8  XOR 12 = 4    错误
16 XOR 12 = 28   错误，落到 inactive lane
```

所以“XOR 能交换两半”成立的前提是 half span 为 2 的幂，而不是任意偶数。

## 10. 新 pair mapping

PR 用 compile-time condition 保留 fast path：

```python
if is_power_of_two(vecs_per_half):
    pair_lane = tid ^ vecs_per_half
else:
    pair_lane = tid < vecs_per_half ? tid + vecs_per_half
                                     : tid - vecs_per_half
```

D=96/wave32 得到：

```text
0  <-> 12
1  <-> 13
...
11 <-> 23
```

这正好对应 element ranges：

```text
lane 0 的 elements 0..3   <-> lane 12 的 48..51
lane 1 的 elements 4..7   <-> lane 13 的 52..55
...
lane 11 的 44..47         <-> lane 23 的 92..95
```

`const_expr` 会在 build 时选择分支。D=64/128/256 的 `vecs_per_half=16`，最终仍生成原 XOR
路径，不在 runtime 每次多做 power-of-two 判断。

## 11. `ds_bpermute_b32` 搬的是 peer lane 的 VGPR

尽管指令名字带 `ds`，这里没有为 RoPE pair 分配 LDS array。语义是：当前 lane 提供 byte
address，并读取目标 lane 的一个 32-bit VGPR 值：

```text
pair_byte_addr = pair_lane × 4
```

对不同 vector width：

| VEC_WIDTH | vector bits | bpermute calls/lane |
|---:|---:|---:|
| 1 | 16 | 1，先 zero-extend 到 i32，再 truncate |
| 2 | 32 | 1 |
| 4 | 64 | 2 |
| 8 | 128 | 4 |

D=96 的四元素 vector 被 bitcast 成两个 i32 chunks；每个 chunk 对同一个 `pair_lane` 做一次
`ds_bpermute`，再拼回四元素 peer vector。这样每个 lane 获得 partner lane 对应的完整连续四
元素，而不是只获得第一个 32-bit chunk。

## 12. Cos/Sin vector 如何与两半对应

代码：

```python
is_first_half = tid < vecs_per_half
cos_vec_idx = tid % vecs_per_half if reuse_freqs_front_part else tid
```

当 cos/sin cache 只保存 D/2 列时，两半 lane 通过 `% vecs_per_half` 读取同一频率 vector：

```text
lane 0  与 lane 12 -> cos/sin vector 0
lane 1  与 lane 13 -> cos/sin vector 1
...
```

若 cache 保存完整 D 列，则按 tid 读取各自位置。最终符号：

```python
sin_term = first_half ? -pair*sin : pair*sin
rotated  = self*cos + sin_term
```

正好实现 NeoX 的 `[-x2, +x1]` 结构。

## 13. Q 路径逐 lane 做什么

对 `head_idx<QH` 的 block：

```text
Q[token,head, tid*VEC : (tid+1)*VEC]
  -> register vector q_e

peer lane q_e
  -> ds_bpermute chunks
  -> q_pair_e

q_rot = q_e*cos ± q_pair_e*sin
  -> Q_out 对应 vector
```

每个输入 Q vector只从 global 读一次。partner 不再次从 global memory 读取，而是通过 wave 内
VGPR exchange 获取。PR 没有改变这一资源模型，只让它对 VEC4/非 2 次幂 lane span 正确工作。

## 14. K 路径与 cache write

对 `head_idx<KH`：

```text
K vector -> 同样的 RoPE -> K_out
V vector -> 延迟读取
slot_mapping[token] -> cache block/position
rotated K -> key cache
V         -> value cache
```

当 `slot<0` 时，Q/K rotation 仍可完成，但 cache write 被跳过。这适用于 padding、被丢弃 token
或上层不希望写 cache 的位置。

V 的读取被放到确认 `slot>=0` 后，并且靠后创建 buffer descriptor，以缩短相关 SGPR live
range。这是既有 kernel 设计，PR #1113 没有重新设计它。

## 15. 两种 KV cache layout

Flash layout：

```text
KeyCache/ValueCache [num_blocks, block_size, KH, D]
```

同一 `(block,position,head)` 的 D 连续，K/V 都可以沿 head dimension 做 vector store。

Non-flash/ATOM layout：

```text
KeyCache   [num_blocks, KH, D/16, block_size, 16]
ValueCache [num_blocks, KH, D,    block_size]
```

Key 按 16-element x-pack 排列；Value 是 dim-major，当前 vector 最终逐元素 scatter 到不同 D
row。D=96 可被 16 整除，因此也满足 non-flash key layout contract。

新测试同时覆盖 flash/non-flash，意味着修复不只验证了 `Q_out/K_out`，也验证了两种 layout 下
key/value cache 的最终地址映射。

## 16. FP8 cache path 与本 PR 的关系

当 `apply_scale=True`，kernel 会把旋转后的 K 和 V 乘 reciprocal scale，再用 AMD FP8 pack
intrinsic 写 raw cache bytes。VEC4/VEC8 可以每四元素 pack 一个 i32；窄 vector 则逐元素取
低 byte。

PR 新增的 D=96 tests 使用默认 `apply_scale=False`，也就是 BF16/F16 cache。测试报告不能被
扩展成“D=96 FP8 cache 在所有架构上也已验证”。现有测试甚至明确说明 RDNA FP8 cache encoding
与 CDNA production dtype 不同，并在 RDNA 上跳过该类别。

## 17. 为什么旧 D=64/128/256 路径没有回归

在 gfx1151/wave32：

```text
D=64  old/new VEC_WIDTH = 2
D=128 old/new VEC_WIDTH = 4
D=256 old/new VEC_WIDTH = 8
```

它们的 `vecs_per_half` 都是 16，compile-time pair branch 继续选择 XOR。因此这些配置的 copy
atom、active lanes、peer mapping 和算术主体都没有改变。

PR 报告最终 gfx1151 生成 ISA 对 D=64/128/256 与 `origin/main` byte-for-byte identical；
interleaved CUDA Graph 测量也没有发现性能回归。这比“代码看起来走同一分支”更强，因为它
验证了 compiler 没因重构产生额外指令。

## 18. D=96 的资源形态

归档的 gfx1151 D=96 kernel metadata：

```text
VGPR：35
SGPR：26
LDS：0
scratch：0
register spill：0
```

“LDS=0”与使用 `ds_bpermute_b32` 不矛盾。该指令借用 DS cross-lane 数据通路读取 peer lane
VGPR，但这里没有显式分配 shared/LDS storage。

24/32 active lanes意味着每个 wave 有 8 lanes 不参与 head vector work。它不是满 lane 利用率，
但相比不可编译的 VEC3，这是满足 native copy/shuffle contract 的可运行映射；是否值得设计更
复杂的跨 head packing 来填满剩余 lanes，需要完全不同的 block decomposition 和实测。

## 19. 正确性测试到底覆盖了什么

新增参数化测试：

```text
T=32
D=96
QH=32
KH=32
dtype = BF16 / FP16
layout = flash / non-flash
```

共四个组合。`run_test()` 与 PyTorch reference 比较：

- `Q_out`；
- `K_out`；
- key cache；
- value cache。

reference 明确按：

```text
[q1*cos-q2*sin, q2*cos+q1*sin]
```

计算，并按两种 cache layout 写入。整体 suite 记录为：

```text
74 passed, 248 skipped
```

大量 skipped 来自硬件/可选 AITER/配置 gating，不能解读为 248 个失败。

本 PR 的新增测试没有单独覆盖 D=96 的 negative slot、i64 positions、full-width cos/sin 或 FP8
cache；这些 contract 已由其他 D 的既有测试覆盖，但 D=96 与它们的组合仍属于可扩展测试空间。

## 20. 性能数据应如何解释

gfx1151、BF16、`QH=KH=32`、CUDA Graph replay：

| Tokens | Latency |
|---:|---:|
| 1 | 4.025 µs |
| 128 | 10.637 µs |
| 512 | 35.252 µs |
| 2048 | 456.657 µs |
| 4096 | 909.765 µs |

这些数据证明新 D=96 kernel 可稳定生成并执行，也给出绝对 latency；它们不是相对旧 FlyDSL
kernel 的 speedup，因为旧版本无法编译 D=96。

AITER 的公开 Triton fused kernel也拒绝 non-power-of-two head dimension，所以当前没有直接
AITER D=96 baseline。不能从这张表声称“比 AITER 快多少”。

对于已有 D=64/128/256，证据是 ISA identical 与交错 graph 测量无回归，而不是把 D=96 的
absolute latency外推到它们。

## 21. 这项修复为什么不只是多加一个 shape 特例

可以写死：

```python
if head_dim == 96 and WARP_SIZE == 32:
    VEC_WIDTH = 4
```

但那只修当前 symptom。最终规则表达的是 kernel 的真实构造条件：

```text
原生 vector width
完整 head 可整除
NeoX half 可整除
所有 chunks 能放进单 wave
```

pair mapping 也根据 half span 的数学性质选择 XOR 或显式 offset。这样其他能满足这些约束的
non-power-of-two head dimension 可自然复用，而不需要继续积累 shape-specific patch。

## 22. 常见误读

### 误读一：D=96 的旧问题只是 XOR 配错

不是。旧 VEC3 先在 peer vector reconstruction 阶段产生 2-vs-3 shape compile error；改成 VEC4
后才出现需要处理的 12-lane non-power-of-two pairing。

### 误读二：XOR 可以交换任意等长两半

只有 half span 是 2 的幂时成立。span=12 必须用 `+12/-12`。

### 误读三：`ds_bpermute` 表示 kernel 使用 LDS buffer

这里它做 wave 内 VGPR cross-lane read，kernel metadata 的 LDS 使用仍是 0。

### 误读四：VEC_WIDTH 越小越好，因为 active lanes 更多

width 还必须符合 copy atom和 32-bit shuffle chunk 的表示。VEC3 虽用满 32 lanes，却不可被当前
helper 正确表达。

### 误读五：PR 优化了已有 D=64/128/256 性能

没有。它的目标是让 D=96 正确编译和执行；已有三条路径被刻意保持，并验证 ISA 不变。

### 误读六：测试证明所有 fused RoPE 配置都支持 D=96

新增测试覆盖 BF16/F16 和两种 layout，但没有覆盖 partial rotary、非 NeoX、D=96 FP8 cache 等；
其中 partial rotary 与非 NeoX 本来就被 builder 明确拒绝。

## 23. 如果从零定位和修复这个问题

建议按以下顺序：

1. 从报错的 vector shapes 反推每 lane 的真实 payload，而不是先怀疑 RoPE 数学。
2. 把 `head_dim / wave_size`、`VEC_WIDTH`、`vecs_per_head`、`vecs_per_half` 全部列成表。
3. 检查 copy atom 与 shuffle primitive 的原生 bit widths，不允许任意 ceil 结果直接进入类型构造。
4. 证明一个 head 必须落在单 wave，加入 `head_dim/width <= WARP_SIZE`。
5. 分别证明完整 head 和 half head 都能被 width 整除。
6. 修改 width 后重新计算 partner span；不要假设旧 XOR 仍成立。
7. 用具体 lane table验证前半和后半的双向配对。
8. 对 vector 按 32-bit chunks 做 peer exchange，并证明能无损重建全部元素。
9. 同时检查 Q/K rotation 与两种 KV cache layout，不只检查 kernel 能编译。
10. 对未修改的常见 shape 比较生成 ISA，确认 compile-time branch 没给热路径增加成本。

## 24. 当前边界与后续方向

- 只支持 NeoX-style RoPE；`is_neox=False` 明确报错。
- 只支持 full-head rotation；`rotary_dim != head_dim` 明确报错。
- 输入 dtype 仅 BF16/F16。
- 一个 head 的 vector chunks 必须放入单 wave；找不到 `{1,2,4,8}` 合法 width 时拒绝构建。
- non-flash layout要求 D 可被 x-pack 16 整除。
- D=96 的 runtime 验证集中在 gfx1151/wave32；其他 RDNA/CDNA 目标需分别编译和运行确认。
- 24 active lanes 留下 8 个 inactive lanes；进一步利用它们需要改变“一 wave 一 head”的映射，
  并重新评估 cos/sin、Q/K 共享、cache address 和 SGPR/VGPR。
- 如果未来支持 interleaved/Llama-style RoPE 或 partial rotary，pair mapping 与频率索引都需要重新
  定义，不能直接复用本 PR 的 NeoX half-offset 逻辑。

## 25. 最终应记住的一句话

> PR #1113 修复的不是一个孤立的 `head_dim==96` if：它把 vector width 从任意 ceil 值收紧为
> 能整除 head/half、可在单 wave 内执行的原生 `{1,2,4,8}` copy width；D=96/wave32 因而从
> 不可重建的三元素 vector 改为 24 lanes × 4 elements。由于每半只剩 12 lanes，kernel 又把
> 只适用于 2 次幂 span 的 XOR peer mapping 推广为 XOR-or-`+/-span`，最终在不改变 D=64/128/256
> 生成路径的前提下正确支持 Phi-3 Mini 使用的 96-dimensional head。
