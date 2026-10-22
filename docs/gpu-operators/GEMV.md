---
title: "GEMV / decode 算子（split-K + W8A16 反量化融合）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">GEMV / decode 算子（split-K + W8A16 反量化融合）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P15</span><span>gemv</span><span>decode</span><span>split-K</span><span>weight-only 量化</span><span>memory-bound</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

`y[M] = W[M,K] × x[K]`，row-major。测试 M=K=8192。

<dl class="gpu-operator__io">
  <dt>为什么重要</dt><dd><strong>LLM decode 阶段 batch=1 时，所有 Linear 层都退化成 GEMV</strong>。
算术强度 = 2MK/4MK = <strong>0.5 flop/byte</strong>，比 GEMM 低两个数量级 —— 纯 memory-bound。</dd>
  <dt>唯一目标</dt><dd>把权重矩阵 W 的读取带宽打满。<strong>算力完全不重要。</strong></dd>
  <dt>三个考点</dt><dd>① 线程映射（一行一线程 vs 一行一 warp）② M 小时的 split-K ③ 权重量化后的融合反量化</dd>
</dl>

GEMM 考的是"你会不会优化 compute-bound"，**GEMV 考的是"你懂不懂 decode"**。 推理岗面试里，后者往往问得更细。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void gemv(const float* W, const float* x, float* y, int M, int K);

// weight-only 量化版（int8 权重 + per-output-channel scale）
void gemv_w8a16(const int8_t* Wq, const float* scale,
                const float* x, float* y, int M, int K);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先写"一个线程一行"，然后**立刻分析访存**：warp 里 32 个线程读的是 W 的 32 个不同行，间隔 K×4 字节 —— 完全不合并。
- 改成 **一个 warp 一行**：warp 沿 K 方向分工，读的就是连续的 128 字节，最后 `warpReduceSum`。
- 再上 float4：一个 warp 一条指令覆盖 512 字节。
- M 很小（输出维度小）时 block 数不够 → **沿 K 切 split-K**，和 flash-decoding 一个套路。
- 量化版：**反量化必须融合在 GEMV 里**。per-channel scale 可以提到最后乘一次，内层循环不受影响。
- ⚠ int8 也要按 **16 字节**一次读（16 个权重）；用 char4 只读 4 字节的话，一个 warp 一条指令才 128 字节，白白浪费带宽。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define FULL 0xffffffffu
// y[M] = W[M,K] * x[K]，row-major
// decode 阶段的所有 Linear 层都长这样（batch=1 时 GEMM 退化成 GEMV）。
// 算术强度 = 2MK flop / 4MK byte = 0.5 flop/byte → 纯 memory-bound，
// 唯一目标：把权重矩阵 W 的读取带宽打满。
__device__ __forceinline__ float warpReduceSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 一个线程一行"

    最直观：线程 i 算 `y[i] = Σ W[i][k]·x[k]`。

    **但访存彻底不合并**：同一时刻，warp 里的 32 个线程分别在读 `W[0][k], W[1][k], ..., W[31][k]`，**地址间隔 K×4 = 16 KB**，一个 warp 要发 32 个独立的内存事务，每个事务 128 B 里只用了 4 B。**有效带宽只有峰值的 1/32。**

    ```cpp
    __global__ void gemv_v0(const float* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int M, int K){
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M) return;
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += W[(size_t)row * K + k] * x[k];
        y[row] = acc;
    }
    ```

=== "v1 一个 warp 一行 + warp reduce"

    换个分工：**一整个 warp 负责一行**，32 个 lane 沿 K 方向分别累加自己那份，最后 `warpReduceSum` 合并。

    此时同一时刻 warp 读的是 `W[row][k], W[row][k+1], ..., W[row][k+31]` —— **连续 128 字节，完美合并**。

    这是 GEMV 的标准形态，一行代码没变多，带宽却能提升一个数量级。

    ```cpp
    __global__ void gemv_v1(const float* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int M, int K){
        const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        const int lane = threadIdx.x & 31;
        if (warpId >= M) return;
        const float* w = W + (size_t)warpId * K;
        float acc = 0.f;
        for (int k = lane; k < K; k += 32) acc += w[k] * x[k];   // 一个 warp 读连续 128B
        acc = warpReduceSum(acc);
        if (lane == 0) y[warpId] = acc;
    }
    ```

=== "v2 float4 向量化 + x 进 shared"

    两个改进：

    ① **float4**：每个 lane 一次读 4 个权重，一个 warp 一条指令覆盖 512 字节；

    ② **x 放进 shared**：x 只有 K 个元素，却被 block 里所有行重复读——先合作搬进 shared（或者直接靠 L2 也行，因为它很小很热）。

    这一版基本能吃满 HBM 带宽，**也就到了 GEMV 的理论上限**。

    ```cpp
    template<int TH>
    __global__ void gemv_v2(const float* __restrict__ W, const float* __restrict__ x,
                            float* __restrict__ y, int M, int K){
        extern __shared__ float xs[];
        for (int i = threadIdx.x; i < K; i += TH) xs[i] = x[i];
        __syncthreads();
        const int warpsPerBlock = TH / 32;
        const int warpId = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
        const int lane = threadIdx.x & 31;
        if (warpId >= M) return;
        const float4* w4 = reinterpret_cast<const float4*>(W + (size_t)warpId * K);
        const float4* x4 = reinterpret_cast<const float4*>(xs);
        float acc = 0.f;
        for (int k = lane; k < K / 4; k += 32){
            float4 a = w4[k], b = x4[k];
            acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
        acc = warpReduceSum(acc);
        if (lane == 0) y[warpId] = acc;
    }
    ```

=== "v3 split-K（M 很小时救命）"

    当输出维度 M 很小（比如 LLM 最后一层之前的某些投影、或 MoE 单个 expert），`grid = M/warp数` 会**少到填不满 GPU**。

    解法和 flash-decoding 一样：**沿 K 再切 nsplit 份**，每份算一个部分和写进 workspace，第二个 kernel 求和。

    并行度从 M 提升到 M×nsplit。**代价是多一次中间往返，所以只在 M 小的时候用**——判据是 `M / (blockDim/32) < SM 数 × 常数`。

    ```cpp
    __global__ void gemv_splitk(const float* __restrict__ W, const float* __restrict__ x,
                                float* __restrict__ partial, int M, int K, int nsplit){
        const int row = blockIdx.x, sp = blockIdx.y;
        const int per = CEIL_DIV(K, nsplit);
        const int k0 = sp * per, k1 = min(K, k0 + per);
        const float* w = W + (size_t)row * K;
        float acc = 0.f;
        for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) acc += w[k] * x[k];
        // block 内归约
        __shared__ float sm[32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        acc = warpReduceSum(acc);
        if (lane == 0) sm[wid] = acc;
        __syncthreads();
        if (wid == 0){
            float v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
            v = warpReduceSum(v);
            if (lane == 0) partial[(size_t)row * nsplit + sp] = v;
        }
    }
    __global__ void splitk_reduce(const float* __restrict__ partial, float* __restrict__ y,
                                  int M, int nsplit){
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M) return;
        float s = 0.f;
        for (int i = 0; i < nsplit; i++) s += partial[(size_t)row * nsplit + i];
        y[row] = s;
    }
    ```

=== "v4 W8A16 反量化融合（推理主力）"

    LLM 权重量化到 int8（甚至 int4）后，**weight-only 量化**的推理路径是：

    `读 int8 权重 → 反量化成 fp32/fp16 → 和激活做乘加`

    **关键：反量化必须融合在 GEMV 里，不能先反量化成一个 fp16 大矩阵再算**——那样就白省了带宽。

    收益直接：**权重的 HBM 读取量降到 1/4（int8）或 1/8（int4）**，而 GEMV 是 memory-bound 的，**所以速度差不多就是 4 倍 / 8 倍**。

    量化粒度用 **per-output-channel（每行一个 scale）**：反量化可以提到最后做一次（`y = scale · Σ q·x`），内层循环里一次乘法都不多。

    ```cpp
    template<int R>            // R = 一个 warp 同时算几行（x 只读一次，服务 R 行）
    __global__ void gemv_w8a16(const int8_t* __restrict__ Wq, const float* __restrict__ scale,
                               const float* __restrict__ x, float* __restrict__ y, int M, int K){
        const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        const int lane = threadIdx.x & 31;
        const int row0 = warpId * R;
        if (row0 >= M) return;
        // ⚠ 两个关键点：
        //  ① 仍然按 **16 字节** 一次读：int8 时 16 B = 16 个权重。只用 char4（4 B）
        //     的话一个 warp 一条指令才 128 B，白白浪费一半以上的带宽。
        //  ② 一个 warp 同时算 R 行：**x 从 cache 读一次就服务 R 行**。
        //     不这么做的话 x 的重复读取（M×K×4 字节）会超过权重本身，量化白省了。
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const int4* w16[R];
        #pragma unroll
        for (int r = 0; r < R; r++)
            w16[r] = reinterpret_cast<const int4*>(Wq + (size_t)min(row0 + r, M - 1) * K);
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.f;

        for (int k = lane; k < K / 16; k += 32){
            float4 b[4];
            #pragma unroll
            for (int t = 0; t < 4; t++) b[t] = x4[k * 4 + t];      // x 读一次
            #pragma unroll
            for (int r = 0; r < R; r++){
                int4 packed = w16[r][k];                 // 16 个 int8 权重，一条 128-bit load
                const int8_t* q = reinterpret_cast<const int8_t*>(&packed);
                #pragma unroll
                for (int t = 0; t < 4; t++)
                    acc[r] += (float)q[t*4+0]*b[t].x + (float)q[t*4+1]*b[t].y
                            + (float)q[t*4+2]*b[t].z + (float)q[t*4+3]*b[t].w;
            }
        }
        #pragma unroll
        for (int r = 0; r < R; r++){
            float v = warpReduceSum(acc[r]);
            if (lane == 0 && row0 + r < M) y[row0 + r] = v * scale[row0 + r];  // per-channel scale
        }
    }
    ```

</div>
