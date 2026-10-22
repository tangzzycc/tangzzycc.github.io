---
title: "朴素 Attention（三段式 → 融合，理解 O(S²) 的痛点）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">朴素 Attention（三段式 → 融合，理解 O(S²) 的痛点）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P11</span><span>attention</span><span>QK^T</span><span>softmax</span><span>铺垫 FlashAttention</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

实现标准的 scaled dot-product attention（带 causal mask）：

`O = softmax(QKᵀ / √D + mask) · V`

<dl class="gpu-operator__io">
  <dt>输入</dt><dd>Q, K, V: [B, H, S, D]，测试用 B=2, H=8, S=1024, D=64</dd>
  <dt>输出</dt><dd>O: [B, H, S, D]</dd>
  <dt>关键观察</dt><dd>中间的 score 矩阵 <strong>P 是 [B,H,S,S]</strong> —— 它随 S <strong>平方增长</strong></dd>
</dl>

这题本身不难（就是两个 GEMM 夹一个 softmax），**它存在的意义是让你亲手算出那个平方项有多可怕**： S=4096、B=8、H=32 时 P 有 **17 GB**，显存都放不下。 理解了这一点，才能理解 FlashAttention 为什么值一篇论文——**它解决的不是"算得慢"，而是"中间结果太大"。**

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void attention(const float* Q, const float* K, const float* V,
               float* O, int B, int H, int S, int D,
               float scale, bool causal);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先按定义拆成三个 kernel：QKᵀ → softmax → PV。**写完立刻算一下 P 占多少显存**。
- QKᵀ 就是一个 GEMM（把 K 当成转置的右矩阵），可以直接套 shared tile 的 GEMM 写法。
- causal mask：`j > i` 的位置填 `-INFINITY`，softmax 之后自然变成 0。
- softmax 复用 P07 的 safe softmax（一个 block 一行）。
- 想想：P 的一行只被**一个 query**用，为什么要写进 HBM？把它留在 shared 里就地做完 softmax 和 PV。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define FULL 0xffffffffu
// 布局：Q/K/V/O 都是 [B, H, S, D]（head-major，PyTorch 参考实现的排布）
// score 矩阵 P 是 [B, H, S, S]  ← 问题就出在它身上

__device__ __forceinline__ float warpAllMax(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float warpAllSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float blockAllMax(float v){
    __shared__ float sm[32]; __shared__ float r;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllMax(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : -INFINITY; t = warpAllMax(t); if (lane == 0) r = t; }
    __syncthreads();
    return r;
}
__device__ __forceinline__ float blockAllSum(float v){
    __shared__ float sm[32]; __shared__ float r;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllSum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : 0.f; t = warpAllSum(t); if (lane == 0) r = t; }
    __syncthreads();
    return r;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 三段式（三个 kernel）"

    最直白的实现，也是 PyTorch 早期的做法：

    ① `P = QKᵀ·scale`（+ mask）② `P = softmax(P)` ③ `O = PV`

    **致命问题：中间的 P 是 [B,H,S,S] 的大矩阵，要完整写进 HBM 再读出来两次。**

    S=4096、B=8、H=32 时 P 有 8×32×4096×4096×4B = **17 GB**——显存都放不下，更别说带宽。

    这就是 FlashAttention 要解决的问题：**不是算得不够快，是中间结果太大。**

    ```cpp
    __global__ void qk_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                              float* __restrict__ P, int S, int D, float scale, bool causal){
        const int bh = blockIdx.y, i = blockIdx.x;              // 第 bh 个 (batch,head)，第 i 个 query
        const float* q = Q + ((size_t)bh * S + i) * D;
        float* p = P + ((size_t)bh * S + i) * S;
        for (int j = threadIdx.x; j < S; j += blockDim.x){
            const float* k = K + ((size_t)bh * S + j) * D;
            float acc = 0.f;
            for (int d = 0; d < D; d++) acc += q[d] * k[d];
            p[j] = (causal && j > i) ? -INFINITY : acc * scale;   // causal: 只能看自己和之前
        }
    }
    __global__ void softmax_kernel(float* __restrict__ P, int S){
        float* p = P + (size_t)blockIdx.x * S;                  // 一行 = 一个 query 的全部 score
        float m = -INFINITY;
        for (int j = threadIdx.x; j < S; j += blockDim.x) m = fmaxf(m, p[j]);
        m = blockAllMax(m);
        float l = 0.f;
        for (int j = threadIdx.x; j < S; j += blockDim.x) l += __expf(p[j] - m);
        l = blockAllSum(l);
        float inv = __fdividef(1.0f, l);
        for (int j = threadIdx.x; j < S; j += blockDim.x) p[j] = __expf(p[j] - m) * inv;
    }
    __global__ void pv_kernel(const float* __restrict__ P, const float* __restrict__ V,
                              float* __restrict__ O, int S, int D){
        const int bh = blockIdx.y, i = blockIdx.x;
        const float* p = P + ((size_t)bh * S + i) * S;
        float* o = O + ((size_t)bh * S + i) * D;
        for (int d = threadIdx.x; d < D; d += blockDim.x){
            float acc = 0.f;
            for (int j = 0; j < S; j++) acc += p[j] * V[((size_t)bh * S + j) * D + d];
            o[d] = acc;
        }
    }
    ```

=== "v1 QKᵀ 用 shared tile"

    v0 的 `qk_kernel` 里，**每个 query 都要把整个 K 读一遍** → 总访存 O(S²·D)。

    把它当成 GEMM 来做：**一个 block 负责 32 个 query × 32 个 key 的小块**，Q tile 和 K tile 先搬进 shared，块内复用。访存降到 O(S²·D/TILE)。

    这一版本质上就是 P14 的 sgemm v2，**attention 的前半段就是一个 GEMM**——面试时点出这一点很重要。

    ```cpp
    template<int TILE>
    __global__ void qk_tiled(const float* __restrict__ Q, const float* __restrict__ K,
                             float* __restrict__ P, int S, int D, float scale, bool causal){
        __shared__ float Qs[TILE][TILE + 1], Ks[TILE][TILE + 1];
        const int bh = blockIdx.z;
        const int i = blockIdx.y * TILE + threadIdx.y;          // query 行
        const int j = blockIdx.x * TILE + threadIdx.x;          // key 行
        const float* Qb = Q + (size_t)bh * S * D;
        const float* Kb = K + (size_t)bh * S * D;
        float acc = 0.f;
        for (int d0 = 0; d0 < D; d0 += TILE){
            Qs[threadIdx.y][threadIdx.x] = (i < S && d0 + threadIdx.x < D) ? Qb[(size_t)i * D + d0 + threadIdx.x] : 0.f;
            // K 也按 [key][dim] 读进来，注意这里 threadIdx.y 走的是 key 维
            Ks[threadIdx.y][threadIdx.x] = (blockIdx.x * TILE + threadIdx.y < S && d0 + threadIdx.x < D)
                                         ? Kb[((size_t)blockIdx.x * TILE + threadIdx.y) * D + d0 + threadIdx.x] : 0.f;
            __syncthreads();
            #pragma unroll
            for (int t = 0; t < TILE; t++) acc += Qs[threadIdx.y][t] * Ks[threadIdx.x][t];
            __syncthreads();
        }
        if (i < S && j < S)
            P[((size_t)bh * S + i) * S + j] = (causal && j > i) ? -INFINITY : acc * scale;
    }
    ```

=== "v2 融合成一个 kernel（S 存 shared）"

    既然 P 的一行只被这一个 query 用，**为什么要写进 HBM？**

    一个 block 负责**一个 query 行**：算出整行 score 存进 **shared**，就地 softmax，再乘 V 得到输出。**P 完全不落 HBM**，从"读写 3 遍 S² 数据"变成 0 遍。

    限制：**shared 只有 48~228 KB，装不下长序列的一整行**（S=8192 时一行就要 32 KB，还只能放一个 query）。

    **这一版已经很接近 FlashAttention 的思想了，差的就是"当一行也放不下时怎么办"——答案是把 KV 也分块，用 online softmax 增量合并。**

    ```cpp
    template<int D, int NC>            // NC = blockDim.x / D，PV 阶段沿 key 维再切 NC 份
    __global__ void attn_fused_row(const float* __restrict__ Q, const float* __restrict__ K,
                                   const float* __restrict__ V, float* __restrict__ O,
                                   int S, float scale, bool causal){
        extern __shared__ float smem[];                         // [D] 放 q，后面 [S] 放 score
        float* qs = smem;
        float* s  = smem + D;
        __shared__ float part[NC][D];
        const int bh = blockIdx.y, i = blockIdx.x, tid = threadIdx.x;
        const float* q  = Q + ((size_t)bh * S + i) * D;
        const float* Kb = K + (size_t)bh * S * D;
        const float* Vb = V + (size_t)bh * S * D;
        float* o = O + ((size_t)bh * S + i) * D;

        for (int d = tid; d < D; d += blockDim.x) qs[d] = q[d];  // q 被全 block 复用 → 先进 shared
        __syncthreads();

        float m = -INFINITY;
        for (int j = tid; j < S; j += blockDim.x){               // ① score 存 shared，不落 HBM
            const float* k = Kb + (size_t)j * D;
            float acc = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d++) acc += qs[d] * k[d];
            float v = (causal && j > i) ? -INFINITY : acc * scale;
            s[j] = v; m = fmaxf(m, v);
        }
        m = blockAllMax(m);                                      // ② 就地 softmax
        float l = 0.f;
        for (int j = tid; j < S; j += blockDim.x){ float e = __expf(s[j] - m); s[j] = e; l += e; }
        l = blockAllSum(l);
        const float inv = __fdividef(1.0f, l);

        const int d = tid % D, jc = tid / D;                     // ③ PV：64 个 d × NC 个 key 分片
        float acc = 0.f;
        for (int j = jc; j < S; j += NC) acc += s[j] * Vb[(size_t)j * D + d];
        part[jc][d] = acc;
        __syncthreads();
        if (tid < D){
            float t = 0.f;
            #pragma unroll
            for (int c = 0; c < NC; c++) t += part[c][tid];
            o[tid] = t * inv;
        }
    }
    ```

</div>
