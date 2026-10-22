---
title: "Softmax（safe / online 一遍法 / 寄存器缓存）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Softmax（safe / online 一遍法 / 寄存器缓存）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P07</span><span>softmax</span><span>数值稳定</span><span>online</span><span>FlashAttention 前置</span><span>必考</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

对 `x[M][N]` 的**每一行**做 softmax，M=8192，N=4096。

`y[i][j] = exp(x[i][j]) / Σⱼ exp(x[i][j])`

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>x</code>: [M, N] float32，<strong>数值范围不受限</strong>（测试里故意放了 120 这样的大值）</dd>
  <dt>输出</dt><dd><code>y</code>: [M, N] float32，每行和为 1</dd>
  <dt>考点</dt><dd>① 数值稳定（减 max）② online 一遍法 ③ 减少 global 读写次数</dd>
  <dt>瓶颈</dt><dd>memory-bound，下界是&quot;读一遍写一遍&quot;</dd>
</dl>

softmax 是**面试必考**，而且是 attention 的前置：**online softmax 就是 FlashAttention 的数学核心**， 不会这个，后面的 flash attention 一定写不出来。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void softmax(const float* x,   // [M, N]
             float* y,         // [M, N]
             int M, int N);    // 沿 N 维做 softmax
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 第一版可以直接按定义写，但**写完要主动说"这里 exp 会溢出"**——这是面试官在等的话。
- safe softmax：`softmax(x) = softmax(x - max)`，减完之后 exp 的参数 ≤ 0，永不上溢（下溢成 0 是安全的）。
- 一个 block 处理一行，用 `blockAllReduce`（结果要广播给所有线程，因为每个线程都要减 max、除 sum）。
- online softmax：边扫边维护 (m, l)，新来一个值时 `l = l·exp(m_old - m_new) + exp(x - m_new)`。这样 m 和 l 一趟就出来了。
- 如果 N 不大，让每个线程**把自己那份数据留在寄存器里**，就能做到 global 只读 1 遍写 1 遍——这是最快的形态。
- **向量化时索引要写成 `threadIdx.x + j*blockDim.x`**（相邻线程读相邻 float4），不要写成每线程一段连续区间。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define FULL 0xffffffffu

__device__ __forceinline__ float warpAllMax(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;                                   // 全 lane 有效
}
__device__ __forceinline__ float warpAllSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
// 全 block 可见的归约：softmax 每个线程都要用 max 和 sum，所以必须广播
__device__ __forceinline__ float blockAllMax(float v){
    __shared__ float sm[32]; __shared__ float res;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllMax(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : -INFINITY; t = warpAllMax(t); if (lane == 0) res = t; }
    __syncthreads();
    return res;
}
__device__ __forceinline__ float blockAllSum(float v){
    __shared__ float sm[32]; __shared__ float res;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllSum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : 0.0f; t = warpAllSum(t); if (lane == 0) res = t; }
    __syncthreads();
    return res;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 三趟朴素（数值不安全）"

    直接按定义写：求 sum(exp(x))，再除。**这一版会溢出**：x 只要超过 88，`expf` 就返回 inf，inf/inf = NaN。LLM 的 logits 经常上百，attention score 在长序列里也不小。

    面试时可以先写它，然后**主动说"这里会溢出，我要减 max"** —— 这是考察数值稳定性意识的标准套路。

    ```cpp
    __global__ void softmax_v0(const float* __restrict__ x, float* __restrict__ y, int N){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float s = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) s += __expf(xr[i]);   // ← 会 inf
        s = blockAllSum(s);
        for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = __expf(xr[i]) / s;
    }
    ```

=== "v1 safe softmax（减 max）"

    **softmax(x) = softmax(x - c)** 对任意常数 c 成立，取 c = max(x) 就能保证 exp 的参数 ≤ 0、结果落在 (0,1]，永不溢出（下溢成 0 是安全的）。

    代价是**多一趟归约**：先扫一遍求 max，再扫一遍求 sum，最后一遍写出 → **读 3 遍 global**。这是最标准的"正确但不够快"的答案。

    ```cpp
    __global__ void softmax_v1(const float* __restrict__ x, float* __restrict__ y, int N){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float m = -INFINITY;
        for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, xr[i]);
        m = blockAllMax(m);                                   // ① 全行 max
        float s = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) s += __expf(xr[i] - m);
        s = blockAllSum(s);                                   // ② 全行 sum
        float inv = __fdividef(1.0f, s);            // 除法换成一次倒数 + 乘法
        for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = __expf(xr[i] - m) * inv;
    }
    ```

=== "v2 online softmax（一趟出 m 和 l）"

    **FlashAttention 的核心数学**。边扫边维护 (m, l)：读到新值 x 时

    `m_new = max(m, x)`，`l_new = l·exp(m - m_new) + exp(x - m_new)`

    关键在 **`exp(m - m_new)` 这个 rescale 因子**：旧的 l 是以旧 max 为基准算的，换基准时整体缩放一下即可。

    这样**求 m 和 l 只需要扫一趟**（3 趟 → 2 趟）。归约时两个 warp 的 (m,l) 合并也用同一个公式，所以要自己写合并逻辑，不能直接用 blockReduce。

    ```cpp
    __device__ __forceinline__ void mergeML(float& m, float& l, float m2, float l2){
        float mn = fmaxf(m, m2);
        // ⚠ 两边都还是空的（m = m2 = -inf）时，(-inf) - (-inf) = NaN，会污染整行。
        //   FlashAttention 里"这个 KV block 全被 mask 掉"就是这个情况，必须单独处理。
        if (mn == -INFINITY){ m = mn; l = 0.f; return; }
        l = l * __expf(m - mn) + l2 * __expf(m2 - mn);        // 两边都缩到新基准
        m = mn;
    }
    __global__ void softmax_v2(const float* __restrict__ x, float* __restrict__ y, int N){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float m = -INFINITY, l = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x){    // ① 一趟同时得到 m 和 l
            float v = xr[i];
            float mn = fmaxf(m, v);
            l = l * __expf(m - mn) + __expf(v - mn);
            m = mn;
        }
        // warp / block 内合并 (m,l)：不能分别 reduce，必须按上面的公式两两合并
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1){
            float m2 = __shfl_xor_sync(FULL, m, o), l2 = __shfl_xor_sync(FULL, l, o);
            mergeML(m, l, m2, l2);
        }
        __shared__ float sm[32], sl[32]; __shared__ float rm, rl;
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
        if (lane == 0){ sm[wid] = m; sl[wid] = l; }
        __syncthreads();
        if (wid == 0){
            float mm = (lane < nw) ? sm[lane] : -INFINITY, ll = (lane < nw) ? sl[lane] : 0.f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1){
                float m2 = __shfl_xor_sync(FULL, mm, o), l2 = __shfl_xor_sync(FULL, ll, o);
                mergeML(mm, ll, m2, l2);
            }
            if (lane == 0){ rm = mm; rl = ll; }
        }
        __syncthreads();
        float inv = __fdividef(1.0f, rl);
        for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = __expf(xr[i] - rm) * inv;   // ② 第二趟写出
    }
    ```

=== "v3 寄存器缓存 + float4（最快）"

    前面几版都要**把 x 从 HBM 读两三遍**。如果一行不太长（N ≤ 几千），可以让每个线程**把自己负责的元素一次性读进寄存器**：

    `读一遍 → 求 max → 求 sum → 直接用寄存器里的值写出`，**global 只读 1 遍、写 1 遍，达到 memory-bound 下界**。

    限制：`EPT = N/blockDim` 必须是编译期常量且不能太大（寄存器 255 个上限），否则会 spill 到 local memory 反而更慢。**这是 vLLM / TensorRT 里 softmax 的真实形态。**

    ```cpp
    template<int V>                                   // V = 每线程的 float4 个数
    __global__ void softmax_v3(const float* __restrict__ x, float* __restrict__ y, int N){
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * N);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * N);
        float4 r[V];
        #pragma unroll                                    // 相邻线程读相邻 float4 → 完全合并
        for (int j = 0; j < V; j++) r[j] = xr[threadIdx.x + j * blockDim.x];
        float* rf = reinterpret_cast<float*>(r);          // 当成 4V 个标量用

        float m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++) m = fmaxf(m, rf[k]);
        m = blockAllMax(m);

        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++){ rf[k] = __expf(rf[k] - m); s += rf[k]; }
        s = blockAllSum(s);

        float inv = __fdividef(1.0f, s);
        #pragma unroll
        for (int k = 0; k < 4 * V; k++) rf[k] *= inv;
        #pragma unroll
        for (int j = 0; j < V; j++) yr[threadIdx.x + j * blockDim.x] = r[j];
    }
    ```

=== "v4 一个 warp 一行（N 小时）"

    当 N 只有几十~几百（**MoE 的 router、多头 attention 的短序列、beam search 的候选打分**），一个 block 处理一行就太浪费了：block 内跨 warp 的归约要走 shared + syncthreads，而数据量还不够填满一个 warp。

    改成 **一个 warp 负责一行**：归约全在 `shfl` 里完成，**零 shared、零 \_\_syncthreads**，一个 block（256 线程）同时处理 8 行。

    ```cpp
    __global__ void softmax_v4(const float* __restrict__ x, float* __restrict__ y, int M, int N){
        const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 全局第几个 warp
        const int lane   = threadIdx.x & 31;
        if (warpId >= M) return;                              // 注意：这里 return 是安全的，因为没有 __syncthreads
        const float* xr = x + (size_t)warpId * N;
        float*       yr = y + (size_t)warpId * N;

        float m = -INFINITY;
        for (int i = lane; i < N; i += 32) m = fmaxf(m, xr[i]);
        m = warpAllMax(m);                          // 纯 shuffle，不用 shared
        float s = 0.f;
        for (int i = lane; i < N; i += 32) s += __expf(xr[i] - m);
        s = warpAllSum(s);
        float inv = __fdividef(1.0f, s);
        for (int i = lane; i < N; i += 32) yr[i] = __expf(xr[i] - m) * inv;
    }
    ```

</div>
