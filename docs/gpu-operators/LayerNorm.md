---
title: "LayerNorm（一趟 vs Welford 的数值取舍 + 融合 residual）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">LayerNorm（一趟 vs Welford 的数值取舍 + 融合 residual）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P08</span><span>layernorm</span><span>Welford</span><span>灾难性抵消</span><span>kernel fusion</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

对 `x[M][N]` 每一行做 LayerNorm，M=8192，N=4096。

`y = (x - E[x]) / sqrt(Var[x] + ε) · γ + β`

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>x</code>: [M, N]，<code>γ</code>/<code>β</code>: [N]，ε=1e-5</dd>
  <dt>输出</dt><dd><code>y</code>: [M, N]</dd>
  <dt>考点</dt><dd>① 一趟求 sum 和 sum² 的<strong>数值风险</strong> ② Welford ③ 和 residual add 的融合</dd>
  <dt>瓶颈</dt><dd>memory-bound；关键是&quot;x 读几遍&quot;</dd>
</dl>

LayerNorm 的代码本身不难（会了 reduce 就会），**面试真正问的是数值稳定性**： "你用 E[x²]-E[x]² 一趟算方差，什么时候会出问题？" —— 通过构造输入即可观察到这个数值稳定性问题。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void layernorm(const float* x,      // [M, N]
               const float* gamma,  // [N]
               const float* beta,   // [N]
               float* y,            // [M, N]
               int M, int N, float eps);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 最直接的写法是两趟：先归约求 mean，再归约求 `Σ(x-mean)²`。这版数值最稳但要读 3 遍 x。
- 一趟写法：同时累加 `Σx` 和 `Σx²`，用 `Var = E[x²] - E[x]²`。**两个量打包成 float2 一次归约**，别调两次 blockReduce。
- 一趟法有**灾难性抵消**风险：均值远大于标准差时，两个大数相减会丢光有效位，甚至算出负方差。
- Welford 是"一趟且稳定"的解：只操作偏差量。并行版需要一个合并公式把两个部分统计量拼起来。
- 生产形态：**把数据留在寄存器 + float4 向量化 + 和 residual add 融合**。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define FULL 0xffffffffu

__device__ __forceinline__ float warpAllSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
// 一次归约两个量（sum 和 sum²）：LayerNorm 的标准做法
__device__ __forceinline__ float2 blockAllSum2(float2 v){
    __shared__ float2 sm[32]; __shared__ float2 res;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v.x = warpAllSum(v.x); v.y = warpAllSum(v.y);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){
        float2 t = (lane < nw) ? sm[lane] : make_float2(0.f, 0.f);
        t.x = warpAllSum(t.x); t.y = warpAllSum(t.y);
        if (lane == 0) res = t;
    }
    __syncthreads();
    return res;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 两趟（教科书定义）"

    严格按定义来：**第一趟求均值，第二趟求方差**，第三趟归一化写出。

    优点是数值最稳（方差用 `(x-mean)²` 直接算，没有大数相减）；缺点是 **x 要从 HBM 读 3 遍**。LayerNorm 是彻头彻尾的 memory-bound 算子，读 3 遍就是慢 3 倍。

    ```cpp
    __global__ void ln_v0(const float* __restrict__ x, const float* __restrict__ g,
                          const float* __restrict__ b, float* __restrict__ y, int N, float eps){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float s = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) s += xr[i];
        float mean = blockAllSum2(make_float2(s, 0.f)).x / N;      // ① 均值
        float v = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x){ float d = xr[i] - mean; v += d * d; }
        float var = blockAllSum2(make_float2(v, 0.f)).x / N;       // ② 方差
        float inv = rsqrtf(var + eps);
        for (int i = threadIdx.x; i < N; i += blockDim.x)
            yr[i] = (xr[i] - mean) * inv * g[i] + b[i];            // ③ 写出
    }
    ```

=== "v1 一趟 sum + sum²（要小心）"

    用 **Var[x] = E[x²] - E[x]²** 就能一趟同时累加 `sum` 和 `sum²`，读 2 遍而不是 3 遍，而且两个量**打包成 float2 一次归约**。

    ⚠ **陷阱**：当均值远大于标准差时（比如 x ≈ 1000 ± 1），E[x²] 和 E[x]² 都是 1e6 量级而它们的差只有 0.33，**float 的 7 位有效数字被吃光，方差算出来可能是负数**——这叫灾难性抵消。

    实践中 LayerNorm 前面通常有 residual，输入大致零均值，所以这版能用；但**面试必须主动说出这个风险**。

    ```cpp
    __global__ void ln_v1(const float* __restrict__ x, const float* __restrict__ g,
                          const float* __restrict__ b, float* __restrict__ y, int N, float eps){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float s = 0.f, ss = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x){ float v = xr[i]; s += v; ss += v * v; }
        float2 r = blockAllSum2(make_float2(s, ss));     // 一次归约拿两个量
        float mean = r.x / N;
        float var  = r.y / N - mean * mean;              // ← 灾难性抵消就发生在这里
        float inv  = rsqrtf(fmaxf(var, 0.f) + eps);                // 夹一下，防止算出负方差
        for (int i = threadIdx.x; i < N; i += blockDim.x)
            yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
    }
    ```

=== "v2 Welford（一趟且数值稳定）"

    Welford 增量公式：

    `n += 1; δ = x - mean; mean += δ/n; M2 += δ·(x - mean_new)`

    **全程只操作"偏差"，不会出现大数相减**，所以既是一趟又稳定。并行版需要一个**合并公式**把两个部分统计量拼起来：

    `δ = mb - ma; mean = ma + δ·nb/n; M2 = M2a + M2b + δ²·na·nb/n`

    代价：每次更新多一次除法，比 v1 慢一点。**PyTorch 的 LayerNorm、以及所有训练框架的 BatchNorm 统计都用 Welford。**

    ```cpp
    struct WF { float mean, m2; int n; };
    __device__ __forceinline__ void wfMerge(WF& a, WF b){
        if (b.n == 0) return;
        int n = a.n + b.n;
        float d = b.mean - a.mean;
        float nb_over_n = (float)b.n / (float)n;
        a.mean += d * nb_over_n;
        a.m2   += b.m2 + d * d * (float)a.n * nb_over_n;
        a.n = n;
    }
    __global__ void ln_v2(const float* __restrict__ x, const float* __restrict__ g,
                          const float* __restrict__ b, float* __restrict__ y, int N, float eps){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        WF w{0.f, 0.f, 0};
        for (int i = threadIdx.x; i < N; i += blockDim.x){          // ① 线程内 Welford
            float v = xr[i];
            w.n++;
            float d = v - w.mean;
            w.mean += d / w.n;
            w.m2   += d * (v - w.mean);
        }
        #pragma unroll                                              // ② warp 内合并
        for (int o = 16; o > 0; o >>= 1){
            WF t; t.mean = __shfl_xor_sync(FULL, w.mean, o);
                  t.m2   = __shfl_xor_sync(FULL, w.m2, o);
                  t.n    = __shfl_xor_sync(FULL, w.n, o);
            wfMerge(w, t);
        }
        __shared__ WF sw[32]; __shared__ WF res;
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
        if (lane == 0) sw[wid] = w;
        __syncthreads();
        if (wid == 0){                                              // ③ block 内合并
            WF t = (lane < nw) ? sw[lane] : WF{0.f, 0.f, 0};
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1){
                WF u; u.mean = __shfl_xor_sync(FULL, t.mean, o);
                      u.m2   = __shfl_xor_sync(FULL, t.m2, o);
                      u.n    = __shfl_xor_sync(FULL, t.n, o);
                wfMerge(t, u);
            }
            if (lane == 0) res = t;
        }
        __syncthreads();
        float mean = res.mean, inv = rsqrtf(res.m2 / N + eps);
        for (int i = threadIdx.x; i < N; i += blockDim.x)
            yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
    }
    ```

=== "v3 寄存器缓存 + float4"

    和 softmax 一样的招：**一次把自己负责的元素读进寄存器，算完直接写出**，global 只读 1 遍写 1 遍，达到 memory-bound 的下界。

    `gamma` / `beta` 只有 N 个元素、被所有行复用，**会常驻 L2**，不算额外带宽。这版是推理框架里 LayerNorm 的真实形态。

    ```cpp
    template<int V>                                    // V = 每线程负责的 float4 个数
    __global__ void ln_v3(const float* __restrict__ x, const float* __restrict__ g,
                          const float* __restrict__ b, float* __restrict__ y, int N, float eps){
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * N);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * N);
        const float4* g4 = reinterpret_cast<const float4*>(g);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        float4 r[V];
        #pragma unroll                                     // 相邻线程读相邻 float4 → 完全合并
        for (int j = 0; j < V; j++) r[j] = xr[threadIdx.x + j * blockDim.x];
        float* rf = reinterpret_cast<float*>(r);

        float s = 0.f, ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++){ s += rf[k]; ss += rf[k] * rf[k]; }
        float2 t = blockAllSum2(make_float2(s, ss));
        float mean = t.x / N, inv = rsqrtf(fmaxf(t.y / N - mean * mean, 0.f) + eps);

        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 gg = g4[c], bb = b4[c];                 // gamma/beta 只有 N 个元素，常驻 L2
            r[j].x = (r[j].x - mean) * inv * gg.x + bb.x;
            r[j].y = (r[j].y - mean) * inv * gg.y + bb.y;
            r[j].z = (r[j].z - mean) * inv * gg.z + bb.z;
            r[j].w = (r[j].w - mean) * inv * gg.w + bb.w;
            yr[c] = r[j];
        }
    }
    ```

=== "v4 融合 residual add（推理形态）"

    Transformer 每层都是 `h = h + Attn(...)` → `LN(h)`。**拆成两个 kernel 要多读写一遍 h**（对 7B 模型的 hidden state 就是几十 MB×层数）。

    融合后一个 kernel 干三件事：**加残差 → 把结果写回给下一层用 → 顺手归一化输出**。

    这是推理优化里**最典型、收益最直接的 kernel fusion**，面试时能主动提出来就说明你懂推理而不只是懂 CUDA。

    ```cpp
    template<int V>
    __global__ void ln_v4_fused(const float* __restrict__ x, const float* __restrict__ res,
                                const float* __restrict__ g, const float* __restrict__ b,
                                float* __restrict__ out_res, float* __restrict__ y, int N, float eps){
        const size_t off4 = (size_t)blockIdx.x * (N / 4);
        const float4* xr = reinterpret_cast<const float4*>(x) + off4;
        const float4* rr = reinterpret_cast<const float4*>(res) + off4;
        float4* orr = reinterpret_cast<float4*>(out_res) + off4;
        float4* yr  = reinterpret_cast<float4*>(y) + off4;
        const float4* g4 = reinterpret_cast<const float4*>(g);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        float4 r[V];
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 a = xr[c], d = rr[c];
            a.x += d.x; a.y += d.y; a.z += d.z; a.w += d.w;   // ① 残差相加
            r[j] = a;
            orr[c] = a;                             // ② 写回给下一层用
        }
        float* rf = reinterpret_cast<float*>(r);
        float s = 0.f, ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++){ s += rf[k]; ss += rf[k] * rf[k]; }
        float2 t = blockAllSum2(make_float2(s, ss));
        float mean = t.x / N, inv = rsqrtf(fmaxf(t.y / N - mean * mean, 0.f) + eps);
        #pragma unroll
        for (int j = 0; j < V; j++){                          // ③ 归一化输出
            int c = threadIdx.x + j * blockDim.x;
            float4 gg = g4[c], bb = b4[c];
            float4 o;
            o.x = (r[j].x - mean) * inv * gg.x + bb.x;
            o.y = (r[j].y - mean) * inv * gg.y + bb.y;
            o.z = (r[j].z - mean) * inv * gg.z + bb.z;
            o.w = (r[j].w - mean) * inv * gg.w + bb.w;
            yr[c] = o;
        }
    }
    ```

</div>
