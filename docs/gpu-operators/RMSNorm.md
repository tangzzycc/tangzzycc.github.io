---
title: "RMSNorm（LLaMA 形态：融合 residual + half IO）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">RMSNorm（LLaMA 形态：融合 residual + half IO）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P09</span><span>rmsnorm</span><span>LLaMA</span><span>kernel fusion</span><span>fp16</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

实现 RMSNorm，并做两个推理里必做的事：**融合 residual add**、**支持 half 输入输出**。

`y = x · rsqrt(mean(x²) + ε) · γ`

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>x</code>: [M, N]，<code>γ</code>: [N]，ε=1e-6；M=8192, N=4096</dd>
  <dt>输出</dt><dd><code>y</code>: [M, N]；融合版还要输出 <code>x + residual</code> 供下一层用</dd>
  <dt>与 LayerNorm 的区别</dt><dd><strong>不减均值、没有 β</strong> → 少一次归约、少一次减法</dd>
</dl>

LLaMA / Qwen / Mistral 全系用 RMSNorm，所以**推理岗面试问 norm 时，大概率问的是这个**。 代码比 LayerNorm 更简单，考点集中在**融合**和**低精度**上。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void rmsnorm(const float* x, const float* gamma,
             float* y, int M, int N, float eps);

// 融合版（vLLM 里的 fused_add_rms_norm）
void fused_add_rmsnorm(const half* x, const half* residual, const half* gamma,
                       half* out_residual, half* y, int M, int N, float eps);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 比 LayerNorm 少一次归约：只需要 `Σx²`，不需要 `Σx`。
- 用 `rsqrtf(ss/N + eps)`，一条硬件指令搞定"求和 → 平均 → 开方 → 取倒数"的最后两步。
- 寄存器缓存 + float4 是标配：读 1 遍写 1 遍。索引记得写成 `threadIdx.x + j*blockDim.x`。
- 融合 residual：**加完之后要把和写回去**（下一层的 residual 分支要用），再归一化输出——一个 kernel 三件事。
- half 版：一个 `float4` 装 8 个 half（访存再省一半），但**平方和必须用 fp32 累加**。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define FULL 0xffffffffu

__device__ __forceinline__ float warpAllSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float blockAllSum(float v){
    __shared__ float sm[32]; __shared__ float res;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllSum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : 0.f; t = warpAllSum(t); if (lane == 0) res = t; }
    __syncthreads();
    return res;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 朴素两趟"

    RMSNorm 比 LayerNorm 简单：**不减均值、没有 beta**，只有

    `y = x · rsqrt(mean(x²) + ε) · γ`

    省掉了求均值那一趟归约和一次减法，这也是 LLaMA / Qwen 系列全面改用它的原因（**效果几乎不变，算子快 20~30%**）。

    这一版读 2 遍 global：一遍求平方和，一遍写出。

    ```cpp
    __global__ void rms_v0(const float* __restrict__ x, const float* __restrict__ g,
                           float* __restrict__ y, int N, float eps){
        const float* xr = x + (size_t)blockIdx.x * N;
        float*       yr = y + (size_t)blockIdx.x * N;
        float ss = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x){ float v = xr[i]; ss += v * v; }
        ss = blockAllSum(ss);
        float inv = rsqrtf(ss / N + eps);         // 用 rsqrtf，不要写 1.f/sqrtf
        for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = xr[i] * inv * g[i];
    }
    ```

=== "v1 寄存器缓存 + float4"

    和 softmax / LayerNorm 完全一样的套路：**一次读进寄存器 → 归约 → 直接写出**，global 读 1 遍写 1 遍，打满带宽。

    注意索引写成 `threadIdx.x + j*blockDim.x`（相邻线程读相邻 float4）而**不是** `threadIdx.x*V + j`——后者虽然"每线程一段连续"看起来自然，但**一个 warp 的 32 个线程会跨 2 KB，访存不合并，**

    ```cpp
    template<int V>
    __global__ void rms_v1(const float* __restrict__ x, const float* __restrict__ g,
                           float* __restrict__ y, int N, float eps){
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * N);
        float4*       yr = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * N);
        const float4* g4 = reinterpret_cast<const float4*>(g);
        float4 r[V];
        #pragma unroll
        for (int j = 0; j < V; j++) r[j] = xr[threadIdx.x + j * blockDim.x];
        float* rf = reinterpret_cast<float*>(r);
        float ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++) ss += rf[k] * rf[k];
        float inv = rsqrtf(blockAllSum(ss) / N + eps);
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 gg = g4[c];
            r[j].x *= inv * gg.x; r[j].y *= inv * gg.y;
            r[j].z *= inv * gg.z; r[j].w *= inv * gg.w;
            yr[c] = r[j];
        }
    }
    ```

=== "v2 融合 residual add（LLaMA 形态）"

    LLaMA 的每一层是

    `h = h + Attn(RMSNorm(h))` → `h = h + MLP(RMSNorm(h))`

    所以 **"加残差"和"归一化"永远成对出现**。拆成两个 kernel 要多读写一遍 hidden state（7B 模型每层几 MB × 32 层 × 2 次/层）。

    融合后一个 kernel：**加残差 → 把和写回（下一个残差要用）→ 归一化输出**。这是推理框架里必做的融合，vLLM 的 `fused_add_rms_norm` 就是它。

    ```cpp
    template<int V>
    __global__ void rms_v2_fused(const float* __restrict__ x, const float* __restrict__ res,
                                 const float* __restrict__ g, float* __restrict__ out_res,
                                 float* __restrict__ y, int N, float eps){
        const size_t o4 = (size_t)blockIdx.x * (N / 4);
        const float4* xr = reinterpret_cast<const float4*>(x) + o4;
        const float4* rr = reinterpret_cast<const float4*>(res) + o4;
        float4* orr = reinterpret_cast<float4*>(out_res) + o4;
        float4* yr  = reinterpret_cast<float4*>(y) + o4;
        const float4* g4 = reinterpret_cast<const float4*>(g);
        float4 r[V];
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 a = xr[c], d = rr[c];
            a.x += d.x; a.y += d.y; a.z += d.z; a.w += d.w;
            r[j] = a;
            orr[c] = a;                            // 残差流写回，给下一层用
        }
        float* rf = reinterpret_cast<float*>(r);
        float ss = 0.f;
        #pragma unroll
        for (int k = 0; k < 4 * V; k++) ss += rf[k] * rf[k];
        float inv = rsqrtf(blockAllSum(ss) / N + eps);
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 gg = g4[c], o;
            o.x = r[j].x * inv * gg.x; o.y = r[j].y * inv * gg.y;
            o.z = r[j].z * inv * gg.z; o.w = r[j].w * inv * gg.w;
            yr[c] = o;
        }
    }
    ```

=== "v3 half IO + fp32 累加（真实推理形态）"

    推理里 hidden state 是 **fp16/bf16**。两个关键点：

    ① **访存仍然按 16 字节走**：一个 `float4` 装 **8 个 half**，比 fp32 版本再省一半带宽；

    ② **累加必须用 fp32**：half 尾数只有 10 bit，4096 个数的平方和累加会严重失真，`x²` 还容易溢出 half 的 65504 上限。

    口诀：**"存储用低精度，累加用高精度"**。bf16 更狠——尾数只有 7 bit，但动态范围和 fp32 一样，所以不怕溢出、只怕精度，累加同样要 fp32。

    ```cpp
    template<int V>       // V = 每线程的 float4 个数，一个 float4 = 8 个 half
    __global__ void rms_v3_half(const __half* __restrict__ x, const __half* __restrict__ res,
                                const __half* __restrict__ g, __half* __restrict__ out_res,
                                __half* __restrict__ y, int N, float eps){
        const size_t o4 = (size_t)blockIdx.x * (N / 8);
        const float4* xr = reinterpret_cast<const float4*>(x) + o4;
        const float4* rr = reinterpret_cast<const float4*>(res) + o4;
        float4* orr = reinterpret_cast<float4*>(out_res) + o4;
        float4* yr  = reinterpret_cast<float4*>(y) + o4;
        const float4* g4 = reinterpret_cast<const float4*>(g);

        float4 buf[V];
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 a = xr[c], d = rr[c];
            __half2* pa = reinterpret_cast<__half2*>(&a);
            __half2* pd = reinterpret_cast<__half2*>(&d);
            #pragma unroll
            for (int t = 0; t < 4; t++) pa[t] = __hadd2(pa[t], pd[t]);   // 8 个 half 一次加完
            buf[j] = a;
            orr[c] = a;
        }
        float ss = 0.f;
        __half2* h2 = reinterpret_cast<__half2*>(buf);
        #pragma unroll
        for (int k = 0; k < 4 * V; k++){
            float2 f = __half22float2(h2[k]);      // 转 fp32 再平方累加
            ss += f.x * f.x + f.y * f.y;
        }
        float inv = rsqrtf(blockAllSum(ss) / N + eps);
        #pragma unroll
        for (int j = 0; j < V; j++){
            int c = threadIdx.x + j * blockDim.x;
            float4 gv = g4[c], o;
            __half2* pg = reinterpret_cast<__half2*>(&gv);
            __half2* pb = reinterpret_cast<__half2*>(&buf[j]);
            __half2* po = reinterpret_cast<__half2*>(&o);
            #pragma unroll
            for (int t = 0; t < 4; t++){
                float2 b = __half22float2(pb[t]), gg = __half22float2(pg[t]);
                po[t] = __floats2half2_rn(b.x * inv * gg.x, b.y * inv * gg.y);
            }
            yr[c] = o;
        }
    }
    ```

</div>
