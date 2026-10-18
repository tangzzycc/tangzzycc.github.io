---
title: "Fused Bias + GELU（向量化 elementwise）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Fused Bias + GELU（向量化 elementwise）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>float32</span><span>float4</span><span>grid-stride</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

给定按行存储的矩阵 `x[M][N]` 和按列广播的 `bias[N]`，将 Bias 加法与 GELU 融合在一个 Kernel 中，计算：

$$
y_{i,j}=\operatorname{GELU}(x_{i,j}+\operatorname{bias}_j)
$$

GELU 使用 tanh 近似：

$$
\operatorname{GELU}(v)
=\frac{v}{2}\left[1+\tanh\!\left(\sqrt{\frac{2}{\pi}}\left(v+0.044715v^3\right)\right)\right]
$$

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>x</code>：[M, N]，<code>bias</code>：[N]，均为 float32；示例取 M = N = 4096。</dd>
  <dt>输出</dt><dd><code>y</code>：[M, N]，float32。</dd>
  <dt>约定</dt><dd>M、N 为正，M × N 不超出 int 范围；N 是 4 的倍数。输入、输出不重叠，三个指针均为 cudaMalloc 返回的起始地址。</dd>
  <dt>目标</dt><dd>从标量实现逐步优化到向量化实现，减少访存指令和索引计算开销。</dd>
</dl>

主要数据流量来自读取 `x` 和写入 `y`。若 `bias` 可被缓存复用，有效带宽可按下式估算：

$$
B_{\mathrm{eff}}\approx\frac{2MN\cdot\operatorname{sizeof}(\mathrm{float})}{t}
$$

这个指标用来比较各版本的运行时间，并不等于实际 HBM 流量。GELU 包含特殊函数运算，具体瓶颈仍需结合实测判断。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void gelu_bias(const float* x,      // [M, N]
               const float* bias,   // [N]
               float* y,            // [M, N]
               int M, int N);       // 假设 N % 4 == 0
```

下面各版本的 `run_v0`～`run_v3` 使用相同的参数，分别启动对应 Kernel。

<p class="gpu-operator__label" id="tips">Tips</p>

- **先展开成一维索引**：`idx = blockIdx.x * blockDim.x + threadIdx.x`，用 `idx < M * N` 处理边界。
- **Bias 按列广播**：从线性下标恢复列号，使用 `idx % N`，而不是 `idx % M`。
- **用 grid-stride loop 解耦数据规模与启动配置**：每轮跨过整个 Grid 的线程数；Grid 大小结合 SM 数量与实际测试选择。
- **float4 需要对齐**：一次处理四个 float，要求指针 16 字节对齐且 N 能被 4 整除。切片指针不一定满足这个条件。
- **向量化减少的是指令开销**：处理相同数量的元素时，总数据量没有减少；不要把它理解为 HBM 流量直接降为四分之一。

<p class="gpu-operator__label" id="prelude">公共前置</p>

各版本共用以下头文件、宏和 GELU 函数。

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))

// GELU tanh 近似：0.5x(1+tanh(0.7978845608*(x+0.044715x^3)))
__device__ __forceinline__ float gelu(float x){
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c0 * (x + c1 * x * x * x)));
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 每线程一元素"

    **基线：每个线程处理一个元素。** Warp 内相邻线程访问相邻元素，访存已经合并。通过边界判断处理最后一个不完整的 Block；后面的版本以它为对照。

    ```cpp
    __global__ void gelu_bias_v0(const float* __restrict__ x, const float* __restrict__ b,
                                 float* __restrict__ y, int M, int N){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线性下标
        if (idx >= M * N) return;                          // 边界：一定要写
        int col = idx % N;                       // bias 按列广播
        y[idx] = gelu(x[idx] + b[col]);
    }
    void run_v0(const float* x,const float* b,float* y,int M,int N){
        int n = M * N, T = 256;
        gelu_bias_v0<<<CEIL_DIV(n,T), T>>>(x,b,y,M,N);
    }
    ```

=== "v1 grid-stride loop"

    **固定 Grid，用循环覆盖全部元素。** 步长为 `gridDim.x * blockDim.x`，相邻线程每轮仍访问相邻元素。示例取 1024 个 Block；实际应按 GPU 和数据规模调整，并不保证比 v0 更快。

    ```cpp
    __global__ void gelu_bias_v1(const float* __restrict__ x, const float* __restrict__ b,
                                 float* __restrict__ y, int M, int N){
        int n = M * N;
        int stride = gridDim.x * blockDim.x;     // 一整个 grid 的线程数
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride){
            y[i] = gelu(x[i] + b[i % N]);
        }
    }
    void run_v1(const float* x,const float* b,float* y,int M,int N){
        int T = 256, G = 1024;                             // 示例启动配置，实际结合 GPU 调整
        gelu_bias_v1<<<G,T>>>(x,b,y,M,N);
    }
    ```

=== "v2 float4 向量化"

    **每个线程一次加载、计算并写回四个元素。** `i` 是 float4 索引，`(i * 4) % N` 得到起始列，`col / 4` 得到对应的 Bias 向量索引。满足对齐且编译器生成向量访存时，可减少加载、存储指令以及循环与地址计算开销。

    ```cpp
    __global__ void gelu_bias_v2(const float* __restrict__ x, const float* __restrict__ b,
                                 float* __restrict__ y, int M, int N){
        int n4 = M * N / 4;                                // 以 float4 为单位的元素数
        int stride = gridDim.x * blockDim.x;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4*       y4 = reinterpret_cast<float4*>(y);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride){
            float4 v = x4[i];                    // 128-bit 一次读回
            int col = (i * 4) % N;                         // 该 float4 起始列
            float4 bb = reinterpret_cast<const float4*>(b)[col / 4];
            v.x = gelu(v.x + bb.x);  v.y = gelu(v.y + bb.y);
            v.z = gelu(v.z + bb.z);  v.w = gelu(v.w + bb.w);
            y4[i] = v;
        }
    }
    void run_v2(const float* x,const float* b,float* y,int M,int N){
        gelu_bias_v2<<<1024,256>>>(x,b,y,M,N);
    }
    ```

=== "v3 float4 + 快速数学"

    **在 v2 上尝试更换 GELU 的计算方式。** 利用 $\tanh(u)=1-2/(e^{2u}+1)$，用 `__expf` 构造近似结果。额外的除法、寄存器占用和近似误差可能抵消收益，是否采用需要同时比较运行时间与误差。

    ```cpp
    __device__ __forceinline__ float gelu_fast(float x){
        const float c0 = 0.7978845608028654f, c1 = 0.044715f;
        float u = c0 * (x + c1 * x * x * x);
        float t = 1.0f - 2.0f / (__expf(2.0f * u) + 1.0f);   // = tanh(u)
        return 0.5f * x * (1.0f + t);
    }
    __global__ void gelu_bias_v3(const float* __restrict__ x, const float* __restrict__ b,
                                 float* __restrict__ y, int M, int N){
        int n4 = M * N / 4, stride = gridDim.x * blockDim.x;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4*       y4 = reinterpret_cast<float4*>(y);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride){
            float4 v = x4[i];
            float4 bb = reinterpret_cast<const float4*>(b)[((i * 4) % N) / 4];
            v.x = gelu_fast(v.x + bb.x);  v.y = gelu_fast(v.y + bb.y);
            v.z = gelu_fast(v.z + bb.z);  v.w = gelu_fast(v.w + bb.w);
            y4[i] = v;
        }
    }
    void run_v3(const float* x,const float* b,float* y,int M,int N){
        gelu_bias_v3<<<1024,256>>>(x,b,y,M,N);
    }
    ```

</div>
