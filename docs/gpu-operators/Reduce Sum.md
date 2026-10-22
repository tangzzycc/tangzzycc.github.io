---
title: "Reduce Sum 七连击（归约优化全流程）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Reduce Sum 七连击（归约优化全流程）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P03</span><span>reduce</span><span>warp shuffle</span><span>bank conflict</span><span>atomic</span><span>必考</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

求 `float` 数组 `in[n]` 的和，`n = 2²⁴`（16M 个元素，64 MB）。

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>in</code>: [n] float32，n 不保证是 blockDim 的整数倍</dd>
  <dt>输出</dt><dd><code>out</code>: 单个 float32（device 上）</dd>
  <dt>约束</dt><dd>不许用 cub / thrust；要能说清每一版比上一版快在哪</dd>
  <dt>目标</dt><dd>下界就是&quot;把数据读一遍&quot;的时间 → 有效带宽 = n·4/t 应逼近 HBM 带宽</dd>
</dl>

**这是 GPU 面试出现频率最高的一道题**，也是 NVIDIA 那份著名 slides《Optimizing Parallel Reduction in CUDA》的内容。 面试官真正想看的不是最终版本，而是**你能不能一层层说出每次改动消除了什么硬件层面的浪费**。 建议把 v0→v5 的演进背成肌肉记忆，30 分钟内能默写到 v5。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
// 两段式：kernel 1 每个 block 求出部分和，kernel 2 把部分和归约成一个数
__global__ void reduce(const float* in, float* out, int n);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先想清楚数据流：**global → shared →（树形归约）→ 每 block 一个数 → 第二个 kernel 再归约一次**。
- v0 用 `tid % (2*stride) == 0` 会造成 warp 内分支发散；改成 `idx = 2*stride*tid` 让活跃线程连续。
- 再改成**步长从大到小**（`stride = blockDim/2; stride >>= 1`），这一步同时干掉分支发散和 bank conflict，是标准骨架。
- 第一轮迭代有一半线程闲着 → 让每个 block 负责 **2×blockDim** 个元素，**在从 global 读的时候就先加一次**。
- 最后 32 个线程时 `__syncthreads()` 是浪费 → 用 `__shfl_down_sync` 直接在寄存器里归约。**务必背熟 warpReduceSum / blockReduceSum 这两个模板**，后面 softmax / layernorm / attention 全要用。
- `__syncthreads()` 必须放在 if 外面：block 内所有线程都必须到达，否则 UB / 死锁。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define T 256                                   // block 线程数
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 交错寻址"

    教科书第一版：步长 1,2,4… 每轮只有 `tid%(2s)==0` 的线程干活。两个问题：**(1) warp 内一半线程走 if 的不同分支 → 分支发散**，warp 里活跃线程越来越少但整个 warp 还得跑完；**(2) `s[tid]` 与 `s[tid+stride]` 的访问模式造成 shared bank conflict**。这版能过，但是性能反面教材。

    ```cpp
    __global__ void reduce_v0(const float* in, float* out, int n){
        extern __shared__ float s[];
        int tid = threadIdx.x;
        int i   = blockIdx.x * blockDim.x + threadIdx.x;
        s[tid] = (i < n) ? in[i] : 0.0f;             // 越界补 0，别用 return（会漏 syncthreads）
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2){
            if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
            __syncthreads();                          // 必须在 if 外面：所有线程都要到达
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v1 消除分支发散"

    把"谁干活"从 `tid%2s==0` 改成 **让前 k 个连续线程干活**：`idx = 2*stride*tid`。同样的加法树，但活跃线程在 warp 里是连续的 —— 当活跃线程数 ≥32 时**整个 warp 要么全干活要么全空闲，分支发散没了**。代价：`s[idx]` 步长 2/4/8… 的访问带来严重 bank conflict。

    ```cpp
    __global__ void reduce_v1(const float* in, float* out, int n){
        extern __shared__ float s[];
        int tid = threadIdx.x;
        int i   = blockIdx.x * blockDim.x + threadIdx.x;
        s[tid] = (i < n) ? in[i] : 0.0f;
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2){
            int idx = 2 * stride * tid;     // 活跃线程连续
            if (idx + stride < blockDim.x) s[idx] += s[idx + stride];
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v2 顺序寻址"

    把步长**从大到小**：`stride = blockDim/2, /2, …`，永远是 `tid < stride` 的前一半线程读 `s[tid]` 和 `s[tid+stride]`。**两个地址都是连续的 → bank conflict 彻底消失**，分支发散也没有。这是 reduce 的标准骨架，后面所有变体都基于它。

    ```cpp
    __global__ void reduce_v2(const float* in, float* out, int n){
        extern __shared__ float s[];
        int tid = threadIdx.x;
        int i   = blockIdx.x * blockDim.x + threadIdx.x;
        s[tid] = (i < n) ? in[i] : 0.0f;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v3 首次加载即相加"

    v2 的第一轮迭代里有一半线程完全闲着——那不如**在从 global 读的时候就先加一次**：每个 block 负责 2\*blockDim 个元素，grid 减半。这一步同时把 kernel 从"访存+同步"变成"访存"主导：**block 数减半 = 调度开销和第二段 reduce 的输入都减半**。

    ```cpp
    __global__ void reduce_v3(const float* in, float* out, int n){
        extern __shared__ float s[];
        int tid = threadIdx.x;
        int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        float a = (i < n)             ? in[i]             : 0.0f;
        float b = (i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f;
        s[tid] = a + b;                     // 读进来就先加
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v4 展开最后一个 warp"

    当 stride ≤ 32 时只剩一个 warp 在干活，此时 `__syncthreads()` 是纯浪费（warp 内本来就同步执行）。把最后 6 轮**手工展开**并用 `__syncwarp()` 代替。注意：**Volta 之后 warp 内不再保证锁步**，只写 `volatile` 是不够的，必须加 `__syncwarp()`，否则可能读到旧值。

    ```cpp
    __device__ __forceinline__ void warpReduceShared(volatile float* s, int tid){
        s[tid] += s[tid + 32]; __syncwarp();
        s[tid] += s[tid + 16]; __syncwarp();
        s[tid] += s[tid +  8]; __syncwarp();
        s[tid] += s[tid +  4]; __syncwarp();
        s[tid] += s[tid +  2]; __syncwarp();
        s[tid] += s[tid +  1]; __syncwarp();
    }
    __global__ void reduce_v4(const float* in, float* out, int n){
        extern __shared__ float s[];
        int tid = threadIdx.x;
        int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        s[tid] = ((i < n) ? in[i] : 0.0f) + ((i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f);
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 32; stride >>= 1){
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        if (tid < 32) warpReduceShared(s, tid);
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v5 warp shuffle（面试标准答案）"

    `__shfl_down_sync` 直接在**寄存器**之间交换数据：不走 shared、不用 `__syncthreads`、没有 bank conflict。结构是固定的两级：**warp 内 shuffle 归约 → 每个 warp 的 lane0 写 shared（最多 32 个数）→ 第一个 warp 再 shuffle 归约一次**。`blockReduceSum` 这个模板后面 softmax / layernorm / attention 全都要复用，务必背熟。

    ```cpp
    __device__ __forceinline__ float warpReduceSum(float v){
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        return v;                                     // lane 0 持有结果
    }
    __device__ __forceinline__ float blockReduceSum(float v){
        static __shared__ float warp[32];             // 一个 block 最多 32 个 warp
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        v = warpReduceSum(v);
        if (lane == 0) warp[wid] = v;
        __syncthreads();
        int nw = blockDim.x >> 5;
        v = (threadIdx.x < nw) ? warp[lane] : 0.0f;   // 只让第一个 warp 装载
        if (wid == 0) v = warpReduceSum(v);
        return v;                                     // thread 0 持有结果
    }
    __global__ void reduce_v5(const float* in, float* out, int n){
        int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
        float v = ((i < n) ? in[i] : 0.0f) + ((i + blockDim.x < n) ? in[i + blockDim.x] : 0.0f);
        v = blockReduceSum(v);
        if (threadIdx.x == 0) out[blockIdx.x] = v;
    }
    ```

=== "v6 grid-stride + float4 + atomic"

    生产写法：**固定 grid（≈ SM数×每SM block数），每线程用 float4 grid-stride 扫全数组**，块内 shuffle 归约，最后 `atomicAdd` 到单个 float —— 一次 kernel 出结果，省掉第二段。注意 atomicAdd(float) 的**累加顺序不确定 → 结果不可复现**；要 bitwise 可复现就退回两段式（确定性的树形归约）。

    ```cpp
    __global__ void reduce_v6(const float* __restrict__ in, float* __restrict__ out, int n){
        float v = 0.0f;
        int n4 = n / 4, stride = gridDim.x * blockDim.x;
        const float4* in4 = reinterpret_cast<const float4*>(in);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride){
            float4 t = in4[i];
            v += t.x + t.y + t.z + t.w;               // 每线程先在寄存器里累加
        }
        for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) v += in[i]; // 尾巴
        v = blockReduceSum(v);
        if (threadIdx.x == 0) atomicAdd(out, v);
    }
    ```

</div>
