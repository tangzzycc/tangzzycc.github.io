---
title: "Reduce Max / ArgMax（含 float atomicMax 技巧）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Reduce Max / ArgMax（含 float atomicMax 技巧）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P05</span><span>argmax</span><span>atomic</span><span>位技巧</span><span>tie-break</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

求 `float` 数组的最大值，以及**最大值所在的下标**（并列时取最小下标）。

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>in</code>: [n] float32，<strong>可能全是负数</strong>，且<strong>可能有并列的最大值</strong></dd>
  <dt>输出</dt><dd>最大值 + 下标</dd>
  <dt>陷阱</dt><dd>① 越界补 0（全负数就错了）② 并列时下标不确定 ③ CUDA 没有 <code>atomicMax(float*)</code></dd>
</dl>

argmax 在推理里到处都是：**greedy sampling 取 top-1 token、beam search、softmax 的 max 归约、 TopK 的第一步、量化的 absmax 统计**。而且它比 sum 多两个考点：**单位元**和 **tie-break**。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void reduce_max(const float* in, float* out_val, int n);
void arg_max(const float* in, float* out_val, int* out_idx, int n);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 和 sum 完全同构，只有两处不同：`+` → `fmaxf`，单位元 `0` → **`-INFINITY`**。
- 要下标就得**同时搬运 (value, index) 两个量**：两次 shfl 换出邻居的 v 和 i，再按比较结果整体替换。
- **一定要定义 tie-break 规则**（值相等时取小下标），否则结果依赖于调度顺序，不可复现。
- CUDA 没有 `atomicMax(float*)`：把 float 转成**保序的 uint32**（正数置符号位、负数全取反），再用整型 atomicMax。
- argmax 的全局合并**不能分两次原子**（会拿到 A 的值配 B 的下标）——把 (保序值 << 32 | ~idx) 打包成 64 位做一次 atomicMax。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define FULL 0xffffffffu
#define T 256
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 block max（shared）"

    和 sum 完全同构，只是把 `+` 换成 `fmaxf`、单位元从 0 换成 **`-INFINITY`**。这里最常见的错是**越界补 0** —— 如果数据全是负数，补 0 会让结果错成 0。归约的单位元必须是该运算的 identity。

    ```cpp
    __global__ void max_v0(const float* in, float* out, int n){
        __shared__ float s[T];
        int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
        s[tid] = (i < n) ? in[i] : -INFINITY;    // ← 不是 0！
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
            if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
            __syncthreads();
        }
        if (tid == 0) out[blockIdx.x] = s[0];
    }
    ```

=== "v1 warp shuffle max"

    标准两级 shuffle 版本。`fmaxf` 对 NaN 的行为要清楚：`fmaxf(NaN, x) = x`（IEEE 的 maxNum 语义），所以 **NaN 会被"吞掉"而不是传播**；如果业务要求检测 NaN 就不能靠 fmaxf。

    ```cpp
    __device__ __forceinline__ float warpReduceMax(float v){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(FULL, v, off));   // xor：全 lane 都拿到结果
        return v;
    }
    __device__ __forceinline__ float blockReduceMax(float v){
        __shared__ float smem[32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
        v = warpReduceMax(v);
        if (lane == 0) smem[wid] = v;
        __syncthreads();
        v = (lane < nw) ? smem[lane] : -INFINITY;
        if (wid == 0) v = warpReduceMax(v);
        return v;
    }
    __global__ void max_v1(const float* in, float* out, int n){
        float v = -INFINITY;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            v = fmaxf(v, in[i]);
        v = blockReduceMax(v);
        if (threadIdx.x == 0) out[blockIdx.x] = v;
    }
    ```

=== "v2 max + argmax（配对 shuffle）"

    要下标就得**同时搬运 value 和 index**：两次 shfl 换出邻居的 (v,i)，再按比较结果整体替换。**必须定义 tie-break 规则**（这里"值相等取小下标"），否则同一份输入在不同 block 配置下会给出不同下标，测试就不可复现。这也是 topk / beam search 里最常见的坑。

    ```cpp
    __device__ __forceinline__ void warpReduceArgMax(float& v, int& idx){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1){
            float ov = __shfl_xor_sync(FULL, v,   off);
            int   oi = __shfl_xor_sync(FULL, idx, off);
            if (ov > v || (ov == v && oi < idx)){ v = ov; idx = oi; }
        }
    }
    __device__ __forceinline__ void blockReduceArgMax(float& v, int& idx){
        __shared__ float sv[32]; __shared__ int si[32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
        warpReduceArgMax(v, idx);
        if (lane == 0){ sv[wid] = v; si[wid] = idx; }
        __syncthreads();
        v   = (lane < nw) ? sv[lane] : -INFINITY;
        idx = (lane < nw) ? si[lane] : INT_MAX;
        if (wid == 0) warpReduceArgMax(v, idx);
    }
    __global__ void argmax_v2(const float* in, float* outv, int* outi, int n){
        float v = -INFINITY; int idx = INT_MAX;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
            float t = in[i];
            if (t > v || (t == v && i < idx)){ v = t; idx = i; }
        }
        blockReduceArgMax(v, idx);
        if (threadIdx.x == 0){ outv[blockIdx.x] = v; outi[blockIdx.x] = idx; }
    }
    ```

=== "v3 float 的 atomicMax 技巧"

    CUDA **没有 `atomicMax(float*)`**（支持浮点 atomicMax 的硬件支持出现前）。经典做法：把 float 变成**保序的 uint** 再用整型 atomicMax。变换规则：正数把符号位置 1（变成大数），负数**全取反**（大的负数变小）。这样 uint 的大小顺序 == float 的大小顺序，一次 `atomicMax` 搞定。反变换同理。（另一个流派是 `atomicCAS` 自旋比较，竞争大时慢很多。）

    ```cpp
    __device__ __forceinline__ unsigned f2ord(float f){
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    __device__ __forceinline__ float ord2f(unsigned u){
        unsigned r = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
        return __uint_as_float(r);
    }
    __global__ void max_v3(const float* in, unsigned* out, int n){
        float v = -INFINITY;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            v = fmaxf(v, in[i]);
        v = blockReduceMax(v);
        if (threadIdx.x == 0) atomicMax(out, f2ord(v));     // 全局只有 gridDim 次原子操作
    }
    ```

=== "v4 打包成 64 位一次原子"

    argmax 的全局合并要同时更新 (值, 下标)，分两次原子就**不原子了**（可能拿到 A 的值配 B 的下标）。技巧：**把保序 uint 放高 32 位、下标放低 32 位**，凑成 `unsigned long long` 做一次 `atomicMax`；比较自然是"先比值、值相等再比下标"。想让小下标优先就存 `~idx`。

    ```cpp
    __global__ void argmax_v4(const float* in, unsigned long long* out, int n){
        float v = -INFINITY; int idx = INT_MAX;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
            float t = in[i];
            if (t > v || (t == v && i < idx)){ v = t; idx = i; }
        }
        blockReduceArgMax(v, idx);
        if (threadIdx.x == 0){
            // 高 32 位：保序值；低 32 位：~idx（这样值相等时【小下标】的 key 更大）
            unsigned long long key = ((unsigned long long)f2ord(v) << 32) | (unsigned)(~idx);
            atomicMax(out, key);
        }
    }
    ```

</div>
