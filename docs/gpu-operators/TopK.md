---
title: "TopK（迭代 / 局部堆 / warp bitonic / radix select）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">TopK（迭代 / 局部堆 / warp bitonic / radix select）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P16</span><span>topk</span><span>radix select</span><span>bitonic</span><span>寄存器 spill</span><span>LLM 采样</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

对 `x[ROWS][n]` 的每一行，求最大的 k 个元素**及其下标**。测试 ROWS=2048, n=8192, k=32。

<dl class="gpu-operator__io">
  <dt>应用</dt><dd>LLM 的 top-k / top-p 采样（vocab 32k~128k）、MoE 的 router 选 top-2 expert、
beam search、向量检索的候选打分</dd>
  <dt>四种解法</dt><dd>各有各的适用区间，<strong>面试真正考的是&quot;你能不能说出什么时候用哪个&quot;</strong></dd>
  <dt>陷阱</dt><dd>并列怎么办、下标要不要稳定、k 很大时前三种都会崩</dd>
</dl>

这题的技术点密度很高：**局部堆、warp shuffle 归并、bitonic 网络、radix select、以及"寄存器数组不能用动态下标"这个致命细节**。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void topk(const float* x,   // [rows, n]
          float* out_val,   // [rows, k] 降序
          int*   out_idx,   // [rows, k]
          int rows, int n, int k);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先说清楚 k 的量级：**k ≤ 8 / k ≤ 32 / k 很大，是三种完全不同的解法**。
- 最朴素：求 max → 记下 → 置 -inf → 重复 k 次。O(k·n)，k 小时够用。
- 正统：每线程扫自己那份，在寄存器里维护有序的 top-K，最后两两归并（归并两个有序 K 数组是 O(K)）。
- ⚠ **寄存器数组一旦被动态下标访问就会 spill 到 local memory**（其实是 global），插入要写成"全展开 + 静态下标"的移位形式。
- k ≤ 32 时可以让 warp 的 32 个 lane 各持一个候选，用 `shfl_xor` 的 bitonic 网络排序，零 shared。
- k 很大（几百上千）时只能用 **radix select**：按 8 位做直方图找出第 k 大的**阈值**，4 轮定位，最后扫一遍收集。**复杂度与 k 无关。**

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define FULL 0xffffffffu
// 每行 n 个元素，求每行最大的 k 个（不要求有序，但本题按降序输出方便对拍）
// 场景：LLM 采样的 top-k / top-p、MoE 的 router 选 expert、beam search、检索的 ANN 打分

// float → 保序 uint（见 P05）：正数置符号位，负数全取反
__device__ __forceinline__ unsigned f2ord(float f){
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned u){
    unsigned r = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    return __uint_as_float(r);
}
__device__ __forceinline__ float warpAllMax(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
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
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 迭代 k 次求最大"

    最容易想到：**求最大值 → 记录 → 把它置成 -inf → 再求**，重复 k 次。

    正确、简单、k 很小（≤8）时完全够用。但复杂度是 **O(k·n)**：k=64 就要把整行扫 64 遍，**访存量放大 64 倍**。

    面试可以用它开场，但一定要主动说"这是 O(k·n)，k 大就不行了"。

    ```cpp
    __global__ void topk_v0(const float* __restrict__ x, float* __restrict__ outv,
                            int* __restrict__ outi, int n, int k){
        extern __shared__ float s[];
        const float* row = x + (size_t)blockIdx.x * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = row[i];
        __syncthreads();
        for (int t = 0; t < k; t++){
            float m = -INFINITY;
            for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, s[i]);
            m = blockAllMax(m);                              // 全行最大值（所有线程可见）
            // 找到它的下标（取最小的那个），并置为 -inf
            __shared__ int pos;
            if (threadIdx.x == 0) pos = INT_MAX;
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                if (s[i] == m) atomicMin(&pos, i);
            __syncthreads();
            if (threadIdx.x == 0){
                outv[(size_t)blockIdx.x * k + t] = m;
                outi[(size_t)blockIdx.x * k + t] = pos;
                s[pos] = -INFINITY;                // 划掉，下一轮找次大
            }
            __syncthreads();
        }
    }
    ```

=== "v1 每线程局部 top-k + 归并"

    正统解法：**每个线程先扫自己那份数据，在寄存器里维护一个大小 K 的有序数组**（新元素用插入排序放进去，比最小的还小就直接扔）。

    一趟扫完，全 block 有 `blockDim × K` 个候选；再**两两归并**（归并两个有序的 K 数组只要 O(K)），log 步之后就是答案。

    复杂度 **O(n + p·K·log p)**，数据只扫一遍。这是 k 中等（几十）时的最优解。

    ```cpp
    template<int K>
    __global__ void topk_v1(const float* __restrict__ x, float* __restrict__ outv,
                            int* __restrict__ outi, int n, int k){
        const float* row = x + (size_t)blockIdx.x * n;
        float lv[K]; int li[K];
        #pragma unroll
        for (int t = 0; t < K; t++){ lv[t] = -INFINITY; li[t] = -1; }

        for (int i = threadIdx.x; i < n; i += blockDim.x){    // ① 线程内维护有序 top-K
            float v = row[i];
            if (v <= lv[K - 1]) continue;           // 比最小的还小 → 直接扔（绝大多数走这里）
            // ⚠ 千万别写成 while 循环 + 动态下标 lv[p]！
            //   寄存器数组一旦被动态下标访问，编译器只能把它放到 local memory（其实是 global），
            //   要写成【全展开 + 静态下标】的移位插入：
            int cur_i = i;
            #pragma unroll
            for (int t = 0; t < K; t++){
                bool gt = v > lv[t];
                float tv = gt ? v : lv[t];  int ti = gt ? cur_i : li[t];
                v = gt ? lv[t] : v;  cur_i = gt ? li[t] : cur_i;   // 被挤出来的往后传
                lv[t] = tv; li[t] = ti;
            }
        }
        // ② 用 shared 做两两归并（每轮活跃线程减半）
        extern __shared__ char raw[];
        float* sv = reinterpret_cast<float*>(raw);
        int*   si = reinterpret_cast<int*>(sv + (size_t)blockDim.x * K);
        #pragma unroll
        for (int t = 0; t < K; t++){ sv[threadIdx.x * K + t] = lv[t]; si[threadIdx.x * K + t] = li[t]; }
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
            if (threadIdx.x < stride){
                float* a = sv + threadIdx.x * K;          int* ai = si + threadIdx.x * K;
                float* b = sv + (threadIdx.x + stride) * K; int* bi = si + (threadIdx.x + stride) * K;
                float mv[K]; int mi[K];
                int p = 0, q = 0;
                #pragma unroll
                for (int t = 0; t < K; t++){               // 归并两个有序数组，只取前 K
                    if (a[p] >= b[q]){ mv[t] = a[p]; mi[t] = ai[p]; p++; }
                    else                        { mv[t] = b[q]; mi[t] = bi[q]; q++; }
                }
                #pragma unroll
                for (int t = 0; t < K; t++){ a[t] = mv[t]; ai[t] = mi[t]; }
            }
            __syncthreads();
        }
        for (int t = threadIdx.x; t < k; t += blockDim.x){
            outv[(size_t)blockIdx.x * k + t] = sv[t];
            outi[(size_t)blockIdx.x * k + t] = si[t];
        }
    }
    ```

=== "v2 warp 内 shuffle 归并（k ≤ 32）"

    当 k 不超过 32 时，可以让**一个 warp 的 32 个 lane 各持有一个候选**，归并完全靠 `shfl`，**零 shared、零 syncthreads**。

    做法：每个 lane 先扫自己那份数据保留一个最大值 → 用 **bitonic 网络**（`shfl_xor` 的经典应用）把 32 个 lane 的值排序 → 一个 warp 就得到 top-32。

    多个 warp 的结果再归并一次即可。**这是 k 小、行多的场景（MoE router 选 top-2 expert）的最优形态。**

    ```cpp
    __device__ __forceinline__ void bitonicWarp(float& v, int& idx){
        // 32 个 lane 的 bitonic sort（降序）：shfl_xor 的经典应用，log²32 = 15 步
        const int lane = threadIdx.x & 31;
        #pragma unroll
        for (int size = 2; size <= 32; size <<= 1){
            #pragma unroll
            for (int stride = size >> 1; stride > 0; stride >>= 1){
                float ov = __shfl_xor_sync(FULL, v, stride);
                int   oi = __shfl_xor_sync(FULL, idx, stride);
                // 每个 size 段的排序方向要交替，最后一段（size=32）才是全局降序
                const bool desc   = ((lane & size) == 0);
                const bool higher = (lane & stride) != 0;      // 我是这一对里下标大的那个吗
                const bool wantLarger = desc ? !higher : higher;
                const bool takeOther  = wantLarger ? (ov > v) : (ov < v);
                if (takeOther){ v = ov; idx = oi; }
            }
        }
    }
    __global__ void topk_v2_warp(const float* __restrict__ x, float* __restrict__ outv,
                                 int* __restrict__ outi, int n, int k){
        const float* row = x + (size_t)blockIdx.x * n;
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
        float best = -INFINITY; int bi = -1;                  // 每个 lane 持有 top-32 中的一个（降序）

        for (int base = wid * 32; base < n; base += nw * 32){
            float v = (base + lane < n) ? row[base + lane] : -INFINITY;
            int idx = base + lane;
            // ⚠ 剪枝：整块都比当前第 32 名还小就跳过（实际数据里绝大多数块都会被跳掉）
            float chunkMax = v;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) chunkMax = fmaxf(chunkMax, __shfl_xor_sync(FULL, chunkMax, o));
            if (chunkMax <= __shfl_sync(FULL, best, 31)) continue;   // lane31 持有当前最小值

            bitonicWarp(v, idx);                              // ① 新块排成降序
            // ② 合并两个有序 32 元组、只留 top-32：把其中一个反转后逐元素取大者 → 得到 bitonic 序列
            float rv = __shfl_sync(FULL, v,   31 - lane);
            int   ri = __shfl_sync(FULL, idx, 31 - lane);
            if (rv > best){ best = rv; bi = ri; }
            bitonicWarp(best, bi);                            // ③ 再排一次
        }
        // 跨 warp 合并：每个 warp 的有序 32 元组写进 shared，由 warp0 依次并进来
        __shared__ float sv[32 * 32]; __shared__ int si[32 * 32];
        sv[wid * 32 + lane] = best; si[wid * 32 + lane] = bi;
        __syncthreads();
        if (wid == 0){
            for (int w = 1; w < nw; w++){
                float ov = sv[w * 32 + (31 - lane)];          // 反转读
                int   oi = si[w * 32 + (31 - lane)];
                if (ov > best){ best = ov; bi = oi; }
                bitonicWarp(best, bi);
            }
            if (lane < k){ outv[(size_t)blockIdx.x * k + lane] = best; outi[(size_t)blockIdx.x * k + lane] = bi; }
        }
    }
    ```

=== "v3 radix select（k 很大时唯一可行）"

    前面几版的代价都随 k 增长。**radix select 的复杂度与 k 无关**：

    把 float 变成保序 uint，**从最高 8 位开始做直方图**（256 个桶）→ 从大到小累加，找出"第 k 大"落在哪个桶 → **把前缀固定住，再对下一个 8 位做直方图**，4 轮之后就精确得到**第 k 大的值（阈值）**。

    然后再扫一遍收集所有 ≥ 阈值的元素。**总共 5 遍扫描，与 k 无关**。

    这是 **vocab=128k 做 top-k 采样**的标准做法（也是 cub 的 `RadixSelect`、PyTorch `torch.topk` 大 k 时的实现）。

    ```cpp
    __global__ void topk_v3_radix(const float* __restrict__ x, float* __restrict__ outv,
                                  int* __restrict__ outi, int n, int k){
        const float* row = x + (size_t)blockIdx.x * n;
        __shared__ int hist[256];
        __shared__ unsigned prefix;          // 已确定的高位
        __shared__ int remaining;            // 还要在当前桶里找几个
        __shared__ int cnt;
        if (threadIdx.x == 0){ prefix = 0; remaining = k; }
        __syncthreads();

        for (int pass = 0; pass < 4; pass++){
            const int shift = 24 - 8 * pass;
            const unsigned hiMask = (shift + 8 >= 32) ? 0u : (0xffffffffu << (shift + 8));
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += blockDim.x){
                unsigned u = f2ord(row[i]);
                if ((u & hiMask) == (prefix & hiMask))       // 高位和已定前缀一致的才统计
                    atomicAdd(&hist[(u >> shift) & 0xff], 1);
            }
            __syncthreads();
            if (threadIdx.x == 0){                            // 从大到小累加，找第 k 大所在的桶
                int c = 0, b = 255;
                for (; b >= 0; b--){ if (c + hist[b] >= remaining) break; c += hist[b]; }
                remaining -= c;                      // 在这个桶里还要挑 remaining 个
                prefix |= ((unsigned)b << shift);
            }
            __syncthreads();
        }
        // prefix 就是第 k 大元素的保序 uint 值 → 阈值
        const float thr = ord2f(prefix);
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x){     // 先收严格大于阈值的
            if (row[i] > thr){
                int p = atomicAdd(&cnt, 1);
                if (p < k){ outv[(size_t)blockIdx.x * k + p] = row[i]; outi[(size_t)blockIdx.x * k + p] = i; }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x){     // 再用等于阈值的补满（处理并列）
            if (row[i] == thr){
                int p = atomicAdd(&cnt, 1);
                if (p < k){ outv[(size_t)blockIdx.x * k + p] = row[i]; outi[(size_t)blockIdx.x * k + p] = i; }
            }
        }
    }
    ```

</div>
