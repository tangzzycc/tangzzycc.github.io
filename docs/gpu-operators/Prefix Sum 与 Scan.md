---
title: "Prefix Sum / Scan（Hillis-Steele → Blelloch → decoupled lookback）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Prefix Sum / Scan（Hillis-Steele → Blelloch → decoupled lookback）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P06</span><span>scan</span><span>shfl_up</span><span>work-efficient</span><span>lookback</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

计算 inclusive 前缀和 `out[i] = in[0] + in[1] + ... + in[i]`，n = 2²²。

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>in</code>: [n] float32</dd>
  <dt>输出</dt><dd><code>out</code>: [n] float32</dd>
  <dt>难点</dt><dd>scan 有<strong>串行依赖</strong>：后一个元素依赖前面所有元素。并行化必须靠&quot;分块局部 scan + 块间传播&quot;两级结构。</dd>
  <dt>目标</dt><dd>下界 = 读一遍 + 写一遍，即 2n·4 字节的带宽时间</dd>
</dl>

scan 是**并行算法的第二块基石**（第一块是 reduce）。它本身出现在面试里的频率不如 reduce， 但 **radix sort、TopK、MoE 的 token 路由分桶、变长 batch 的 offset 计算、稀疏 attention 的索引构建** 统统建立在它上面，所以必须会。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void scan_inclusive(const float* in, float* out, int n);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先只考虑 **一个 block 内**的 scan，多 block 的传播最后再说。
- Hillis-Steele：第 d 轮 `s[i] += s[i-2^d]`。注意**同一块 shared 边读边写会 RAW 冲突**，要用两块 buffer 乒乓。
- warp 级用 `__shfl_up_sync`：`t = shfl_up(v, off); if (lane >= off) v += t;` —— **那个 if 不能少**，lane < off 没有前驱。
- 生产写法是"每线程先串行 scan 自己的 4 个元素，再用块内 scan 求出前面所有线程的和，最后加回去"——通信量降到 1/4。
- 多 block：最简单是**三段式**（局部 scan → 扫 tile 和 → 加偏移），但要读写 3 遍数据；cub 用 **decoupled lookback** 单趟搞定。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define FULL 0xffffffffu
#define TB   256          // 每 block 线程数
#define EPT  4            // 每线程元素数 → 一个 tile = 1024 个元素

// warp 内 inclusive scan：shfl_up 的经典五轮
__device__ __forceinline__ float warpScanIncl(float v){
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1){
        float t = __shfl_up_sync(FULL, v, off);
        if (lane >= off) v += t;              // lane < off 的不能加（没有前驱）
    }
    return v;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 Hillis-Steele"

    最直观的一版：第 d 轮把 `s[i]` 加上 `s[i-2^d]`，**log₂n 轮就完事**。致命细节：**同一块 shared 既读又写会 RAW 冲突**，所以要么用两块 buffer 乒乓，要么"先读到寄存器 → sync → 再写回"。

    工作量 O(n log n)，比串行的 O(n) 还多，属于 **work-inefficient**；但步数只有 log n，小数组反而最快。

    ```cpp
    __global__ void scan_v0(const float* in, float* out, float* tilesum, int n){
        __shared__ float buf[2][1024];
        int tid = threadIdx.x, base = blockIdx.x * blockDim.x;
        int i = base + tid;
        int pin = 0, pout = 1;
        buf[pin][tid] = (i < n) ? in[i] : 0.0f;
        __syncthreads();
        for (int off = 1; off < blockDim.x; off <<= 1){
            if (tid >= off) buf[pout][tid] = buf[pin][tid] + buf[pin][tid - off];
            else            buf[pout][tid] = buf[pin][tid];      // 没有前驱的照抄
            __syncthreads();
            pin ^= 1; pout ^= 1;                       // 乒乓，避免读写同一块
        }
        if (i < n) out[i] = buf[pin][tid];
        if (tid == blockDim.x - 1) tilesum[blockIdx.x] = buf[pin][tid];
    }
    ```

=== "v1 Blelloch（work-efficient）"

    两趟：**up-sweep 建和树**（把部分和往上归约），**把根置 0，再 down-sweep 下推**。总工作量 O(n)，是理论最优，但要 2·log n 轮 + 两次 sync，而且**产出的是 exclusive scan**（想要 inclusive 得最后再加一次原值）。

    面试常问"哪个更好"：**GPU 上元素少时 Hillis-Steele 更快（步数少），元素多/带宽紧张时 Blelloch 更优**；实际库里用的是下面 v3 那种混合形态。

    ```cpp
    __global__ void scan_v1(const float* in, float* out, float* tilesum, int n){
        __shared__ float s[1024];
        int tid = threadIdx.x, T = blockDim.x, base = blockIdx.x * T;
        float mine = (base + tid < n) ? in[base + tid] : 0.0f;
        s[tid] = mine;
        __syncthreads();
        // up-sweep：s[2^(d+1)k + 2^(d+1) - 1] += s[2^(d+1)k + 2^d - 1]
        for (int d = 1; d < T; d <<= 1){
            int idx = (tid + 1) * (d << 1) - 1;
            if (idx < T) s[idx] += s[idx - d];
            __syncthreads();
        }
        if (tid == 0){ tilesum[blockIdx.x] = s[T - 1]; s[T - 1] = 0.0f; }  // 根清零
        __syncthreads();
        // down-sweep：左孩子拿父亲的值，父亲变成"父+左"
        for (int d = T >> 1; d >= 1; d >>= 1){
            int idx = (tid + 1) * (d << 1) - 1;
            if (idx < T){
                float t = s[idx - d];
                s[idx - d] = s[idx];
                s[idx] += t;
            }
            __syncthreads();
        }
        if (base + tid < n) out[base + tid] = s[tid] + mine;  // exclusive → inclusive
    }
    ```

=== "v2 warp shfl_up + 两级"

    和 reduce 一样，scan 也是"**warp 内 shuffle → warp 之间再 scan 一次 → 加回去**"的两级结构。`__shfl_up_sync` 取前驱 lane 的值，**lane < off 的线程不能加**（它没有前驱）——这是最容易写错的一行。

    零 bank conflict、零 shared 往返，是 warp 级 scan 的标准形态。

    ```cpp
    __device__ __forceinline__ float blockScanIncl(float v){
        __shared__ float warpSum[32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        v = warpScanIncl(v);                                 // ① warp 内 scan
        if (lane == 31) warpSum[wid] = v;                    // ② 每个 warp 的总和
        __syncthreads();
        if (wid == 0){                                       // ③ 对 32 个 warp 和再 scan 一次
            float w = (lane < (blockDim.x >> 5)) ? warpSum[lane] : 0.0f;
            w = warpScanIncl(w);
            warpSum[lane] = w;
        }
        __syncthreads();
        if (wid > 0) v += warpSum[wid - 1];        // ④ 加上前面所有 warp 的和
        return v;
    }
    __global__ void scan_v2(const float* in, float* out, float* tilesum, int n){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float v = (i < n) ? in[i] : 0.0f;
        v = blockScanIncl(v);
        if (i < n) out[i] = v;
        if (threadIdx.x == blockDim.x - 1) tilesum[blockIdx.x] = v;
    }
    ```

=== "v3 每线程 4 元素（生产形态）"

    一个线程只处理 1 个元素太浪费。标准做法：**每线程用 float4 读 4 个连续元素 → 线程内串行 scan（3 次加法，零通信）→ 用块内 scan 求出"我前面所有线程的总和" → 加回自己的 4 个值**。

    通信量降到 1/4，且访存全部向量化。这就是 cub::BlockScan 的 `BLOCK_SCAN_WARP_SCANS` 算法。

    ```cpp
    __global__ void scan_v3(const float* __restrict__ in, float* __restrict__ out,
                            float* __restrict__ tilesum, int n){
        const int tile = blockIdx.x * (blockDim.x * EPT);
        const int idx  = tile + threadIdx.x * EPT;
        float4 v = make_float4(0,0,0,0);
        if (idx + 3 < n) v = *reinterpret_cast<const float4*>(in + idx);
        else for (int k = 0; k < EPT; k++) ((float*)&v)[k] = (idx + k < n) ? in[idx + k] : 0.0f;

        // ① 线程内串行 scan
        v.y += v.x; v.z += v.y; v.w += v.z;
        // ② 块内 scan 我的总和 → 得到 inclusive，减去自己就是 exclusive 前缀
        float excl = blockScanIncl(v.w) - v.w;
        // ③ 加回去
        v.x += excl; v.y += excl; v.z += excl; v.w += excl;

        if (idx + 3 < n) *reinterpret_cast<float4*>(out + idx) = v;
        else for (int k = 0; k < EPT; k++) if (idx + k < n) out[idx + k] = ((float*)&v)[k];
        if (threadIdx.x == blockDim.x - 1) tilesum[blockIdx.x] = v.w;
    }
    ```

=== "v4 decoupled lookback（单 kernel）"

    三段式（局部 scan → 扫 tile 和 → 加偏移）要**读写三遍数据**，带宽白白翻倍。工业界（cub）的做法是 **decoupled lookback**：单 kernel 一趟搞定。

    每个 tile 算完局部和后立刻发布 **A（aggregate，只是我自己的和）**状态，然后**回头看前一个 tile**：看到 P（inclusive prefix，含全部前驱）就停下、看到 A 就累加继续往前看。拿到自己的前缀后发布 P，后继就能立刻停在我这。

    关键前提：**tile 号必须用 `atomicAdd` 动态领取**，不能用 blockIdx——否则可能有 block 在等一个还没被调度的前驱，直接死锁。状态和值打包成 64 位一次原子读写，省掉 `__threadfence` 的正确性推理。

    ```cpp
    __device__ __forceinline__ unsigned long long pack(int flag, float v){
        return ((unsigned long long)flag << 32) | (unsigned long long)__float_as_uint(v);
    }
    __global__ void scan_v4(const float* __restrict__ in, float* __restrict__ out,
                            unsigned long long* state, int* counter, int n){
        __shared__ int s_tile;
        __shared__ float s_excl;
        if (threadIdx.x == 0) s_tile = atomicAdd(counter, 1);   // 动态领号，保证前驱一定已启动
        __syncthreads();
        const int t = s_tile;

        const int base = t * (blockDim.x * EPT);
        const int idx  = base + threadIdx.x * EPT;
        float4 v = make_float4(0,0,0,0);
        for (int k = 0; k < EPT; k++) ((float*)&v)[k] = (idx + k < n) ? in[idx + k] : 0.0f;
        v.y += v.x; v.z += v.y; v.w += v.z;
        float incl = blockScanIncl(v.w);
        float excl_in_block = incl - v.w;

        if (threadIdx.x == blockDim.x - 1){
            float agg = incl;                                  // 本 tile 总和
            if (t == 0){
                atomicExch(&state[0], pack(2, agg));            // 0 号 tile 的前缀就是自己
                s_excl = 0.0f;
            } else {
                atomicExch(&state[t], pack(1, agg));  // 先发布 aggregate，别人可以开始用
                float exclusive = 0.0f;
                int look = t - 1;
                while (look >= 0){
                    unsigned long long st;
                    do { st = atomicAdd(&state[look], 0ULL); } while ((int)(st >> 32) == 0);  // 自旋等它有效
                    float val = __uint_as_float((unsigned)(st & 0xffffffffu));
                    exclusive += val;
                    if ((int)(st >> 32) == 2) break;  // 拿到 inclusive prefix，可以收工
                    look--;                                     // 只是 aggregate，继续往前看
                }
                atomicExch(&state[t], pack(2, exclusive + agg)); // 发布自己的 inclusive prefix
                s_excl = exclusive;
            }
        }
        __syncthreads();
        float off = s_excl + excl_in_block;
        v.x += off; v.y += off; v.z += off; v.w += off;
        for (int k = 0; k < EPT; k++) if (idx + k < n) out[idx + k] = ((float*)&v)[k];
    }
    ```

</div>
