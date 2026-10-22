---
title: "warpReduce / blockReduce 模板工具箱"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">warpReduce / blockReduce 模板工具箱</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P04</span><span>模板</span><span>shfl_xor</span><span>cooperative groups</span><span>必背</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

**这不是一道"题"，是你必须随手就能默写出来的一套模板。** 后面 softmax、LayerNorm、RMSNorm、attention、TopK、量化统计——**全都建立在这几段代码上**。 建议单独抽 10 分钟，把 v0~v3 蒙着写三遍，直到不用想就能写对。

<dl class="gpu-operator__io">
  <dt>要背的</dt><dd>① <code>warpReduceSum</code>（shfl_down 版）② 蝶形 <code>shfl_xor</code> 版
③ <code>blockReduce&lt;Op&gt;</code> 通用模板 ④ <code>blockAllReduce</code>（全 block 可见）
⑤ <code>blockReduceSum2</code>（一次归约两个量）</dd>
  <dt>要能讲的</dt><dd>shfl_down vs shfl_xor 的选择、单位元、smem 复用的 sync 陷阱、
warp0-only 有效 vs 全 block 可见的区别</dd>
</dl>

**最容易被追问的三个点**：(1) 为什么归约结果只有 thread 0 有效？(2) 同一 kernel 里连着调两次 blockReduce 会怎样？ (3) 为什么 LayerNorm 要把 sum 和 sum² 塞进一次归约？

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
__device__ float warpReduceSum(float v);              // lane 0 有效
__device__ float warpAllReduceSum(float v);           // 全 lane 有效
template<class Op>
__device__ float blockReduce(float v, Op op, float ident);   // warp 0 有效
__device__ float blockAllReduceSum(float v);          // 全 block 有效
__device__ float2 blockReduceSum2(float2 v);          // 一次归约两个量
```

<p class="gpu-operator__label" id="tips">Tips</p>

- warp 内归约的骨架永远是 `for (off = 16; off > 0; off >>= 1)` 五轮。
- block 内归约的骨架永远是两级：**warp 内 shuffle → 每 warp 的 lane0 写 smem[32] → syncthreads → 第一个 warp 再归约一次**。
- 装载 `smem[lane]` 时，超出实际 warp 数的 lane 必须填**单位元**（sum→0，max→-INFINITY，min→+INFINITY，prod→1）。
- 要让**整个 block**都拿到结果，必须再过一次 shared 广播——shfl 只在 warp 内有效，跨 warp 传不了。
- 一次归约多个量：把它们打包成 `float2`，两条 shfl 会被编译器打包调度，几乎不增加开销。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define FULL 0xffffffffu                          // 32 个 lane 全参与的掩码
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 warpReduceSum（shfl_down）"

    最常写的一段。`__shfl_down_sync(mask, v, off)` = 我拿 **lane+off** 那个线程的 v。5 轮 16/8/4/2/1 之后 **lane 0 持有全 warp 的和**（其他 lane 的值是垃圾，别用）。mask 必须包含所有参与线程，Volta 之后**不写 `_sync` 后缀的老 API 已废弃**。

    ```cpp
    __device__ __forceinline__ float warpReduceSum(float v){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(FULL, v, off);
        return v;                                     // 只有 lane 0 有效
    }
    ```

=== "v1 蝶形 shfl_xor（全 lane 拿结果）"

    `__shfl_xor_sync` 让 lane i 和 lane i^off 互换，是**对称**的蝶形网络：5 轮之后 **32 个 lane 都持有完整结果**，省掉一次 `__shfl_sync(v,0)` 广播。什么时候用它？**归约结果后面每个线程都要用**时（softmax 要减 max、layernorm 要减均值、RMSNorm 要乘 rsqrt）。纯粹取一个标量写回则 `shfl_down` 更省。

    ```cpp
    __device__ __forceinline__ float warpAllReduceSum(float v){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(FULL, v, off);
        return v;                                     // 全部 32 个 lane 都有效
    }
    ```

=== "v2 blockReduce 通用模板"

    把 op 抽成 functor，一份代码复用给 sum/max/min。**两级结构是固定套路**：warp 内 shuffle → 每 warp 的 lane0 写进 `smem[32]` → `__syncthreads()` → 第一个 warp 再归约一次。两个坑：**(1) 装载 smem 时越界的 lane 要填 op 的单位元**（sum 填 0、max 填 -INF）；**(2) 如果同一 kernel 里连续调用两次，中间必须再 `__syncthreads()`**，否则第二次的写会覆盖第一次还没读完的 smem。

    ```cpp
    struct SumOp{ __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };
    struct MaxOp{ __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); } };

    template<class Op>
    __device__ __forceinline__ float warpReduce(float v, Op op){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = op(v, __shfl_xor_sync(FULL, v, off));
        return v;
    }
    template<class Op>
    __device__ __forceinline__ float blockReduce(float v, Op op, float ident){
        __shared__ float smem[32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (blockDim.x + 31) >> 5;
        v = warpReduce(v, op);
        if (lane == 0) smem[wid] = v;
        __syncthreads();
        v = (lane < nwarp) ? smem[lane] : ident;   // 单位元：sum→0，max→-INF，min→+INF
        if (wid == 0) v = warpReduce(v, op);
        return v;                                            // ⚠ 只有 warp 0 的线程有效
    }
    // 如果【整个 block 的每个线程】都要用这个结果（softmax 减 max、layernorm 减均值），
    // 必须再过一次 shared 广播，不能靠 shfl（shfl 只在 warp 内部有效）：
    __device__ __forceinline__ float blockAllReduceSum(float v){
        __shared__ float smem[32];
        __shared__ float result;
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (blockDim.x + 31) >> 5;
        v = warpAllReduceSum(v);
        if (lane == 0) smem[wid] = v;
        __syncthreads();
        if (wid == 0){
            float t = (lane < nwarp) ? smem[lane] : 0.0f;
            t = warpAllReduceSum(t);
            if (lane == 0) result = t;
        }
        __syncthreads();
        return result;                                 // 全 block 可见
    }
    ```

=== "v3 一次归约多个量"

    LayerNorm 要同时要 `sum` 和 `sum²`，attention 要同时要 `max` 和 `sum`。**千万别调两次 blockReduce**（两倍 shuffle + 两次 syncthreads）：把多个量塞进一次 shuffle，指令数几乎不变。`float2` 打包最省事；Welford 那种非交换的更新也能这么写。

    ```cpp
    __device__ __forceinline__ float2 warpReduceSum2(float2 v){
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1){
            v.x += __shfl_xor_sync(FULL, v.x, off);
            v.y += __shfl_xor_sync(FULL, v.y, off);   // 编译器会把两条 shfl 打包调度
        }
        return v;
    }
    __device__ __forceinline__ float2 blockReduceSum2(float2 v){
        __shared__ float2 smem[32];
        const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        const int nwarp = (blockDim.x + 31) >> 5;
        v = warpReduceSum2(v);
        if (lane == 0) smem[wid] = v;
        __syncthreads();
        v = (lane < nwarp) ? smem[lane] : make_float2(0.0f, 0.0f);
        if (wid == 0) v = warpReduceSum2(v);
        return v;                                      // warp0 全 lane 有效
    }
    ```

=== "v4 硬件指令与库"

    面试可以顺口提一句"其实有现成的"：

    • `__reduce_add_sync`（sm\_80+，**仅整型**）一条指令做完 warp 归约，int8 量化统计里很有用；

    • `cooperative_groups::reduce` 支持 float，编译器会自动选最优实现，还能改 tile 大小（`tiled_partition<16>`）；

    • `cub::BlockReduce` / `cub::WarpReduce` 是 CUDA 自带的高度调优实现，实际项目直接用。手写只是为了面试和极端定制场景。

    ```cpp
    __device__ __forceinline__ int warpReduceSumInt(int v){
    #if __CUDA_ARCH__ >= 800
        return __reduce_add_sync(FULL, v);   // 支持该指令的 CUDA 架构 硬件归约指令
    #else
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
        return v;
    #endif
    }
    __device__ __forceinline__ float cgWarpSum(float v){
        auto tile = cg::tiled_partition<32>(cg::this_thread_block());
        return cg::reduce(tile, v, cg::plus<float>());  // 全 lane 都拿到结果
    }
    ```

</div>
