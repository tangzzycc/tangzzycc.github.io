---
title: "RoPE 旋转位置编码（两种配对 + GQA + KV cache 偏移）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">RoPE 旋转位置编码（两种配对 + GQA + KV cache 偏移）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P10</span><span>rope</span><span>position encoding</span><span>GQA</span><span>layout</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

对 Q 和 K 施加旋转位置编码。把 head\_dim 维度**两两配对**，每对做一次二维旋转：

`[xₐ, x_b] → [xₐ·cosθ − x_b·sinθ, x_b·cosθ + xₐ·sinθ]`， θ = pos · base<sup>−2i/D</sup>，base = 10000

<dl class="gpu-operator__io">
  <dt>输入</dt><dd><code>q</code>: [B, S, H_q, D]，<code>k</code>: [B, S, H_k, D]（GQA：H_q=32, H_k=8, D=128）</dd>
  <dt>输出</dt><dd>就地覆盖（in-place）</dd>
  <dt>考点</dt><dd>① 两种配对约定 ② cos/sin 要不要预计算 ③ <strong>V 要不要转</strong> ④ decode 时 pos 从哪来</dd>
</dl>

RoPE 本身是 memory-bound 的小算子，但它**特别能区分"背过题"和"写过推理"**： 配对约定、GQA 的 head 数不同、position 来自调度器而不是 `t % S`——这三点答对就赢了。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void rope(float* q, float* k,            // in-place
          const float* cos_cache,        // [max_seq, D/2]
          const float* sin_cache,        // [max_seq, D/2]
          const int* positions,          // [num_tokens] 每个 token 的绝对位置
          int num_tokens, int H_q, int H_k, int D);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先写最朴素的：一个线程一对，现场 `powf` + `sincosf` 算 θ。然后**立刻指出这里应该查表**。
- θ 只和 (pos, i) 有关，**与 batch 和 head 无关** → 预计算一张 [max\_seq, D/2] 的 cos/sin 表。
- 两种配对：**rotate\_half**（HF/GPT-NeoX，配 i 和 i+D/2）和 **interleaved**（LLaMA 原版，配 2i 和 2i+1）。等价，差一个置换。
- 让 `threadIdx.x` 直接等于"第几对"、`threadIdx.y` 等于 head 偏移，**消掉所有整数除法和取模**。
- block 别开太小：`blockDim=(64,1)` 会产生几十万个小 block，凑到 256 线程。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
// 布局：x[B][S][H][D]（token-major，融合 kernel 里最常见的排布）
// 一个 token 的一个 head 内，把 D 维两两配对做二维旋转：
//   [x_a, x_b] → [x_a·cosθ - x_b·sinθ,  x_b·cosθ + x_a·sinθ]
// θ = pos · base^(-2i/D)，i 是"第几对"
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 朴素（现算 sin/cos）"

    直接按公式来，每个线程负责一对。**问题在 `powf` 和 `sincosf`**：这两个是几十个周期的超越函数，而 RoPE 本身是纯 memory-bound 算子（读一个数写一个数），**算 θ 的开销直接变成瓶颈**。

    更要命的是：同一个 pos 的 θ 被 **所有 head 重复算了 H 遍**，整个 batch 又重复了 B 遍。

    ```cpp
    __global__ void rope_v0(float* __restrict__ x, int B, int S, int H, int D, float base){
        int t = blockIdx.x;                       // 第几个 token（跨 batch 展平）
        int pos = t % S;                          // 这个 token 在序列里的位置
        int half = D / 2;
        for (int k = threadIdx.x; k < H * half; k += blockDim.x){
            int h = k / half, i = k % half;       // 第 h 个 head 的第 i 对
            float theta = pos * powf(base, -2.0f * i / D);
            float c, s; __sincosf(theta, &s, &c);
            float* p = x + ((size_t)t * H + h) * D;
            float a = p[i], b = p[i + half];      // rotate_half 配对：(i, i+D/2)
            p[i]        = a * c - b * s;
            p[i + half] = b * c + a * s;
        }
    }
    ```

=== "v1 预计算 cos/sin cache"

    θ 只依赖 **(pos, i)** 两个量，与 batch、head 无关 → **提前算好一张 [S, D/2] 的表**（LLaMA 7B: 4096×64×2×4B = 2 MB，常驻 L2），kernel 里只查表。

    超越函数全部消失，kernel 退化成纯访存。**所有推理框架都是这么做的**：cos/sin cache 在模型初始化时建一次，长上下文扩展（NTK / YaRN / 动态缩放）也只是换一张表，kernel 完全不用改。

    ```cpp
    __global__ void rope_v1(float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, int B, int S, int H, int D){
        int t = blockIdx.x, pos = t % S, half = D / 2;
        for (int k = threadIdx.x; k < H * half; k += blockDim.x){
            int h = k / half, i = k % half;
            float c = cosT[(size_t)pos * half + i];   // 查表，与 head 无关
            float s = sinT[(size_t)pos * half + i];
            float* p = x + ((size_t)t * H + h) * D;
            float a = p[i], b = p[i + half];
            p[i]        = a * c - b * s;
            p[i + half] = b * c + a * s;
        }
    }
    ```

=== "v2 interleaved 配对（LLaMA 原版）"

    上面用的是 **rotate\_half**（HF / GPT-NeoX 风格）：把第 i 维和第 i+D/2 维配对。

    而 **原始论文和 LLaMA 官方实现是 interleaved**：把**相邻**的 (2i, 2i+1) 配成一对。

    两者**数学上等价**，差别只是 head\_dim 内部的一个置换——所以权重转换时把 W\_q / W\_k 的输出维度按同样规则重排一次，就能互换。**面试问"这两种有什么区别"，答"等价，差一个 permutation，权重转换时吸收掉"就满分。**

    实现上 interleaved 更友好：一对元素**物理相邻**，可以用 `float2` 一次读写。

    ```cpp
    __global__ void rope_v2_interleaved(float* __restrict__ x, const float* __restrict__ cosT,
                                        const float* __restrict__ sinT, int B, int S, int H, int D){
        int t = blockIdx.x, pos = t % S, half = D / 2;
        for (int k = threadIdx.x; k < H * half; k += blockDim.x){
            int h = k / half, i = k % half;
            float c = cosT[(size_t)pos * half + i], s = sinT[(size_t)pos * half + i];
            float2* p = reinterpret_cast<float2*>(x + ((size_t)t * H + h) * D);
            float2 v = p[i];                          // (x[2i], x[2i+1]) 一次读回
            p[i] = make_float2(v.x * c - v.y * s, v.y * c + v.x * s);
        }
    }
    ```

=== "v3 二维 block（消除除法）"

    前面几版用 `k / half`、`k % half` 反解 head 和对号，**整数除法要几十条指令**。

    这一版把索引**交给 launch 配置**：`blockDim=(D/2, 4)`、`grid=(num_tokens, H/4)`，于是 `threadIdx.x` 直接就是 i、`threadIdx.y` 直接就是 head 偏移，**零除法、零取模**，一个 warp 访问连续的 32 个 float → 完全合并。

    ⚠ block 别开太小：最早写成 `blockDim=(64,1)`、`grid=(NT,H)`（26 万个 64 线程的小 block），**改成 256 线程/block 立刻正常。**

    ```cpp
    __global__ void rope_v3(float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, int S, int H, int D){
        const int t = blockIdx.x;                                    // 第几个 token
        const int h = blockIdx.y * blockDim.y + threadIdx.y;         // 第几个 head
        const int i = threadIdx.x;                                   // 第几对（直接就是 i）
        const int half = D / 2, pos = t % S;
        if (h >= H) return;
        float c = cosT[(size_t)pos * half + i], s_ = sinT[(size_t)pos * half + i];
        float* p = x + ((size_t)t * H + h) * D;
        float a = p[i], b = p[i + half];
        p[i]        = a * c - b * s_;
        p[i + half] = b * c + a * s_;
    }
    // 启动：blockDim = (D/2, 4)，grid = (num_tokens, H/4) —— 一个 block 256 线程，处理 4 个 head
    ```

=== "v4 融合 Q/K + GQA + KV cache 偏移（生产形态）"

    真实推理里要处理三件事：

    ① **Q 和 K 一起转**（V 不转！），一个 kernel 搞定，省一次 launch 和一遍访存；

    ② **GQA/MQA**：K 的 head 数比 Q 少（LLaMA-70B 是 64:8），两者 head 数不同要分别循环；

    ③ **位置来自 `positions[]` 数组而不是 `t % S`**——decode 阶段每个请求的 KV cache 长度不同、prefix caching / chunked prefill 时起始位置也不同，**位置必须由调度器传进来**。

    这三点是"写过推理框架"和"只做过题"的分水岭。

    ```cpp
    __global__ void rope_v4_fused(float* __restrict__ q, float* __restrict__ k,
                                  const float* __restrict__ cosT, const float* __restrict__ sinT,
                                  const int* __restrict__ positions,   // [num_tokens]，由调度器给
                                  int HQ, int HK, int D){
        const int t = blockIdx.x, i = threadIdx.x, half = D / 2;
        const int h = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = positions[t];                 // 不是 t % S！
        const float c = cosT[(size_t)pos * half + i], s = sinT[(size_t)pos * half + i];
        if (h < HQ){                                            // Q 的 head
            float* p = q + ((size_t)t * HQ + h) * D;
            float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;  p[i + half] = b * c + a * s;
        }
        if (h < HK){                                            // K 的 head（GQA 时少得多）
            float* p = k + ((size_t)t * HK + h) * D;
            float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;  p[i + half] = b * c + a * s;
        }
        // V 不做 RoPE！这是最容易答错的点
    }
    ```

</div>
