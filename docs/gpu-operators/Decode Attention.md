---
title: "Decode Attention（KV cache / flash-decoding / paged / GQA 共享读）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">Decode Attention（KV cache / flash-decoding / paged / GQA 共享读）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P13</span><span>decode</span><span>KV cache</span><span>flash-decoding</span><span>paged attention</span><span>vLLM</span><span>推理核心</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

LLM 自回归解码时的 attention —— **Q 只有 1 个 token**，K/V 是完整的历史 cache。

<dl class="gpu-operator__io">
  <dt>输入</dt><dd>Q: [B, H_q, <strong>1</strong>, D]，KV cache: [B, H_kv, S_kv, D]；测试 B=4, H_q=32, H_kv=8, D=128, S_kv=4096</dd>
  <dt>输出</dt><dd>O: [B, H_q, 1, D]</dd>
  <dt>性质变了</dt><dd>不再是 GEMM 而是 <strong>GEMV</strong>：算术强度 ≈ 2，<strong>100% memory-bound</strong>。
唯一目标是把 KV cache 的读取带宽打满，并且别让 GPU 空转。</dd>
</dl>

**这是推理优化岗最核心的一道题。**训练看 prefill（compute-bound，拼 Tensor Core）， **推理看 decode（memory-bound，拼带宽和并行度）**。 flash-decoding、paged attention、GQA 共享读——这三个词能讲明白，这道题就赢了。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void decode_attention(const float* q,           // [B, H_q, D]
                      const float* k_cache,     // paged: [num_blocks, H_kv, PAGE, D]
                      const float* v_cache,
                      const int* block_table,   // [B, max_blocks] 逻辑块 → 物理块
                      float* out,               // [B, H_q, D]
                      int B, int H_q, int H_kv, int D,
                      int seq_len, float scale);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先写最直白的：`grid = B × H_q`，一个 block 扫完自己那条 KV。**然后算一下 block 数够不够填满 GPU**。
- batch=1 时只有 32 个 block，而 GPU 有一百多个 SM —— **这就是要沿 KV 长度再切一刀的理由**。
- flash-decoding：每个 split 输出 **未归一化**的 `o_i` 和它的 `(m_i, l_i)`，第二个 kernel 用 online softmax 公式合并。
- **阶段一千万别除以 l**——除完就没法合并了。
- GQA：一个 block 服务同一组 KV 的多个 query head，**内层把 k[d] 读一次同时喂给 G 个点积**，KV 带宽需求除以 G。
- paged cache：KV 不连续，`物理块 = block_table[seq][j / PAGE]`，块内偏移 `j % PAGE`。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#define FULL 0xffffffffu
// decode 阶段：Q 只有 1 个 token —— [B, HQ, 1, D]
//              KV cache 有全部历史 —— [B, HKV, S_kv, D]
// 这不再是 GEMM，而是 **GEMV**：算术强度 ≈ 2，彻头彻尾 memory-bound。
// 唯一的目标：把 KV cache 的读取带宽打满，并且让 GPU 别空转。

__device__ __forceinline__ float warpAllMax(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float warpAllSum(float v){
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
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
__device__ __forceinline__ float blockAllSum(float v){
    __shared__ float sm[32]; __shared__ float r;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warpAllSum(v);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0){ float t = (lane < nw) ? sm[lane] : 0.f; t = warpAllSum(t); if (lane == 0) r = t; }
    __syncthreads();
    return r;
}
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 一个 block 一个 (batch, head)"

    最直接的映射：`grid = B × HQ`，每个 block 扫完自己那条 KV cache。三段：算 score 存 shared → softmax → 乘 V。

    **致命问题：并行度只有 B×HQ。**batch=1 时只有 32 个 block，而现代 GPU 的 SM 数通常远多于此，**会有大量计算资源空转**。

    decode 阶段 batch 通常很小（低延迟场景就是 1），所以这版在真实服务里几乎不可用。**这正是 flash-decoding 要解决的问题：并行度不够就沿 KV 长度再切一刀。**

    ```cpp
    __global__ void decode_v0(const float* __restrict__ Q, const float* __restrict__ Kc,
                              const float* __restrict__ Vc, float* __restrict__ O,
                              int HQ, int HKV, int Skv, int D, float scale){
        extern __shared__ float sh[];                   // [D] 放 q，后面 [Skv] 放 score
        float* qs = sh; float* sc = sh + D;
        const int bh = blockIdx.x, b = bh / HQ, hq = bh % HQ;
        const int hkv = hq / (HQ / HKV);                // GQA 映射
        const float* q  = Q  + (size_t)bh * D;
        const float* Kb = Kc + ((size_t)b * HKV + hkv) * Skv * D;
        const float* Vb = Vc + ((size_t)b * HKV + hkv) * Skv * D;
        float* o = O + (size_t)bh * D;

        for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = q[d];
        __syncthreads();
        float m = -INFINITY;
        for (int j = threadIdx.x; j < Skv; j += blockDim.x){
            const float* k = Kb + (size_t)j * D;
            float a = 0.f;
            for (int d = 0; d < D; d++) a += qs[d] * k[d];
            sc[j] = a * scale; m = fmaxf(m, sc[j]);
        }
        m = blockAllMax(m);
        float l = 0.f;
        for (int j = threadIdx.x; j < Skv; j += blockDim.x){ float e = __expf(sc[j] - m); sc[j] = e; l += e; }
        l = blockAllSum(l);
        const float inv = __fdividef(1.0f, l);
        for (int d = threadIdx.x; d < D; d += blockDim.x){
            float acc = 0.f;
            for (int j = 0; j < Skv; j++) acc += sc[j] * Vb[(size_t)j * D + d];
            o[d] = acc * inv;
        }
    }
    ```

=== "v1 float4 向量化 + 分工"

    同样的结构，但把两个热点都向量化：

    ① **score 阶段**：`q·k` 用 float4 读，一条指令 4 个乘加；

    ② **PV 阶段**：把 `blockDim` 拆成 `(D/4) × NC`，每个线程负责 4 个输出维度和 1/NC 的 key，最后 shared 归约 —— **PV 阶段不再只有 D 个线程干活**。

    decode 是纯 memory-bound，所以这两处都是**直接换来带宽**。

    ```cpp
    template<int D, int NC>                              // NC = 沿 key 维再切几份
    __global__ void decode_v1(const float* __restrict__ Q, const float* __restrict__ Kc,
                              const float* __restrict__ Vc, float* __restrict__ O,
                              int HQ, int HKV, int Skv, float scale){
        extern __shared__ float sh[];
        float* qs = sh; float* sc = sh + D;
        __shared__ float part[NC][D];
        const int bh = blockIdx.x, b = bh / HQ, hq = bh % HQ, hkv = hq / (HQ / HKV);
        const float* Kb = Kc + ((size_t)b * HKV + hkv) * Skv * D;
        const float* Vb = Vc + ((size_t)b * HKV + hkv) * Skv * D;
        const int tid = threadIdx.x;

        for (int d = tid; d < D; d += blockDim.x) qs[d] = Q[(size_t)bh * D + d];
        __syncthreads();
        const float4* q4 = reinterpret_cast<const float4*>(qs);
        float m = -INFINITY;
        for (int j = tid; j < Skv; j += blockDim.x){
            const float4* k4 = reinterpret_cast<const float4*>(Kb + (size_t)j * D);
            float a = 0.f;
            #pragma unroll
            for (int d = 0; d < D / 4; d++){
                float4 kk = k4[d], qq = q4[d];
                a += kk.x*qq.x + kk.y*qq.y + kk.z*qq.z + kk.w*qq.w;
            }
            sc[j] = a * scale; m = fmaxf(m, sc[j]);
        }
        m = blockAllMax(m);
        float l = 0.f;
        for (int j = tid; j < Skv; j += blockDim.x){ float e = __expf(sc[j] - m); sc[j] = e; l += e; }
        l = blockAllSum(l);
        const float inv = __fdividef(1.0f, l);

        const int dv = tid % (D / 4), jc = tid / (D / 4);      // 每线程负责 4 个输出维度
        float4 acc = make_float4(0,0,0,0);
        for (int j = jc; j < Skv; j += NC){
            float4 vv = reinterpret_cast<const float4*>(Vb + (size_t)j * D)[dv];
            float p = sc[j];
            acc.x += p*vv.x; acc.y += p*vv.y; acc.z += p*vv.z; acc.w += p*vv.w;
        }
        reinterpret_cast<float4*>(&part[jc][0])[dv] = acc;
        __syncthreads();
        if (tid < D / 4){
            float4 t = make_float4(0,0,0,0);
            #pragma unroll
            for (int c = 0; c < NC; c++){
                float4 u = reinterpret_cast<float4*>(&part[c][0])[tid];
                t.x += u.x; t.y += u.y; t.z += u.z; t.w += u.w;
            }
            t.x *= inv; t.y *= inv; t.z *= inv; t.w *= inv;
            reinterpret_cast<float4*>(O + (size_t)bh * D)[tid] = t;
        }
    }
    ```

=== "v2 flash-decoding（split-KV 两阶段）"

    **decode 优化的核心技巧。**并行度不够（batch 小）时，沿 **KV 长度**再切 `nsplit` 份：

    • **阶段一**：`grid = B×HQ×nsplit`，每个 block 只扫自己那段 KV，输出**未归一化**的 `o_i`、以及该段的 `m_i` 和 `l_i`；

    • **阶段二**：按 online softmax 的合并公式把 nsplit 份拼起来：

    `m = max(mᵢ)`，`O = Σᵢ e^{mᵢ-m}·oᵢ / Σᵢ e^{mᵢ-m}·lᵢ`

    **并行度从 B×HQ 提升到 B×HQ×nsplit**，batch=1 也能填满 GPU。代价是多写一份中间结果（很小：nsplit×D）。

    这就是 FlashAttention 仓库里的 `flash_decoding`，也是 vLLM 的 `paged_attention_v2`。

    ```cpp
    template<int D>
    __global__ void decode_split(const float* __restrict__ Q, const float* __restrict__ Kc,
                                 const float* __restrict__ Vc,
                                 float* __restrict__ pO, float* __restrict__ pM, float* __restrict__ pL,
                                 int HQ, int HKV, int Skv, int nsplit, float scale){
        extern __shared__ float sh[];
        float* qs = sh; float* sc = sh + D;
        const int bh = blockIdx.x, sp = blockIdx.y;
        const int b = bh / HQ, hq = bh % HQ, hkv = hq / (HQ / HKV);
        const int per = CEIL_DIV(Skv, nsplit);
        const int j0 = sp * per, j1 = min(Skv, j0 + per);       // 本 block 负责 [j0, j1)
        const float* Kb = Kc + ((size_t)b * HKV + hkv) * Skv * D;
        const float* Vb = Vc + ((size_t)b * HKV + hkv) * Skv * D;
        const int tid = threadIdx.x;

        for (int d = tid; d < D; d += blockDim.x) qs[d] = Q[(size_t)bh * D + d];
        __syncthreads();
        float m = -INFINITY;
        for (int j = j0 + tid; j < j1; j += blockDim.x){
            const float* k = Kb + (size_t)j * D;
            float a = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d++) a += qs[d] * k[d];
            sc[j - j0] = a * scale; m = fmaxf(m, sc[j - j0]);
        }
        m = blockAllMax(m);
        float l = 0.f;
        for (int j = j0 + tid; j < j1; j += blockDim.x){ float e = __expf(sc[j - j0] - m); sc[j - j0] = e; l += e; }
        l = blockAllSum(l);
        for (int d = tid; d < D; d += blockDim.x){
            float acc = 0.f;
            for (int j = j0; j < j1; j++) acc += sc[j - j0] * Vb[(size_t)j * D + d];
            pO[((size_t)bh * nsplit + sp) * D + d] = acc;       // ⚠ 不除 l！留给阶段二
        }
        if (tid == 0){ pM[(size_t)bh * nsplit + sp] = m; pL[(size_t)bh * nsplit + sp] = l; }
    }
    // 阶段二：按 (m, l) 重加权合并
    template<int D>
    __global__ void decode_combine(const float* __restrict__ pO, const float* __restrict__ pM,
                                   const float* __restrict__ pL, float* __restrict__ O, int nsplit){
        const int bh = blockIdx.x, d = threadIdx.x;
        float m = -INFINITY;
        for (int i = 0; i < nsplit; i++) m = fmaxf(m, pM[(size_t)bh * nsplit + i]);   // 全局 max
        float num = 0.f, den = 0.f;
        for (int i = 0; i < nsplit; i++){
            float w = __expf(pM[(size_t)bh * nsplit + i] - m);              // 重加权系数
            num += w * pO[((size_t)bh * nsplit + i) * D + d];
            den += w * pL[(size_t)bh * nsplit + i];
        }
        O[(size_t)bh * D + d] = num / den;
    }
    ```

=== "v3 paged + GQA 共享读（vLLM paged_attention_v2 形态）"

    把前面所有招数合在一起，就是生产级 decode kernel：

    ① **split-KV**：`grid = (B×HKV, nsplit)`，并行度足够；

    ② **GQA 共享读**：一个 block 服务同一组 KV 的全部 G 个 Q head —— 内层循环里 **`k[d]` 只从 HBM 读一次，同时喂给 G 个 query 的点积**。LLaMA-70B 的 G=8，**KV 带宽需求直接除以 8**。这才是 GQA 在算子层面的真正收益（不只是省显存）。

    ③ **paged cache**：KV 不连续，按固定大小的 page（vLLM 默认 16 token）存放，靠 `block_table[seq][逻辑块] → 物理块` 查表。好处是**零碎片 + prefix 可共享**，代价是多一层间址、且 kernel 必须按 page 组织循环。

    输出仍然是未归一化的 (o, m, l)，交给阶段二合并。

    ```cpp
    template<int D, int G, int P, int PAGE>
    __global__ void decode_paged_v2(const float* __restrict__ Q, const float* __restrict__ Kc,
                                    const float* __restrict__ Vc, const int* __restrict__ blockTable,
                                    float* __restrict__ pO, float* __restrict__ pM, float* __restrict__ pL,
                                    int HQ, int HKV, int Skv, int maxBlocks, int nsplit, float scale){
        __shared__ float qs[G][D];
        __shared__ float sc[G][P];                              // 本 split 内 G 个 head 的 score
        const int bkv = blockIdx.x, sp = blockIdx.y;
        const int b = bkv / HKV, hkv = bkv % HKV, tid = threadIdx.x;
        const int j0 = sp * P, j1 = min(Skv, j0 + P);
        const int* bt = blockTable + (size_t)b * maxBlocks;

        for (int i = tid; i < G * D; i += blockDim.x)
            qs[i / D][i % D] = Q[((size_t)b * HQ + hkv * G + i / D) * D + i % D];
        __syncthreads();

        float mloc[G];
        #pragma unroll
        for (int g = 0; g < G; g++) mloc[g] = -INFINITY;
        for (int j = j0 + tid; j < j1; j += blockDim.x){
            int phys = bt[j / PAGE];                  // 逻辑块 → 物理块
            const float* k = Kc + ((size_t)phys * HKV + hkv) * PAGE * D + (size_t)(j % PAGE) * D;
            float a[G];
            #pragma unroll
            for (int g = 0; g < G; g++) a[g] = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d++){
                float kd = k[d];                                // ★ k 只读一次，服务 G 个 query head
                #pragma unroll
                for (int g = 0; g < G; g++) a[g] += qs[g][d] * kd;
            }
            #pragma unroll
            for (int g = 0; g < G; g++){ a[g] *= scale; sc[g][j - j0] = a[g]; mloc[g] = fmaxf(mloc[g], a[g]); }
        }
        for (int g = 0; g < G; g++){                            // 每个 head 各自 softmax + PV
            float m = blockAllMax(mloc[g]);
            float l = 0.f;
            for (int j = j0 + tid; j < j1; j += blockDim.x){ float e = __expf(sc[g][j - j0] - m); sc[g][j - j0] = e; l += e; }
            l = blockAllSum(l);
            const int hq = hkv * G + g;
            for (int d = tid; d < D; d += blockDim.x){
                float acc = 0.f;
                for (int j = j0; j < j1; j++){
                    int phys = bt[j / PAGE];
                    acc += sc[g][j - j0] * Vc[((size_t)phys * HKV + hkv) * PAGE * D + (size_t)(j % PAGE) * D + d];
                }
                pO[(((size_t)b * HQ + hq) * nsplit + sp) * D + d] = acc;   // 不除 l，留给阶段二
            }
            if (tid == 0){ pM[((size_t)b * HQ + hq) * nsplit + sp] = m; pL[((size_t)b * HQ + hq) * nsplit + sp] = l; }
            __syncthreads();
        }
    }
    ```

</div>
