---
title: "FlashAttention（分块 + online softmax + causal 跳块）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">FlashAttention（分块 + online softmax + causal 跳块）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P12</span><span>flash attention</span><span>online softmax</span><span>tiling</span><span>causal</span><span>GQA</span><span>压轴</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

不产生 S×S 的中间矩阵，实现 attention。

<dl class="gpu-operator__io">
  <dt>核心思想</dt><dd>① 一个 block 负责 <strong>BR 个 query 行</strong>，输出累加器 O 全程待在寄存器；<br>
② 外层循环遍历 <strong>KV 的分块</strong>（每块 BC 行），K/V tile 搬进 shared；<br>
③ 每处理完一块，用 <strong>online softmax</strong> 更新 (m, l) 并 rescale 已累积的 O。</dd>
  <dt>结果</dt><dd>S×S 的 score 矩阵<strong>从来没有完整存在过</strong>，HBM 访存从 O(S²) 降到 O(S·D)</dd>
  <dt>要点</dt><dd>rescale 公式、causal 整块跳过、空块的 NaN、LSE 的两个用途、GQA 映射</dd>
</dl>

**这是攻推理岗最值得投入时间的一题。**30 分钟写不完完整版，但**写出核心循环 + 讲清 rescale 的数学**是完全可能的， 而这已经能把大部分候选人甩开。

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
// 一个 block 负责 BR 个 query；grid = (ceil(S/BR), B*H)
template<int D, int BR, int BC>
__global__ void flash_attention(const float* Q, const float* K, const float* V,
                                float* O, float* LSE,   // LSE 可选
                                int S, float scale, bool causal);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- 先写外层骨架：`for (KV 块) { 搬 K/V 进 shared; 算 score; online 更新 (m,l,O); }`
- 每个线程负责**一个 query 行**，它的 `q[D]` 和输出 `o[D]` 都放寄存器（D 是模板常量才行）。
- rescale 公式：`m' = max(m, m_tile)`，`corr = exp(m - m')`，`l = l·corr + l_tile`，`o = o·corr + Σ p·v`。
- **先算完一整块的 score 再统一 rescale 一次**，不要每个 key 都 rescale（那样 o 的缩放开销会翻倍）。
- causal：外层循环上界改成 `min(S, qbase+BR)`，**右上角的 KV 块整块不用算**——这直接省掉一半计算。
- ⚠ **整块被 mask 掉时（m\_tile = -inf）必须跳过**，否则 `exp(-inf-(-inf)) = NaN` 污染整行。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
// 布局 [B, H, S, D]。核心思想：
//   ① 一个 block 负责 BR 个 query 行，输出累加器 O 一直待在寄存器里；
//   ② 外层循环遍历 KV 的分块（每块 BC 行），K/V tile 搬进 shared；
//   ③ 每处理完一块就用 online softmax 更新 (m, l) 并 rescale 已有的 O。
// 于是 S×S 的 score 矩阵【从来没有完整存在过】，HBM 访存从 O(S²) 降到 O(S·D)。
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v1 分块 + 逐 key online"

    最直白的 flash：**一个线程负责一个 query 行**，q 和输出累加器 o 全程待在寄存器；外层遍历 KV 分块，K/V 搬进 shared 后，对块里的每个 key 立刻做一次 online 更新：

    `m' = max(m, s); corr = exp(m - m'); l = l·corr + exp(s - m'); o = o·corr + exp(s-m')·v`

    **正确但浪费**：每来一个 key 就要把整个 `o[D]` 缩放一遍 —— D 次乘法/key，和算 score 的 D 次 FMA 一样多，**等于把计算量翻倍**。

    ```cpp
    template<int D, int BR, int BC>
    __global__ void flash_v1(const float* __restrict__ Q, const float* __restrict__ K,
                             const float* __restrict__ V, float* __restrict__ O,
                             int S, float scale, bool causal){
        __shared__ float Ks[BC][D], Vs[BC][D];
        const int bh = blockIdx.y, qbase = blockIdx.x * BR, tid = threadIdx.x;
        const int qi = qbase + tid;
        const float* Qb = Q + (size_t)bh * S * D;
        const float* Kb = K + (size_t)bh * S * D;
        const float* Vb = V + (size_t)bh * S * D;

        float q[D], o[D];
        #pragma unroll
        for (int d = 0; d < D; d++){ q[d] = (qi < S) ? Qb[(size_t)qi * D + d] : 0.f; o[d] = 0.f; }
        float m = -INFINITY, l = 0.f;

        const int jend = causal ? min(S, qbase + BR) : S;      // causal：后面的 KV 块整块不用算
        for (int j0 = 0; j0 < jend; j0 += BC){
            for (int t = tid; t < BC * D; t += BR){            // 协作搬 K/V tile
                int jj = t / D, dd = t % D, gj = j0 + jj;
                Ks[jj][dd] = (gj < S) ? Kb[(size_t)gj * D + dd] : 0.f;
                Vs[jj][dd] = (gj < S) ? Vb[(size_t)gj * D + dd] : 0.f;
            }
            __syncthreads();
            for (int jj = 0; jj < BC; jj++){
                int gj = j0 + jj;
                if (gj >= S) break;
                if (causal && gj > qi) break;        // key 有序，越界即可整体退出
                float a = 0.f;
                #pragma unroll
                for (int d = 0; d < D; d++) a += q[d] * Ks[jj][d];
                float s = a * scale;
                float mn = fmaxf(m, s);
                float corr = __expf(m - mn);         // 旧的 l 和 o 换基准
                float p    = __expf(s - mn);
                l = l * corr + p;
                #pragma unroll
                for (int d = 0; d < D; d++) o[d] = o[d] * corr + p * Vs[jj][d];   // ← 每个 key 都缩放，浪费
                m = mn;
            }
            __syncthreads();
        }
        if (qi < S){
            float inv = __fdividef(1.0f, l);
            #pragma unroll
            for (int d = 0; d < D; d++) O[((size_t)bh * S + qi) * D + d] = o[d] * inv;
        }
    }
    ```

=== "v2 每块只 rescale 一次（真正的 FlashAttention）"

    关键改进：**先把整个 KV 块的 score 算完存在寄存器里，求出块内最大值，再统一做一次 rescale**。

    `m' = max(m, max_j s_j)；o = o·exp(m-m') + Σⱼ exp(sⱼ-m')·vⱼ`

    于是 **o 的缩放从"每 key 一次"降到"每块一次"**，计算量降回 1×。这就是论文里那个 rescale 公式的真实意义。

    代价是每线程多 BC 个寄存器存 score，所以 **BC 不能太大**（这里 32）——**BR/BC 的选择本质是"寄存器/shared 容量"和"重用率"的权衡**，也是 FlashAttention 调参的全部内容。

    ```cpp
    template<int D, int BR, int BC>
    __global__ void flash_v2(const float* __restrict__ Q, const float* __restrict__ K,
                             const float* __restrict__ V, float* __restrict__ O,
                             int S, float scale, bool causal){
        __shared__ float Ks[BC][D], Vs[BC][D];
        const int bh = blockIdx.y, qbase = blockIdx.x * BR, tid = threadIdx.x;
        const int qi = qbase + tid;
        const float* Qb = Q + (size_t)bh * S * D;
        const float* Kb = K + (size_t)bh * S * D;
        const float* Vb = V + (size_t)bh * S * D;

        float q[D], o[D], s[BC];
        #pragma unroll
        for (int d = 0; d < D; d++){ q[d] = (qi < S) ? Qb[(size_t)qi * D + d] : 0.f; o[d] = 0.f; }
        float m = -INFINITY, l = 0.f;

        const int jend = causal ? min(S, qbase + BR) : S;
        for (int j0 = 0; j0 < jend; j0 += BC){
            for (int t = tid; t < BC * D; t += BR){
                int jj = t / D, dd = t % D, gj = j0 + jj;
                Ks[jj][dd] = (gj < S) ? Kb[(size_t)gj * D + dd] : 0.f;
                Vs[jj][dd] = (gj < S) ? Vb[(size_t)gj * D + dd] : 0.f;
            }
            __syncthreads();

            float mtile = -INFINITY;                            // ① 先算完整块的 score
            #pragma unroll
            for (int jj = 0; jj < BC; jj++){
                int gj = j0 + jj;
                bool valid = (gj < S) && !(causal && gj > qi);
                float a = 0.f;
                if (valid){
                    #pragma unroll
                    for (int d = 0; d < D; d++) a += q[d] * Ks[jj][d];
                    a *= scale;
                } else a = -INFINITY;
                s[jj] = a; mtile = fmaxf(mtile, a);
            }
            if (mtile > -INFINITY){                             // 整块被 mask 掉就直接跳过（否则 exp 出 NaN）
                float mn   = fmaxf(m, mtile);                   // ② 只在这里 rescale 一次
                float corr = __expf(m - mn);
                float ltile = 0.f;
                #pragma unroll
                for (int jj = 0; jj < BC; jj++){ s[jj] = (s[jj] == -INFINITY) ? 0.f : __expf(s[jj] - mn); ltile += s[jj]; }
                l = l * corr + ltile;
                #pragma unroll
                for (int d = 0; d < D; d++){
                    float acc = 0.f;
                    #pragma unroll
                    for (int jj = 0; jj < BC; jj++) acc += s[jj] * Vs[jj][d];
                    o[d] = o[d] * corr + acc;
                }
                m = mn;
            }
            __syncthreads();
        }
        if (qi < S){
            float inv = __fdividef(1.0f, l);
            #pragma unroll
            for (int d = 0; d < D; d++) O[((size_t)bh * S + qi) * D + d] = o[d] * inv;
        }
    }
    ```

=== "v3 causal 的三种块 + 输出 LSE"

    真实实现会把 KV 块分成三类：

    • **全在对角线左下** → 不用 mask，走最快的路径；

    • **跨对角线** → 需要逐元素判断 `gj > qi`；

    • **全在右上** → **整块跳过**（causal 下这是一半的计算量，所以 causal attention 应该比 non-causal 快近一倍）。

    另外输出 **LSE = m + log(l)**：反向传播要用它重算概率（不存 P 矩阵），**flash-decoding 的分块合并也要用它**。这两个用途决定了它必须输出。

    ```cpp
    template<int D, int BR, int BC>
    __global__ void flash_v3(const float* __restrict__ Q, const float* __restrict__ K,
                             const float* __restrict__ V, float* __restrict__ O,
                             float* __restrict__ LSE, int S, float scale){
        __shared__ float Ks[BC][D], Vs[BC][D];
        const int bh = blockIdx.y, qbase = blockIdx.x * BR, tid = threadIdx.x;
        const int qi = qbase + tid;
        const float* Qb = Q + (size_t)bh * S * D;
        const float* Kb = K + (size_t)bh * S * D;
        const float* Vb = V + (size_t)bh * S * D;

        float q[D], o[D], s[BC];
        #pragma unroll
        for (int d = 0; d < D; d++){ q[d] = (qi < S) ? Qb[(size_t)qi * D + d] : 0.f; o[d] = 0.f; }
        float m = -INFINITY, l = 0.f;

        const int jend = min(S, qbase + BR);                    // 整块跳过：只到本 query 块的末尾
        for (int j0 = 0; j0 < jend; j0 += BC){
            for (int t = tid; t < BC * D; t += BR){
                int jj = t / D, dd = t % D, gj = j0 + jj;
                Ks[jj][dd] = (gj < S) ? Kb[(size_t)gj * D + dd] : 0.f;
                Vs[jj][dd] = (gj < S) ? Vb[(size_t)gj * D + dd] : 0.f;
            }
            __syncthreads();
            // 这一块整体在对角线左下方吗？是的话完全不用逐元素判断
            const bool full = (j0 + BC - 1) <= qbase;
            float mtile = -INFINITY;
            #pragma unroll
            for (int jj = 0; jj < BC; jj++){
                int gj = j0 + jj;
                float a;
                if (full || ((gj < S) && gj <= qi)){
                    a = 0.f;
                    #pragma unroll
                    for (int d = 0; d < D; d++) a += q[d] * Ks[jj][d];
                    a *= scale;
                } else a = -INFINITY;
                s[jj] = a; mtile = fmaxf(mtile, a);
            }
            if (mtile > -INFINITY){
                float mn = fmaxf(m, mtile), corr = __expf(m - mn), ltile = 0.f;
                #pragma unroll
                for (int jj = 0; jj < BC; jj++){ s[jj] = (s[jj] == -INFINITY) ? 0.f : __expf(s[jj] - mn); ltile += s[jj]; }
                l = l * corr + ltile;
                #pragma unroll
                for (int d = 0; d < D; d++){
                    float acc = 0.f;
                    #pragma unroll
                    for (int jj = 0; jj < BC; jj++) acc += s[jj] * Vs[jj][d];
                    o[d] = o[d] * corr + acc;
                }
                m = mn;
            }
            __syncthreads();
        }
        if (qi < S){
            float inv = __fdividef(1.0f, l);
            #pragma unroll
            for (int d = 0; d < D; d++) O[((size_t)bh * S + qi) * D + d] = o[d] * inv;
            if (LSE) LSE[(size_t)bh * S + qi] = m + __logf(l);   // log-sum-exp
        }
    }
    ```

=== "v4 GQA / MQA（K、V 的 head 更少）"

    LLaMA-70B 是 **64 个 Q head 共享 8 个 KV head**（GQA-8），MQA 更极端（所有 Q head 共享 1 组 KV）。

    改动其实只有一行：**`kv_head = q_head / (HQ / HKV)`**，K/V 的指针按 kv\_head 走。

    但意义很大：**KV cache 的显存和访存量直接除以 8**，decode 阶段（完全 memory-bound）几乎线性提速。

    再进一步就是 **MLA**（DeepSeek）：把 KV 压成一个低秩隐向量，cache 再小一个数量级。**"为什么现在的模型都用 GQA"——答案是 decode 的瓶颈是 KV cache 的带宽，不是算力。**

    ```cpp
    template<int D, int BR, int BC>
    __global__ void flash_v4_gqa(const float* __restrict__ Q, const float* __restrict__ K,
                                 const float* __restrict__ V, float* __restrict__ O,
                                 int S, int HQ, int HKV, float scale, bool causal){
        __shared__ float Ks[BC][D], Vs[BC][D];
        const int bh = blockIdx.y;                              // = b * HQ + hq
        const int b = bh / HQ, hq = bh % HQ;
        const int hkv = hq / (HQ / HKV);              // 多个 Q head 映射到同一组 KV
        const int qbase = blockIdx.x * BR, tid = threadIdx.x, qi = qbase + tid;
        const float* Qb = Q + ((size_t)b * HQ  + hq)  * S * D;
        const float* Kb = K + ((size_t)b * HKV + hkv) * S * D;
        const float* Vb = V + ((size_t)b * HKV + hkv) * S * D;

        float q[D], o[D], s[BC];
        #pragma unroll
        for (int d = 0; d < D; d++){ q[d] = (qi < S) ? Qb[(size_t)qi * D + d] : 0.f; o[d] = 0.f; }
        float m = -INFINITY, l = 0.f;
        const int jend = causal ? min(S, qbase + BR) : S;
        for (int j0 = 0; j0 < jend; j0 += BC){
            for (int t = tid; t < BC * D; t += BR){
                int jj = t / D, dd = t % D, gj = j0 + jj;
                Ks[jj][dd] = (gj < S) ? Kb[(size_t)gj * D + dd] : 0.f;
                Vs[jj][dd] = (gj < S) ? Vb[(size_t)gj * D + dd] : 0.f;
            }
            __syncthreads();
            float mtile = -INFINITY;
            #pragma unroll
            for (int jj = 0; jj < BC; jj++){
                int gj = j0 + jj; float a;
                if ((gj < S) && !(causal && gj > qi)){
                    a = 0.f;
                    #pragma unroll
                    for (int d = 0; d < D; d++) a += q[d] * Ks[jj][d];
                    a *= scale;
                } else a = -INFINITY;
                s[jj] = a; mtile = fmaxf(mtile, a);
            }
            if (mtile > -INFINITY){
                float mn = fmaxf(m, mtile), corr = __expf(m - mn), ltile = 0.f;
                #pragma unroll
                for (int jj = 0; jj < BC; jj++){ s[jj] = (s[jj] == -INFINITY) ? 0.f : __expf(s[jj] - mn); ltile += s[jj]; }
                l = l * corr + ltile;
                #pragma unroll
                for (int d = 0; d < D; d++){
                    float acc = 0.f;
                    #pragma unroll
                    for (int jj = 0; jj < BC; jj++) acc += s[jj] * Vs[jj][d];
                    o[d] = o[d] * corr + acc;
                }
                m = mn;
            }
            __syncthreads();
        }
        if (qi < S){
            float inv = __fdividef(1.0f, l);
            #pragma unroll
            for (int d = 0; d < D; d++) O[((size_t)bh * S + qi) * D + d] = o[d] * inv;
        }
    }
    ```

</div>
