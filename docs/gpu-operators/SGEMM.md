---
title: "SGEMM 逐级优化（naive → shared → register tile → 向量化 → 双缓冲 → wmma）"
code_fold: false
hide_title: true
hide:
  - toc
  - footer
---

<div class="gpu-operator" markdown="1">

<p class="gpu-operator__title">SGEMM 逐级优化（naive → shared → register tile → 向量化 → 双缓冲 → wmma）</p>
<div class="gpu-operator__meta">
  <span>CUDA</span><span>P14</span><span>gemm</span><span>tiling</span><span>register tile</span><span>double buffer</span><span>tensor core</span><span>压轴</span>
</div>

<p class="gpu-operator__label" id="problem">题目</p>

`C[M,N] = A[M,K] × B[K,N]`，全部 row-major，fp32。测试 M=N=K=2048。

<dl class="gpu-operator__io">
  <dt>核心指标</dt><dd><strong>计算访存比</strong> = FLOP / 从 global 读的字节数。naive 是 0.25 flop/byte，
而 GPU 的 ridge point 在 100 量级 —— <strong>差 400 倍，所以纯卡在访存上</strong>。
每一版优化的本质都是&quot;<strong>把这个比值提上去</strong>&quot;。</dd>
  <dt>目标</dt><dd>逼近 cuBLAS。手写能到 <strong>cuBLAS 的 50~70%</strong> 就是很好的水平了。</dd>
</dl>

**这是 GPU 面试的绝对主角。**不要求你 30 分钟写出 cuBLAS，但要求你**说得出每一步优化解决了什么瓶颈**， 并且能把 register tile 那一版写对。**建议把 v0→v3 练到闭着眼能写。**

<p class="gpu-operator__label" id="signature">给定签名</p>

```cpp
void sgemm(const float* A,   // [M, K] row-major
           const float* B,   // [K, N] row-major
           float* C,         // [M, N] row-major
           int M, int N, int K);
```

<p class="gpu-operator__label" id="tips">Tips</p>

- naive 写完先算计算访存比：每个 C 元素读 2K 个 float（8K 字节）做 2K 次 flop → **0.25 flop/byte**。
- 第一步永远是 shared tile：搬进来的元素被复用 TILE 次，比值提到 TILE/4。**两个 `__syncthreads()` 一个都不能少**。
- 第二步是 register tile：让一个线程算 TM×TN 个输出，**从 shared 读 TM+TN 个数却做 TM×TN 次 FMA**。这一步收益最大。
- 经典配置背下来：**BM=BN=128, BK=8, TM=TN=8, blockDim=256**。shared 只要 8 KB，每线程 64 个累加器。
- **A 在写进 shared 时就转置**成 `As[k][m]`，这样内层按 k 取 A 的一列时是连续的，可以用 float4 读。
- 最后是双缓冲：开两块 shared，**算当前块的同时把下一块搬进来**，每轮只需要一次 `__syncthreads()`。

<p class="gpu-operator__label" id="prelude">公共前置</p>

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
// C[M,N] = A[M,K] * B[K,N]，全部 row-major
// 关键指标：计算访存比 = FLOP / 从 global 读的字节数。
//   naive     : 每个 C 元素读 2K 个 float 做 2K flop → 比值 0.25 flop/byte（惨）
//   shared tile: 比值 ≈ TILE/4
//   register tile: 比值 ≈ (TM*TN)/(TM+TN)/4 * BK... 见各版说明
```

<p class="gpu-operator__label" id="implementations">分级实现</p>

=== "v0 naive"

    一个线程算一个 C 元素，从 global 一路读到底。**计算访存比 = 2K flop / 8K byte = 0.25**，

    ```cpp
    // 典型现代 GPU 的 ridge point 在数十到数百 flop/byte 量级 —— 差了 400 倍，<b>完全卡在访存上</b>。
    __global__ void sgemm_v0(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N){
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
            C[(size_t)row * N + col] = acc;
        }
    }
    ```

=== "v1 修正线程映射（合并访存）"

    v0 里 `row` 跟着 `threadIdx.y`、`col` 跟着 `threadIdx.x`，看起来没问题——但如果不小心写反（`row = threadIdx.x`），一个 warp 的 32 个线程就会访问 B 的**同一列**（间隔 N×4 字节），**访存完全不合并，慢 5~10 倍**。

    正确的写法是：**让 `threadIdx.x`（warp 内连续变化的那一维）对应输出矩阵的列**，这样同一 warp 读 B 的一行连续元素、写 C 的一行连续元素。

    这一版把线程映射摊平成一维再手工拆，是为了让这件事显式可见。

    ```cpp
    __global__ void sgemm_v1(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int row = tid / N;                    // 慢维
        int col = tid % N;                    // 快维 ← 必须是 warp 内连续的那一维
        if (row < M && col < N){
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
            C[(size_t)row * N + col] = acc;
        }
    }
    ```

=== "v2 shared memory tile"

    把 K 维切成 TILE 大小的片，每次把 A 的 TILE×TILE 块和 B 的 TILE×TILE 块搬进 shared，块内所有线程复用。

    **计算访存比从 0.25 提到 TILE/4（TILE=32 时是 8）**，因为搬进来的每个元素被复用了 TILE 次。

    三个必写的细节：① 搬完要 `__syncthreads()`；② 算完、下一轮覆盖前**还要一次** `__syncthreads()`；③ 边界不足时补 0。

    ```cpp
    template<int TILE>
    __global__ void sgemm_v2(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        __shared__ float As[TILE][TILE], Bs[TILE][TILE];
        int tx = threadIdx.x, ty = threadIdx.y;
        int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
        float acc = 0.f;
        for (int k0 = 0; k0 < K; k0 += TILE){
            As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
            Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
            __syncthreads();                  // ← 漏掉这句会读到下一轮覆盖的数据
        }
        if (row < M && col < N) C[(size_t)row * N + col] = acc;
    }
    ```

=== "v3 register tile（一个线程算 TM×TN 个）"

    v2 里每个线程只算 1 个输出，**从 shared 读 2 个数只做 1 次 FMA**，shared 带宽成了新瓶颈。

    让**一个线程负责 TM×TN 的小块**：从 shared 读 TM+TN 个数，却能做 TM×TN 次 FMA —— **shared 的计算访存比从 1:2 提到 TM×TN : (TM+TN)**（4×4 时是 2:1，8×8 时是 4:1）。

    累加器 `acc[TM][TN]` 全在寄存器里。**这一步通常带来最大的性能跃升。**

    ```cpp
    template<int BM, int BN, int BK, int TM, int TN>
    __global__ void sgemm_v3(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        __shared__ float As[BK][BM], Bs[BK][BN];         // As 存成转置：[k][m]，方便按 k 取一列
        const int tid = threadIdx.x;
        const int tx = tid % (BN / TN), ty = tid / (BN / TN);
        const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
        float acc[TM][TN] = {0.f};
        float fa[TM], fb[TN];

        for (int k0 = 0; k0 < K; k0 += BK){
            for (int i = tid; i < BM * BK; i += blockDim.x){       // 搬 A：[BM][BK] → As[BK][BM]
                int r = i / BK, c = i % BK;
                As[c][r] = (cRow + r < M && k0 + c < K) ? A[(size_t)(cRow + r) * K + k0 + c] : 0.f;
            }
            for (int i = tid; i < BK * BN; i += blockDim.x){       // 搬 B：[BK][BN]
                int r = i / BN, c = i % BN;
                Bs[r][c] = (k0 + r < K && cCol + c < N) ? B[(size_t)(k0 + r) * N + cCol + c] : 0.f;
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < BK; k++){
                #pragma unroll
                for (int i = 0; i < TM; i++) fa[i] = As[k][ty * TM + i];   // 一次读 TM 个
                #pragma unroll
                for (int j = 0; j < TN; j++) fb[j] = Bs[k][tx * TN + j];   // 一次读 TN 个
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) acc[i][j] += fa[i] * fb[j];  // TM*TN 次 FMA
            }
            __syncthreads();
        }
        for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++){
                int r = cRow + ty * TM + i, c = cCol + tx * TN + j;
                if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
            }
    }
    ```

=== "v4 float4 向量化 + A 转置存 shared"

    三处一起改：

    ① **global→shared 用 float4**：一条 128-bit 指令搬 4 个元素，搬运指令数降到 1/4；

    ② **A 在写入 shared 时就转置**（`As[k][m]`）：这样内层循环按 k 取 A 的一整列时是连续的，**可以用 float4 从 shared 读**；

    ③ **shared→register 也用 float4**。

    配置用经典的 **BM=BN=128, BK=8, TM=TN=8, 256 线程**：每个 block 算 128×128 的输出块，每线程 8×8，**shared 只要 8 KB**，寄存器 64 个累加器 —— 这是 SGEMM 的黄金配置。

    ```cpp
    template<int BM, int BN, int BK, int TM, int TN>
    __global__ void sgemm_v4(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        __shared__ float As[BK][BM], Bs[BK][BN];
        const int tid = threadIdx.x;
        const int tx = tid % (BN / TN), ty = tid / (BN / TN);
        const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
        A += (size_t)cRow * K;  B += cCol;  C += (size_t)cRow * N + cCol;

        // 每个线程搬 A 的一个 float4（BM*BK/4 = 256 个 float4，正好 256 线程）
        const int aRow = tid / (BK / 4), aCol = (tid % (BK / 4)) * 4;
        const int bRow = tid / (BN / 4), bCol = (tid % (BN / 4)) * 4;
        const int bStride = blockDim.x / (BN / 4);
        float acc[TM][TN] = {0.f};
        float fa[TM], fb[TN];

        for (int k0 = 0; k0 < K; k0 += BK){
            float4 a = *reinterpret_cast<const float4*>(&A[(size_t)aRow * K + aCol]);
            As[aCol + 0][aRow] = a.x;  As[aCol + 1][aRow] = a.y;
            As[aCol + 2][aRow] = a.z;  As[aCol + 3][aRow] = a.w;   // 转置写入
            #pragma unroll
            for (int r = 0; r < BK; r += bStride)
                *reinterpret_cast<float4*>(&Bs[bRow + r][bCol]) =
                    *reinterpret_cast<const float4*>(&B[(size_t)(bRow + r) * N + bCol]);
            __syncthreads();
            A += BK;  B += (size_t)BK * N;
            #pragma unroll
            for (int k = 0; k < BK; k++){
                #pragma unroll
                for (int i = 0; i < TM; i += 4)
                    *reinterpret_cast<float4*>(&fa[i]) = *reinterpret_cast<float4*>(&As[k][ty * TM + i]);
                #pragma unroll
                for (int j = 0; j < TN; j += 4)
                    *reinterpret_cast<float4*>(&fb[j]) = *reinterpret_cast<float4*>(&Bs[k][tx * TN + j]);
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) acc[i][j] += fa[i] * fb[j];
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&C[(size_t)(ty * TM + i) * N + tx * TN + j]) =
                    *reinterpret_cast<float4*>(&acc[i][j]);
    }
    ```

=== "v5 double buffering（双缓冲）"

    v4 的循环是**串行的**：`搬数据 → sync → 计算 → sync → 搬数据…`，搬的时候计算单元闲着、算的时候访存单元闲着。

    双缓冲：**开两块 shared，算第 i 块的同时把第 i+1 块搬进另一块**，让访存延迟被计算掩盖。

    关键收益：**每轮只需要一次 `__syncthreads()`**（不用等"大家都算完才能覆盖"），而且 global load 提前发射，延迟被隐藏。

    更进一步是 **支持异步拷贝的 CUDA 架构中的 `cp.async`**：直接 global→shared 异步拷贝，不占寄存器、不需要中转，配合 `cp.async.wait_group` 做多级流水；更新 CUDA 架构 上则是 **TMA + wgmma**。

    ```cpp
    template<int BM, int BN, int BK, int TM, int TN>
    __global__ void sgemm_v5(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K){
        __shared__ float As[2][BK][BM], Bs[2][BK][BN];   // ← 两块，乒乓
        const int tid = threadIdx.x;
        const int tx = tid % (BN / TN), ty = tid / (BN / TN);
        const int cRow = blockIdx.y * BM, cCol = blockIdx.x * BN;
        A += (size_t)cRow * K;  B += cCol;  C += (size_t)cRow * N + cCol;
        const int aRow = tid / (BK / 4), aCol = (tid % (BK / 4)) * 4;
        const int bRow = tid / (BN / 4), bCol = (tid % (BN / 4)) * 4;
        const int bStride = blockDim.x / (BN / 4);
        float acc[TM][TN] = {0.f};
        float fa[TM], fb[TN];

        // 预取第 0 块
        {
            float4 a = *reinterpret_cast<const float4*>(&A[(size_t)aRow * K + aCol]);
            As[0][aCol+0][aRow]=a.x; As[0][aCol+1][aRow]=a.y; As[0][aCol+2][aRow]=a.z; As[0][aCol+3][aRow]=a.w;
            #pragma unroll
            for (int r = 0; r < BK; r += bStride)
                *reinterpret_cast<float4*>(&Bs[0][bRow+r][bCol]) =
                    *reinterpret_cast<const float4*>(&B[(size_t)(bRow+r)*N + bCol]);
        }
        __syncthreads();
        int cur = 0;
        for (int k0 = 0; k0 < K; k0 += BK){
            float4 na; float4 nb[BK / 8 > 0 ? BK / 8 : 1];
            const bool hasNext = (k0 + BK) < K;
            if (hasNext){                                 // ① 先把下一块的 global load 发出去
                const float* An = A + BK;
                const float* Bn = B + (size_t)BK * N;
                na = *reinterpret_cast<const float4*>(&An[(size_t)aRow * K + aCol]);
                int c = 0;
                #pragma unroll
                for (int r = 0; r < BK; r += bStride)
                    nb[c++] = *reinterpret_cast<const float4*>(&Bn[(size_t)(bRow+r)*N + bCol]);
            }
            #pragma unroll
            for (int k = 0; k < BK; k++){                 // ② 用当前块做计算（延迟被 ① 掩盖）
                #pragma unroll
                for (int i = 0; i < TM; i += 4)
                    *reinterpret_cast<float4*>(&fa[i]) = *reinterpret_cast<float4*>(&As[cur][k][ty*TM+i]);
                #pragma unroll
                for (int j = 0; j < TN; j += 4)
                    *reinterpret_cast<float4*>(&fb[j]) = *reinterpret_cast<float4*>(&Bs[cur][k][tx*TN+j]);
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) acc[i][j] += fa[i] * fb[j];
            }
            if (hasNext){                                 // ③ 写进另一块 buffer
                int nxt = cur ^ 1;
                As[nxt][aCol+0][aRow]=na.x; As[nxt][aCol+1][aRow]=na.y;
                As[nxt][aCol+2][aRow]=na.z; As[nxt][aCol+3][aRow]=na.w;
                int c = 0;
                #pragma unroll
                for (int r = 0; r < BK; r += bStride)
                    *reinterpret_cast<float4*>(&Bs[nxt][bRow+r][bCol]) = nb[c++];
                A += BK;  B += (size_t)BK * N;
                __syncthreads();                // 每轮只要一次 sync
                cur = nxt;
            }
        }
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&C[(size_t)(ty*TM+i)*N + tx*TN + j]) =
                    *reinterpret_cast<float4*>(&acc[i][j]);
    }
    ```

=== "v6 wmma Tensor Core（加分项）"

    前面五版全在用 CUDA Core 的 FMA。**Tensor Core 是另一套硬件**：一条 `mma` 指令由**一整个 warp 协作**完成一个 16×16×16 的小矩阵乘，吞吐是 CUDA Core 的 8~16 倍。

    `wmma` 是它的高层 C++ 接口：**fragment（片段）**是分布在 warp 32 个线程寄存器里的一块矩阵，`load_matrix_sync` / `mma_sync` / `store_matrix_sync` 都是**warp 级**操作，整个 warp 必须一起执行。

    注意：这里给的是**最简版**（直接从 global 读 fragment，没有 shared tiling），**只用来演示 API**——真正快的实现要配合 shared tile、`ldmatrix`、swizzle 和 `cp.async`，那已经是 cutlass 的范畴了。**面试能讲清"fragment 是 warp 级的、mma 一条指令干 16×16×16"就够了。**

    ```cpp
    using namespace nvcuda;
    __global__ void sgemm_v6_wmma(const half* __restrict__ A, const half* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K){
        const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;   // 每个 warp 算一个 16x16 块
        const int warpN = blockIdx.y * blockDim.y + threadIdx.y;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (int k = 0; k < K; k += 16){
            int aRow = warpM * 16, bCol = warpN * 16;
            if (aRow < M && bCol < N && k < K){
                wmma::load_matrix_sync(a_frag, A + (size_t)aRow * K + k, K);      // 第三个参数是 leading dim
                wmma::load_matrix_sync(b_frag, B + (size_t)k * N + bCol, N);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);         // D = A*B + C
            }
        }
        int cRow = warpM * 16, cCol = warpN * 16;
        if (cRow < M && cCol < N)
            wmma::store_matrix_sync(C + (size_t)cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
    ```

</div>
