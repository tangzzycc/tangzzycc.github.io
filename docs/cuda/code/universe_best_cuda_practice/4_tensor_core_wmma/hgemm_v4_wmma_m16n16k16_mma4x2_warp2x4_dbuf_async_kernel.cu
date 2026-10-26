
#include <iostream>
#include <cuda_runtime.h>

#include "common/tester.h"
#include "common/common.h"

#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

using namespace nvcuda;

// Double buffers
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int OFFSET = 0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(
    half *A, half *B, half *C, int M, int N, int K)
{
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 16
    // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
    __shared__ half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET];

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                   wmma::row_major>
        B_frag[WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // k = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[0][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[0][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

#pragma unroll
    for (int k = 1; k < NUM_K_TILES; ++k)
    {                               // start from 1
        int smem_sel = (k - 1) & 1; // k 1->0, k 2->1, k 3->0, ...
        int smem_sel_next = k & 1;  // k 1->1, k 2->0, k 3->1, ...

        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); //!!!!
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    // processing last k tile
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j)
            {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j)
        {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(half *A, half *B, half *C, int M, int N, int K)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;

    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;

    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[])
{
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async");
    return 0;
}