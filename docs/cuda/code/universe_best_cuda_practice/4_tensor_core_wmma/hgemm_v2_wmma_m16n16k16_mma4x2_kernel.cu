
#include <iostream>
#include <cuda_runtime.h>

#include "common/tester.h"
#include "common/common.h"

#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])

using namespace nvcuda;

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(
    half *A, half *B, half *C, int M, int N, int K)
{
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M;      // 16x4=64
    constexpr int BN = WMMA_N * WMMA_TILE_N;      // 16x2=32
    constexpr int BK = WMMA_K;                    // 16
    __shared__ half s_a[BM][BK], s_b[WMMA_K][BN]; // 64x16x2=2KB, 16x32x2=1KB

    // 要保证相同的warp下thread执行相同的指令
    // warp_id 0 -> warp_m 0, warp_n 0
    // warp_id 1 -> warp_m 0, warp_n 1
    // warp_id 2 -> warp_m 1, warp_n 0
    // warp_id 3 -> warp_m 1, warp_n 1
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 256线程分别load s_a=64x16, s_b=16x32
    // 64*16/256=4, half4, 16x32/256=2, half2
    // s_a, 64*16, 每个线程load 4 half, 每行需要4线程，64行，共256线程
    const int load_smem_a_m = tid / 4;       // 0~63
    const int load_smem_a_k = (tid % 4) * 4; // 0,4,12,...
    // s_b, 16x32, 每个线程load 2 half, 每行需要8线程，32行，共256线程
    const int load_smem_b_k = tid / 16;                // 0~16
    const int load_smem_b_n = (tid % 16) * 2;          // 0,2,4,...,32
    const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
    const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n

    if (load_gmem_a_m >= M && load_gmem_b_n >= N)
        return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // 64 bits sync memory issues gmem_a -> smem_a.
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST64BITS(A[load_gmem_a_addr]));
        // 32 bits sync memory issues gmem_b -> smem_b.
        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST32BITS(B[load_gmem_b_addr]));
        __syncthreads();

        wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK); // BM*BK, BK=WMMA_K
        wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN); // BK=BN, BK=WMMA_K

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N,
                            wmma::mem_row_major);
}

void hgemm_wmma_m16n16k16_mma4x2(half *A, half *B, half *C, int M, int N, int K)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[])
{
    Tester tester(512, 2048, 1024, 1, 10, 100, false);
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");
    return 0;
}