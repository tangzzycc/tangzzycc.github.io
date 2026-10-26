

#include <iostream>
#include <cuda_runtime.h>

#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

__device__ __forceinline__ void ld_st_128bit(void *dst, void *src)
{
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

__global__ void shared_memory_wmma_kernel(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];

    int tx = threadIdx.x;
    ld_st_128bit(smem_a + 8 * tx, A + 8 * tx);
    ld_st_128bit(smem_b + 8 * tx, B + 8 * tx);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(smem_c, c_frag, 16, wmma::mem_row_major);

    // sync threads not necessary when only 1 warp, but we will generalize it in
    // the future, so just keep it here
    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

void shared_memory_wmma(half *A, half *B, half *C, int M, int N, int K)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    dim3 block(32);
    dim3 grid(1);

    shared_memory_wmma_kernel<<<grid, block>>>(A, B, C);
}

int main(int argc, char *argv[])
{
    Tester tester(16, 16, 16, 1, 10, 100, true);
    tester.evaluate(shared_memory_wmma, "shared_memory_wmma");
    return 0;
}