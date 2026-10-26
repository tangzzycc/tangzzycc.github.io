

#include <iostream>
#include <cuda_runtime.h>

#include "common/tester.h"
#include "common/common.h"

using namespace nvcuda;

__device__ __forceinline__ void ld_st_128bit(void *dst, void *src)
{
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

#define REG(val) (*reinterpret_cast<uint32_t *>(&(val)))
#define HALF2(val) (*reinterpret_cast<half2 *>(&val))

__device__ __forceinline__ void ldmatrix_sync(half *dst, void *addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void ldmatrix_trans_sync(half *dst, void *addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(REG(dst[0])),
          "=r"(REG(dst[2])),
          "=r"(REG(dst[4])),
          "=r"(REG(dst[6]))
        : "l"(__cvta_generic_to_shared(addr)));
}

__device__ __forceinline__ void stmatrix_sync(half *dst, half *src)
{
    // ! Ampere doesn't have stmatrix.sync, we should simulate it
    uint64_t private_addr = (uint64_t)dst;
    uint64_t shared_addr[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        shared_addr[i] =
            __shfl_sync(0xFFFFFFFF, private_addr, i * 8 + threadIdx.x / 4);
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        *(reinterpret_cast<half2 *>(shared_addr[i]) + threadIdx.x % 4) =
            HALF2(src[2 * i]);
    }
}
/**
 * \tparam S: SShift, right shift the addr for swizzling
 * \tparam B: BShift, bits to be swizzled
 * \tparam M: MBase, bits keep the same
 */
template <uint32_t S, uint32_t B, uint32_t M>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr)
{
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}

__global__ void shared_memory_mmma_swizzle_kernel(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];
    // swizzle load A and B
    int tx = threadIdx.x;
    // each thread load 8 bytes, so tx * 8 is the offset
    uint32_t gAddr = tx * 8;
    auto g2sAddr = swizzle<3, 1, 3>(gAddr);
    ld_st_128bit(smem_a + g2sAddr, A + gAddr);
    ld_st_128bit(smem_b + g2sAddr, B + gAddr);
    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);

    // swizzle load frag a and b
    uint32_t rAddr = (tx % 16) * 16 + (tx / 16) * 8;
    auto r2sAddr = swizzle<3, 1, 3>(rAddr);

    ldmatrix_sync(a_frag.x, smem_a + r2sAddr);
    ldmatrix_trans_sync(b_frag.x, smem_b + r2sAddr);

    // calc and store
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store can also be swizzle, but we are interested in LDSM only
    // store_matrix_sync(smem_c, c_frag, 16, mem_row_major);
    // __syncthreads();
    // ld_st_128bit(c + 8 * threadIdx.x, smem_c + 8 * threadIdx.x);

    stmatrix_sync(smem_c + r2sAddr, c_frag.x);
    ld_st_128bit(C + gAddr, smem_c + g2sAddr);
}

void shared_memory_mma_swizzle(half *A, half *B, half *C, int M, int N, int K)
{
    dim3 block(32);
    dim3 grid(1);

    shared_memory_mmma_swizzle_kernel<<<grid, block>>>(A, B, C);
}

int main(int argc, char *argv[])
{
    Tester tester(16, 16, 16, 1, 10, 100, false);
    tester.evaluate(shared_memory_mma_swizzle, "shared_memory_mma");
    return 0;
}