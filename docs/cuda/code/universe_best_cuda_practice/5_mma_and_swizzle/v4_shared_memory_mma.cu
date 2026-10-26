

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

__device__ __forceinline__ void mma_sync_m16n8k16(half *c, half *a, half *b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3, %4, %5}, "
                 "{%6, %7}, "
                 "{%8, %9};"
                 : "=r"(REG(c[0])), "=r"(REG(c[2]))
                 : "r"(REG(a[0])),
                   "r"(REG(a[2])),
                   "r"(REG(a[4])),
                   "r"(REG(a[6])),
                   "r"(REG(b[0])),
                   "r"(REG(b[2])),
                   "r"(0),
                   "r"(0));
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

__global__ void shared_memory_mma_kernel(half *A, half *B, half *C)
{
    __shared__ half smem_a[16 * 16];
    __shared__ half smem_b[16 * 16];
    __shared__ half smem_c[16 * 16];

    int tx = threadIdx.x;
    uint32_t gAddr = tx * 8;
    ld_st_128bit(smem_a + gAddr, A + gAddr);
    ld_st_128bit(smem_b + gAddr, B + gAddr);
    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    fill_fragment(c_frag, 0.0f);

    uint32_t row = tx % 16;
    uint32_t col = tx / 16;
    ldmatrix_sync(a_frag.x, smem_a + row * 16 + col * 8);
    ldmatrix_trans_sync(b_frag.x, smem_b + row * 16 + col * 8);

    // 2 m16n8k16 HMMA to achieve m16n16k16 matrix multiplication
    mma_sync_m16n8k16(c_frag.x, a_frag.x, b_frag.x);
    mma_sync_m16n8k16(c_frag.x + 4, a_frag.x, b_frag.x + 4);
    // store the result back to shared memory, this can be hand coded, but we
    // are interested in LDSM now

    // store_matrix_sync(smem_c, c_frag, 16, mem_row_major);
    stmatrix_sync(smem_c + row * 16 + col * 8, c_frag.x);

    __syncthreads();
    ld_st_128bit(C + 8 * tx, smem_c + 8 * tx);
}

void shared_memory_mma(half *A, half *B, half *C, int M, int N, int K)
{
    dim3 block(32);
    dim3 grid(1);

    shared_memory_mma_kernel<<<grid, block>>>(A, B, C);
}

int main(int argc, char *argv[])
{
    Tester tester(16, 16, 16, 1, 10, 100, true);
    tester.evaluate(shared_memory_mma, "shared_memory_wmma");
    return 0;
}