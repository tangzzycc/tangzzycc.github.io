// GFX11 Wave32: one workgroup of 32 threads computes one 16x16 output tile.
// A[M,K], B[K,N], C[M,N] are row-major; FP16 inputs/output, FP32 accumulation.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
static_assert(__AMDGCN_WAVEFRONT_SIZE == 32, "WMMA requires Wave32 here");
#endif

__global__ void hgemm_kernel(const __half* __restrict__ a,
                             const __half* __restrict__ b,
                             __half* __restrict__ c, int m, int n, int k) {
    const int lane = threadIdx.x; // Exactly 32 threads in a one-dimensional block.
    const int t = lane % 16;
    const int row0 = blockIdx.y * 16;
    const int col0 = blockIdx.x * 16;
    float8 accum = {};

    for (int k0 = 0; k0 < k; k0 += 16) {
        half16 a_frag, b_frag;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int p = k0 + e;
            a_frag[e] = (row0 + t < m && p < k)
                ? static_cast<_Float16>(a[static_cast<size_t>(row0 + t) * k + p])
                : static_cast<_Float16>(0);
            b_frag[e] = (p < k && col0 + t < n)
                ? static_cast<_Float16>(b[static_cast<size_t>(p) * n + col0 + t])
                : static_cast<_Float16>(0);
        }
        accum = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, accum);
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int row = row0 + 2 * e + lane / 16;
        const int col = col0 + t;
        if (row < m && col < n)
            c[static_cast<size_t>(row) * n + col] = __float2half(accum[e]);
    }
}
