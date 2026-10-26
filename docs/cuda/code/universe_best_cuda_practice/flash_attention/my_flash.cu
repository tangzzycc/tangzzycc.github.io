#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_forward_kernel(const float *Q, const float *K, const float *V, const int N, const int d,
                                  const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale, float *L, float *M,
                                  float *O)
{
    const int thread_num_per_block = blockDim.x * blockDim.y;
    int tid = threadIdx.x;
    int bid = gridDim.x * blockIdx.y + blockIdx.x;

    const float *Q_start = Q + bid * d * N;
    const float *K_start = K + bid * d * N;
    const float *V_start = V + bid * d * N;
    float *O_start = O + bid * d * N;

    extern __shared__ float smem[];
    int offset = 0;
    float *Qs = &smem[offset];
    offset += d * Br;
    float *Ks = &smem[offset];
    offset += d * Bc;
    float *Vs = &smem[offset];
    offset += d * Bc;
    float *Ss = &smem[offset];

    for (int i = 0; i < Tc; i++)
    {
        // K V -> smem
        for (int j = 0; j < d * Bc; j += thread_num_per_block)
        {
            Ks[j + tid] = K_start[i * d * Bc + j + tid];
            Vs[j + tid] = V_start[i * d * Bc + j + tid];
        }
        __syncthreads();

        for (int j = 0; j < Tr; j++)
        {
            // Q -> smem
            for (int k = 0; k < d * Br; k += thread_num_per_block)
            {
                Qs[k + tid] = Q_start[j * d * Br + k + tid];
            }
            __syncthreads();
            float row_m = -INFINITY;
            // float row_m_prev = -INFINITY;
            for (int l = 0; l < Bc; l++)
            {
                for (int m = 0; m < d; m++)
                {
                    Ss[tid * Bc + l] += Qs[tid * d + m] * Ks[tid * d + m];
                }
                Ss[tid * Bc + l] *= softmax_scale;
                if (Ss[tid * Bc + l] > row_m)
                    row_m = Ss[tid * Bc + l];
            }
            float row_l = 0;
            // float row_l_prev = 0;
            for (int k = 0; k < Bc; k++)
            {
                Ss[tid * Bc + k] = __expf(Ss[tid * Bc + k] - row_m);
                row_l += Ss[tid * Bc + k];
            }

            float row_m_prev = M[(Br * j) + tid];
            float row_l_prev = L[(Br * j) + tid];

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int k = 0; k < d; k++)
            {
                float pv = 0;
                for (int l = 0; l < Bc; l++)
                {
                    pv += Ss[(Br * tid) + l] * Vs[(l * d) + k];
                }
                O_start[j * Br + tid * Br + k] =
                    (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[j * Br + tid * Br + k]) + (__expf(row_m - row_m_new) * pv));
            }
            M[tid + j * Br] = row_m_new;
            L[tid + j * Br] = row_l_new;
        }
        // __syncthreads(); // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor my_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});
    auto M = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    L = L.to(device);
    M = M.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Br * d * sizeof(float)) + (Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Bc);   // Bc threads per block

    my_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, L.data_ptr<float>(), M.data_ptr<float>(), O.data_ptr<float>());
    return O;
}