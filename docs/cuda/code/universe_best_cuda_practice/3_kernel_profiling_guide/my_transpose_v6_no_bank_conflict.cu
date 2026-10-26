#include <iostream>
#include <cuda_runtime.h>

class Perf
{
public:
    Perf(const std::string &name)
    {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }

    ~Perf()
    {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
}; // class Perf

bool check(float *cpu_result, float *gpu_result, const int M, const int N)
{
    const int size = M * N;
    for (int i = 0; i < size; i++)
    {
        if (cpu_result[i] != gpu_result[i])
        {
            printf("error index is :%d\n", i);
            return false;
        }
    }
    return true;
}
__global__ void mat_transpose_kernel_v0(const float *idata, float *odata, int M, int N)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N)
    {
        odata[x * M + y] = idata[y * N + x];
    }
}

template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v6(float *input, float *output, int M, int N)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int x = blockDim.x * blockIdx.x + tx;
    int y = blockDim.y * blockIdx.y + ty;
    float *input_start = input + N * blockIdx.y * BLOCK_SZ + blockIdx.x * BLOCK_SZ;
    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ + 1];

    if (y < M && x < N)
    {
        sdata[tx][ty] = input_start[ty * N + tx];
    }
    __syncthreads();
    x = blockDim.y * blockIdx.y + tx;
    y = blockDim.x * blockIdx.x + ty;
    float *output_start = output + M * blockIdx.x * BLOCK_SZ + blockIdx.y * BLOCK_SZ;
    if (y < N && x < M)
    {
        output_start[ty * M + tx] = sdata[ty][tx];
    }
}

void transpose_cpu(float *input, float *output, const int M, const int N)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            const int input_index = m * N + n;
            const int output_index = n * M + m;
            output[output_index] = input[input_index];
        }
    }
}

int main(int argc, char *argv[])
{
    // const int MATRIX_M = 2300;
    // const int MATRIX_N = 1500;

    const int MATRIX_M = 16;
    const int MATRIX_N = 16;

    const size_t size = MATRIX_M * MATRIX_N;

    float *input_host = (float *)malloc(size * sizeof(float));
    float *output_host_cpu_calc = (float *)malloc(size * sizeof(float));
    float *output_host_gpu_calc = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        input_host[i] = 2.0 * (float)drand48() - 1.0;
    }

    transpose_cpu(input_host, output_host_cpu_calc, MATRIX_M, MATRIX_N);
    float *input_device, *output_device;

    cudaMalloc(&input_device, size * sizeof(float));
    cudaMemcpy(input_device, input_host, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&output_device, size * sizeof(float));

    // ==================
    cudaMemset(output_device, 0, size * sizeof(float));
    for (int i = 0; i < 5; i++)
    {
        Perf perf("mat_transpose_kernel_v6");
        constexpr int BLOCK_SZ = 16;
        dim3 block(BLOCK_SZ, BLOCK_SZ);
        dim3 grid((MATRIX_N + BLOCK_SZ - 1) / BLOCK_SZ, (MATRIX_M + BLOCK_SZ - 1) / BLOCK_SZ);
        mat_transpose_kernel_v6<BLOCK_SZ><<<grid, block>>>(input_device, output_device, MATRIX_M, MATRIX_N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(output_host_gpu_calc, output_device,
               size * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(output_host_cpu_calc, output_host_gpu_calc, MATRIX_M, MATRIX_N))
    {
        std::cout << "right!" << std::endl;
    }

    return 0;
}