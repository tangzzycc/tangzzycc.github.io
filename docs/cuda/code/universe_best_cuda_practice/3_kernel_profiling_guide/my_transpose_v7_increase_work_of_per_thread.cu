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

template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v7(float *input, float *output, int M, int N)
{
    const int STEP = BLOCK_SZ / NUM_PER_THREAD;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int x = blockDim.x * blockIdx.x + tx;
    int y = blockDim.y * blockIdx.y + ty;
    float *input_start = input + by * BLOCK_SZ * N + bx * BLOCK_SZ;

    __shared__ float smem[BLOCK_SZ][BLOCK_SZ + 1];
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        if (x + i * STEP < N && y < M)
            smem[tx + i * STEP][ty] = input_start[ty * N + tx + i * STEP];
    }

    x = blockDim.y * blockIdx.y + tx;
    y = blockDim.x * blockIdx.x + ty;
    __syncthreads();

    float *output_start = output + M * bx * BLOCK_SZ + by * BLOCK_SZ;
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        if (x + i * STEP < M && y < N)
            output_start[ty * M + tx + i * STEP] = smem[ty][tx + i * STEP];
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
    const int MATRIX_M = 2300;
    const int MATRIX_N = 1500;

    // const int MATRIX_M = 2048;
    // const int MATRIX_N = 1024;

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
        constexpr int BLOCK_SZ = 32;
        constexpr int NUM_PER_THREAD = 4;
        dim3 block(BLOCK_SZ / NUM_PER_THREAD, BLOCK_SZ);
        dim3 grid((MATRIX_N + BLOCK_SZ - 1) / BLOCK_SZ, (MATRIX_M + BLOCK_SZ - 1) / BLOCK_SZ);
        mat_transpose_kernel_v7<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(input_device, output_device, MATRIX_M, MATRIX_N);
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