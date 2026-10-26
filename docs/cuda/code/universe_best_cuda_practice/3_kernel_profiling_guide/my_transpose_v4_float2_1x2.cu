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
            return false;
        }
    }
    return true;
}

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
template <
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X> // width of block of C that each thread calculate
__global__ void transpose_float2_inner_1x2(float *input,
                                           float *output, const int M, const int N)
{
    float src_transpose[2];
    float dst_transpose[2];
    float *input_start = input + N * blockIdx.y * THREAD_SIZE_Y + blockIdx.x * THREAD_SIZE_X;

    src_transpose[0] = input_start[2 * threadIdx.y * N + threadIdx.x];
    src_transpose[1] = input_start[(2 * threadIdx.y + 1) * N + threadIdx.x];

    FETCH_FLOAT2(dst_transpose[0]) = make_float2(src_transpose[0], src_transpose[1]);

    float *output_start = output + M * blockIdx.x * THREAD_SIZE_X + blockIdx.y * THREAD_SIZE_Y;

    FETCH_FLOAT2(output_start[threadIdx.x * M + threadIdx.y * 2]) = FETCH_FLOAT2(dst_transpose[0]);
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
    const int MATRIX_M = 2048;
    const int MATRIX_N = 512;
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
        Perf perf("transpose_8_32");
        dim3 block_size(8, 32);
        dim3 grid_size((MATRIX_N - 1) / block_size.x + 1, ((MATRIX_M >> 1) - 1) / block_size.y + 1);
        constexpr int THREAD_SIZE_Y = 32 * 2;
        constexpr int THREAD_SIZE_X = 8;
        transpose_float2_inner_1x2<THREAD_SIZE_Y, THREAD_SIZE_X><<<grid_size, block_size>>>(input_device, output_device, MATRIX_M, MATRIX_N);
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