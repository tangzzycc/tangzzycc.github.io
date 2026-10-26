#include <bits/stdc++.h>

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <ctime>
#include <sys/time.h>

// #include <cudnn.h>
#include <cublas_v2.h>

/**
 * @brief transpose with cuda
 *
 */

using namespace std;

#define PERF(name) Perf perf_##name##__COUNTER__(#name)
#define PERF_CPU(name) PerfCPU perf_CPU_##name##__COUNTER__(#name)

class PerfCPU
{
public:
    PerfCPU(const std::string &name)
    {
        m_name = name;
        gettimeofday(&m_start, NULL);
    }

    ~PerfCPU()
    {
        gettimeofday(&m_end, NULL);
        float elapsed_time = (m_end.tv_sec - m_start.tv_sec) * 1000.0 + (m_end.tv_usec - m_start.tv_usec) / 1000.0;
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    struct timeval m_start, m_end;
}; // class PerfCPU

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

const int WIDTH_BLOCK_SIZE = 32;
const int HEIGHT_BLOCK_SIZE = 8;

const int MATRIX_M = 2048;
const int MATRIX_N = 512;

/**
 * https://blog.csdn.net/LostUnravel/article/details/137613493 使用shm最好的测试效果
 */
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v3(const float *idata, float *odata, int M, int N)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ + 1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if (x < N)
    {
#pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE)
        {
            if (y + y_off < M)
            {
                sdata[ty + y_off][tx] = idata[(y + y_off) * N + x];
            }
        }
    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (x < M)
    {
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE)
        {
            if (y + y_off < N)
            {
                odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        }
    }
}

__global__ void transpose_col4row1(const float *input,
                                   float *__restrict__ output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx >= (MATRIX_N >> 2) or row_idx >= MATRIX_M)
    {
        return;
    }

    int offset = row_idx * MATRIX_N + (col_idx << 2);
    // printf("offset=%d, col_idx=%d, row_idx=%d\n", offset, col_idx, row_idx);
    const float4 *input_v2 = reinterpret_cast<const float4 *>(input + offset);

    float4 src = input_v2[0];

    __shared__ float sdata[4];

    sdata[0] = src.x;
    sdata[1] = src.y;
    sdata[2] = src.z;
    sdata[3] = src.w;

    // printf("offset=%d, src.x = %f, src.y = %f\n", offset, src.x, src.y);

    offset = ((col_idx * MATRIX_M) << 2) + row_idx;
    float *dst = output + offset;
    dst[0] = sdata[0];
    dst[MATRIX_M] = sdata[1];
    dst[MATRIX_M << 1] = sdata[2];
    dst[(MATRIX_M << 1) + MATRIX_M] = sdata[3];
}

__global__ void transpose_col2row1(const float *input,
                                   float *__restrict__ output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx >= (MATRIX_N >> 1) or row_idx >= MATRIX_M)
    {
        return;
    }

    int offset = row_idx * MATRIX_N + (col_idx << 1);
    // printf("offset=%d, col_idx=%d, row_idx=%d\n", offset, col_idx, row_idx);
    const float2 *input_v2 = reinterpret_cast<const float2 *>(input + offset);

    float2 src = input_v2[0];

    // printf("offset=%d, src.x = %f, src.y = %f\n", offset, src.x, src.y);

    offset = ((col_idx * MATRIX_M) << 1) + row_idx;
    float *dst = output + offset;
    dst[0] = src.x;
    dst[MATRIX_M] = src.y;
}

__global__ void transpose_float2(const float *input,
                                 float *__restrict__ output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx >= (MATRIX_N >> 1) or row_idx >= (MATRIX_M >> 1))
    {
        return;
    }

    int offset = (row_idx * MATRIX_N + col_idx) << 1;
    const float2 *input_v2 = reinterpret_cast<const float2 *>(input + offset);

    // 4 * 4
    float2 src_row0 = input_v2[0];
    float2 src_row1 = input_v2[MATRIX_N >> 1];

    // 将4列transpose为4行
    float2 dst_row0 = make_float2(src_row0.x, src_row1.x);
    float2 dst_row1 = make_float2(src_row0.y, src_row1.y);

    // 将4行写入到output
    offset = (col_idx * MATRIX_M + row_idx) << 1;
    float2 *dst_v2 = reinterpret_cast<float2 *>(output + offset);
    dst_v2[0] = dst_row0;
    dst_v2[MATRIX_M >> 1] = dst_row1;
}

__global__ void transpose_perfect(const float *input,
                                  float *__restrict__ output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx >= (MATRIX_N >> 2) or row_idx >= (MATRIX_M >> 2))
    {
        return;
    }

    int offset = (row_idx * MATRIX_N + col_idx) << 2; // * 4
    const float4 *input_v4 = reinterpret_cast<const float4 *>(input + offset);

    // 4 * 4
    float4 src_row0 = input_v4[0];
    float4 src_row1 = input_v4[MATRIX_N >> 2];
    float4 src_row2 = input_v4[MATRIX_N >> 1];
    float4 src_row3 = input_v4[(MATRIX_N >> 2) * 3];

    // 将4列transpose为4行
    float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    // 将4行写入到output
    offset = (col_idx * MATRIX_M + row_idx) << 2;
    float4 *dst_v4 = reinterpret_cast<float4 *>(output + offset);
    dst_v4[0] = dst_row0;
    dst_v4[MATRIX_M >> 2] = dst_row1;
    dst_v4[MATRIX_M >> 1] = dst_row2;
    dst_v4[(MATRIX_M >> 2) * 3] = dst_row3;
}

__global__ void transpose_inplace(float *input)
{
}

__global__ void transpose_naive(float *input, float *output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < MATRIX_N && row_idx < MATRIX_M)
    {
        int idx = row_idx * MATRIX_N + col_idx;
        int trans_idx = col_idx * MATRIX_M + row_idx;
        output[trans_idx] = input[idx];
    }
}

void AllocABCAndInit(float *&matrix_a)
{
    matrix_a = new float[MATRIX_M * MATRIX_N];

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int i = 0; i < MATRIX_M * MATRIX_N; ++i)
    {
        matrix_a[i] = distribution(generator);
    }
}

#define OPEN_PRINT 0

#if OPEN_PRINT
void PrettyPrint(float *matrix, int row, int col)
{
    for (int r = 0; r < row; ++r)
    {
        for (int c = 0; c < col; ++c)
        {
            std::cout << matrix[r * col + c] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
#else
void PrettyPrint(float *matrix, int row, int col) {}
#endif // OPEN_PRINT

int main(int argc, char *argv[])
{
    float *matrix, *matrix_output;
    AllocABCAndInit(matrix);

    matrix_output = new float[MATRIX_M * MATRIX_N];

    PrettyPrint(matrix, MATRIX_M, MATRIX_N);
    float *d_matrix, *d_matrix_output;
    cudaMalloc(&d_matrix, MATRIX_M * MATRIX_N * sizeof(float));
    cudaMemcpy(d_matrix, matrix, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_matrix_output, MATRIX_M * MATRIX_N * sizeof(float));

    {
        Perf perf("transpose");
        dim3 block_size(WIDTH_BLOCK_SIZE, HEIGHT_BLOCK_SIZE);
        dim3 grid_size((MATRIX_N - 1) / block_size.x + 1, (MATRIX_M - 1) / block_size.y + 1);
        transpose_naive<<<grid_size, block_size>>>(d_matrix, d_matrix_output);
        cudaDeviceSynchronize();
        cudaMemcpy(matrix_output, d_matrix_output,
                   MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    PrettyPrint(matrix_output, MATRIX_N, MATRIX_M);

    float *perfect_output;
    cudaMalloc(&perfect_output, MATRIX_M * MATRIX_N * sizeof(float));
    float *cpu_perfect = new float[MATRIX_M * MATRIX_N];

    {

        Perf perf("transpose_perfect");
        dim3 block_size(WIDTH_BLOCK_SIZE, HEIGHT_BLOCK_SIZE);
        dim3 grid_size(((MATRIX_N >> 2) - 1) / block_size.x + 1, (MATRIX_M >> 2 - 1) / block_size.y + 1);
        std::cout << "grid_size: " << grid_size.x << " " << grid_size.y << std::endl;
        transpose_perfect<<<grid_size, block_size>>>(d_matrix, perfect_output);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_perfect, perfect_output,
                   MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    std::cout << "matrix:" << std::endl;
    std::cout << std::endl;
    PrettyPrint(matrix_output, MATRIX_N, MATRIX_M);

    float *floatv2_output;
    cudaMalloc(&floatv2_output, MATRIX_M * MATRIX_N * sizeof(float));
    float *cpu_floatv2 = new float[MATRIX_M * MATRIX_N];
    {
        Perf perf("transpose_floatv2");
        dim3 block_size(WIDTH_BLOCK_SIZE, HEIGHT_BLOCK_SIZE);
        dim3 grid_size(((MATRIX_N >> 1) - 1) / block_size.x + 1, (MATRIX_M >> 1 - 1) / block_size.y + 1);
        transpose_float2<<<grid_size, block_size>>>(d_matrix, floatv2_output);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_floatv2, floatv2_output,
                   MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float *float_col4row1;
    cudaMalloc(&float_col4row1, MATRIX_M * MATRIX_N * sizeof(float));
    float *cpu_float_col4row1 = new float[MATRIX_M * MATRIX_N];
    {
        Perf perf("transpose_float_col4row1");
        dim3 block_size(WIDTH_BLOCK_SIZE, HEIGHT_BLOCK_SIZE);
        int grid_x = ((MATRIX_N >> 2) - 1) / block_size.x + 1;
        int grid_y = (MATRIX_M - 1) / block_size.y + 1;
        dim3 grid_size;
        grid_size.x = grid_x;
        grid_size.y = grid_y;
        std::cout << "transpose float MATRIX_N: " << MATRIX_N << ", MATRIX_M: " << MATRIX_M << std::endl;
        std::cout << " transpose_float_col2row1, grid_size: " << grid_x << " " << grid_y << std::endl;
        transpose_col4row1<<<grid_size, block_size>>>(d_matrix, float_col4row1);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_float_col4row1, float_col4row1,
                   MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    std::cout << std::endl;
    PrettyPrint(cpu_float_col4row1, MATRIX_N, MATRIX_M);

    float *float_col2row1;
    cudaMalloc(&float_col2row1, MATRIX_M * MATRIX_N * sizeof(float));
    float *cpu_float_col2row1 = new float[MATRIX_M * MATRIX_N];
    {
        Perf perf("transpose_float_col2row1");
        dim3 block_size(WIDTH_BLOCK_SIZE, HEIGHT_BLOCK_SIZE);
        int grid_x = ((MATRIX_N >> 1) - 1) / block_size.x + 1;
        int grid_y = (MATRIX_M - 1) / block_size.y + 1;
        dim3 grid_size;
        grid_size.x = grid_x;
        grid_size.y = grid_y;
        std::cout << "transpose float MATRIX_N: " << MATRIX_N << ", MATRIX_M: " << MATRIX_M << std::endl;
        std::cout << " transpose_float_col2row1, grid_size: " << grid_x << " " << grid_y << std::endl;
        transpose_col2row1<<<grid_size, block_size>>>(d_matrix, float_col2row1);
        cudaDeviceSynchronize();
        cudaMemcpy(cpu_float_col2row1, float_col2row1,
                   MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }
    std::cout << std::endl;
    PrettyPrint(cpu_float_col2row1, MATRIX_N, MATRIX_M);

    float *cpu_csdn_shm = new float[MATRIX_M * MATRIX_N];
    memset(cpu_csdn_shm, 0, MATRIX_M * MATRIX_N * sizeof(float));
    // cpu_csdn_shm = perfect_output;
    {
        float *csdn_shm;
        cudaMalloc(&csdn_shm, MATRIX_M * MATRIX_N * sizeof(float));
        cudaMemset(csdn_shm, 0, MATRIX_M * MATRIX_N * sizeof(float));
        {
            Perf perf("transpose_csdn_shm");
            constexpr int BLOCK_SZ = 32;
            constexpr int NUM_PER_THREAD = 4;

            dim3 block_size(BLOCK_SZ, BLOCK_SZ / NUM_PER_THREAD);
            dim3 grid_size((MATRIX_N - 1) / block_size.x + 1, (MATRIX_M / NUM_PER_THREAD - 1) / block_size.y + 1);
            mat_transpose_kernel_v3<BLOCK_SZ, NUM_PER_THREAD>
                <<<grid_size, block_size>>>(d_matrix, csdn_shm, MATRIX_M, MATRIX_N);
            cudaDeviceSynchronize();
            cudaMemcpy(cpu_csdn_shm, csdn_shm,
                       MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    int print_count = 64;
    for (int i = 0; i < MATRIX_M * MATRIX_N; ++i)
    {
        if (std::abs(cpu_csdn_shm[i] - matrix_output[i]) > 10e-3)
        {
            std::cout << "Error: cpu_csdn_shm[" << i << "] = " << cpu_csdn_shm[i] << ", matrix_output[" << i << "] = " << matrix[i] << ", diff: " << cpu_csdn_shm[i] - matrix_output[i] << std::endl;
        }
    }
    return 0;
}