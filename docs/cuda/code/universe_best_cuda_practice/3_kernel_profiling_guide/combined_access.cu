#include <bits/stdc++.h>

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <ctime>
#include <sys/time.h>

// #include <cudnn.h>
#include <cublas_v2.h>

void __global__ add1(float *x, float *y, float *z)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

void __global__ add2(float *x, float *y, float *z)
{

    int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
    z[n] = x[n] + y[n];
}

void __global__ add3(float *x, float *y, float *z)
{
    int tid_permuted = threadIdx.x ^ 0x1;
    int n = tid_permuted + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

void __global__ add4(float *x, float *y, float *z)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_idx = n / 32;
    z[warp_idx] = x[warp_idx] + y[warp_idx];
}

void __global__ add5(float *x, float *y, float *z)
{
    int n = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    z[n] = x[n] + y[n];
}

void __global__ add6(float *x, float *y, float *z)
{
    if (blockIdx.x == 0 && threadIdx.x < 32)
    {
        int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
        z[n] = x[n] + y[n];
    }
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input_x = (float *)malloc(N * sizeof(float));
    float *input_y = (float *)malloc(N * sizeof(float));
    float *d_input_x;
    float *d_input_y;
    cudaMalloc((void **)&d_input_x, N * sizeof(float));
    cudaMalloc((void **)&d_input_y, N * sizeof(float));
    cudaMemcpy(d_input_x, input_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *output = (float *)malloc(N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, N * sizeof(float));

    dim3 Grid(N / 256);
    dim3 Block(64);

    for (int i = 0; i < 2; i++)
    {
        add1<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add2<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add3<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add4<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add5<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 2; i++)
    {
        add6<<<Grid, Block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }

    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);

    free(input_x);
    free(input_y);
    return 0;
}