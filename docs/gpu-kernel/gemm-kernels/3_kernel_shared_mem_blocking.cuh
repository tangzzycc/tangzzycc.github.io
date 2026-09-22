#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // 当前 Thread Block 负责计算的输出块坐标
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // 在高速 Shared Memory 中为当前块分配缓冲区
  // 同一 Block 内的所有线程共享这部分内存
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // 当前线程在 tile 内访问的行列坐标
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // 将指针移动到当前 Block tile 的起始位置
  A += cRow * BLOCKSIZE * K;                    // 行=cRow，列=0
  B += cCol * BLOCKSIZE;                        // 行=0，列=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // 行=cRow，列=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // 每个线程分别从 A 和 B 加载一个元素
    // 让 threadCol（即 threadIdx.x）作为连续变化的索引，
    // 以便合并 Global Memory 访问
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // 等待 Block 内所有线程完成 Shared Memory 填充
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // 在当前缓存的 tile 上执行点积
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // 结束时再次同步，防止较快的线程在较慢线程完成计算前
    // 就把下一个 tile 写入缓存
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
