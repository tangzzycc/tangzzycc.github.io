#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // 若在这里交换 x 和 y，大矩阵上的性能会下降约 30%。
  // 当前配置使 Block ID 连续的 Block 顺序访问 B 的各列，
  // 同时复用 A 的同一行，因此性能更高。
  // 较慢的配置会复用 A 的列，但对 B 的访问并不连续。
  // 当前配置具有更好的空间局部性，
  // 因而能获得更高的 L2 命中率。
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 每个 Warp 计算 32*TM 个元素，其中 32 对应列方向
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // 在 Shared Memory 中为当前 Block tile 分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 将指针移动到 A 对应行与 B 对应列的起始位置
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // TODO：调整为每个线程加载多个元素，
  // 以更充分地利用缓存容量
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // 在 Warp 层面合并 Global Memory 访问
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // 在 Warp 层面合并 Global Memory 访问
  const uint innerRowB = threadIdx.x / BN;

  // 在寄存器文件中为每个线程分配结果缓存
  float threadResults[TM] = {0.0};

  // 遍历 K 维上的 Block tile
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 协作填充 Shared Memory 缓存
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // 将指针推进到下一个 Block tile
    A += BK;
    B += BK * N;

    // 计算当前线程负责的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 将点积循环放在外层，便于复用 Bs 元素，
      // 并将其缓存在临时变量中
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // 写回计算结果
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
