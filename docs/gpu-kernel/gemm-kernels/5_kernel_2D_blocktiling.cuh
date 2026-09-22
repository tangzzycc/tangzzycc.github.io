#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // 每个线程负责计算 Block tile 中的 TM*TN 个元素
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // 每个 Block 的结果数 / 每个线程的结果数 = 每个 Block 的线程数
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN 表示列方向上的线程数
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 在 Shared Memory 中为当前 Block tile 分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 将指针移动到 A 对应行与 B 对应列的起始位置
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 计算当前线程加载到 Shared Memory 的索引
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // 计算一个 Block 在单次加载中搬运的 As 行数
  // 该步长由同一 Block 内的线程数决定
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // 对 As 和 Bs，都让每次加载覆盖完整的列宽，
  // 以获得更好的 Global Memory 合并访问；若覆盖完整行宽并
  // 沿列迭代，访问模式会更差
  const uint strideB = numThreadsBlocktile / BN;

  // 在寄存器文件中为每个线程分配结果缓存
  float threadResults[TM * TN] = {0.0};
  // 为 As 和 Bs 分配寄存器缓存
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 遍历 K 维 Block tile 的最外层循环
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 协作填充 Shared Memory 缓存
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // 将指针推进到下一个 Block tile
    A += BK;     // 向右移动 BK 列
    B += BK * N; // 向下移动 BK 行

    // 计算当前线程负责的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 将 Shared Memory 数据加载到寄存器
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // 写回计算结果
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
