# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Adapted for the notebook from ROCm/FlyDSL at
# v0.5.0:
# tests/kernels/test_rdna3_wmma_atom.py::_compile_tiled_wmma.
# Changes: fixed 64x64x32 FP16 case, M-first wave order, and standalone checks.

"""RDNA 3 / 3.5: a register-only tiled WMMA, C = A @ B.T."""

import flydsl  # Preload the compiler runtime before importing torch.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
import torch

M, N, K = 64, 64, 32


@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, K), (K, 1)))
    bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, K), (K, 1)))
    bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Float16, fx.Float32))
    tiled_mma = fx.make_tiled_mma(atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.Float16)
    copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    copy_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
    copy_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
    copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

    fx.copy(copy_ab, copy_A.partition_S(bA), copy_A.retile(frag_A))
    fx.copy(copy_ab, copy_B.partition_S(bB), copy_B.retile(frag_B))
    frag_C.fill(0)
    for ki in fx.range_constexpr(K // 16):
        fx.gemm(tiled_mma, frag_C, frag_A[None, None, ki], frag_B[None, None, ki], frag_C)
    fx.copy(copy_c, copy_C.retile(frag_C), copy_C.partition_D(bC))


@flyc.jit
def tiled_wmma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


def main():
    arch = get_rocm_arch()
    if not torch.cuda.is_available() or not arch.startswith("gfx11"):
        raise RuntimeError(f"This example requires a gfx11 RDNA 3 / 3.5 GPU; detected {arch!r}")
    torch.manual_seed(0)
    # Small integers make layout errors easy to distinguish from rounding.
    A = (torch.randn(M, K, device="cuda") * 4).round().to(torch.float16)
    B = (torch.randn(N, K, device="cuda") * 4).round().to(torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()
    tiled_wmma(A, B, C, stream=stream)  # Warm up before graph capture.
    torch.cuda.synchronize()
    reference = A.float() @ B.float().T
    torch.testing.assert_close(C, reference, atol=1e-3, rtol=1e-6)
    eager_error = (C - reference).abs().max().item()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tiled_wmma(A, B, C, stream=torch.cuda.current_stream())
    C.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(C, reference, atol=1e-3, rtol=1e-6)
    print(f"{arch}: 64x64x32, FP16 -> FP32, 4 Wave32 / 128 threads")
    print(f"Eager PASS; max absolute error = {eager_error:g}")
    print(f"Graph replay PASS; max absolute error = {(C-reference).abs().max().item():g}")


if __name__ == "__main__":
    main()
