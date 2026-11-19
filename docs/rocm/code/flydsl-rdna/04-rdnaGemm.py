# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Notebook driver for ROCm/FlyDSL at 605d7f5fee6bed9a1ddaf6c3e4f96e79a70484d5.
# The kernel remains in the upstream source checkout; put that checkout on PYTHONPATH.

import flydsl
import torch
from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module


def main():
    torch.manual_seed(0)
    M, N, K = 256, 256, 64
    A = (torch.randn(M, K, device="cuda") * 4).round().to(torch.float16)
    B_T = (torch.randn(N, K, device="cuda") * 4).round().to(torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    launch, bm, bn, bk = create_wmma_gemm_module(M, N, K, in_dtype="f16", out_dtype="f32")
    launch(C, A, B_T, torch.cuda.current_stream())
    torch.cuda.synchronize()
    reference = A.float() @ B_T.float().T
    torch.testing.assert_close(C, reference, atol=1e-3, rtol=1e-6)
    print(f"GEMM {M}x{N}x{K}; block tile {bm}x{bn}x{bk}; PASS")
    print(f"Max absolute error = {(C-reference).abs().max().item():g}")


if __name__ == "__main__":
    main()
