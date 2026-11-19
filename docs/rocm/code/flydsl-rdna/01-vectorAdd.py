# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Adapted from the scalar vectorAdd example discussed at
# https://jinnpan.com/sources/flydsl.html, using the current FlyDSL expression API.
# Changes: complete the omitted B/C partitions, fix graph stream selection,
# and limit this teaching example to complete 64-element blocks.

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


@flyc.kernel
def vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, block_dim: fx.Constexpr[int]):
    bid, tid = fx.block_idx.x, fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)

    def thread_tiles(t):
        t = fx.logical_divide(t, fx.make_layout(block_dim, 1))
        t = fx.slice(t, (None, bid))
        return fx.logical_divide(t, fx.make_layout(1, 1))

    tA, tB, tC = thread_tiles(A), thread_tiles(B), thread_tiles(C)
    buffer_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    universal_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.copy_atom_call(buffer_atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(universal_atom, fx.slice(tB, (None, tid)), rB)
    rC.store(rA.load() + rB.load())
    fx.copy_atom_call(universal_atom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, n: fx.Int32,
               block_dim: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    grid_x = (n + block_dim - 1) // block_dim
    vector_add_kernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


def main():
    torch.manual_seed(0)
    A = torch.randn(128, device="cuda", dtype=torch.float32)
    B = torch.randn_like(A)
    C = torch.empty_like(A)
    vector_add(A, B, C, 128, 64, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A + B)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        vector_add(A, B, C, 128, 64, stream=torch.cuda.current_stream())
    C.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(C, A + B)
    print("Eager PASS; Graph replay PASS")


if __name__ == "__main__":
    main()
