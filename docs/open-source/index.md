# Kernel PR 深读

这里记录对 GPU kernel、量化、GEMM、Triton 和编译器 dispatch 相关 PR 的源码复盘。
每篇笔记都固定到对应 PR 的公开 head，先说明数学与 tensor layout，再沿 launch、线程映射、
片上存储、边界条件和测试路径展开。性能数字只保留 PR 或源码能支撑的范围，不把局部 benchmark
直接等同于端到端吞吐。

## Comfy-Kitchen

- [PR #127：Rowwise INT8 向量化 Kernel](Comfy-Kitchen-127.md)
  - 16-byte typed load、寄存器展开，以及为什么 reduction 本身没有改变。
- [PR #129：SVDQuant INT4 Quantizer 与 W4A4 GEMM](Comfy-Kitchen-129-SVDQuant-INT4-Quantizer.md)
  - 16-lane activation quantization、nibble packing、WMMA LoRA-down 与 epilogue。
- [PR #134：AWQ W4A16 的 GEMV、WMMA 与三级 Dispatch](Comfy-Kitchen-134-AWQ-W4A16-WMMA.md)
  - small-M GEMV、medium-M fused WMMA、large-M dequant + library GEMM 的路径选择。
- [PR #157：Triton Stochastic FP8 Rounding](Comfy-Kitchen-157-Triton-Stochastic-FP8-Rounding.md)
  - 单 pass 随机舍入、FP8 编码、RNG storage 复用和 Triton registry。

## FlyDSL

- [PR #1057：RDNA3 GEMM Kernel 与 gfx1151 Tile Policy](FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)
  - gfx1151 的 tile heuristic、workgroup 数、LDS layout 与 architecture cache key。
- [PR #1113：Fused RoPE + KV Cache 的 Head-Dim 96 修复](FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)
  - 合法 vector width、非 2 次幂 half-span 的 peer lane 映射和 D=96 正确性。

## 阅读入口

建议先读 PR #127，建立 rowwise quantization 与 GPU 线程/内存映射的基本概念；再按主题继续：

- 量化与低比特矩阵计算：PR #129、PR #134、PR #157
- RDNA3/FlyDSL kernel 设计：PR #1057、PR #1113

所有来源链接、commit head、核心文件和测试入口都保留在对应文档开头，便于回到原始代码核对。
