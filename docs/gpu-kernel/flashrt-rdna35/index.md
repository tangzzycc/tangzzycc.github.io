# FlashRT RDNA 3.5 Kernel 源码精读

[FlashRT PR #213](https://github.com/flashrt-project/FlashRT/pull/213) 为 Pi0.5 增加了面向 `gfx1151` 的 BF16 推理后端。`open-source/FlashRT-213` 中的笔记关注 PR 的模型、Frontend 和端到端数据流；这个子系列只关注其中的 RDNA HIP Kernel，目标是能从源码中还原每个 Kernel 的 Thread Ownership、Memory Access、Reduction、Synchronization 和 BF16 Rounding Boundary。

- PR 合并后固定版本：[`e2cf5d1`](https://github.com/flashrt-project/FlashRT/commit/e2cf5d1fe7209e3f4fbf212c5b6321467e8f4874)
- PR 基线：[`dd549e2`](https://github.com/flashrt-project/FlashRT/commit/dd549e2)
- 目标硬件：AMD RDNA 3.5，`gfx1151`，Wave32
- 数值路径：BF16 Input/Output，FP32 Accumulation

> 正文代码块中的中文注释由本系列补充，用于标记 Ownership、索引单位、数据生命周期与同步边界，不属于上游源码原注释。未加说明的代码仍对应固定提交中的实现；代码块中若出现 `...`，表示为了聚焦当前机制而明确省略的源码。

## 系列阅读顺序

| 章节 | 回答的核心问题 |
| --- | --- |
| [00 · AMD GPU 硬件模型与 Kernel 阅读术语](00-AMD-GPU硬件模型与术语.md) | Grid、Workgroup、Wavefront、Lane 如何映射到 WGP、CU、SIMD，VGPR/SGPR/LDS/Cache 又分别由谁拥有？ |
| [01 · HIP 执行模型与 Wave32 归约](01-HIP执行模型与Wave32归约.md) | 如何从 `grid/block/thread` 还原数据所有权，`__shfl_*`、LDS 和 `__syncthreads()` 如何组成 Block Reduction？ |
| [02 · Activation 与 Residual Kernel](02-Activation与Residual-Kernel.md) | Scalar/Packed 路径如何分流，Merged GeGLU 和 Gated Residual 如何对应内存布局？ |
| [03 · Normalization Kernel](03-Normalization-Kernel.md) | 8 个 Norm Kernel 怎样完成行内归约，固定宽度为什么能用 Register 换掉重读？ |
| [04 · QKV Split 与 RoPE Kernel](04-QKV-Split与RoPE-Kernel.md) | 一个 Thread 如何拥有一个 Complex Pair，并复用一组 Cos/Sin 处理 8Q/1K/1V？ |
| [05 · Encoder FlashAttention Kernel](05-Encoder-FlashAttention-Kernel.md) | 两个 Wave 如何协作完成 `16×16` QK、Online Softmax 和 `P×V`？ |
| [06 · Decoder GQA Kernel](06-Decoder-GQA-Kernel.md) | Split-key 如何将 KV 分给 8 Waves，为什么还保留一条 Row-owned 实现？ |
| [07 · Small-M WMMA GEMM Kernel](07-SmallM-WMMA-GEMM-Kernel.md) | Narrow-N Split-K 与 Wider-N LDS Tiling 如何共用 WMMA Mapping，又如何与 hipBLASLt 分工？ |

## PR #213 的 Kernel 清单

PR 新增了 24 个 `__global__` Kernel body。这里按源码定义计数；Template 可能在编译时实例化出多个不同的 Machine Kernel。

| 实现族 | `__global__` Kernel | 数量 |
| --- | --- | ---: |
| Activation | `gelu_rdna_kernel`、`gelu_packed_rdna_kernel`、`gelu_mul_rdna_kernel`、`gelu_mul_packed_rdna_kernel`、`gelu_mul_merged_rdna_kernel`、`gelu_mul_merged_packed_rdna_kernel`、`silu_rdna_kernel`、`silu_packed_rdna_kernel` | 8 |
| Residual | `residual_scalar_rdna_kernel`、`residual_rdna_kernel<HasGate>` | 2 |
| Normalization | `layer_norm_wide_rdna_kernel`、`rms_norm_wide_rdna_kernel`、`residual_rms_wide_rdna_kernel`、`layer_norm_rdna_kernel`、`rms_norm_rdna_kernel`、`adarms_rdna_kernel`、`residual_rms_rdna_kernel`、`residual_adarms_rdna_kernel` | 8 |
| RoPE | `qkv_rope_rdna_kernel` | 1 |
| Encoder Attention | `attention_encoder_gqa_rdna_kernel` | 1 |
| Decoder Attention | `attention_decoder_gqa_splitkey_rdna_kernel`、`attention_decoder_gqa_rdna_kernel` | 2 |
| Small-M GEMM | `smallm_wmma_bf16_tiled_rdna_kernel`、`smallm_wmma_bf16_rdna_kernel` | 2 |

`hipblaslt_runner_rdna.hip` 不定义 `__global__` Function，而是通过 hipBLASLt 调用 Library Kernel。它仍然会在第 07 章中讲解，因为只有明确通用 GEMM 与手写 Small-M Kernel 的分界，才能理解 PR 为什么不用 WMMA Kernel 接管所有 Linear。

## 每个 Kernel 都用同一组问题阅读

后续章节不会只把 C++ 代码改写成中文。对每个 Kernel，都会明确回答：

1. 输入、输出和 Shape Contract 是什么。
2. `gridDim`、`blockDim`、`blockIdx`、`threadIdx`、Wave 和 Lane 分别表示什么。
3. 一个 Thread/Wave/Workgroup 拥有哪些 Value，覆盖是否无重叠且无遗漏。
4. 每次 Global Memory 或 LDS 访问有多宽，是否连续、对齐、需要 Tail Guard。
5. 哪些 Value 位于 Register，哪些位于 LDS，什么时候必须 Barrier。
6. FP32 Accumulation 和 BF16 Cast 发生在哪里，融合前后是否保留同一 Rounding Boundary。
7. Host Launcher 与 Python Provider 在什么条件下选择它，什么时候走 Fallback。

每篇正文会先列出本章覆盖的 `__global__` Kernel、相关 Host Launcher/Helper 和推荐阅读入口，再把固定提交中的关键 Kernel Body 直接放进笔记，贴着源码解释实现。源码阅读顺序默认是 **Launch 与 Shape Contract → Index/Ownership → GMEM/LDS/Register Dataflow → Barrier/Reduction → Epilogue/Rounding → Provider Dispatch**；这样不需要在笔记和仓库之间频繁切换，也能解释 Kernel 为何只在特定 Shape 下出现。

这种读法会把“算子的数学定义”与“GPU 上的物理执行”分开，再通过数据所有权把两者接回来。比如 `GELU(x)` 的数学式没有 Thread，但 `gelu_packed_rdna_kernel` 的物理实现是每个 Thread 拥有一个 BF16x2；Attention 的数学式只有 $QK^T$ 和 Softmax，但 Encoder Kernel 中的真实单位是协作的 Wave Pair。

## 源码与验证边界

本系列将 PR 合并后提交 `e2cf5d1` 作为唯一实现依据，不用当前 `main` 或后续 PR 的代码补齐细节。每篇会同时检查 HIP Function、Host Dispatch、Python Provider 与对应 Test，因为单独看 `__global__` Function 无法得知它在生产路径中是否会被选中。

算子测试能证明指定 Shape 与容差下的 HIP/PyTorch 对齐，但它不能自动证明“所有 Shape 都适用”或“所有机器上都更快”。本系列因此只说明已经写入源码的 Contract、Dispatch 和数值语义，不把某组开发机 Benchmark 外推为通用性能结论。
