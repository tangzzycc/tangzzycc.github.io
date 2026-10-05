# vLLM 笔记

这里整理 vLLM 的架构、推理机制与实践记录。

## 基础篇

- [vLLM V1 整体架构](基础篇/vLLM V1整体架构.md)
- [请求的数据形态：从 Client 到 GPU](基础篇/请求的数据形态：从Client到GPU.md)
- [请求预处理：Chat Template 与 Tokenization](基础篇/请求预处理：Chat Template与Tokenization.md)
- [EngineCore](基础篇/EngineCore.md)
- [从 EngineCoreRequest 到 Request](基础篇/从EngineCoreRequest到Request.md)
- [KVCacheConfig](基础篇/KVCacheConfig.md)
- [KVCacheConfig 的产生](基础篇/KVCacheConfig的产生.md)
- [KVCacheConfig 的 Core 侧消费：逻辑资源初始化](基础篇/KVCacheConfig的Core侧消费：逻辑资源初始化.md)
- [KVCacheConfig 的 Worker 侧消费：GPU KV Cache 初始化](基础篇/KVCacheConfig的Worker侧消费：GPU KV Cache初始化.md)
- [Continuous Batching：动态更新执行批次](基础篇/Continuous Batching：动态更新执行批次.md)
- [Chunked Prefill](基础篇/Chunked Prefill.md)
- [PD 分离与 KV 传输](基础篇/PD分离与KV传输.md)
- [请求变成 GPU 输入](基础篇/请求变成GPU输入.md)
- [Attention 的计算机理与 vLLM 实现](基础篇/Attention的计算机理与vLLM实现.md)
- [从 Hidden States 到 Output Token](基础篇/从Hidden States到Output Token.md)

## 进阶篇

- [FlashAttention v1：从完整矩阵到分块精确计算](进阶篇/FlashAttention：分块计算与Online Softmax.md)
- [FlashAttention-2：并行划分与非矩阵运算优化](进阶篇/FlashAttention-2：线程块与Warp的工作划分.md)
- [FlashAttention-3：异步搬运与计算流水线](进阶篇/FlashAttention-3：异步搬运与计算流水线.md)
- [Decode Attention：Paged KV、GQA 与 Split-KV](进阶篇/Decode Attention：Paged KV、GQA与Split-KV.md)
- [MLA：低秩压缩、矩阵吸收与解耦 RoPE](进阶篇/MLA：低秩压缩、矩阵吸收与解耦RoPE.md)
- [稀疏 Attention：选择性访问与高效计算](进阶篇/稀疏Attention：选择性访问与高效计算.md)

## 开发实践

- [CogAgent-18B 在 vLLM 中的模型接入与推理优化](开发实践/CogAgent-18B在vLLM中的模型接入与推理优化.md)
- [RDNA 3.5 上的 vLLM LoRA Kernel 优化](开发实践/RDNA3.5上的vLLM-LoRA-Kernel优化.md)

围绕开源 PR，沿实际调用链学习算子分发、量化加载、Attention 与投机解码：

- [单输出 GEMM 与 Shared Expert 门控的 ROCm 优化（PR #53283）](开发实践/单输出GEMM与Shared-Expert门控的ROCm优化.md)
- [RDNA W4A16 的形状分发与 LDS 边界（PR #52619）](开发实践/RDNA-W4A16的形状分发与LDS边界.md)
- [GPTQ 权重加载与 RDNA INT4 布局重排（PR #52637）](开发实践/GPTQ权重加载与RDNA-INT4布局重排.md)
- [Channelwise GPTQ MoE 的加载与 Triton 执行（PR #58513）](开发实践/Channelwise-GPTQ-MoE的加载与Triton执行.md)
- [ROCm Attention 滑动窗口的参数与边界（PR #59550）](开发实践/ROCm-Attention滑动窗口的参数与边界.md)
- [多模态 Prefix-LM 在 ROCm Attention 中的实现（PR #60040）](开发实践/多模态Prefix-LM在ROCm-Attention中的实现.md)
- [MRV2 投机解码中的 ShortConv 历史恢复（PR #59600）](开发实践/MRV2投机解码中的ShortConv历史恢复.md)
- [MRV2 的 PIECEWISE 填充与投机解码启动 Profiling（PR #60049）](开发实践/MRV2的PIECEWISE填充与投机解码启动Profiling.md)

## 杂记

- [vLLM Multi-LoRA 推理](杂记/vLLM Multi-LoRA推理.md)
