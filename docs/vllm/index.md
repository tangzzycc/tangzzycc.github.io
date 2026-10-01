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

## 开发实践

- [CogAgent-18B 在 vLLM 中的模型接入与推理优化](开发实践/CogAgent-18B在vLLM中的模型接入与推理优化.md)
- [RDNA 3.5 上的 vLLM LoRA Kernel 优化](开发实践/RDNA3.5上的vLLM-LoRA-Kernel优化.md)

## 杂记

- [vLLM Multi-LoRA 推理](杂记/vLLM Multi-LoRA推理.md)
