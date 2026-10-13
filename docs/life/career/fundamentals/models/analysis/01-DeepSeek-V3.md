---
title: "DeepSeek-V3：模型结构与参数量"
---

# DeepSeek-V3：模型结构与参数量

## 1. DeepSeek-V3 的结构和参数量是什么？

以下对应 **原版 DeepSeek-V3**，依据官方仓库及模型配置，不将 V3.2 等后续版本混入。

| 项目 | 参数 |
| --- | --- |
| 主模型参数 | 总计约 671B；每 token 激活约 37B |
| 层数与隐藏维度 | 61 层；hidden size 7168 |
| FFN 结构 | 前 3 层 Dense，后 58 层 MoE；SwiGLU |
| MoE 专家 | 每层 256 routed experts，Top-8；1 shared expert |
| FFN 中间维度 | Dense 18432；单专家 2048 |
| MLA | 128 个 Query head；Q 低秩维度 1536；KV 低秩维度 512 |
| Attention head 维度 | Q/K 内容维度 128、RoPE 维度 64；V 维度 128 |
| 词表与上下文 | 词表 129280；官方报告上下文 128K |
| MTP | 官方完整权重另含约 14B MTP，合计约 685B |

1. **MoE 降低激活计算，不按同样比例减少显存。** 每 token 只执行 Top-8 路由专家及共享专家，但不同 token 可能选择其他专家。仅按主模型参数粗算，BF16 权重约 1.34 TB、FP8 约 671 GB，还需计入未量化参数、scale、MTP、KV 和 workspace。
2. **MLA 压缩 KV。** 常见压缩缓存保存 512 维 KV latent 加 64 维解耦 RoPE Key，即每层每 token 576 个元素；Q latent 不进入历史 KV。收益需结合具体 backend 的压缩缓存与矩阵吸收路径。
3. **路由与 MTP。** 原配置使用 sigmoid 分数、分组限制和负载均衡策略；MTP 是多 token 预测训练目标，相关模块可用于投机解码，不代表普通自回归推理默认每次输出多个 token。
4. **优化重点。** MLA Attention、FP8/Grouped GEMM、路由与 gather/scatter，以及 EP dispatch/combine。37B 激活量不能单独预测吞吐，因为 batch 会改变实际触及的专家数量与通信。

官方报告的 128K 上下文与 HF 配置的 `max_position_embeddings=163840` 是不同口径，回答标称上下文时使用 128K。

来源：[DeepSeek-V3 官方说明](https://github.com/deepseek-ai/DeepSeek-V3)、[官方推理配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json)、[HF 模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。扩展阅读：[MLA 笔记](../../../../../vllm/进阶篇/MLA：低秩压缩、矩阵吸收与解耦RoPE.md)。
