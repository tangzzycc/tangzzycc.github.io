---
title: "模型结构：面试问题与示例回答"
---

# 模型结构：面试问题与示例回答

??? question "1. DeepSeek-V3 的结构和参数量是什么？"

    原版 DeepSeek-V3 是采用 **MLA 和 DeepSeekMoE**  的自回归 Transformer，主模型**总参数约 671B，每 token 激活约 37B** 。它有 61 层、隐藏维度 7168，前 3 层是 Dense FFN，后 58 层是 MoE；每个 MoE 层有 256 个路由专家，每 token 选 8 个，另有一个始终执行的共享专家。MLA 用低秩 KV 表示降低缓存开销，训练还使用 MTP。37B 代表每 token 的激活规模，**部署通常仍需保存全部专家权重** 。
