---
title: "AI Infra 面试基础"
---

# AI Infra 面试基础

按[当前简历](../resume/resume_ng_zh.tex)整理复习顺序：先学能解释自己实现的基础，再补部署与架构进阶。仍分为 **GPU与算子优化、推理优化、模型结构** 三个模块，每个模块保留一个问答总文件，具体分析放在 `analysis/`。

回答直接展示，关键结论加粗，并用绿色突出；蓝色链接直达每题的原理分析，红色提醒易错边界。先读一组、尝试口述，再点击题后的分析看推导、算例和执行约束；分析小节也可返回对应面试题。

## 从简历主线开始

| 简历里的工作 | 先读哪些基础 | 再接个人实现 |
| --- | --- | --- |
| vLLM P/D 解耦、HIP Graph 与动态资源调度 | [请求与指标](inference/index.md#request-metrics) → [组批与异步执行](inference/index.md#batching-execution) → [P/D 解耦](inference/index.md#pd-disaggregation) | [P/D 项目问答](../projects/vllm-pd/面试问题与回答.md) |
| Adaptive KV Offload、Load-vs-Recompute | [KV 容量、加载与重算](inference/index.md#kv-reuse) | [KV 分层与成本](../projects/vllm-adaptive-kv/01-KV分层缓存与重算成本.md) |
| ROCm/CogAgent 接入、Snapdragon/QNN 部署 | [模型适配与端侧执行](inference/index.md#model-backend) → [Attention 状态](models/index.md#attention-state) | [ROCm 项目问答](../projects/vllm-rocm/面试问题与回答.md) |
| FlashRT/FlyDSL、LoRA、GPTQ/AWQ 算子 | [GEMM](gpu-kernels/index.md#gemm) → [量化执行](gpu-kernels/index.md#quantization) → [访存布局](gpu-kernels/index.md#memory-layout) → [性能证据](gpu-kernels/index.md#profiling) | [FlashRT 问答](../development/flashrt/FlashRT-面试问题与回答.md) · [FlyDSL 贡献](../development/flydsl/FlyDSL-PR.md) |
| 混合 Attention 状态与投机解码修复 | [混合状态](models/index.md#hybrid-attention) → [投机与状态正确性](inference/index.md#speculation) | [vLLM 开源问答](../development/vllm/vLLM-面试问题与回答.md) |

## GPU与算子优化

[进入问答总文件](gpu-kernels/index.md) · 64 道题，8 个主题。

第一轮按 **GEMM → 流水线 → 量化 → 访存布局 → 性能分析** 阅读，再练 Attention 与归约；结构化稀疏、Hopper/Blackwell 和直方图选择按目标岗位补充。

详细分析入口：[GEMM 切分与 Persistent](gpu-kernels/analysis/14-GEMM切分与Persistent调度.md)、[量化与稀疏](gpu-kernels/analysis/01-量化与稀疏.md)、[NCU 与性能分析](gpu-kernels/analysis/04-Kernel性能分析.md)。每道题的回答后均有对应分析链接，也可从侧栏选择专题。

## 推理优化

[进入问答总文件](inference/index.md) · 102 道题，12 个主题。

第一轮按 **请求与指标 → 组批/Graph → P/D → KV → 模型适配 → 投机推理** 阅读。接着补 TP/DP、通信和 MoE；路由、PP/CP、仿真与配置搜索排在后面，按面试方向深入。

详细分析入口：[P/D 分离](inference/analysis/05-PD分离.md)、[KV 恢复与重算](inference/analysis/20-RadixCache与分层恢复.md)、[执行适配](inference/analysis/15-芯片适配与执行优化.md)、[投机验证](inference/analysis/17-投机推理与验证.md)。

## 模型结构

[进入问答总文件](models/index.md) · 14 道题，6 个主题。

先看 **图像怎样经过 Patch Embedding、ViT 和 connector 接入 LLM**，再比较 MHA/GQA/MLA 的缓存表示与混合状态；MoE 要讲清逻辑路由与物理放置，MTP 要区分候选与最终提交 token。DSA、CSA/HCA 等具体变体作为最后一组补充。

详细分析入口：[图像、ViT 与 LLM](models/analysis/05-Image-Patch-ViT与LLM.md)、[Attention 与缓存](models/analysis/01-MHA-GQA与MLA.md)、[线性与稀疏 Attention](models/analysis/03-稀疏压缩与线性Attention.md)、[MoE](models/analysis/02-MoE与路由.md)、[MTP](models/analysis/04-MTP与半自回归.md)。

[项目与开源贡献](../projects/index.md) · [返回求职与面试准备](../index.md)
