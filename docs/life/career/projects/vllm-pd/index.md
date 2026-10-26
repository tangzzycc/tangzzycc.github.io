---
title: "vLLM P/D 解耦与调度优化"
---

# vLLM P/D 解耦与调度优化

沿 Prefill/Decode 的干扰问题，复习论文思路、V1/MRV2 实现与实际性能取舍。

1. [面试问题与回答](面试问题与回答.md)：按最新 `e3e07d9` 梳理九个核心问题，每题包含 1 分钟简答和 5 分钟详细阐述，覆盖开发链、[GPU IPC 原理与共享正确性](面试问题与回答.md#gpu-ipc)、TP、DeepSeek 与测试验证。
2. [论文与推理指标](01-论文与推理指标.md)：Prefill/Decode 的资源特点、P/D 部署方式、TTFT/ITL 与 goodput。
3. [P/D 解耦项目详解](02-PD解耦项目详解.md)：围绕最终实现讲解 P/D 架构、模块协作与请求执行，详细展开 DeepSeek-V3 的 MLA、MoE、FP8 共享，以及调度交接、TP 完成和 KV 回收。
4. [RAPID-Serve 论文完整中文翻译](RAPID-Serve论文完整中文翻译.md)：按 v1 原文章节与段落对应翻译，包含公式、全部 12 幅原图、中文图注和 19 条参考文献。
5. [ATOM 框架层开发与多卡 DeepSeek 适配](ATOM-RAPIDServe框架层开发讲解.md)：讲解 PR #860 的双引擎、TP rank 配对、MLA/MoE 共享与执行路径，并核对公开性能数据、InferenceX 方法和论文实验。

原有实验汇总、配置与图片保存在 [evidence 清单](evidence/manifest.json)所列的本地归档中。

[返回项目导航](../index.md)
