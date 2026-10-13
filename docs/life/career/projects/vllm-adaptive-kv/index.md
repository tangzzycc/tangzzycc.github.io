---
title: "vLLM Adaptive KV Offload"
---

# vLLM Adaptive KV Offload

沿 KV 读取与重算的成本选择，复习缓存层级、调度决策及异步资源生命周期。

1. [KV 分层缓存与重算成本](01-KV分层缓存与重算成本.md)：缓存对象、分层命中与成本判断。
2. [调度接入与异步生命周期](02-调度接入与异步生命周期.md)：预算、异步加载、状态变化与取消回收。
3. [性能证据与面试问答](03-性能证据与面试问答.md)：正式负载、历史测试与归因边界。

原有 benchmark、配置与分析保存在 [evidence 清单](evidence/manifest.json)所列的本地归档中。

[返回项目导航](../index.md)
