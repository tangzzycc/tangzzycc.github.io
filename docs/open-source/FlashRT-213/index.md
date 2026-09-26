# FlashRT Pi0.5：RDNA 3.5 BF16 后端与 Temporal K/V

[FlashRT PR #213](https://github.com/flashrt-project/FlashRT/pull/213) 为 Pi0.5 增加了面向 `gfx1151` 的 BF16 推理实现，补齐从公开 API、checkpoint 转换、静态 tensor pipeline，到融合算子、Attention 和 GEMM provider 的完整执行链。[PR #215](https://github.com/flashrt-project/FlashRT/pull/215) 继续在这条 shared pipeline 上增加 Temporal K/V，让连续帧可以按显式周期复用最近一次完整推理产生的 Prefix K/V。

- PR #213 固定源码版本：[`e2cf5d1`](https://github.com/flashrt-project/FlashRT/commit/e2cf5d1fe7209e3f4fbf212c5b6321467e8f4874)
- PR #215 固定源码版本：[`839b159`](https://github.com/flashrt-project/FlashRT/commit/839b15977f0ddf4e525e3511ec3b67b157b25acf)
- PR #213 变更规模：45 个文件，新增 6,858 行、删除 413 行
- PR #215 变更规模：8 个文件，新增 541 行、删除 47 行
- 目标设备：AMD RDNA 3.5，`gfx1151`，wave32
- 数值范围：当前生产路径为 BF16

## 阅读顺序

| 笔记 | 主要问题 | 读完能够解释 |
| --- | --- | --- |
| [01 · Frontend 与 Pipeline](01-Pi05-Frontend与Pipeline.md) | FlashRT 如何把 OpenPI 模型变成静态、可捕获的推理流程？ | API、frontend、shared pipeline、provider 和 checkpoint conversion 的职责边界 |
| [02 · 模型结构与推理流程](02-Pi05模型结构与推理流程.md) | Pi0.5 怎样从图像、指令和随机噪声生成连续动作？ | Prefix、action expert、AdaRMS 与 flow matching 的数据流和 shape |
| [03 · 带宽型融合算子](03-RDNA3.5融合算子.md) | 小算子为什么值得手写 HIP，并且怎样保持 BF16 语义？ | QKV+RoPE、GeGLU、residual+norm 等融合点及舍入边界 |
| [04 · Attention Kernel](04-RDNA3.5-Attention-Kernel.md) | Encoder 长序列与 Decoder 短 Query 为什么需要两种调度？ | wave-pair encoder、split-key decoder 与在线 softmax 合并 |
| [05 · GEMM 实现](05-RDNA3.5-GEMM实现.md) | 通用 hipBLASLt 与 small-M WMMA 如何分工？ | descriptor cache、lazy autotune、持久转置与两类 WMMA kernel |
| [06 · Temporal K/V 跨帧复用](06-Temporal-KV跨帧复用.md) | 连续帧如何复用 Prefix K/V，同时保持失败恢复与 graph refresh 正确？ | Full/Decode-only 周期、cache commit、地址复用、失效边界与 stale-context 语义 |

## 范围与证据

01–05 固定到 PR #213 的 `e2cf5d1`；06 固定到 PR #215 合入官方 main 后的 `839b159`。性能数字没有作为主线：这组笔记关注模型语义、实现边界、kernel 组织方式与 cache 状态变化。

PR 中的 source-only 测试和共享 pipeline 回归能够证明接口与参考路径的一致性；只有使用独立 OpenPI fixture 的比较才属于外部模型正确性证据。本文不会把 optimized/fallback 两条同源 pipeline 的对比写成独立 oracle，也不会把 PR 早期实验值写成最终性能结论。
