---
title: "计算图执行：CUDA Graph 与 torch.compile"
---

# 计算图执行：CUDA Graph 与 torch.compile

## 1. CUDA Graph 的加速原理是什么？

1. **执行过程。** 先 warmup，再通过 stream capture 或显式建图记录 kernel、memcpy 等节点和依赖，实例化可执行图，后续反复 replay。计算仍在 GPU 上执行，重放的是操作与依赖，不是上一次结果。
2. **省掉什么。** 常规 replay 绕过被捕获区域的逐算子 host 执行，降低提交成本与 kernel 间 CPU 空隙；图中通常仍有多个 kernel，不能说变成一个融合 kernel。
3. **为何小 Decode 常受益。** 短 kernel 的提交开销占比高，更容易受 CPU 供给速度限制；大 GEMM 已由 GPU 计算主导，Graph 收益可能较小。Eager 的同 stream kernel 也无需逐个同步，收益来自减少提交，而非移除不存在的同步。
4. **动态输入与限制。** 常见做法是固定输入 buffer、更新内容，并为不同 batch 捕获 buckets、配合 padding。数据相关的 CPU 控制流和同步等会限制捕获；Graph API 可更新部分节点参数，但不是任意 shape 都能直接重放。捕获、显存与 padding 也有成本。

扩展阅读：[CUDA/HIP Graph、Compile 与 Fusion（第 10.12 节）](../../../projects/vllm-pd/03-性能证据与面试问答.md)。

## 2. torch.compiler / torch.compile 是什么？与 CUDA Graph 有什么区别？

1. **入口与流程。** 常用 `compiled_model = torch.compile(model)`。Dynamo 通过 Python frame/bytecode 机制提取 FX 图并建立 guards，Inductor 通常为 GPU 生成 Triton kernel，或调用现有高效库；不能说整个模型都会变成一个 Triton kernel。
2. **加速来源。** 比如 `y = relu(x + bias)` 可融合成一个 kernel，减少一次 launch 和中间结果写读；还可优化调度、布局与内存复用。大型 GEMM 可能仍调用 cuBLAS 等库，收益需看完整执行链。
3. **动态图的代价。** Shape、dtype 或控制条件变化可能触发 guards 失败与重编译；不支持的 Python 行为可能造成 graph break，默认情况下图段与 eager 交替执行。动态 shape 支持可以减少部分重编译，但不是所有控制流都能捕获。
4. **与 Graph 配合。** 常见顺序是先编译优化，再捕获稳定的 GPU 执行进行重放；某些 compile 模式和后端会使用 CUDA Graph。`reduce-overhead` 模式可在适用条件下尝试通过 CUDA Graph 降低 Python 开销，具体行为依版本和 backend，不代表所有模型必然使用 Graph。

| 技术 | 主要优化对象 | 是否自动融合算子 |
| --- | --- | --- |
| CUDA Graph | 重复的 CPU 分发、提交和 GPU 等待空隙 | Graph 本身不自动融合 |
| torch.compile | 捕获图的实现、生成代码和数据流 | 编译后端可进行融合 |

扩展阅读：[PyTorch torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)、[torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html)。
