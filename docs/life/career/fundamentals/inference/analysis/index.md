# vLLM 推理框架：分析导航与源码版本

沿请求执行与数据流复习 vLLM 的调度、KV、Attention、投机解码、量化与 MoE。问题与面试示例回答集中在[推理优化问答](../index.md)；本目录保留具体分析、算例与源码追问。

按下面的框架主题复习；需要结合个人贡献时，回查[贡献介绍与各 PR 问答](../../../projects/open-source/vLLM-PR.md)。

| 面试官的切入点 | 可以展开的框架知识 | 笔记 |
|---|---|---|
| #60049：为什么 dummy profiling 会越过预算？ | 请求生命周期、Scheduler、MRV2、Chunked Prefill、Graph、KV 分配与回收 | [02 请求调度、MRV2 与 KV Cache](02-请求调度-MRV2与KVCache.md) |
| #60040 / #59550：Attention mask 为什么不对？ | QKV、GQA、RoPE、PagedAttention、滑动窗口、Prefix-LM、Online Softmax | [03 Attention 与多模态执行](03-Attention与多模态执行.md) |
| #59600：拒绝候选后为什么还会污染状态？ | Target/Drafter、验证与采样、混合模型、ShortConv、状态恢复、请求身份 | [04 投机解码与混合模型状态](04-投机解码与混合模型状态.md) |
| #52637：checkpoint 能加载为什么还不能正确推理？ | 模型加载、Linear、GPTQ、packed layout、Tensor Parallel、权重生命周期 | [05 权重加载、量化与 Linear](05-权重加载-量化与Linear.md) |
| #58513：`group_size=-1` 为什么要修两处？ | MoE 路由、token/expert pair、两次 GEMM、fallback、专家并行 | [06 MoE 路由与量化执行](06-MoE路由与量化执行.md) |
| #53283 / #52619：几行分发为什么能影响性能？ | ROCm 算子入口、GEMM 形状、LDS、Roofline、Amdahl、Profiler、验证与评审 | [07 ROCm 算子与性能追问](07-ROCm算子与性能追问.md) |

## 一次完整技术面可以怎样展开

**从 #60049 开始：** “有效 token 与 padding 容量混淆” → Scheduler 输出哪些量 → 请求怎样展开成 GPU 行 → `query_start_loc` 如何划分请求 → Graph 为什么需要稳定 buffer → draft correction 怎样占预算 → KV 何时可以释放。

**从 #60040 开始：** “图像内部要双向 Attention” → QKV 怎样形成 → 为什么允许未来图像但不允许未来文本 → mask 放开后为何还要扩大遍历 → block table 与 slot mapping → 滑动窗口边界 → NaN 与读地址保护 → 为什么数值测试还不够。

**从 #59600 开始：** “拒绝 token 后卷积历史没有恢复” → 投机解码怎么验证 → target 为何可以一次算多行 → Attention KV 与滚动状态有何区别 → checkpoint 为什么在 correction 前 → 连续两轮与 batch 重排 → 为什么不直接宣称支持 prefix caching。

**从量化或 kernel PR 开始：** “配置与 GPU 契约不一致” → checkpoint 到 runtime 的加载流程 → INT4 的数值与布局 → TP 如何改变 K/N → MoE 如何改变行数 → dispatch 条件与硬件边界 → kernel 加速如何传导到模型收益 → 测试是否能证明整个支持范围。

## 回答时如何把基础知识连接到自己的贡献

一个可展开的回答通常有四层：

1. **先说可见问题：** 哪个模型、配置或 shape 触发了错误，用户看到什么。
2. **再说框架职责：** 谁产生这个量，谁拥有状态，谁执行 GPU 操作。
3. **用一个算例定位：** 哪个索引、窗口、维度或可信边界被解释错了。
4. **最后说明证据：** 回归测试覆盖什么，数值、模型与性能验证各证明什么。

若被问到一个未读过的版本或 backend，可以先说明已核对的实现范围，再沿公开接口解释需要继续确认的部分。将“我复用了已有 kernel”“我补齐了元数据链”“我实现了 checkpoint 恢复”说准确，比把每项贡献都说成重写引擎更容易经得起追问。

## 源码版本与资料入口

整理日期为 **2026-10-11** 。各 PR 来自不同时间的源码，六篇笔记分别使用下面的固定提交；这些提交用于核对实现，不代表六项修改都包含在同一个部署中，也不代表 PR 当前状态。

| 笔记 | 固定提交 | 核对重点 |
|---|---|---|
| 02 | [`aeaa5018bc`](https://github.com/vllm-project/vllm/commit/aeaa5018bc7a4f901e3a44298ec8d11541ec01cd) | #60049 与该版本 V1/MRV2 主链 |
| 03 | [`309e622d88`](https://github.com/vllm-project/vllm/commit/309e622d88ff6606e5d71519c8a2326c1f102044) | #60040、已包含的窗口转换与 ROCM_ATTN |
| 04 | [`359db8bc61`](https://github.com/vllm-project/vllm/commit/359db8bc6143f1d89fd6390b5ac30f6bc85b0165) | #59600、独立 draft-model 与 ShortConv |
| 05 | [`510e42aeb1`](https://github.com/vllm-project/vllm/commit/510e42aeb1c7abcda0d31d826aadbe6e34ba8100) | #52637、GPTQ loader 与 Linear |
| 06 | [`a08f2cb3c5`](https://github.com/vllm-project/vllm/commit/a08f2cb3c54605c8306f42d4cf00b62322bf804f) | #58513、WNA16 Triton MoE fallback |
| 07 | [`e8b32863a7`](https://github.com/vllm-project/vllm/commit/e8b32863a7e70f8331446ffe43192cf293053603) | #52619、已包含的单输出分发与 HIP 实现 |

已有的[基础篇](<../../../../../vllm/基础篇/vLLM V1整体架构.md>)、[Attention 实现](../../../../../vllm/基础篇/Attention的计算机理与vLLM实现.md)与[Decode Attention](<../../../../../vllm/进阶篇/Decode Attention：Paged KV、GQA与Split-KV.md>)适合回查完整推导。项目经验另见 [P/D](../../../projects/vllm-pd/02-V1-MRV2实现与设计取舍.md)、[ROCm 模型适配](../../../projects/vllm-rocm/01-CogAgent模型接入与量化推理.md)、[Adaptive KV](../../../projects/vllm-adaptive-kv/02-调度接入与异步生命周期.md)；面试中应分别介绍项目分支能力与上游 PR 改动。
