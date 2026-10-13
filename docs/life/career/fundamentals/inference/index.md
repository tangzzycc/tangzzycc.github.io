---
title: "推理优化：面试问题与示例回答"
---

# 推理优化：面试问题与示例回答

??? question "1. CUDA Graph 的加速原理是什么？"

    普通 eager 执行会由 CPU 逐个分发算子、提交 GPU kernel。CUDA Graph 先记录一段 GPU 操作及其依赖，实例化后通过**一次 graph launch 重放** ，**省去重复的 Python、框架分发和逐 kernel 提交开销** ，减少 GPU 等待 CPU 的空隙。它特别适合执行结构稳定、包含大量短 kernel 的推理，例如小 batch Decode。Graph 本身不会自动融合算子或减少计算量；输入内容可以变化，但常规 replay 要**保持被捕获的地址和执行结构稳定** 。

??? question "2. torch.compiler / torch.compile 是什么？与 CUDA Graph 有什么区别？"

    `torch.compiler` 是 PyTorch 编译相关功能的命名空间，常用入口是 `torch.compile`，用于编译模型或函数。在常见默认路径中，**TorchDynamo 捕获 Python 执行中的 tensor 计算图** ，**TorchInductor 做图优化与代码生成** ，比如融合逐元素算子、减少中间访存；训练还会涉及 AOTAutograd。**Compile 主要优化计算的实现，CUDA Graph 主要优化重复提交** ，两者可以结合。编译收益取决于算子、shape 和 backend，首次编译、graph break 和重编译也有成本。

??? question "3. 一个请求进入 vLLM 后，经过哪些核心模块？"

    前端先把输入转成 token 与引擎请求，EngineCore 接收后交给 Scheduler。**Scheduler 决定本轮推进哪些请求、各推进多少 token** ，并**向 KVCacheManager 申请空间** 。Executor 把调度结果交给 Worker；MRV2 根据请求状态**准备张量与 Attention metadata** ，模型和 backend 执行 GPU 计算，再由采样与输出处理更新请求状态并返回结果。

??? question "4. V1 Scheduler 是怎么统一 Prefill 和 Decode 的？"

    该版本主要围绕**请求已经计算到哪里** ，以及目前有哪些 token 等待计算来调度。Prefill 是还有一段 prompt 没计算，普通 Decode 通常是已采样出一个新 token、但它的 KV 还没计算。两者都表现为**待推进的 token 差额** ，再受 **token budget、请求数、KV 空间和可执行条件** 约束。

??? question "5. Continuous Batching 和 Chunked Prefill 为什么能改善服务？"

    Continuous Batching 允许**每轮重新组成批次** ，完成的请求退出，新的请求进入；Chunked Prefill **将长 prompt 分成多轮** ，使它不能一次占满很长的模型执行时间。它们**改善批次利用和请求公平性** ，但同轮 Prefill 计算仍可能拉长 Decode 的 token 间隔。

??? question "6. MRV2 怎么把多个请求变成一批 GPU 输入？"

    Runner 先把本轮请求映射到**稳定的请求状态槽** ，再将各请求调度的 token **展平成连续行** 。**`query_start_loc`**  给出每条请求在这些行中的边界，**`positions`**  给出 token 的绝对位置，**`seq_lens`**  描述它本轮可使用的序列范围；之后准备 block table、slot mapping 和 backend metadata。

??? question "7. PR #60049 为什么没有直接增大 buffer？"

    Buffer 容量并没有错，错的是 PIECEWISE dummy 构造**把 padding 当作有效 Query** ，随后 drafter 又给每个请求补 correction 行。修复保留**有效计数与 padded 容量的区别** ，让**预算仍约束真实展开工作** ；简单增大 buffer 会掩盖计数错误，还可能继续扭曲 profiling。

??? question "8. KVCacheManager 管的是真实 K/V 张量吗？"

    它主要管理请求与缓存块的**逻辑归属、命中、分配和释放** 。**真实 K/V 在 Worker 初始化的 GPU tensor 中** ，block IDs 与 metadata 指示 kernel 读写其中哪些位置。`KVCacheBlock` 是**资源元数据对象** ，创建它不会自动为每个对象单独分配一个 GPU tensor。

??? question "9. Prefix caching 怎么复用，为什么不能只对当前块 hash？"

    Causal 模型某位置的 **KV 依赖此前上下文** 。同样一段 token 接在不同前缀后，其 KV 可能不同，所以缓存身份需要表达**前缀链及影响计算的附加信息** 。命中已有块后，请求**获得对它们的引用** ，再为新后缀分配空间；命中不是把完整 KV 复制一份给新请求。

??? question "10. 抢占、取消与异步调度最容易出什么错？"

    抢占会改变计算进度与缓存归属，取消会终止业务请求，但**已经提交的 GPU 工作可能仍在读写** 。必须保证**状态重建与资源回收都落在正确边界** ；如果立即把块或状态槽分给另一个请求，就可能把旧请求的写入带到新请求中。

??? question "11. 模型产生 hidden states 后，怎样得到用户看到的 token？"

    Runner 选出需要输出 logits 的 hidden rows，经 **LM head 得到词表分数** ，再应用**约束、penalty、temperature 与 Top-k/Top-p**  等采样规则。普通 Decode 每条请求通常采一行，投机验证可能需要多行 logits。采样后**更新引擎请求** ，前端 detokenize 并形成用户输出。

??? question "12. vLLM 的 Attention 层与 backend 分别负责什么？"

    模型层根据 hidden states 产生 Q/K/V，完成模型要求的归一化和位置编码，再调用通用 Attention 层。Backend 负责设备适配、metadata、KV layout 和 GPU 算子执行。**模型决定计算语义** ，**backend 将这些语义实现为当前硬件上的读写与计算** 。

??? question "13. GQA、MQA 与 RoPE 会影响哪些缓存契约？"

    MHA 的每个 Query head 有对应 KV head，MQA 共享一组 KV，GQA 则让**若干 Query heads 共享一个 KV head** 。**KV head 数决定缓存载荷** 。RoPE 按请求内部 position 旋转 Q/K，历史缓存中的 K 必须与其**真实位置一致** ；batch 重排不能改变这些位置。

??? question "14. PagedAttention 与 FlashAttention 有什么区别？"

    **分页 KV 主要解决缓存存储、分配与非连续历史访问** ，FlashAttention 主要通过**分块和 Online Softmax**  减少 Attention 中间矩阵的显存读写。**二者可以同时使用** ：kernel 依据 block table 读取分页 K/V，再分 tile 完成 softmax 与加权求和。

??? question "15. 滑动窗口为什么会差一个 token？"

    模型给的是**包含当前位置的窗口总长 W** ，而接口可能传**左跨度 W-1** 。若 kernel 用 `q-k<W`，就需要总长 W；把左跨度直接传入会少保留最旧的那个合法 token。修复位置应是**单位与语义转换的接口** 。

??? question "16. 图像内部双向为什么不会泄漏未来回答？"

    图像 embedding 是**已经给定的输入条件** ；允许同一图像范围内部互相读取，不等于允许读取尚未生成的文本。Prefix-LM 在 causal 可见性上增加**同一图像区间** ，必须按**每张图的具体范围** 判断，不能把所有 image tokens 连成一个双向集合。

??? question "17. 为什么只改 mask，结果仍然可能错误？"

    Mask 只能在实际加载和计算的候选 Key 中决定可见性。原 causal 遍历可能根本没访问未来同图像 Key，所以要**先扩大 tile 遍历** ；扩大后又可能访问不合法页表，必须**保护地址读取** ，并处理缓存残留 NaN。**元数据、遍历边界与数值 mask 要一起成立** 。

??? question "18. 支持 Attention 后为什么还要验证 Graph 与完整模型？"

    **Kernel 对照** 证明算子在给定输入与 metadata 下数值正确，**模型对照** 证明元数据及模型接入链正确，**真实 replay**  则证明 capture/replay 的输入布局和分支可复用。**三者覆盖不同风险** 。

??? question "19. 投机解码为什么可能更快，最终由谁决定输出？"

    小 drafter 先提出几个候选，**target 一次前向验证多个位置** ，接受符合验证规则的前缀并产生 correction 或 bonus token。用户得到的是 **target 验证后的输出** 。它希望用较便宜的提案换取每次昂贵 target 执行提交更多 token，但**收益要覆盖 drafter、验证、采样和状态管理成本** 。

??? question "20. 投机解码怎样保持 target 的输出语义？"

    Greedy 模式逐个比较候选与**对应 target argmax** ，第一处不一致后停止接受并取 target 结果。随机采样需要使用**正确的接受概率和拒绝后的校正分布** ；“从 target 再随便采一次”不能自动保持原分布。

??? question "21. 接受率高就一定加速吗？"

    接受率影响每轮提交多少 token，但**不决定整轮成本** 。提案长度增加会增加 drafter 时间、target Query 行、logits 和 workspace；大 batch 下 target 已有较好的矩阵利用率，额外验证可能降低收益。要比较**每提交 token 的总时间** 。

??? question "22. LFM2.5 为什么不能只恢复 Attention KV 的长度？"

    LFM2.5 交替使用 Attention 和 ShortConv。Attention KV 按 token 位置组织，可以通过有效长度和后续覆盖排除拒绝尾部；**ShortConv 保存滚动窗口** ，候选推进会覆盖之前的数值。把长度缩回去不会自动找回已经**被覆盖的历史** ，必须**恢复可信窗口并重放有效输入** 。

??? question "23. Checkpoint 为什么必须在 correction 前？"

    下一轮 drafter 会从 correction token 重新消费输入，所以 checkpoint 应保存**尚未消费 correction 的可信历史** ；当前轮继续提出候选又需要**已经消费 correction 的 runtime state** 。**两份历史相差一步** ，分别服务下一轮恢复和当前轮继续预测。

??? question "24. Expanded batch 为什么还有被拒绝位置？"

    当前执行 buffer 为每条请求保留 target Query 加一行 correction 的布局。拒绝尾部可能仍作为**临时槽位** 存在，但 checkpoint 和最后有效输入索引**按接受/拒绝数确定** 。**Tensor 长度不能直接当作已提交进度** 。

??? question "25. Chunked Prefill 没有采样时，下一输入从哪里来？"

    Prompt 还没算完，下一 token 已知，应**从 prompt lookahead 取** ，不应读可能陈旧的最后采样 token。只有当前请求**确实采样过** ，才使用 last_sampled。Prefill 与生成边界必须由**真实状态判断** 。

??? question "26. 如何证明连续多轮与请求重排都正确？"

    单轮只能证明局部恢复，**连续轮次** 才能证明新 checkpoint 能成为下一轮的可信起点；**batch 重排、请求退出与状态槽复用** 能证明身份没有错。还应对照不启用投机的有效前缀状态，覆盖**零接受、全接受与部分接受** 。

??? question "27. vLLM 加载一个量化 Linear 的流程是什么？"

    量化配置先为该 Linear **选择 quant method** ，创建参数与 loader 属性；checkpoint loader 将文件权重写入对应参数；加载后的统一处理调用 **`process_weights_after_loading`** ，转成选定 backend 的布局。正常 forward 再通过 **`quant_method.apply`**  使用运行时参数。

??? question "28. W4A16、GPTQ、AWQ 分别说明什么？"

    W4A16 描述权重 4 bit、激活 16 bit 的**执行格式** ，GPTQ 与 AWQ 描述权重量化的**算法或 checkpoint 体系** 。GPTQ 利用近似二阶信息降低逐层量化误差；AWQ 利用激活统计对重要通道做缩放保护。相同 W4A16 标签**不能保证文件布局或运行 kernel 一样** 。

??? question "29. checkpoint 里有 qzeros，为什么要删除？"

    是否消费显式 zero point 由**量化语义和 backend 配置决定** ，不能由文件里是否存在 qzeros 决定。对称路径可能将中心编码在 weight type 中，checkpoint 仍保留冗余 qzeros。应在**加载后、backend 处理前** 清理错误的张量引用。

??? question "30. Packed layout 变了，量化值会变吗？"

    正确重排**只改变 bit 的存放位置** ，不改逻辑 q、scale 和解码后的权重。GPU kernel 按自己的布局解码，所以 loader 和 kernel 必须对**同一个 nibble 顺序** 达成一致。这是**表示转换** ，不是重新量化。

??? question "31. transpose、contiguous、clone、view 哪些会复制？"

    **transpose 通常只改 shape/stride** ；**contiguous 在布局不连续时复制** ；**clone 创建独立存储** ；**view 在满足条件时重新解释同一存储** 。原地 shuffle 要求连续 packed tensor，并且不能污染仍被其他参数引用的源存储，所以这里先 clone。

??? question "32. Tensor Parallel 的 Row/Column 为什么会影响量化 kernel？"

    **ColumnParallel 按输出通道分权重** ，各 rank 产生输出分片；**RowParallel 按输入特征分权重** ，各 rank 产生部分和后再归约。它们改变**本设备真实 K/N** ，因此影响 group 对齐、packed 轴和 kernel dispatch，不能只把 TP 当作多跑几份同形状 GEMM。

??? question "33. 量化 PR 怎样验证才经得起追问？"

    **先验证加载契约，再验证布局和数值，最后验证模型** 。零点修复要覆盖冗余 qzeros 与真正需要显式零点两种情况；packed shuffle 要对照**逻辑解码** ，包含轴方向与**源存储保护** ；模型验证则检查实际加载、生成与必要的精度评测。

??? question "34. MoE 跟 Dense MLP 的区别在哪里？"

    Dense MLP 对每个 token 使用同一组参数，MoE 的 **router 为每个 token 选若干专家** ，分别计算再**按权重组合** 。激活参数量减少不意味着总参数显存按相同比例减少，因为部署通常**仍需保存全部专家** ，执行还增加路由、分组与可能的跨卡通信。

??? question "35. token/expert pair 怎样进入两个 GEMM？"

    每个 token 的每个 **Top-k 槽形成一个 pair** ，再按 expert 将 pair IDs 分组。**第一 GEMM 根据 pair 找原始 hidden row** ，结果按 pair 保存；激活后**第二 GEMM 输入已经每 pair 一行** ，不能再除一次原 Top-k。最后按路由权重还原每 token 输出。

??? question "36. 为什么 w13 与 w2 的输入 K 不一样？"

    gate/up 合并投影把 **hidden H 变成 2I** ，**激活变成 I** ，down projection 再从 **I 回到 H** 。Channelwise scale 每个输出通道一组，第一组覆盖 H，第二组覆盖 I；所以配置 -1 要在**每次 GEMM 看实际 K 后解析** 。

??? question "37. PR 为什么既修改分配，也修改 runtime wrapper？"

    分配阶段要把 -1 解释成**单组数量** ，否则 scale/zero point 出现负维度；执行阶段要把 -1 解释成**本次 GEMM 的 K** ，否则 group 索引仍错误。两处修复解决**不同消费者对同一 sentinel 的理解** 。

??? question "38. Triton 如何反量化当前专家的 INT4 权重？"

    根据 expert ID、输出 n 和输入 k 寻址 packed byte，**抽取对应半字节** ，再**按 k/G 找 scale 与可选 zero point** ，生成**当前 tile 的计算值做 dot** 。Accumulator 通常保留较高精度，最终按输出类型写回。

??? question "39. Expert Parallel、TP 和 shared expert 有什么区别？"

    **TP 拆分专家矩阵的特征轴** ，**EP 将专家分布到不同 rank** ，需要把 token 送到对应专家并组合结果。**Shared expert 是模型结构里参与共享计算的分支** ，不等于 routed Top-k 里的普通专家；是否有独立 gate、如何加权由模型定义。

??? question "40. 应怎样验证这个修复？"

    先验证单组参数分配，再用 **H 与 I 不相等** 的专家检查两个 GEMM 都能正确解析 K，覆盖**有/无 zero point 和非均匀 scale** ；同时对照**路由结果与 dense dequant reference** 。实际模型启动和生成用于检查整条 fallback 链。

??? question "41. 从模型 Linear 到 ROCm kernel 有哪些边界？"

    模型调用 Linear，quant method 或未量化 GEMM 实现选择平台 wrapper，wrapper 依据 **shape/dtype/layout**  调用 custom op。C++ host 函数**检查契约并 launch HIP kernel** ，GPU 线程执行真实运算。**Python dispatch 与 C++ 支持范围必须一致** 。

??? question "42. 单输出 GEMM 的 M/N 为什么容易说反？"

    我先统一成 `X[M,K]、W[N,K]、Y[M,N]`，**M 是 token 行数，N 是输出通道数** 。这个 PR 的**单输出是 N=1** ，不一定是 batch=1。源码 wrapper 变量 m 指输出通道、n 指行数，必须**按 tensor shape 还原含义** 。

??? question "43. 小 GEMM 为什么不能只依赖通用大矩阵 kernel？"

    M 或 N 很小时，通用 tile 可能有**大量空闲 lane** ，并且调用和 **launch 固定成本占比高** 。专门 kernel 根据输出和 K 方向分工、复用输入并减少无效工作，能改善特定 shape。但它的**收益范围受模板、对齐和资源限制约束** 。

??? question "44. medium-skinny 为什么允许输入超过 64 KiB LDS？"

    64 KiB 是此实现**每 workgroup 的 LDS 使用容量** ，不是总工作集必须全部放入片上的条件。Medium kernel 把**激活前缀放进 LDS，尾部从 global memory 读取** ，所以允许总激活元素数略大于 LDS。PR 补齐这个已有能力的 Python 门槛。

??? question "45. Prefill 是 compute-bound，Decode 是 memory-bound，可以直接这样回答吗？"

    这是**常见趋势** ，不能当作所有算子的固定标签。**Prefill 的 M 较大，能复用权重** ，通常更容易发挥矩阵算力；**小 batch Decode 的 M 较小** ，权重读带宽与 launch 占比常更高。大 batch Decode、长上下文 Attention、MoE 路由和小 gate 都可能有不同瓶颈。

??? question "46. 37.8 倍 kernel 加速为什么只有 12.79% 模型收益？"

    Kernel 只占模型执行的一部分，Attention、专家矩阵、其他层和框架开销没有按同一比例缩短。需要看它在**关键路径的占比与重复次数** ，使用 **Amdahl 分析** ，再做**模型实测** 。

??? question "47. 怎么用 profiler 证明优化真的生效？"

    先确认模型触发了**预期 shape** ，再确认 trace 中执行的是**新的 dispatch 路径** ，并比较**同一工作负载** 的 kernel 时间、调用次数、CPU 空洞及整体迭代时间。ROCm GPU profiler 与框架 trace 提供不同粒度证据，时间线上看见 kernel 名还不等于证明整个模型收益。

??? question "48. TTFT、ITL、TPOT、throughput 和 goodput 怎么区分？"

    TTFT 是**从请求提交到首 token**  的时间；ITL 是**相邻输出 token 的间隔** ；TPOT 是**生成区间的平均每 token 时间** ；throughput 是系统单位时间完成的工作量；goodput 只计算**满足指定延迟 SLO**  的有效工作。不同指标回答不同问题，不能用吞吐替代尾延迟。

??? question "49. 怎样设计 A/B，避免把启动和 batch 变化当作 kernel 收益？"

    **固定模型权重、硬件、输入输出长度** 、采样、cache/Graph/并行配置与请求负载，明确 **warmup 和计量窗口** 。微基准固定同一 shape 测算子；模型测试记录实际批次分布与多轮结果，解释收益究竟来自**同 shape 加速还是批次利用变化** 。

??? question "50. 面试官问“为什么 upstream 会接受这类 PR”，怎么答？"

    先说明**问题能复现** 且没有重复修复，再说明改动落在**共享契约边界** 、**范围明确并保留 fallback** 。正确性修复用**行为与数值对照** ，性能修改用支持边界和模型证据；提交后要能解释每行改动及评审取舍。
