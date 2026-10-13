---
title: "推理优化：面试问题与示例回答"
study_guide: true
---

# 推理优化

<div class="study-notes" markdown="1">

先沿请求执行理解组批、P/D 与 KV，再进入模型适配和投机推理。这些主题直接对应简历中的 vLLM 调度、Adaptive KV Offload、ROCm 接入与投机解码修复。

102 道题 · 12 个主题 · 答案展开 · 每题附原理分析。
{.study-meta}

前六组优先复习；TP/DP 与通信是部署基础，路由、PP/CP 和仿真放在后面深入。
{.study-intro}

## 一、请求流程、阶段特征与性能指标 {#request-metrics}

### 01. 一个请求进入 SGLang / vLLM 一类推理框架后，会经历什么流程？ {#q8 .study-question}

请求先经过 API 解析、模板处理和 tokenization，部署有 Router 时再路由到目标实例。Scheduler 根据队列、token 预算和 KV 空间安排执行，Worker 准备模型输入与 Attention 元数据，执行前向和采样。随后更新请求状态，返回 token，并在 <strong class="study-key">执行完成或安全取消后回收资源</strong>。

核心循环是 **调度、执行、更新状态**；具体进程和类的划分因框架与部署而异。

[原理分析：一个请求的执行闭环](analysis/03-推理框架请求流程.md#request-lifecycle)
{.study-analysis-link}

### 02. Router、Scheduler、Worker 分别负责什么？ {#q9 .study-question}

Router 主要决定 **请求送到哪个服务实例**，处理负载分配及必要的转发；Scheduler 决定实例内 **本轮执行哪些请求、推进多少 token**，受预算和缓存资源约束。Worker 持有模型与设备资源，准备张量、执行前向和相关采样操作。

路由不等于执行计划，执行计划也不等于 GPU 计算；三者之间需要传递请求状态和完成信息。

[原理分析：三类职责](analysis/03-推理框架请求流程.md#role-boundaries)
{.study-analysis-link}

### 03. Prefill 和 Decode 有什么区别？ {#q4 .study-question}

Prefill 处理已知的 prompt，通常一次计算多个输入 token，建立相应 KV Cache，并在 prompt 完整处理后产生首个生成 token。Decode 再把已生成 token 作为输入，复用历史 KV，继续逐步生成。

核心区别是 <strong class="study-key">每轮可一起计算的 token 数与权重复用程度</strong>；Prefill 更影响首 token 延迟，Decode 更影响 **生成阶段的 token 间隔**。

[原理分析：阶段与指标](analysis/02-Prefill与Decode.md#phase-semantics)
{.study-analysis-link}

### 04. 为什么 Prefill 通常更偏计算受限？ {#q5 .study-question}

Prefill 同时处理较多 token，Linear 的矩阵行数较大，**同一份权重可被多个 token 复用**，每读取一个字节能完成更多运算，更容易发挥矩阵计算单元的吞吐。因此它通常比小 batch Decode 更偏计算受限。

但这只是趋势，短 prompt、小矩阵或某些 Attention 路径仍可能受访存或提交开销限制，需要结合 **实际 shape 和算术强度** 判断。

[原理分析：为什么 Prefill 更偏计算？](analysis/02-Prefill与Decode.md#prefill-intensity)
{.study-analysis-link}

### 05. 为什么小 batch Decode 更容易受内存带宽限制？ {#q6 .study-question}

小 batch Decode 每轮只计算少量 token，却仍要访问很多层的权重，**权重复用少、算术强度低**。计算单元常在等数据，因此减少权重字节数或提高有效带宽可能比增加峰值算力更有效。历史 KV 读取也会随上下文增长成为明显成本。

具体是权重还是 KV 主导，要看 **batch、上下文长度和模型结构**。

[原理分析：为什么小 batch Decode 更偏带宽？](analysis/02-Prefill与Decode.md#decode-bandwidth)
{.study-analysis-link}

### 06. 为什么 GPU 峰值算力很高，Decode 仍然可能跑不满？ {#q7 .study-question}

峰值算力要求足够的数据供给和合适的矩阵形状；小 batch Decode 可能同时存在 **带宽不足、矩阵太小或并行任务不足**，短 kernel 还可能被 CPU 提交间隙影响。另外，自回归下一 token 依赖前一次采样，难以像 Prefill 一样并行处理同一请求的多个真实 token。

因此要区分 **算力利用率、带宽利用率与执行空隙**，再决定优化方向。

[原理分析：高峰值算力为何不等于高 Decode 性能？](analysis/02-Prefill与Decode.md#peak-vs-effective)
{.study-analysis-link}

### 07. TTFT 和 TPOT 分别衡量什么？ {#q11 .study-question}

TTFT 是请求提交到首个输出 token 的时间，包含 <strong class="study-key">排队、预处理、Prefill、采样和返回</strong>。TPOT 常指首 token 之后的平均生成间隔：输出 N 个 token 时，用 **最后 token 时间减首 token 时间，再除以 N−1**。它衡量持续生成速度，但平均值可能掩盖长停顿，还需看 ITL 尾延迟。

不同工具的计量边界要先统一。

[原理分析：TTFT 与 TPOT 的计量](analysis/02-Prefill与Decode.md#latency-metrics)
{.study-analysis-link}

### 08. 哪些优化主要改善 TTFT？ {#q12 .study-question}

先看首 token 关键路径：排队主导时优化 **入场调度、路由和 Prefill 容量**；重复前缀较多时用 prefix caching 减少计算；计算主导时优化 Prefill 的 Attention、GEMM 和并行执行。预处理、提交或传输慢则优化对应阶段。

Chunked Prefill 和 P/D 分离也可能改变 TTFT，但各有额外开销，不能认为开关一开就一定更快。

[原理分析：优化落在什么阶段？](analysis/02-Prefill与Decode.md#optimization-targets)
{.study-analysis-link}

### 09. 哪些优化主要改善 TPOT？ {#q13 .study-question}

优先降低每轮 Decode 的计算、数据读取与等待，例如 **权重或 KV 量化、Decode Attention 优化、Graph 和算子融合**；再通过合适的合批提高复用，用 Chunked Prefill 或 P/D 隔离减轻长 Prefill 干扰。合批过大、通信或资源竞争也可能拉长单轮时间，因此要在 **吞吐和生成延迟目标** 下测量，不能只看总 token/s。

[原理分析：优化落在什么阶段？](analysis/02-Prefill与Decode.md#optimization-targets)
{.study-analysis-link}

### 10. 推理框架优化和单 CUDA Kernel 优化有什么区别？ {#q10 .study-question}

Kernel 优化主要降低给定 shape 下的 **单次算子执行成本**，例如访存、计算和同步。框架优化面向 **请求的整条关键路径与资源利用率**，涉及组批、KV 管理、排队、提交和多实例协作。

局部 kernel 很快，若占比低或排队主导，端到端收益仍可能很小；反过来，框架改变 batch 和 shape，也会改变 kernel 的效率。

[原理分析：框架优化与 Kernel 优化](analysis/03-推理框架请求流程.md#system-vs-kernel)
{.study-analysis-link}

### 11. 如何平衡吞吐、TTFT 和 TPOT？ {#q25 .study-question}

先定义 TTFT、TPOT 与尾延迟目标，再在约束内最大化 <strong class="study-key">达标请求的有效吞吐，也就是 goodput</strong>。用 batch、Prefill token 预算、入场控制、路由和 P/D 配比调节资源：扩大 batch 可能提高吞吐，也可能增加等待和单轮时间。

通过代表性负载及到达率扫描比较曲线，不能只优化平均延迟或单个峰值。

[原理分析：怎样平衡吞吐和延迟？](analysis/06-Serving调度.md#serving-objectives)
{.study-analysis-link}

继续深入：[推理框架](analysis/03-推理框架请求流程.md) · [Prefill 与 Decode](analysis/02-Prefill与Decode.md) · [Serving 调度](analysis/06-Serving调度.md)
{.study-source}

## 二、组批、Graph 与异步执行 {#batching-execution}

### 12. Continuous Batching 解决什么问题，与普通动态组批有什么区别？ {#q76 .study-question}

固定生成批次里，短请求完成后可能留下空位，长请求继续占用，后到请求还要等整批结束。Continuous Batching 在 <strong class="study-key">模型迭代边界更新执行集合</strong>，移除完成请求并让新请求按预算加入，提高设备利用与服务吞吐。

相比只在入场时攒批，它能在生成过程中重组；但仍需维护 <strong class="study-risk">每请求进度、KV 与延迟约束</strong>，不能在任意正在运行的 kernel 中插入请求。

[原理分析：Continuous Batching 动态更新执行集合](analysis/18-组批与收益验证.md#continuous-batch)
{.study-analysis-link}

### 13. Chunked Prefill 是什么？ {#q22 .study-question}

Chunked Prefill 把长 prompt 的处理拆成 <strong class="study-key">多个模型迭代</strong>，每轮推进一段，并复用已算前缀的 KV。它让长 Prefill 在迭代边界提供重新调度机会，可以和 Decode 共同组批。

它不减少 prompt token 数，也不是把每段当成独立请求；正确性仍要维持完整上下文和模型依赖。

[原理分析：Chunked Prefill 的机制](analysis/06-Serving调度.md#chunk-progress)
{.study-analysis-link}

### 14. 为什么 Chunked Prefill 有利于在线 Serving？ {#q23 .study-question}

长 Prefill 一次执行会拉长模型迭代，让正在 Decode 的请求等待。切块后可 **限制每轮 Prefill 工作量**，让 Decode 在较短的迭代边界继续推进，改善生成间隔和公平性。

但小 chunk 增加调度与提交次数，也可能降低矩阵效率、拉长新请求 TTFT；chunk 大小要按 **实际耗时与延迟目标** 调整。

[原理分析：在线 Serving 为什么受益？](analysis/06-Serving调度.md#online-fairness)
{.study-analysis-link}

### 15. CUDA Graph 如何加速推理，动态请求怎么适配？ {#q68 .study-question}

CUDA Graph 记录可重放的设备工作，主要减少 <strong class="study-key">CPU 提交与多个短 kernel 的 launch 开销</strong>，不会自动减少模型算术量。推理常预先捕获若干 shape 桶，在稳定 buffer 上更新本轮数据；动态布局或不支持捕获的路径可分段或回退。

要维护 **地址、执行形状和依赖约束**，计入 padding、捕获内存与启动成本；不同请求可复用执行图，图里不应固化错误的请求状态。

[原理分析：Graph 减少提交成本，融合减少中间开销](analysis/15-芯片适配与执行优化.md#graph-fusion)
{.study-analysis-link}

### 16. 如何实现通信与计算重叠，怎样证明通信被隐藏？ {#q70 .study-question}

先找 **不依赖正在传输结果的计算**，用异步通信、stream/event 和缓冲安排并行推进；真正消费结果前再等待。重叠可能争用 SM、显存或链路带宽，提前同步也会让它失效。

用时间线检查未覆盖的通信等待与后续依赖，并比较关闭重叠时的总耗时；**trace 同时出现计算和通信不等于全部隐藏**，新增工作与资源竞争也要计入。

[原理分析：通信重叠必须有独立工作](analysis/15-芯片适配与执行优化.md#overlap-dependencies)
{.study-analysis-link}

继续深入：[Serving 调度](analysis/06-Serving调度.md) · [芯片适配与执行优化](analysis/15-芯片适配与执行优化.md)
{.study-source}

## 三、P/D 解耦与资源配置 {#pd-disaggregation}

### 17. 什么是 Prefill / Decode 分离？ {#q17 .study-question}

P/D 分离把请求的 prompt 处理和持续生成交给 <strong class="study-key">不同执行实例或资源池</strong>，Prefill 完成后把必要的 KV 和请求状态交接给 Decode 侧。它允许两阶段独立调度、配置与扩缩容，减少互相干扰。

这里改变的是执行边界；**Chunked Prefill 则改变单实例内的执行粒度**，两者可以结合。

[原理分析：分离改变什么？](analysis/05-PD分离.md#pd-boundary)
{.study-analysis-link}

### 18. 为什么 Prefill 和 Decode 适合部署到不同 Worker？ {#q18 .study-question}

两阶段的典型 shape、资源需求和延迟目标不同：Prefill 更重视 prompt 处理速度，Decode 更重视持续生成间隔。分开后可 **分别选择 batch、并行策略和资源数量**，减少大 Prefill 对 Decode 的阻塞。

但并不要求必须使用不同型号 GPU，也不能仅凭“一个偏计算、一个偏带宽”就认定分离一定划算。

[原理分析：分离改变什么？](analysis/05-PD分离.md#pd-boundary)
{.study-analysis-link}

### 19. P/D 分离一定会提升性能吗？ {#q19 .study-question}

不一定，收益取决于 <strong class="study-key">隔离干扰和独立调度的收益，能否覆盖交接与资源成本</strong>。如果原先干扰很小、KV 传输很大或一侧负载不足，分离反而可能降低利用率并增加延迟。应在相同资源或成本预算下，比较吞吐、TTFT、TPOT 和达标请求比例；不能把分离前后的单个峰值直接比较。

[原理分析：为什么不保证更快？](analysis/05-PD分离.md#pd-cost)
{.study-analysis-link}

### 20. P/D 分离会新增哪些通信和调度开销？ {#q20 .study-question}

主要新增 **KV 数据传输、目标缓存分配和请求交接**，还要维护两侧队列、状态、完成通知以及超时取消。缓存布局或并行配置不同可能需要重排，资源池不均衡会产生额外排队。独立实例通常还各自保存模型权重和缓存池。传输可能重叠，但其带宽竞争和生命周期管理成本仍需计入。

[原理分析：通信与调度开销](analysis/05-PD分离.md#communication-scheduling)
{.study-analysis-link}

### 21. Prefill Worker 的 KV 如何传给 Decode Worker？ {#q21 .study-question}

先协商请求身份、缓存布局和目标位置，Decode 侧准备空间；设备侧传输组件再通过本地互联或网络，把各层 KV 复制到目标缓存，采用 push 或 pull 都可以。必须确保 <strong class="study-key">源 KV 已写完、目标 KV 就绪后才读取</strong>，并保持源数据到传输安全结束。

首 token、采样进度和逻辑缓存映射也要交接，<strong class="study-risk">只传 block ID 不能代替 KV 数据</strong>。

[原理分析：KV 怎样安全交接？](analysis/05-PD分离.md#kv-handoff)
{.study-analysis-link}

### 22. Mix Chunk 与 P/D 分离有什么区别，TTFT/TPOT 一定怎样变化吗？ {#q77 .study-question}

若 Mix Chunk 指 **Chunked Prefill 与 Decode 混合调度**，它在同一执行组中限制 Prefill 粒度，给 Decode 留迭代机会；P/D 分离则把阶段交给不同资源池，并新增 KV 交接。前者可能减少生成干扰，也可能因小 chunk 拉长新请求 TTFT；后者可独立配置，但传输或配比失衡也会拖慢。

两者能结合，**指标方向取决于基线、负载和配置**，不是固定一升一降或全面改善。

[原理分析：Mix Chunk、Chunked Prefill 与 P/D 分离](analysis/18-组批与收益验证.md#mixed-chunk)
{.study-analysis-link}

### 23. 怎样决定 Prefill / Decode 的资源配比？ {#q54 .study-question}

先测代表性负载下每个 P/D 执行组的服务能力，按 **需求除以每组能力** 估算组数。若每组 P/D 处理能力为 μP、μD 请求/s，目标到达率 λ，则需让 `λ/(nPμP)`、`λ/(nDμD)` 低于设定利用率上限，近似均衡时 **`nP/nD≈μD/μP`**。

能力随长度、batch 和并行配置变化，还需满足 KV 容量、传输与 TTFT/TPOT，预留突发余量后在固定预算下实测。

[原理分析：配比取决于工作与服务能力](analysis/12-缓存层级与部署配置.md#pd-ratio)
{.study-analysis-link}

### 24. P/D Worker 应该固定绑定，还是按在途负载与 KV 容量选择？ {#q93 .study-question}

固定绑定简化交接与局部性，但可能形成 **局部排队和容量碎片**；动态选择能均衡，也增加目标预留、连接与状态协调。应按执行组估计在途和活动 Decode 工作，在交接前预留目标 KV，并把刚分配但尚未上报的请求计入负载。生成长度未知时保留扩容/抢占方案，取消或失败安全释放。

核心是 **容量准入、负载记账和交接契约**，请求数相同不代表负载相同。

[原理分析：Worker 绑定与目标容量准入](analysis/05-PD分离.md#admission-binding)
{.study-analysis-link}

### 25. P/D 分离下，TP、DP、EP 应该如何配置？ {#q63 .study-question}

P 与 D 侧应分别满足 **模型/KV 容量、计算形状和 TTFT/TPOT 目标**，再按工作负载选择 TP 与副本数；小 batch Decode 未必适合大 TP，Prefill 也不能只按峰值算力配置。MoE 还需明确 Attention 组与专家组，EP 不一定是独立设备维度。

两侧并行度不同可能要 **重新分片或复制 KV**，应计入交接兼容性与通信，在固定总资源下比较 goodput。

[原理分析：P/D 两侧并行配置分别建模](analysis/14-框架选择与并行部署.md#pd-parallel)
{.study-analysis-link}

继续深入：[P/D 分离](analysis/05-PD分离.md) · [缓存层级与部署配置](analysis/12-缓存层级与部署配置.md) · [框架选择与并行部署](analysis/14-框架选择与并行部署.md)
{.study-source}

## 四、KV 容量、加载与重算 {#kv-reuse}

### 26. KV Cache 的作用是什么，为什么能减少 Decode 的重复计算？ {#q14 .study-question}

KV Cache 保存各 Attention 层中历史 token 的 Key 和 Value。在标准因果模型中，历史 token 的表示不依赖未来新 token，因此 Decode 只需计算新增输入的 Q/K/V，<strong class="study-key">复用历史 K/V</strong>，无需重新前向整段历史。

它用缓存空间换重复计算，但新 Query 仍要与可见历史做 Attention，**缓存没有消除历史读取和加权求和**。

[原理分析：为什么历史 K/V 可以复用？](analysis/04-KVCache.md#causal-reuse)
{.study-analysis-link}

### 27. KV Cache 占用如何随序列长度增长？ {#q15 .study-question}

对普通全 Attention 模型，KV 载荷通常随 <strong class="study-key">缓存 token 总数线性增长</strong>。B 个请求均缓存 S 个 token 时，相同层配置主载荷为 `2×B×S×KV heads×head_dim×元素字节数×层数`；MHA 的 KV heads 等于 Q heads。

普通 head TP 可按 head 分片，但复制、前缀共享、量化、MLA 和分页会改变物理占用，应按 **实际布局** 估算。

[原理分析：怎样估算容量？](analysis/04-KVCache.md#kv-capacity)
{.study-analysis-link}

### 28. 长上下文下 KV Cache 为什么会成为瓶颈？ {#q16 .study-question}

一方面，缓存长度增加会占用更多内存，**减少可同时服务的请求数**，严重时需要抢占、重算或卸载。另一方面，dense Decode Attention 需要访问更长的历史，**KV 读取与 Attention 计算成本也增加**。

因此要区分容量瓶颈和执行瓶颈；KV 量化、分页管理或稀疏访问分别解决不同问题，并不是同一种优化。

[原理分析：长上下文的两种瓶颈](analysis/04-KVCache.md#long-context)
{.study-analysis-link}

### 29. 多级 KV / Prefix Cache 解决什么问题，主要代价是什么？ {#q55 .study-question}

把可复用缓存放到 **GPU、主机内存或更慢存储**，用较大容量换取缓存命中，减少重复 Prefill 或缓解 GPU 容量压力。命中后仍需将数据准备到执行所需位置，因此有 **查找、搬运、恢复布局和带宽竞争** 的成本。

Prefix Cache 保存可重用前缀，活动请求的 KV 卸载保存当前状态；两者关联但管理目标不同，应比较节省的计算与恢复成本。

[原理分析：多级缓存怎样节省计算？](analysis/12-缓存层级与部署配置.md#cache-tiers)
{.study-analysis-link}

### 30. CPU、磁盘或远端 Prefix Cache 什么时候比重新 Prefill 划算？ {#q87 .study-question}

对同一可复用前缀，比较 <strong class="study-key">恢复的增量关键路径成本与节省的重算成本</strong>，包括查找、读取、H2D、布局恢复、排队和竞争；可重叠时只计未覆盖部分。加载时间小于对应重算时间是简化条件，写回成本还要由未来命中摊薄。

应使用相同 suffix 和组批条件对照，<strong class="study-risk">多命中不一定更快</strong>，慢层恢复或挤占活动 KV 可能得不偿失。

[原理分析：比较的是恢复与重算的增量关键路径](analysis/20-RadixCache与分层恢复.md#load-vs-recompute)
{.study-analysis-link}

### 31. 多级 KV Cache 的预取、写回与驱逐如何避免拖慢服务？ {#q89 .study-question}

按 <strong class="study-key">预计复用收益、恢复耗时和带宽预算</strong> 决定预取及写回，设置等待边界或超时，低收益时回退重算；无用预取会占用队列和容量。驱逐只能回收未被执行或传输引用的缓存，写回/加载要有就绪状态，部分命中需保持连续有效前缀。

TP 多 rank 还要对齐可用长度与进度，核心是 **兼顾服务关键路径与数据生命周期**，不能把异步提交当作数据已可读。

[原理分析：预取、写回和驱逐相互制约](analysis/20-RadixCache与分层恢复.md#cache-io)
{.study-analysis-link}

### 32. 本地 Prefix Cache 未命中时一定重算吗，远端命中怎样取舍？ {#q102 .study-question}

只管理本地缓存的实现会 **重算未命中的后缀**；接有分层或跨实例恢复的系统还可查询共享目录、搬 KV，或将尚未执行的请求路由到有缓存的实例。需比较恢复、重算及排队的预计成本，并满足模型/布局兼容和容量。

重路由新请求与迁移活动请求是不同操作，后者还要交接进度；**图片中的不迁移只描述原实现**，不是推理框架的普遍规则。

[原理分析：未命中后重算、恢复还是换目标？](analysis/13-CacheAware路由与共享.md#miss-decision)
{.study-analysis-link}

### 33. KV Cache 量化如何落地，与权重量化有什么区别？ {#q69 .study-question}

KV 量化压缩 <strong class="study-key">运行时生成并反复读取的历史状态</strong>，与压缩固定模型权重不同。需要确定 K/V 的格式、scale 粒度和校准方式，在写缓存时量化，在 Attention 读取时按 backend 支持解码或直接低精度计算，并匹配分页及传输布局。

它可增加上下文和并发容量，但有 <strong class="study-risk">误差、scale 与转换成本</strong>；降低缓存字节数不保证单请求更快，应测长上下文质量、实际流量和 TTFT/TPOT。

[原理分析：KV 量化的写入与读取路径](analysis/15-芯片适配与执行优化.md#kv-quantization)
{.study-analysis-link}

### 34. Radix Cache 如何按 Token ID 做最长前缀匹配并管理 KV？ {#q86 .study-question}

Radix tree 用 **一段 token 序列作为边标签**，沿请求开头逐段比较，找到最长连续匹配前缀；分叉落在边中间时可拆节点，共享前缀并关联实际 KV 页/槽。执行前锁定或增加引用，完成后释放引用，淘汰只回收允许驱逐且无在途使用的数据。

匹配还要满足 **模型、adapter、位置及其他输入状态一致**，不是文本相似或任意公共子串就能复用。

[原理分析：Radix Tree 索引的是连续前缀状态](analysis/20-RadixCache与分层恢复.md#radix-prefix) · [引用与驱逐](analysis/20-RadixCache与分层恢复.md#reference-eviction)
{.study-analysis-link}

### 35. HiCache 是什么，与 Radix Cache 及 GPU L1/L2 有什么关系？ {#q88 .study-question}

HiCache 是 SGLang 的 **分层 KV 缓存机制**，将 GPU、主机与可选外部存储的容量组织起来，配合前缀索引、写回、预取和恢复。Radix Cache 负责可复用前缀及状态映射，分层机制负责数据驻留和搬运；具体 backend 和策略以版本为准。

文档中的 L1/L2/L3 通常指 **GPU/Host/外部存储层级**，不是硬件 GPU L1/L2 cache，项目名称也不能证明实现相同。

[原理分析：HiCache 的层级与 GPU Cache 不同](analysis/20-RadixCache与分层恢复.md#hicache-tiers)
{.study-analysis-link}

### 36. 跨实例共享 KV Cache 需要共享什么，主要成本在哪里？ {#q60 .study-question}

共享索引只说明 **缓存在哪里**，执行实例还必须能获得实际 KV 或模型所需状态。可以复制、拉取或从共享层恢复，同时匹配模型、adapter、位置和缓存布局，维护有效长度、引用及就绪通知。

传输、重排与带宽竞争可能超过重算成本；**命中远端缓存不等于本地立即可用**，路由必须计入恢复成本并处理缓存被淘汰或源实例失败的情况。

[原理分析：共享索引与共享数据不同](analysis/13-CacheAware路由与共享.md#directory-vs-data)
{.study-analysis-link}

继续深入：[KV Cache](analysis/04-KVCache.md) · [Radix Cache 与分层恢复](analysis/20-RadixCache与分层恢复.md)
{.study-source}

## 五、模型适配与端侧执行 {#model-backend}

### 37. 开源推理框架如何适配自研芯片，框架岗位主要做什么？ {#q66 .study-question}

框架岗要把 <strong class="study-key">模型、调度、缓存与设备执行接成正确高效的服务闭环</strong>。适配时先对接 device/runtime、内存分配、stream/event，再实现或接入 Attention、Linear、Norm、MoE 等 backend，并匹配权重和 KV 布局。随后补分布式通信、图执行及量化等能力，用基础前向到完整 Serving 分层验证。

核心是 **适配执行契约与生命周期**，不是只把 CUDA 算子换成芯片 API。

[原理分析：从最小可运行模型到完整服务](analysis/15-芯片适配与执行优化.md#backend-contract)
{.study-analysis-link}

### 38. 跨芯片精度对齐应该怎样做，为什么不能只比较生成文本？ {#q67 .study-question}

先统一 <strong class="study-key">权重、dtype、输入处理、位置、mask 和采样设置</strong>，在相同历史上比较关键中间张量与 logits，定位误差最早出现的位置；再比较自由生成和任务质量。浮点归约顺序、量化和 fused kernel 都可能产生容差内差异，而小 logits 差异也可能让后续 token 分叉。

因此要区分 <strong class="study-risk">数值误差、采样差异与质量退化</strong>，性能优化后沿相同边界回归验证，不能把逐 token 完全相同当唯一标准。

[原理分析：精度对齐先定位首次误差](analysis/15-芯片适配与执行优化.md#numerical-alignment)
{.study-analysis-link}

### 39. 为什么要做端侧大模型推理？ {#q1 .study-question}

端侧推理把模型放到设备本地执行，主要价值是 **降低网络依赖和请求往返延迟**，支持离线运行，并减少原始数据上传。代价是设备的 **内存、功耗和散热预算更紧**，所以通常需要选合适的模型规模、量化和执行方式。是否适合端侧，要结合任务质量、响应时间与设备成本判断。

[原理分析：为什么做端侧推理？](analysis/01-端侧推理.md#edge-purpose)
{.study-analysis-link}

### 40. 端侧部署与云端部署最大的区别是什么？ {#q2 .study-question}

最大区别是 **资源约束与服务目标不同**。端侧通常在有限内存和持续功耗预算下，满足本地交互延迟与能耗要求；云端通常利用更大的算力、内存和多卡资源，同时服务多个用户，兼顾吞吐、尾延迟和成本。

两者都要优化计算和访存，但端侧更重视 **单设备可运行性和持续性能**，云端更重视并发调度与资源利用率。

[原理分析：端侧与云端部署的区别](analysis/01-端侧推理.md#edge-cloud)
{.study-analysis-link}

### 41. 端侧大模型推理的性能瓶颈通常在哪里？ {#q3 .study-question}

不能只看设备峰值算力，要按阶段和实际负载判断。小 batch Decode 常受 **权重和 KV 的内存访问** 限制；较长 Prefill 可能受矩阵计算限制，小算子还可能受提交开销影响。端侧另外要考虑 **容量、共享内存带宽和功耗散热**，持续运行可能降频。

应拆开阶段耗时，并结合带宽、算力利用和频率确认瓶颈。

[原理分析：怎样定位端侧瓶颈？](analysis/01-端侧推理.md#edge-bottlenecks)
{.study-analysis-link}

### 42. CPU/GPU/NPU/DLA 异构场景怎样设计统一调度框架？ {#q83 .study-question}

把工作表示为 **带数据依赖的任务 DAG**，为设备 backend 统一提交、完成通知和 buffer 生命周期接口，同时描述算子支持、dtype/layout、内存可达性和队列能力。Scheduler 在依赖满足后，结合预计计算、转换/搬运、排队与 deadline 选设备，资源不足时回压。

要优化 **整条流水线的吞吐、尾延迟和功耗**，不能把不同设备峰值算力相加，或为提高利用率盲目拆分模型。

[原理分析：统一接口，显式保留设备约束](analysis/19-异构调度与跨帧缓冲.md#heterogeneous-dag)
{.study-analysis-link}

### 43. 跨硬件流水线为何会产生跨帧数据读写竞争？ {#q84 .study-question}

多帧复用同一 buffer 时，生产者可能写下一帧，而消费者仍在读上一帧；异步提交后 **Host 函数返回不等于设备读写完成**，不同引擎的完成顺序也未必一致。必须建立写完到读、所有读完到覆盖的依赖，并维护 frame ID、版本与所有权；非一致性内存还需对应可见性协议。

核心是 **按真实设备完成释放 buffer**，只加 CPU 锁或固定延时不足以证明安全。

[原理分析：单缓冲竞争的最小例子](analysis/19-异构调度与跨帧缓冲.md#buffer-race)
{.study-analysis-link}

### 44. 全量拷贝、双缓冲和单缓冲时序管控分别有什么代价？ {#q85 .study-question}

全量拷贝为消费者保留快照，但增加 **搬运字节、暂存与带宽占用**，复制本身也要避开源覆盖；双缓冲交替读写，允许一定跨帧重叠，但增加容量，慢消费者仍会迫使等待。单缓冲可省额外数据副本，通过完成事件保证读完再覆盖；同一区域的生产消费通常会串行，除非能证明区域依赖独立。

应比较 **内存、带宽与可重叠程度**，不能将零拷贝等同于零同步成本。

[原理分析：三种方案放在同一个边界比较](analysis/19-异构调度与跨帧缓冲.md#buffer-strategies)
{.study-analysis-link}

继续深入：[芯片适配与执行优化](analysis/15-芯片适配与执行优化.md) · [端侧推理](analysis/01-端侧推理.md) · [异构调度与跨帧缓冲](analysis/19-异构调度与跨帧缓冲.md)
{.study-source}

## 六、投机推理与状态正确性 {#speculation}

### 45. 投机推理怎样生成和验证多个候选 Token？ {#q78 .study-question}

先由便宜的 drafter 提出候选，目标模型用 <strong class="study-key">已知候选与因果依赖一次验证多个位置</strong>，接受合法前缀，在拒绝处纠正，再提交 token 并修正 KV/状态。它用额外计算换取减少目标模型的串行轮次。

Greedy 可按目标 token 一致性验证；随机采样要使用 **正确的接受与纠正分布** 才能保持目标分布，不能只要求某个概率阈值通过。

[原理分析：为什么验证可以并行多个位置？](analysis/17-投机推理与验证.md#draft-verify) · [接受与纠正规则](analysis/17-投机推理与验证.md#exact-sampling)
{.study-analysis-link}

### 46. 常见投机推理方案主要差在哪里？ {#q79 .study-question}

独立小模型需要自己的权重和状态；EAGLE 类方法利用目标隐藏状态训练 drafter；MTP 利用模型的多 token 预测模块；n-gram/后缀复用从已有 token 模式找候选。候选还可按 **串行链、并行块或树** 组织，验证成本不同。

选择应比较 **draft 成本、连续接受长度、验证开销与资源预算**，具体版本和采样支持另核对，不能仅按方法名字判断哪种最快。

[原理分析：方案差异沿三个维度判断](analysis/17-投机推理与验证.md#proposal-types)
{.study-analysis-link}

### 47. 投机推理什么时候划算，接受率越高一定越快吗？ {#q80 .study-question}

关键是 **每个最终提交 token 的总成本**：若一轮 draft、verify、状态更新耗时为 Td、Tv、To，平均提交 A 个 token，需让 `(Td+Tv+To)/A` 优于普通 Decode 的对应成本。候选多会增加验证、KV 与调度，连续前缀接受长度也不同于平均位置命中率。

低 batch 带宽受限时常有机会，大 batch 计算繁忙时可能回退，最终看 **同负载的延迟、吞吐和质量**。

[原理分析：接受率与总收益不同](analysis/17-投机推理与验证.md#committed-cost)
{.study-analysis-link}

继续深入：[投机推理](analysis/17-投机推理与验证.md)
{.study-source}

## 七、TP/DP 与模型切分 {#tp-dp}

### 48. DP 和 TP 有什么区别，DP 是否复制完整模型？ {#q37 .study-question}

DP 用多个 **逻辑模型副本处理不同请求或数据批次**；TP 用多个 rank 共同切分并执行同一层。纯 DP 通常每卡保存完整模型，而 DP 与 TP/PP 组合时，每个 DP 副本内部仍可分片，完整模型由一组设备共同持有。

单卡放不下模型，**单独增加 DP 不能解决**，需量化、卸载或模型并行等手段。

[原理分析：DP 的复制范围](analysis/09-DP与PipelineParallel.md#dp-replica)
{.study-analysis-link}

### 49. Tensor Parallel 怎么切 Linear、Attention 和 FFN？ {#q26 .study-question}

TP 在同一层内切分参数和计算，多个 rank **共同处理同一批 token**。Linear 可以切输出通道或输入特征；Attention 常按 head 切 QKV 和计算，再合并输出投影的部分和；FFN 常把扩张投影按输出维切、收缩投影按输入维切。

它降低每卡参数和部分计算量，但会新增 **层内集合通信**，不能理解成各卡独立处理不同请求。

[原理分析：Column 与 Row 的逻辑维度](analysis/07-TensorParallel.md#column-row) · [Attention Head 切分](analysis/07-TensorParallel.md#head-sharding)
{.study-analysis-link}

### 50. Column Parallel 与 Row Parallel 分别切什么，通信在哪里？ {#q27 .study-question}

统一写成 `Y=XW`，W 为 `[输入维, 输出维]`。Column Parallel <strong class="study-key">切输出维</strong>，下游能消费分片时不必立即 gather；Row Parallel **切输入维**，计算完整输出的部分和后用 All-Reduce 或 Reduce-Scatter 合并。需要完整且适合复制的较小投影或逐元素路径可保留复制，但应说明布局与成本。

框架权重可能转置存储，所以按逻辑维度判断；Norm、残差等也要看是否配合 SP。

[原理分析：Column 与 Row 的逻辑维度](analysis/07-TensorParallel.md#column-row)
{.study-analysis-link}

### 51. QKV 和 Attention Head 怎么做 TP？ {#q28 .study-question}

QKV 投影通常沿输出通道切分，让每个 rank 持有 **对应的 Query heads 和 KV heads**，本地执行这些 head 的 Attention；输出投影常用 Row Parallel，再归约部分和。MHA 的 head 通常可分片，GQA/MQA 的 KV heads 较少时可能需要 **复制 KV heads 或特殊映射**。

不能假设 Query heads 和 KV heads 总能同样按 TP 等分。

[原理分析：QKV 与 Attention Heads](analysis/07-TensorParallel.md#head-sharding)
{.study-analysis-link}

### 52. FFN 两层如何组合切分？ {#q29 .study-question}

常见 FFN 先用 Column Parallel 将 hidden H 扩张到分片的中间维 F/p，本地执行激活，再用 Row Parallel 从这份分片投回 H，最后 **归约各 rank 的输出部分和**。SwiGLU 的 gate 和 up 要按同一中间通道对应切分，才能本地相乘。

这样激活前通常不需恢复完整 F，减少中间 **All-Gather**。

[原理分析：FFN 的连续切分](analysis/07-TensorParallel.md#ffn-pair)
{.study-analysis-link}

### 53. 为什么经常采用 Column Parallel + Row Parallel？ {#q30 .study-question}

因为前一层按输出维产生的分片，正好能作为后一层按输入维的局部输入，**中间计算保持分片**，避免先 gather 完整激活再重新切开。最终只在需要完整输出或指定布局时合并部分和。这适合 FFN 和 Attention 输出投影等结构，但要确认激活、残差和其他算子能否消费该布局。

[原理分析：FFN 的连续切分](analysis/07-TensorParallel.md#ffn-pair) · [FFN 配对切分](analysis/07-TensorParallel.md#ffn-pair)
{.study-analysis-link}

### 54. TP Size 越大是否一定越快？ {#q31 .study-question}

不一定。更多 TP rank 能减少每卡权重和部分计算，但会让 **本地 GEMM 更小、集合通信参与者更多**，并受到 head、分组与布局限制。小 batch Decode 的通信启动延迟尤其可能难以摊薄；跨节点拓扑也可能使收益回退。

应比较同一工作负载的 **单步延迟、显存和吞吐**，选择满足容量且综合效率合适的 TP。

[原理分析：TP 扩大后的取舍](analysis/07-TensorParallel.md#tp-tradeoff)
{.study-analysis-link}

### 55. 给定 TP=8，怎样推导 Prefill 的 Attention 与 MLP 张量维度？ {#q74 .study-question}

先约定普通 head TP、无 SP/CP，T 为本轮 Prefill token 行数，H 为 hidden，F 为 FFN 中间维。输入通常为 **`[T,H]`**；32 个 Q heads、8 个 KV heads、head_dim=128 时，每卡 Q/K/V 输出为 `[T,512]`、`[T,128]`、`[T,128]`，Attention 输出 `[T,512]` 经 Row 投影产生 `[T,H]` 部分和再归约。

MLP 中间为 **`[T,F/8]`**；若有 KV 复制、EP 或序列分片则按实际布局另推。

[原理分析：TP=8 的 Prefill 维度算例](analysis/07-TensorParallel.md#tp8-shapes)
{.study-analysis-link}

### 56. 为什么 DP 提升吞吐，是否降低单请求延迟？ {#q38 .study-question}

多个副本能并行处理不同请求，增加总服务容量，因此在负载和路由合适时 **提高系统吞吐、减少排队**。但每个请求通常只由一个副本执行，DP 本身不拆它的计算，不直接降低单次 forward 时间。

观察到延迟下降时要区分 **排队改善与模型执行加速**；数据供给、路由和其他共享资源也可能限制扩展。

[原理分析：DP 与 TP 的资源选择](analysis/09-DP与PipelineParallel.md#dp-vs-tp)
{.study-analysis-link}

### 57. 什么时候应该增加 DP，而不是 TP？ {#q39 .study-question}

已有副本能够容纳模型并满足单请求执行延迟，需求主要是更多并发时，通常优先增加 **DP 副本扩服务容量**。如果容量不足或单请求计算需要跨卡协作，再考虑 TP/PP。

过大的 TP 可能让矩阵变小、通信占比上升；在固定 GPU 数下应比较 **少量大 TP 副本与更多小 TP 副本** 的 goodput。

[原理分析：DP 与 TP 的资源选择](analysis/09-DP与PipelineParallel.md#dp-vs-tp)
{.study-analysis-link}

### 58. Serving DP 和训练 DP 有什么不同？ {#q40 .study-question}

训练 DP 各副本用不同数据计算梯度，再 **同步梯度并更新一致参数**；经典复制式 DP 通常复制参数，状态分片方案需另外说明。Serving 通常使用固定权重，各副本独立服务不同请求，**没有每步梯度 All-Reduce**。

但某些模型或引擎仍有路由、MoE 和调度通信，不能说 Serving DP 永远完全不通信。

[原理分析：DP 的复制范围](analysis/09-DP与PipelineParallel.md#dp-replica)
{.study-analysis-link}

### 59. TP=2 的吞吐或效率衰减怎样计算，有没有固定比例？ {#q97 .study-question}

先明确 **单副本时延、系统吞吐与每卡效率**。单卡和 TP=2 同一请求耗时为 t₁、t₂ 时，latency speedup=t₁/t₂，理想两卡效率可写为 speedup/2；容量允许时，固定两卡预算还应与两个单卡 DP 副本比较。

通信、本地 shape、batch 和 KV 复制都会改变收益，**没有通用衰减百分比**，单卡放不下模型时也不能虚构单卡实测基线。

[原理分析：TP=2 的“衰减”先定义基线](analysis/22-配置搜索与吞吐延迟.md#tp-efficiency)
{.study-analysis-link}

继续深入：[Tensor Parallel](analysis/07-TensorParallel.md) · [DP 与 PP](analysis/09-DP与PipelineParallel.md)
{.study-source}

## 八、集合通信与传输成本 {#communication}

### 60. All-Reduce、Reduce-Scatter、All-Gather 分别做什么，TP 为什么需要它们？ {#q32 .study-question}

All-Reduce 将各 rank 的对应元素归约，并让每卡得到完整结果；Reduce-Scatter **归约后只保留结果分片**；All-Gather 将各卡分片拼成完整张量。Row Parallel 产生部分和，需要前两种之一合并；下游需要完整输入时可能 gather。

是否使用三者取决于 **张量布局和并行方案**，并不是每个 TP Linear 后都固定执行三种通信。

[原理分析：归约与布局](analysis/08-集合通信与成本.md#collective-semantics)
{.study-analysis-link}

### 61. 给定 Tensor Shape，如何计算通信多少 Byte？ {#q33 .study-question}

先算 **元素总数乘实际通信 dtype 的字节数**，得到逻辑张量载荷 S；再说明 collective、算法、rank 数以及统计的是每卡发送、收发之和还是全组流量。例如 `[8,128,4096]` 的 BF16 是 8 MiB，但一次 Ring All-Reduce 的每卡发送量不是总等于 8 MiB，而是 **`2(p−1)S/p`**，每卡接收量相同。

[原理分析：明确通信统计口径](analysis/08-集合通信与成本.md#traffic-accounting)
{.study-analysis-link}

### 62. TP=2 与 TP=4 的 All-Reduce 通信量有什么变化？ {#q34 .study-question}

固定逻辑载荷 S，按 Ring All-Reduce，每卡发送量在 TP=2 时为 **S**，TP=4 时为 **1.5S**，接收量分别相同。Ring 的阶段数也从 2 增到 6，因此启动开销会变化。

这个比较是假设算法和载荷不变；真实 backend 可能换算法，且 **通信时间还取决于拓扑、有效带宽与重叠**，不能只按字节比例推断延迟。

[原理分析：Shape 与 TP=2/4 算例](analysis/08-集合通信与成本.md#ring-example)
{.study-analysis-link}

### 63. TP 通信量与 Batch、Sequence Length、Hidden Size 有什么关系？ {#q35 .study-question}

常见输出归约的逻辑形状为 `[M,H]`，载荷是 **M×H×元素字节数**。固定长度 Prefill 中 M 可为 B×S；普通 Decode 中 M 通常约为本轮请求数 B，**不能再乘历史上下文长度**。

真实调度、padding 和分片会改变 M；层数和 collective 次数决定总量，历史长度则主要影响 Attention/KV，CP 等方案需要另算。

[原理分析：Batch、序列和 Hidden 的关系](analysis/08-集合通信与成本.md#token-scaling)
{.study-analysis-link}

### 64. 怎么判断推理已进入 Communication Bound？ {#q36 .study-question}

看关键路径中 **未被计算覆盖的通信等待** 是否主导，而不是只看 trace 里 collective 很长。固定负载比较不同 TP、拓扑和 batch，检查通信字节、时长、计算空隙及端到端时间。

还要区分互联带宽不足、启动延迟、rank 到达不同步和网络竞争；如果增加 TP 后计算缩短，却被通信抵消，说明扩展受到通信限制。

[原理分析：Communication Bound 的证据](analysis/08-集合通信与成本.md#communication-bound)
{.study-analysis-link}

### 65. PCIe、NVLink、RDMA 和 GPUDirect RDMA 有什么区别？ {#q64 .study-question}

PCIe 是设备互联总线；NVLink 是 NVIDIA 的高带宽互联技术，GPU 拓扑与路径因系统而异。RDMA 是 **通过网络直接访问注册内存的机制**，常用于跨节点；GPUDirect RDMA 让支持的 NIC 路径访问 GPU 内存，减少主机暂存。

它们分属链路与访问机制，不能完全互斥分类；NCCL 等库再组织集合通信，**实际性能取决于路径、带宽、消息大小与同步**。

[原理分析：通信方式不是同一层次的选项](analysis/14-框架选择与并行部署.md#transport-layers)
{.study-analysis-link}

### 66. 为什么 KV 字节数除以带宽不足以预测传输延迟，跨 NUMA 为什么要实测？ {#q94 .study-question}

`bytes/bandwidth` 只是 **同路径下的载荷估算**，还缺启动、分块/布局、页锁定或暂存、同步、排队与共享链路竞争；传输重叠时也不等于全部计入请求延迟。跨 NUMA 可能经过远端主机内存、CPU 互联或不同 PCIe/NIC 路径，标称带宽不能代表可达有效速率。

应按 **实际端点、方向、大小和并发** 测传输曲线，并区分独立与负载下的性能。

[原理分析：传输估计需要真实路径](analysis/21-推理仿真与性能模型.md#transport-path)
{.study-analysis-link}

继续深入：[集合通信](analysis/08-集合通信与成本.md) · [框架选择与并行部署](analysis/14-框架选择与并行部署.md)
{.study-source}

## 九、MoE 与专家并行 {#moe-parallel}

### 67. Expert Parallel 是什么，与 TP 有什么区别？ {#q49 .study-question}

EP 把 **不同专家的参数分配到不同 rank**，token 按 Router 的选择送到专家所在设备；TP 则切分同一矩阵，让多卡共同完成计算。EP 常与 Attention 的 DP/TP 组合，专家内部也可以另做 TP。

它能分担专家权重容量，但会新增 **token dispatch/combine 通信和负载不均衡**；不能把 EP 理解为每卡独立保存完整 MoE 模型。

[原理分析：专家切分与 Token 流程](analysis/11-ExpertParallel与负载均衡.md#expert-token-flow)
{.study-analysis-link}

### 68. EP 的 Dispatch、专家计算和 Combine 分别做什么？ {#q50 .study-question}

Router 先选 top-k 专家并给出路由权重，Dispatch 将 **token activation 与路由元数据送到目标专家**，按专家聚集执行 FFN；Combine 把结果送回原 token 所在位置，再按权重合并。常用 All-to-All 或专用后端，交换的不是历史 KV。

一个 token 可有多个专家结果，需维护对应关系；按全组 T、top-k、H、字节数 s，输入与返回各约 `TkHs`，**实际跨卡量还取决于远端比例与实现**。

[原理分析：专家切分与 Token 流程](analysis/11-ExpertParallel与负载均衡.md#expert-token-flow)
{.study-analysis-link}

### 69. EP 为什么容易负载不均衡，应该看哪些指标？ {#q51 .study-question}

Router 的选择随 token 内容变化，少数专家可能变热；即使专家数量均分，每卡收到的 token 数和计算时间仍可能不同。一个 EP 组需要协同完成交换与计算，**慢 rank 会拖长整轮执行**。

应看每专家 token 数、每 rank 耗时、通信等待和尾延迟，并区分 **路由偏斜、专家放置与硬件差异**，不能只看平均吞吐。

[原理分析：负载不均衡在哪里产生？](analysis/11-ExpertParallel与负载均衡.md#rank-skew)
{.study-analysis-link}

### 70. 如何缓解 Serving 中的 EP 负载不均衡？ {#q52 .study-question}

可以根据运行统计 **调整专家到 rank 的映射、复制热点专家并分担 token**，结合组批及通信计算重叠改善执行。复制同一专家并正确路由可保留模型语义，但会增加权重容量、搬运和映射维护成本。

训练中的均衡目标与推理时的专家放置是两层问题；**不能为了均衡任意改 top-k 或丢弃 token**，否则可能改变输出质量。

[原理分析：运行时均衡与模型均衡](analysis/11-ExpertParallel与负载均衡.md#physical-balancing)
{.study-analysis-link}

### 71. MoE 的 DP、TP、EP Size 能直接相乘，是否有通用配置等式？ {#q53 .study-question}

先说明 **各层的通信组由哪些 rank 构成**。在 vLLM 文档描述的 EP 部署中，专家组覆盖 Attention 的 DP×TP ranks，所以 **`EP_size=DP_size×TP_size`**；EP 是复用这些设备，不能再乘一次。8 卡可配置 Attention DP=8、TP=1，或 DP=4、TP=2，两者 EP 都为 8，但计算和路由行为不同，性能也不必相同。

其他框架或专家内部 TP 的关系需另查。

[原理分析：DP、TP、EP 不能盲目相乘](analysis/11-ExpertParallel与负载均衡.md#parallel-groups)
{.study-analysis-link}

### 72. DP Attention 是什么，在 MoE 部署中解决什么问题？ {#q71 .study-question}

DP Attention 让不同 rank 或小 TP 组处理 **不同请求/token 的 Attention 和本地 KV**，专家层则可由更大的 EP 组协作。这样可以避免某些大 TP 的 KV 复制和过细 head/矩阵切分，改善每卡工作与容量。

它是 **按层选择不同并行方式**，不意味着每卡复制全部 MoE 专家；收益还要权衡 Attention 权重复制、入口负载和专家通信。

[原理分析：DP Attention 改变的是层级并行布局](analysis/16-DPAttention与TBO.md#attention-groups)
{.study-analysis-link}

### 73. 为什么某些纯 TP 路径会产生 KV 冗余，所有 TP 都这样吗？ {#q72 .study-question}

普通 MHA 的 head TP 可正常分片 KV；但 **GQA/MQA 的 KV heads 少于 TP** 时，某些实现需要复制 KV heads，多个 rank 保存相同历史。MLA 按 Query heads 切分时也可能让各 rank 缓存同一共享 latent，而未分片布局还会带来其他重复。

因此要按 **实际 KV 布局和复制倍数** 计算物理占用，不能说所有 TP 都复制完整 KV；DP Attention 或 CP 也需按模型与实现评估。

[原理分析：TP 的 KV 冗余有哪些来源？](analysis/16-DPAttention与TBO.md#kv-replication)
{.study-analysis-link}

### 74. TBO（Two Batch Overlap）是什么，拆 Batch 后怎样掩盖通信？ {#q73 .study-question}

TBO 把一批工作拆成 **两个有独立请求/token 状态的 micro-batch**，交错执行，让一份的 Attention/GEMM 等计算覆盖另一份的 dispatch/combine 等通信。它不解除同一份工作的依赖，也不把同一请求未来未知 token 直接并行执行。

拆分会改变 GEMM shape，并增加调度、缓冲与同步成本，所以收益取决于 **独立工作、计算通信配比和资源竞争**，不能保证通信量减少或一定更快。

[原理分析：TBO 怎样提供独立工作？](analysis/16-DPAttention与TBO.md#two-batch-pipeline) · [TBO 的额外成本](analysis/16-DPAttention与TBO.md#overlap-cost)
{.study-analysis-link}

### 75. TP 与 EP 的负载不均衡有什么区别？ {#q65 .study-question}

常规 dense TP 的 rank 处理 **同一批 token 的不同矩阵分片**，维度等分时逻辑工作较规则，偏斜常来自形状、设备、拓扑或到达时间。EP 按内容路由到不同专家，**token 数本身会动态偏斜**，均分专家数也不能保证均衡。

TP 要检查分片与各 rank 关键路径，EP 还要统计专家热度和映射；专家内部 TP 与 EP 组合时，两类问题可能同时存在。

[原理分析：TP 与 EP 偏斜从哪里来？](analysis/14-框架选择与并行部署.md#tp-ep-skew)
{.study-analysis-link}

继续深入：[Expert Parallel](analysis/11-ExpertParallel与负载均衡.md) · [DP Attention 与 TBO](analysis/16-DPAttention与TBO.md)
{.study-source}

## 十、路由与服务负载 {#routing}

### 76. 多个 Decode Worker 应该怎样调度请求？ {#q24 .study-question}

应按实例的 **预计执行时间、KV 容量和交接成本** 选择目标，不能只数请求。长 prompt 增加 Prefill 工作，长历史和长输出增加 Decode 负载；可结合长度分组与实时反馈分配 DP 副本，同时利用前缀缓存局部性。

单个副本可能包含多个 TP rank，请求应路由到副本执行组；迁移已有请求需要搬 KV 和状态，不能只移动一个请求 ID。

[原理分析：多个 Decode 实例如何路由？](analysis/06-Serving调度.md#decode-routing)
{.study-analysis-link}

### 77. Cache-aware 路由为什么可能把实例变成热点？ {#q57 .study-question}

如果总把请求送给前缀命中最多的实例，热门前缀会不断吸引新请求，形成 **缓存局部性与负载集中相互强化**。省下的 Prefill 可能被排队、Decode 干扰和 KV 容量压力抵消。

路由要同时估计 **可复用缓存与已分配、即将进入的工作**，必要时限制过载实例；状态上报滞后时还需及时记录刚分配的请求，避免多个路由器同时涌向同一目标。

[原理分析：省 Prefill 不等于请求更快](analysis/13-CacheAware路由与共享.md#cache-hotspot)
{.study-analysis-link}

### 78. Cache Score 与 Load Score 应该怎样组合？ {#q58 .study-question}

先统一分数方向和尺度：例如归一化命中收益为 C、负载成本为 L，可用 **`score=C−λL`，选高分**。也可直接比较预计完成时间，把排队、未命中 Prefill、缓存恢复和执行成本放到同一时间单位。

负载不能只数请求，要看 token 工作、活动 Decode、KV 和硬件能力；**容量或服务约束应先作为准入条件**，不能仅靠软分数抵消。

[原理分析：评分需要方向、尺度和准入条件](analysis/13-CacheAware路由与共享.md#score-units)
{.study-analysis-link}

### 79. Cache-aware 路由中的 λ 应该如何确定？ {#q59 .study-question}

λ 控制 **缓存收益与负载惩罚的相对权重**，大小依赖分数定义，没有通用常数。先固定单位、标定负载估计，再在有代表性的前缀复用、长度分布和到达率下扫描 λ，用 **TTFT/TPOT 约束内的 goodput** 选值。

还要检查热点突发、低命中和异构实例；动态调参需有平滑与反馈延迟控制，避免负载来回摆动。

[原理分析：λ 与反馈怎样验证？](analysis/13-CacheAware路由与共享.md#lambda-feedback)
{.study-analysis-link}

### 80. Prefix Cache 命中率提高，为什么延迟仍可能变差？ {#q61 .study-question}

命中直接减少 Prefill 工作，可能改善 TTFT，但不直接省去后续 Decode 计算；TPOT 可因 **负载分配与减少 Prefill 干扰** 间接受益。命中率未体现排队、传输和实例偏斜，很多短前缀命中也可能省不了多少工作。

应同时看请求/token 命中、TTFT/TPOT 尾延迟、goodput 和传输，**较低命中但更均衡的路由也可能更快**。

[原理分析：为什么不能只优化命中率？](analysis/13-CacheAware路由与共享.md#hit-vs-latency)
{.study-analysis-link}

### 81. 为什么相比轮询收益很大，相比 Load-only 路由收益却很小？ {#q81 .study-question}

轮询基线可能已有严重负载偏斜，Load-only 已解决大部分排队问题，混合策略的 **额外缓存局部性收益** 就可能较小。应在相同负载下比较轮询、仅负载、仅缓存与混合策略，区分排队和 Prefill 节省，并报告 **P50/P99、goodput 与重复测量的不确定性**。

不同基线的百分比不能直接相加，单个 P99 大幅下降也不能外推到所有请求。

[原理分析：不同基线的百分比为何差很多？](analysis/18-组批与收益验证.md#baseline-gap)
{.study-analysis-link}

### 82. 为什么缓存路由常在长 Prompt 收益更大，能否给出固定长度阈值？ {#q82 .study-question}

较长且可复用的前缀通常能省更多 **Prefill 工作**，更容易覆盖路由、缓存恢复和负载偏斜的成本；短 prompt、低复用或排队主导时增量收益可能很小。但阈值还随模型、batch、硬件、命中比例与传输变化，应按 **长度与复用程度分桶测量**。

面经中的 52K 或 8K 是特定实验边界，不能当通用开关阈值。

[原理分析：长 Prompt 的收益边界](analysis/18-组批与收益验证.md#prefix-length)
{.study-analysis-link}

### 83. Dynamo 中 Router、Worker、Planner 分别处于什么控制粒度？ {#q96 .study-question}

Router 负责 **请求级选址与缓存/负载感知**，Worker 执行 Prefill、Decode 等工作并维护设备状态；Planner 根据观测及性能模型，在较慢时间尺度规划资源或扩缩容，具体能力依版本和配置。Worker 内的 Scheduler 再决定每轮组批，因此路由、迭代调度和容量规划是不同决策。

需说明 **反馈、执行组与配置边界**，不能仅凭这些名称判断系统策略或个人方案差异。

[原理分析：Router、Worker 与 Planner 的控制粒度](analysis/13-CacheAware路由与共享.md#control-granularity)
{.study-analysis-link}

### 84. vLLM 与 SGLang 的核心差异应该怎样回答？ {#q62 .study-question}

可以从代表性设计切入：vLLM 以 **分页 KV 管理与高吞吐执行** 著称，SGLang 的 RadixAttention 强调 **前缀树组织的 KV 复用**，早期也有结构化程序接口。但分页解决存储组织，前缀缓存解决复用，两者可同时存在，vLLM 也支持前缀缓存。

实际比较要固定版本，查看 Scheduler、缓存索引、模型/backend、并行及 P/D 支持，并用相同负载验证，不能仅凭框架名字判断快慢。

[原理分析：vLLM 与 SGLang 比较什么？](analysis/14-框架选择与并行部署.md#framework-comparison)
{.study-analysis-link}

继续深入：[Cache-aware 路由](analysis/13-CacheAware路由与共享.md) · [组批与收益验证](analysis/18-组批与收益验证.md) · [框架选择与并行部署](analysis/14-框架选择与并行部署.md)
{.study-source}

## 十一、PP、CP 与 SP 进阶 {#pp-cp}

### 85. Pipeline Parallel 如何切模型，Stage 怎么划分？ {#q41 .study-question}

PP 沿模型深度把层分到不同 stage，每个 stage 保存并执行自己的层，把 **边界激活传给下一 stage**。划分要考虑层耗时、参数与 KV 容量，以及 embedding、LM head 等非均匀开销。

一个 stage 内还可以使用 TP；通常按 **实际执行成本与资源预算** 平衡，不能只让每卡层数相同。

[原理分析：PP Stage 与 Micro-Batch](analysis/09-DP与PipelineParallel.md#pp-stage)
{.study-analysis-link}

### 86. Micro-Batch 是什么，怎样用于 PP？ {#q42 .study-question}

Micro-batch 是把较大的批次拆成较小执行单元，分别推进各 stage。前一个 micro-batch 到后续 stage 时，第一 stage 可以处理下一个，从而 **让不同 stage 同时处理不同工作**。在推理中可来自不同请求或批次拆分；它不同于 Chunked Prefill 的单请求 token 切块。

切得过小会增加 **提交、通信和小矩阵成本**。

[原理分析：PP Stage 与 Micro-Batch](analysis/09-DP与PipelineParallel.md#pp-stage)
{.study-analysis-link}

### 87. Pipeline Bubble 怎么产生？ {#q43 .study-question}

流水线刚开始时后续 stage 还没拿到数据，结束时前面 stage 已没新任务，形成 **填充和排空的空闲**；stage 不均衡或依赖等待还会增加空闲。均匀、只做前向且通信忽略时，p 个 stage、m 个 micro-batch 的理想空闲比例约为 **`(p−1)/(m+p−1)`**。

这个公式不能直接套到所有训练或在线推理调度。

[原理分析：Bubble 的简化算例](analysis/09-DP与PipelineParallel.md#pipeline-bubble)
{.study-analysis-link}

### 88. 如何减少 Pipeline Bubble？ {#q44 .study-question}

用足够的独立 micro-batch 保持流水线有工作，并 **平衡 stage 耗时、重叠可独立推进的通信与计算**；某些实现还可使用交错 stage。代价是更多调度和缓冲，小 micro-batch 也可能降低 GEMM 效率。

在线推理要同时考虑入场等待与自回归依赖，不能为了填满流水线无限积攒请求。

[原理分析：减少 Bubble 与平衡 Stage](analysis/09-DP与PipelineParallel.md#stage-balance)
{.study-analysis-link}

### 89. PP 的 Stage 不均衡怎么办？ {#q45 .study-question}

先 profile 实际负载下的各 stage，找出慢的是计算、通信还是资源等待；再通过 **重新分配层、调整 stage 内 TP 或资源** 平衡。Embedding、LM head、不同层结构及 Prefill/Decode 的差异都会改变划分。

稳态吞吐受慢 stage 限制，但重新划分还必须满足 **权重与 KV 容量、边界通信成本**。

[原理分析：减少 Bubble 与平衡 Stage](analysis/09-DP与PipelineParallel.md#stage-balance)
{.study-analysis-link}

### 90. 为什么 PP 可能增加延迟，实现又为什么复杂？ {#q46 .study-question}

单个 micro-batch 仍要依次经过各 stage，**流水线主要并行不同工作，不直接消除单请求依赖**；阶段通信、排队和小矩阵开销可能增加延迟。严格延迟目标且独立工作不足时，Bubble 更难摊薄，但容量受限时 PP 仍有价值。

实现还需管理各层 KV、缓冲、取消和末端采样返回，维护 **跨 stage 的执行与资源生命周期**。

[原理分析：在线 PP 的延迟与状态协调](analysis/09-DP与PipelineParallel.md#serving-state)
{.study-analysis-link}

### 91. Context Parallel 为什么适合长 Context？ {#q47 .study-question}

CP 沿 **sequence/context 维切分 token 或历史 KV**，让各卡分担长上下文的缓存、激活和 Attention 工作，缓解单卡容量与读取压力。与 TP 按 head 或特征维切分不同，它直接处理很长的序列维。

代价是跨卡交换 KV 或合并 Attention 状态；收益取决于上下文长度、mask、并行度和互联条件，不能保证所有短请求也更快。

[原理分析：CP 分担什么？](analysis/10-ContextParallel.md#context-sharding)
{.study-analysis-link}

### 92. CP 切开 Sequence 后，Attention 如何跨 GPU 计算？ {#q48 .study-question}

对 dense Attention，每个 Query 仍需覆盖 **所有按 mask 可见的 Key/Value**。一种方法是各卡持有本地 Q 和 KV 分片，沿环传递 KV，逐块计算，并用 **online softmax 的最大值、归一化和输出状态** 合并；不能直接平均分块输出。Decode 也可按历史 KV 分片计算局部状态再归并。

具体通信与缓存布局取决于实现，切序列不等于切断跨分片依赖。

[原理分析：Ring Attention 的执行](analysis/10-ContextParallel.md#ring-attention)
{.study-analysis-link}

### 93. CP、SP 与 TP 的切分维度有什么区别？ {#q75 .study-question}

TP 常按 **特征、head 或矩阵输入输出维** 分片；CP 按上下文位置分担 Attention/KV，还需实现跨分片依赖。SP 的含义因系统而异：Megatron 风格通常把 Norm 等激活沿 token 维分片并配合 TP 的 All-Gather/Reduce-Scatter；有些系统也用 SP 指序列 Attention 并行。

应先明确 **切哪些张量、哪些算子和通信组**，不能仅凭“都切序列”认定 CP 与 SP 等同。

[原理分析：Megatron 风格 SP 与 CP 的边界](analysis/10-ContextParallel.md#sp-vs-cp)
{.study-analysis-link}

继续深入：[DP 与 PP](analysis/09-DP与PipelineParallel.md) · [Context Parallel](analysis/10-ContextParallel.md)
{.study-source}

## 十二、性能建模与配置搜索 {#performance-model}

### 94. 推理仿真如何用虚拟时钟、状态机和多 Replica 建模？ {#q90 .study-question}

用 **离散事件队列** 推进到下一个到达、forward 完成或 KV 搬运完成时刻，不必真实 sleep。每个请求维护等待 Prefill、分块执行、KV 在途、等待 Decode、生成与结束状态；每个执行组维护队列、KV 容量和资源占用，Replica 分别接收工作，TP/PP ranks 则协作执行同一组。

延迟函数随 batch、长度与配置更新，**共享链路、专家组和 Host 调度竞争** 也需建模，不能把所有副本能力直接相加。

[原理分析：单进程也能模拟多执行组](analysis/21-推理仿真与性能模型.md#virtual-clock)
{.study-analysis-link}

### 95. 性能表中的实测、SOL 理论值、拟合与插值应该怎样组织？ {#q91 .study-question}

按 **设备、模型/算子、dtype、shape、并行与执行策略** 建 key，并记录来源、测量边界、样本与版本。实测可标定执行成本；SOL 是资源理想上界，对应耗时下界，不是服务 SLO；拟合和插值是预测，应在同执行区域内验证，越界外推需标明不确定性。Pre-silicon 数据可来自周期级仿真，再按目标频率等转换。

核心是 **保留来源和可信范围**，避免用理论峰值冒充实际延迟。

[原理分析：性能数据必须带来源与适用边界](analysis/21-推理仿真与性能模型.md#latency-provenance)
{.study-analysis-link}

### 96. 怎样用 MAPE 验证延迟预测，低于 10% 能证明什么？ {#q92 .study-question}

对正的实测耗时 t 与预测值 t̂，MAPE 为 **`mean(|t̂−t|/t)×100%`**，要在未参与拟合的 shape、配置或 workload 上计算。短 kernel 的小分母会放大误差，平均值也会掩盖系统偏差和长尾；需同时报告绝对误差、分桶与端到端 TTFT/TPOT。

**MAPE<10% 是指定数据集和口径下的结果**，不证明所有配置或排队预测都准确，零值需明确处理规则。

[原理分析：MAPE 的口径和反例](analysis/21-推理仿真与性能模型.md#mape)
{.study-analysis-link}

### 97. Mock Forward 或 Logits Hook 如何支持仿真，有哪些保真边界？ {#q95 .study-question}

Mock backend 可以保留 **调度、组批和缓存状态接口**，用性能表代替真实 GPU forward，按虚拟完成事件返回，并用受控输出推进请求。Logits hook 若在真实 forward 后执行，仍然支付 GPU 成本，本身不等于跳过计算；输出长度、EOS 和候选会反过来影响负载。

它能验证策略及状态逻辑，但 **不能单独证明真实 kernel、模型质量或硬件竞争准确**，要用真实 trace 和端到端测量校准。

[原理分析：Mock 与真实执行的验证边界](analysis/21-推理仿真与性能模型.md#mock-boundary)
{.study-analysis-link}

### 98. 换 GPU 型号、卡数或并行配置，为什么要重新标定性能曲线？ {#q98 .study-question}

新配置会改变 **计算/带宽比、实际 kernel、分片 shape、拓扑通信和容量**，batch 与排队行为也随之变化，因此原曲线不能只按峰值或卡数线性缩放。应重新测关键 shape、forward 和通信，对可迁移的模型分量做校准，再验证 TTFT/TPOT 与吞吐曲线。

核心是 **验证新环境下的预测区间**，不是默认所有数据都可复用或全部必须从零测量。

[原理分析：换设备与卡数改变曲线](analysis/22-配置搜索与吞吐延迟.md#hardware-curves)
{.study-analysis-link}

### 99. 怎样搜索 TP/DP/PP/EP、P/D 配比和 Batch，目标一定是凸的吗？ {#q99 .study-question}

先按 **设备预算、模型/KV 容量与支持条件** 剪掉无效配置，再做网格或分层搜索，统一 workload 测量/仿真，筛选 SLO 达标的 goodput 或成本最优点。配比与并行度是离散变量，kernel 切换、队列和容量阈值还会形成不连续，**不能默认凸或沿拐点就到全局最优**。

可粗扫后局部细化，并验证多个候选、误差及新负载边界。

[原理分析：搜索空间通常离散且不保证凸](analysis/22-配置搜索与吞吐延迟.md#discrete-search)
{.study-analysis-link}

### 100. 部署优化中的帕累托前沿是什么，如何用它选配置？ {#q56 .study-question}

多个目标相互取舍时，如果一个配置在所有目标上都不差、至少一项更好，它就支配另一个配置；**未被其他配置支配的集合是帕累托前沿**。例如吞吐越大越好、延迟和成本越小越好，应先过滤不满足 TTFT/TPOT 约束的点，再按业务目标选配置。

它用于 **保留有价值的取舍**，不能给出脱离负载和服务目标的唯一最优配置。

[原理分析：帕累托前沿的具体例子](analysis/12-缓存层级与部署配置.md#pareto-example)
{.study-analysis-link}

### 101. Pareto 前沿怎样看，Throughput 是否等于 1/Latency？ {#q100 .study-question}

Pareto 点表示 **没有另一配置在所有目标上不差且至少一项更好**，常在相同成本下比较吞吐、TTFT、TPOT或尾延迟。吞吐是单位时间完成工作，不普遍等于单请求延迟倒数；稳态满足同口径时 Little 定律为 **平均在系统数量=到达/完成率×平均停留时间**。

同固定并发时才可能成相应反比，选择前沿点仍要先满足业务 SLO，不能把图形必然当成双曲线。

[原理分析：Throughput 不普遍等于 Latency 的倒数](analysis/22-配置搜索与吞吐延迟.md#little-law)
{.study-analysis-link}

### 102. 仿真误差为什么可能在 TTFT 更明显，Online/Offline 如何比较？ {#q101 .study-question}

Tokenization、Host 调度、提交、采样和网络若未建模，可能让 **首 token 关键路径** 偏乐观；采样位置因 backend 而异，不一定在 CPU。所谓 Online/Offline 要说明是保留真实框架与实时时钟，还是纯虚拟事件查表，并固定 trace、输入输出长度和计时边界。

更多真实组件可能提高保真度也增加运行成本，但 **不自动保证更准或误差稳定**，要按指标和负载校准。

[原理分析：端到端误差与 Online/Offline 的含义](analysis/22-配置搜索与吞吐延迟.md#online-error)
{.study-analysis-link}

继续深入：[推理仿真与性能模型](analysis/21-推理仿真与性能模型.md) · [配置搜索与吞吐延迟](analysis/22-配置搜索与吞吐延迟.md)
{.study-source}

</div>
