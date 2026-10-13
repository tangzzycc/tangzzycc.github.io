---
title: "FlashAttention 与 Grouped GEMM：融合边界和任务组织"
---

# FlashAttention 与 Grouped GEMM：融合边界和任务组织

<div class="study-notes study-analysis" markdown="1">

来源：提供的「英特尔校招面经」；对应 GPU与算子优化[第 41～42、48 题](../index.md#q38)。两者都改变任务组织，但处理的数学依赖不同。

## 1. FlashAttention 避免完整中间矩阵 {#online-softmax}

普通 Attention 为 `S=QKᵀ/√d`、`P=softmax(S)`、`O=PV`。若完整 S/P 落入设备内存，会增加 `Nq×Nk` 大小的中间数据与读写。

FlashAttention 按 Query/KV tile 计算局部分数和概率，在片上累积输出与归一化状态。对一条 Query 行，已有状态为最大值 m、指数和 l、未归一化输出 u；新块分数 s、对应 V 为 Vⱼ：

$$
\begin{aligned}
m'&=\max(m,\max s),\qquad \alpha=e^{m-m'},\qquad p=e^{s-m'},\\
l'&=\alpha l+\sum p,\qquad u'=\alpha u+pV_j,\qquad O=u/l.
\end{aligned}
$$

最大值变化时，旧分母与旧输出都要重缩放。不能把各块独立 Softmax 的输出直接平均；全 mask 块与初始状态需按实现处理。

用两个很小的块验证重缩放。第一块只有 score=0、value=2，状态为 m=0、l=1、u=2；第二块 score=ln2、value=4，新最大值为 ln2，旧状态乘 α=1/2，所以新 l=1/2+1=1.5，新 u=1+4=5，最终 O=5/1.5=10/3。完整计算的概率为 1/3、2/3，输出同样是 10/3；分别对两块 softmax 后简单平均则为 3。

!!! tip "FlashAttention 省的是中间矩阵读写"
    Online 状态足以恢复整行归一化结果，因而不必保存完整 score/probability 矩阵。<strong class="study-key">数学上仍需处理全部可见 QK 配对</strong>，不是把 dense Attention 改成线性复杂度。浮点求和顺序可以不同，正确性应按明确容差比较。

### Online Softmax 的三项状态为什么足够？

对已处理集合 A，定义 `m_A=max sᵢ`、`l_A=Σ_A exp(sᵢ−m_A)`、`u_A=Σ_A exp(sᵢ−m_A)vᵢ`。与另一集合 B 合并时，全局最大值 m 取二者较大者；把 A/B 都乘 `exp(m_A−m)`、`exp(m_B−m)` 后，分母和未归一化输出变为同一基准下的直接求和，所以最终 u/l 等于完整 Attention。

该表示对数学上的集合合并可重复应用，也适用于不同 tile 或 Split-KV 部分状态；浮点实际顺序仍有差异。状态的大小对每条 Query 为一个 max、一个 sum 和一个输出向量，而不是每个历史位置一个概率。因此 KV tile 处理后可丢弃局部分数，省掉 S/P 的完整物化。

### 数学 tile 与 CTA/warp tile 为什么不是同一件事？

算法只要求局部 QK、状态合并与 PV，GPU 实现还要选择一个 CTA 处理多少 Query、多大 KV tile、哪个 warp 加载或做 MMA，以及累加器怎样跨两次矩阵乘转换布局。更大 tile 提高复用，也增加 score 暂存、shared 和输出寄存器；d 大、mask 特殊、GQA head 分组等都会改变合适配置。

FlashAttention 的 IO 收益不证明 Decode 必然快同样倍数。Decode 的 Query 少，完整 score 本来较小，历史 KV 带宽与 grid 任务不足可能更突出；Split-KV 为一行引入更多任务，必须再合并 max/sum/output。

返回面试题：[41](../index.md#q38)
{.study-source}

## 2. 融合有什么收益和边界？ {#fusion-boundary}

- **核心。** QK、缩放、支持的 mask、Softmax、PV 在局部 tile 路径中结合，避免完整 S/P 的全局写读。
- **资源。** 工作空间不再要求完整二次中间矩阵，但片上 tile 与每行状态仍有预算；不意味着没有 Shared Memory 或寄存器开销。
- **语义。** Dense 可见依赖不变，Prefill 主要运算仍约 `O(NqNk d)`；浮点顺序可能造成容差内差异。
- **边界。** QKV 投影、RoPE、KV 更新和输出投影是否融合需看 backend；GQA、分页、稀疏和低精度支持也不能从名字推断。
- **阶段。** 长 Prefill 的中间矩阵较大；Decode 通常 Nq 很小，重点可能转向 KV 读取、head 复用与 Split-KV，收益不应直接外推。

已有推导：[FlashAttention 分块与 Online Softmax](<../../../../../vllm/进阶篇/FlashAttention：分块计算与Online Softmax.md>)、[Decode Attention](<../../../../../vllm/进阶篇/Decode Attention：Paged KV、GQA与Split-KV.md>)。

返回面试题：[42](../index.md#q39)
{.study-source}

## 3. MoE 怎样形成一组 GEMM？ {#expert-gemm}

约定有 T 个 token、每 token 选择 k 个专家；路由产生 T×k 个 token—expert 对。按专家整理后，专家 e 的输入为 `[Mₑ,H]`，其扩张投影通常是 `[Mₑ,H]×[H,F]`，SwiGLU 可对应 gate/up 双分支；激活后再做 `[Mₑ,F]×[F,H]`。

例如 T=8、k=2，16 对分给 4 个专家，`Mₑ=[8,4,3,1]`。每个专家处理独立矩阵，但 M 不相同。Grouped GEMM 在一个或较少 kernel 中调度多个矩阵的 tile；它不会把不同专家权重变成一份普通大 GEMM。

完整路径还包括 Router、排序/桶化与打包、可能的 EP dispatch、专家 FFN、combine 和逆映射。这些可部分融合，但 grouped 矩阵内核只是其中一段。

Token 排列需要保留逆映射。例如原 token 0 分给专家 1、3，它会在两个专家连续输入区各出现一次；combine 时必须把两份输出写回原 token 0，再乘各自路由权重相加。按 expert 排序只改变执行布局，不改变原 token 的身份或逻辑专家选择。

专家 e 的 tile 数约为 `ceil(Mₑ/BM)×ceil(Nₑ/BN)`。小 Mₑ 即使不是零，也可能只用到 tile 的几行；把许多专家放进同一 launch 能降低提交开销，但不能自动增加每个专家内部的权重复用。Persistent/grouped 调度解决任务分发，EP 均衡解决哪些专家落到哪些设备，二者有不同作用范围。

返回面试题：[48](../index.md#q40)
{.study-source}

## 4. Grouped 调度仍可能低效 {#grouped-bottleneck}

- **小 M。** 每专家 token 少，tile 的有效行比例低；若补到 tile 边界，padding 与空专家需正确跳过。
- **任务偏斜。** Tile 数相同也可能 K 或实际工作不同；persistent 调度能组织任务，但需考虑工作分配、调度元数据与尾部。
- **带宽与格式。** 小 batch 下专家权重复用少，量化解码和 scale 也有成本；数据聚集/返回会增加流量。
- **范围。** 本地 grouped 调度不能自动修复 EP 热点；应分别计量 dispatch、打包、两次投影、激活、combine，以及总 MoE 时间。

资料：[FlashAttention Paper](https://arxiv.org/abs/2205.14135)、[CUTLASS Grouped Kernel Scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)。相关分析：[EP 与负载均衡](../../inference/analysis/11-ExpertParallel与负载均衡.md)。

返回面试题：[48](../index.md#q40)
{.study-source}

</div>
