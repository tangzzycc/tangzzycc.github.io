---
title: "MHA、GQA 与 MLA：共享、压缩和历史缓存"
---

# MHA、GQA 与 MLA：共享、压缩和历史缓存

<div class="study-notes study-analysis" markdown="1">

来源：提供的「算能科技二面」记录；对应模型结构[第 2、4～5 题](../index.md#q1)。MHA→GQA→MLA 可作为比较线索，不表示所有模型都按该顺序升级。

## 1. 从 KV Head 数比较 {#kv-heads}

假设每层 Query head 数 h、GQA KV head 数 g，Key/Value 宽度都为 d，MLA latent 宽度 c、位置 Key 宽度 r。比较每 token、每层的理想缓存元素数：

| 结构 | Query 如何对应 KV | 缓存元素数 |
| --- | --- | --- |
| MHA | 每个 Query head 有独立 KV | `2hd` |
| GQA | 每组 Query heads 共享 KV | `2gd` |
| MQA | 所有 Query heads 共享一组 KV | `2d` |
| MLA | 共享 latent，经各 head 上投影形成内容 KV | `c+r`，采用缓存压缩 latent 的路径 |

取 h=32、g=8、d=128、BF16，MHA 每 token 每层为 16 KiB，GQA 为 4 KiB，MQA 为 512 B。这些只是缓存载荷，不表示完整模型显存、质量或加速比。

Query heads 不必随着 KV heads 减少：h=32、g=8 时，每个 KV head 供一组 4 个 Query heads 使用。各 Query 仍有不同投影并形成不同 Attention 权重；共享 K/V 不意味着把这 4 个 Query 输出平均。输出拼接后仍可保持 32 个 Query head 的总宽度。

<strong class="study-key">GQA 压缩的是每位置的 KV head 数，dense Attention 仍覆盖既定历史位置。</strong>
{.study-core}

减少 KV heads 会改变模型的表达约束，通常需要训练或转换/微调支持，不是随意删除 checkpoint 的几个 head 即可。Kernel 还需映射 Query 到正确 KV 分组，并让组内 KV 加载复用，才能把存储收益转为真实带宽收益。

### MHA/GQA 的完整形状与 Attention 工作有什么变化？

设输入 `[B,S,H]`，h_Q 个 Query heads、h_KV 个 KV heads、head width d。投影后 Q 为 `[B,h_Q,S,d]`，K/V 为 `[B,h_KV,S,d]`。GQA 的每个 Query head i 访问其对应分组 `floor(i/(h_Q/h_KV))` 的 K/V；常规打分输出仍有 h_Q 组 `[S,S]`，所以减少 h_KV 不意味着 QK/PV 主计算按同样比例下降。

它主要压缩 K/V 投影、保存和读取，并为组内 KV 复用创造机会。后端若为每 Q head 重复读同一 KV，L2 或 shared 的供数压力仍存在；质量则取决于共享 KV 的表示约束与训练。MQA 把 h_KV 压到 1，是这条维度上的极端，不是任意 MHA 无损压缩。

### RoPE 怎样让打分表达相对位置？

每个二维特征对乘角度随位置 t 变化的旋转 `R_t`，旋转后的 dot product 为 `(R_tq)ᵀ(R_sk)=qᵀR_tᵀR_sk`。在同一频率定义下，`R_tᵀR_s=R_{s−t}`，使分数含相对位置信息。不同维度对使用不同频率；旋转不等于给每个位置加同一个向量。

实际缓存可能存旋转后的 K；位置偏移、RoPE 参数、插值/扩展方法改变时，旧 K 不必兼容。复用 prefix 时除了 token IDs，还需保持位置和模型配置语义。

返回面试题：[02](../index.md#q1)
{.study-source}

## 2. MLA 共享 Latent，保留不同 Head 的投影 {#shared-latent}

从隐藏状态 x 生成较小 latent c，各 head 用各自的矩阵得到内容 Key/Value。共享的是低维信息，展开的 head 表示可以不同；训练学习这种受低秩约束的表示，不是事后无损压缩任意权重。

为避免混淆向量名与维度，令隐藏状态为行向量 `x[1,H]`，下投影 `W_D[H,c]` 得到 latent `z[1,c]`；第 i 个 head 用 `U_K,i[c,d]`、`U_V,i[c,d]` 得到内容 K/V。缓存只保留 z，历史长度 S 时 latent 矩阵是 `C[S,c]`，各 head 共享 C，但上投影参数不同。

举例 c=512、位置宽度 r=64、BF16，压缩路径每 token 每层理想载荷为 `(512+64)×2=1152 bytes`。这个教学配置说明压缩维度，与前面的 h=32、d=128 的 16 KiB MHA 可比较载荷；不能由比值推断真实模型的质量或加速。

返回面试题：[04](../index.md#q2)
{.study-source}

## 3. 为什么矩阵吸收能避免历史展开？ {#matrix-absorption}

暂不考虑位置分支，历史 latent 矩阵为 C，单 head 的上投影为 Uₖ、Uᵥ：

$$
q(CU_k)^\top=(qU_k^\top)C^\top,\qquad
a(CU_v)=(aC)U_v.
$$

- **Key 侧。** 当前 q 乘 Uₖ 的转置，直接与历史 latent 匹配，不必逐位置展开 Key。
- **Value 侧。** 先得到 latent 的 Attention 加权和，再乘 Uᵥ，不必逐位置展开 Value。
- **边界。** a 是包含完整打分与缩放的同一组 Attention 权重，不能把投影移过 softmax；吸收后维度变化也不意味着随意修改缩放。

形状验证：`q[1,d]U_Kᵀ[d,c]` 得到 `q̃[1,c]`，再乘 `Cᵀ[c,S]` 得到与历史各位置的分数 `[1,S]`。Attention 权重 a 为 `[1,S]`，先做 `aC[1,c]`，再乘 `U_V[c,d]` 得到输出 `[1,d]`。两边都只对当前 Query/加权结果做上投影，而不为全部 S 个历史位置展开 KV。

Value 上投影若与最终输出投影相邻且数学兼容，还可合并固定线性矩阵。Softmax 仍在分数与加权求和之间；吸收只用线性运算的结合律，不能越过非线性，也不能自动把原打分 scale 改成 `1/√c`。

### 矩阵吸收省缓存，为什么可能增加每步算术？

内容分支展开后每 head 的 QK/PV width 为 d，吸收后历史计算围绕 latent width c。若 c>d，单 head 对历史的点积和加权和宽度增加；收益来自避免存储/展开 h 份 K/V，并能在执行路径中复用共享 latent。因而 MLA 的缓存优势不自动表示 FLOPs 优势。

一个 Query 的 Key 变换 `qU_Kᵀ` 与 Value 输出变换发生在当前 token/加权结果，成本不随历史 S 同比例增长；若先为每个历史位置展开 U_K/U_V，投影工作则随 S 增长。Decode 常更重视压缩缓存路径，大 Prefill 可能选择展开或其他矩阵组织以提高计算效率，具体由 backend 决定。

RoPE 分支仍要并入同一完整分数再 softmax；内容 score 不能先独立 softmax 后与位置分支输出拼接。Linearity 只允许重排固定投影，不允许跨越归一化非线性。

返回面试题：[05](../index.md#q3)
{.study-source}

## 4. RoPE 与执行路径 {#decoupled-rope}

位置相关旋转会影响 Key 投影与 Query 的代数重排；DeepSeek-V2/V3 的 MLA 用独立位置 Query/Key 分支，使内容分支保持可吸收。执行后端可能在 Prefill 与 Decode 采用不同展开或吸收路径，还要考虑 TP 对 latent 的复制及缓存布局。

RoPE 的旋转矩阵随位置变化。若历史 Key 写为 `z_s U_K R_s`，每个位置都有不同 R_s，当前 Query 端无法用一个固定矩阵吸收全部 `U_K R_s`。解耦后把打分分成内容与位置两项：内容项使用可吸收的 latent，位置项使用独立旋转的 qᴿ/kᴿ，再按模型定义组合与缩放。

**位置分支不是“去掉 RoPE”，而是让位置变换与内容压缩的代数路径兼容。**
{.study-mechanism}

Prefill 可以为大矩阵效率临时展开部分 K/V；Decode 可以更直接使用吸收路径。临时执行表示和长期缓存表示需要区分，TP 按 Query head 分片也不一定让共享 latent 等比例缩小。

已有推导：[MLA：低秩压缩、矩阵吸收与解耦 RoPE](<../../../../../vllm/进阶篇/MLA：低秩压缩、矩阵吸收与解耦RoPE.md>)。模型资料：[DeepSeek-V2](https://arxiv.org/abs/2405.04434)。

返回面试题：[05](../index.md#q3)
{.study-source}

</div>
