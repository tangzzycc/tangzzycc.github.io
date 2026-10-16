# 7 种推理并行流程图：本地复现

来源：[知乎 @科英的原文](https://zhuanlan.zhihu.com/p/2035106376138085860)。重绘依据是用户在会话中提供的七张参考图和正文。
没有从知乎下载原始图片，也没有将参考图里的实现断言当作已经验证的代码事实。

保留参考图的深色配色、中文标注、纵向步骤、矩阵分片、四 rank 集合通信、十六 rank 专家交换和拓扑网格。
这是可编辑的矢量重绘，非逐像素复制。原图中的明显数字/配置冲突按用户选择修正，下面有完整记录。

## 查看与修改

- 浏览器打开 [gallery.html](gallery.html)：七图导航、放大、宽度缩放及 SVG/PNG 下载。也可用本地 HTTP 服务打开。
- SVG 中的文本、矩阵、框、箭头是独立元素，中文字体经过子集化后内嵌，无外部网络依赖。
- 修改 `scripts/reproduce_parallel_diagrams.py` 的 `tp_figure` / `pp_figure` 等函数，可批量再生成。
- 高清 PNG 每张宽度 `3840` 像素，与 vLLM 笔记的 3840px 宽图采用相同默认规格；SVG 逻辑宽度 1100，放大不失真。
- 正文单栏阅读，`details/` 的 22 张局部图放在对应机制段落；[完整流程图附件](../../../ai-infra/推理并行策略-完整流程图.md)只收七张长图。
- 局部图按完整流程块切分，上下边缘移除外部残余箭头；保留块内以及多块之间的连线。PNG 宽度与完整图相同，另提供可编辑的 SVG。
- 正文与附件均使用普通 Markdown 图片，沿用 vLLM 笔记的页面排版，没有专用工具栏或全屏弹窗。
- [contact-sheet.png](contact-sheet.png) 供快速浏览整体结构，详细文字请看单张图。

| 策略 | SVG | PNG | SVG 尺寸 |
| --- | --- | --- | --- |
| TP · 张量并行 | [01-tp.svg](01-tp.svg) | [01-tp.png](01-tp.png) | 1100 × 2238 |
| PP · 流水线并行 | [02-pp.svg](02-pp.svg) | [02-pp.png](02-pp.png) | 1100 × 2030 |
| DP · 数据并行 | [03-dp.svg](03-dp.svg) | [03-dp.png](03-dp.png) | 1100 × 1861 |
| EP · 专家并行 | [04-ep.svg](04-ep.svg) | [04-ep.png](04-ep.png) | 1100 × 3097 |
| DP Attention · 小 Attention TP + 大 MoE EP | [05-dp-attention.svg](05-dp-attention.svg) | [05-dp-attention.png](05-dp-attention.png) | 1100 × 3174 |
| SP · 序列并行 | [06-sp.svg](06-sp.svg) | [06-sp.png](06-sp.png) | 1100 × 2809 |
| CP · 上下文并行 | [07-cp.svg](07-cp.svg) | [07-cp.png](07-cp.png) | 1100 × 3888 |

## 重新生成

在仓库根目录运行：

```bash
.venv/bin/python scripts/reproduce_parallel_diagrams.py
```

默认输出目录为 `docs/assets/vllm/parallelism/`。可加 `--out` 指定目录、`--png-width 3840` 指定 PNG 宽度，或 `--scale 2` 指定比例，
或 `--svg-only` 只生成 SVG。依赖 `Pillow`、`fonttools`、`CairoSVG`（已存在于当前 `.venv`）。
PNG 渲染使用项目内的 `assets/fonts/LXGWWenKaiGBScreen.ttf`，仅在当前进程注册 fontconfig 字体，不修改系统字体配置。

用 HTTP 预览：

```bash
python3 -m http.server 8765 --directory docs/assets/vllm/parallelism
```

打开 `http://127.0.0.1:8765/gallery.html`。整理后的笔记位于 `docs/ai-infra/推理并行策略.md`，已加入 AI Infra 导航。

## 口径与简化

- 图 1/2/3/6：教学用 Dense MHA，H=8192，64 heads，head_dim=128，非 gated FFN intermediate=32768，权重/示例缓冲 fp16。省略 bias、LN 参数、embedding、LM Head 权重的层权重估算。
- 图 4/5/7：教学用 256 routed experts，`moe_tp=1`；每层 LN、Residual、shared expert、融合及调度元数据简化。top-k 路由会产生多目的 token 传输，图中彩条为按目的 expert 组分类的 token 包示意，非等量实测。
- 箭头表示数据来源/去向。集合通信的全连接图是逻辑数据关系，不意味着 NCCL 使用直接全连接链路；AllReduce 不代表实际 Ring 调度图。
- 图 5/7 的 LM Head 显式选择在 attention TP 小组内切 vocab 的配置，不能视为默认配置；实际推理通常只对需采样位置计算 logits。
- 图 7 按参考图画 **AllGather KV 的 prefill CP**。其每 rank attention 工作区持全量 K/V，不提供 Ring Attention 的分段状态合并，也不证明 decode KV cache 按 CP 分布。
- 图 7 按参考图保留 `MOE_DP = ATTN_CP` 的 gather/scatter 接口；实际 dispatch 去重、数据缓冲和 group 定义必须结合框架版本验证。rank 编号遵循参考图的 CP-major 示例，非所有版本的固定编号规则。
- 图 7 的输入 B=2、DP=2，所以每 DP 组只有一条序列；不使用正文中与 zigzag batch=1 约束冲突的 B=32 例子。
- Sampling 为确定性示意；随机采样需要一致的 RNG 或同步采样结果。图中没有模拟某个框架的随机采样实现。
- 这些图未进行 SGLang/vLLM 截至某 commit 的源码核实；原文行号未被当作本地证据。

## 明确修正记录

| 项目 | 参考图/正文的问题 | 本地复现的处理 |
| --- | --- | --- |
| TP 权重 | 原图底部单层约 384 MB、每卡约 96 MB，与其矩阵尺寸不一致。 | 按 fp16、MHA、intermediate=4H、非 gated FFN：单层 1.5 GiB，TP4 每卡 384 MiB。 |
| PP 权重 | 原图按完整层 384 MB 计算每 stage 6.1 GB。 | 同一 Dense 配置的完整层为 1.5 GiB；16 层每 stage 24 GiB；未计 embedding、LM Head、KV。 |
| PP micro-batch | 原图 M=4 但 stage/P2P 主线仍使用 B=32 的张量。 | 主线使用每 micro-batch Bμ=8，传输 256 MiB/次；四次累计 1 GiB。全局输入/输出保留 B=32。 |
| Attention scores | 正文将 [32,64,2000,2000] 标为约 32 GB，dtype 未明确。 | 用 fp16 逻辑 scores 计为 15.26 GiB（TP4 每卡 3.81 GiB）。fp32 是两倍；Flash Attention 不物化这些 scores。 |
| DP 的 KV | 原图总 KV 容量 N×单卡容易与固定 global batch 下的实际用量混淆。 | 固定 global B 时，请求 KV 分配到各副本，总 KV 不因 DP 倍增；模型权重复制 N 份。 |
| DP Attention 配置 | 原图标题 TP=4、DP=4、EP=16，但 SGLang --tp 指的是副本内 world。 | 明确 attn_tp=4、world=16；SGLang 风格启动 --tp 16 --dp 4 --enable-dp-attention --ep-size 16。 |
| DP Attention 的 LM Head | 原图按局部 4 卡拼 vocab，但正文又说默认全 world 拼接。 | 显式选择 --enable-dp-lm-head 的图示条件；不将此配置写成所有部署的默认行为。 |
| RS / MoE / AG | 将跨非线性 MoE 的 RS→MoE→AG 直接等同于一个 AllReduce 容易误导。 | RS 完成 Attention partial 归约，MoE 对 token 独立路由并计算，AG 拼接 MoE 结果；不声称 MoE 可与求和交换。 |
| 专家大小 | 正文使用每 expert 约 1.5 GB 的无明确层数口径估算。 | 图中只保留 256/EP 个 experts/rank，不复用缺乏 intermediate、层数与 dtype 口径的显存估算。 |
| SP 显存 | 正文把每层激活节省直接乘层数，并将其说成单层峰值÷N。 | 只计算两个指定的 [B,S,H] 缓冲：2 GiB→512 MiB。推理峰值取决于张量生命周期，不能直接乘层数。 |
| CP 实现 | 正文称 Ring，但原图画的是 K/V AllGather→两次本地 FA。 | 按实际参考图画 AllGather CP，明确每 rank 工作区持全量 K/V；不声称它是 Ring Attention 或通用 decode CP。 |
| Sampling | 原图写每 rank 独立采样且无需同步，随机采样未说明 RNG。 | 保留每 rank sampling 布局，标记为确定性示意；随机采样须同步 RNG 或采样结果。 |

## 验证

生成时检查文本/图形画布边界、SVG XML 能否解析，以及每个 PNG 的尺寸。
本次交付另用 Chromium 检查了 7 张 SVG：中文内嵌字体加载成功，文字无越界或互相重叠，图底参考小字已删除。
正文使用 22 张局部图，附件保留 7 张完整图；两页使用普通 Markdown 图片，与 vLLM 笔记一致。
这验证的是本地资产的完整性与可读性，不是对原文所有框架实现断言的验证。
