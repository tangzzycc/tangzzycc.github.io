# 视觉与布局约定

## 用现有图对齐风格

以下路径相对仓库根目录。按任务挑一张查看，无需每次读全部 SVG 源码。

| 参考图 | 适合参考的内容 |
|---|---|
| [FlashAttention 分块尺寸](../../../../assets/cuda/flashattention/flashattention-v1-tile-shapes.svg) | 实色矩阵、颜色身份、紧凑上下标、逻辑范围虚线 |
| [RDNA 3.5 WGP](../../../../assets/rocm/rdna35/rdna35-wgp.svg) | 架构分层、共享资源与小字号中文说明 |
| [Wave 执行](../../../../assets/rocm/rdna35/rdna35-wave-exec.svg) | 时间与执行分组、局部展开 |
| [Scalar / Vector](../../../../assets/rocm/rdna35/rdna35-scalar-vector.svg) | 对象对应关系、中文与英文术语混排 |
| [DFlash](../../../../assets/ai-infra/speculative-decoding/dflash-prefill-draft-verify.svg) | 预填充、并行草拟、验证阶段的连续流程 |
| [DSpark](../../../../assets/ai-infra/speculative-decoding/dspark-backbone-markov-confidence.svg) | 骨干与两个 Head 的来源、输出和验证关系 |

## 字号以页面实际显示为准

换算：`显示字号 = SVG 内字号 × 实际显示宽度 / viewBox 宽度`。

以 1000 宽的画布显示为 800 px 为例，正文 16–18 native px 对应 12.8–14.4 显示 px；辅助说明 14 native px 对应 11.2 显示 px。单字母矩阵标识可用 20 native px，标题最多约 22 native px。不要每个矩形都放一个大标题。

图中所有字默认常规字重。Comic Shanns 内嵌版本为 Regular，不用浏览器合成粗体撑满小框；层次主要靠位置、留白、颜色和有限字号差异。多行说明手动断行并留足行距，避免依赖 SVG 自动换行。

缩到窄屏时允许读者点击原图查看细节；仍须避免页面横向溢出。大图在正常桌面栏宽就过小，应删内容、拆图或通栏，不能只在原图 100% 大小时验证。

## 手绘轮廓

Rough.js 推荐起点：`roughness: 0.60`、`bowing: 0.5`、`strokeWidth: 1.15`、`fillStyle: 'solid'`。外层框线可降至 1.0；精细尺寸线可用 `roughness: 0.25–0.35`、`strokeWidth: 0.6–0.9`、`disableMultiStroke: true`。

这些数值是与当前参考图一致的起点，可以按对象大小微调。很短的尺寸线、箭头端点要比大对象轮廓更稳定。保留足够留白让轻微抖动不会挤进文字。

使用 `generator.toPaths()` 序列化 Rough.js 对象时，虚线属性不会自动进入 SVG。必须把 `drawable.options.strokeLineDash` 显式写成轮廓的 `stroke-dasharray`；填色路径不加虚线。辅助库已处理。

## 配色承担对象身份

以下是已修订 FlashAttention 图的配色，适合作为起点；其他主题按语义选择，不要求每张图套同一套颜色。

| 对象 | 填充 | 边框 |
|---|---|---|
| Q / 蓝色输入 | `#6584b0` | `#5878a3` |
| K / 绿色输入 | `#83a86c` | `#769c5f` |
| V / 橙色输入 | `#d99b6b` | `#c98b5c` |
| O / 青色输出 | `#65a7b4` | `#5899a5` |
| 当前输入与得分块 | `#c95d65` | `#b7555d` |
| 当前输出块 | `#8872aa` | `#7b669c` |
| 逻辑范围虚线 | 无填色 | `#d1b275` |

正文深色可用 `#343a40`，辅助说明可用 `#737b83`。深色实底中少量简短标签用白色；浅底用深色。标签提供身份，避免只靠颜色区分。排线、网格只在需要表示结构、选择区域或未知值时加入。

## 上下标排版

默认主字与上下标的实际 ink 间隔为 **0.35 native px**，上下标大小为主字 **0.66**；下标 baseline 为 `y + size × 0.20`，上标 baseline 为 `y − size × 0.35`。

上下标分别减去自身 glyph 左边界，使其左侧 ink 都对齐到 `baseInkRight + gap`。组合的居中/右对齐使用整个组合的 ink bounds，不能只对齐主字或按等宽 advance 估算。普通后缀与数学组合另外留约 3.5 native px；不要靠前导空格，SVG 默认可能将其折叠。

这些参数适用于工具内的 Comic Shanns 版本。替换字体文件后必须重新生成字形度量。工具处理单个主字或短英文串与上下标，不解析任意 LaTeX；复杂公式按需要拆成明确对象，仍核查数学语义。
