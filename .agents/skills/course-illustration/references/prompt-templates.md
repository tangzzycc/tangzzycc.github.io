# 提示词模板

这些模板用于补齐教学规格，不应逐字套用。不固定阶段数量、横纵版、字号或每图字数；按展示尺寸和内容选择。图内文案要具体到可核对的文字。

## 首轮生成

```text
Create ONE technical teaching illustration in English.
Teaching takeaway: {one sentence the reader should understand}.
Audience and assumed knowledge: {prerequisites}.

Use this continuous example: {input objects, identities, sizes, units,
assumptions, and expected output}. Any illustrative numbers must be labeled.

Reference image role: {visual style only / composition reference}.
Retain its clarity and visual continuity, but design the geometry for this topic.
Style: clean light background, readable headings, fine outlines, consistent
object colors. Concrete data objects and state changes; no decorative people.

Main reading path: {dependency order}.
Stage {n}, title "{exact heading}":
- Draw {concrete objects with visible axes/identity labels}.
- Show {transformation, motion, storage, sharing, or change of state}.
- Connect {source} to {operation/consumer}; show {result} at {destination}.
- Print these required labels exactly: {labels/formulas}.
- Carry the highlighted {object/row/token} into {next stage}.
Give {hardest mechanism} the largest area.

Correctness invariants:
{short list of the actual failure-prone relationships for this topic}.

Layout: {proportions suited to the target document width}.
Keep meaningful information density using objects and transformations, not
paragraphs in boxes. Place each necessary formula beside its operation.
If space is insufficient, expand the relevant area rather than shrink labels.
Request a crisp native high-resolution result; avoid clipped text and crossings.
```

## 布局修订

```text
Edit the supplied image. Preserve the existing mechanism, labels, values,
object identities, color meaning, and all unaffected regions.

Problem: {observable readability problem}.
Change: {specific geometry/layout change}.
For example, move the output-mapping row below the shared-input and execution
lanes and let it span the full width. Enlarge its labels using the extra space.

Keep {specific shared sources, branch count, and identities} unchanged.
Return the entire figure. Do not add new mechanisms or decorative objects.
```

## 正确性修订

```text
Make only these precise corrections to the supplied technical illustration:
1. At {location}, replace {incorrect label} with EXACT "{correct label}".
2. Route {color} arrow from {exact source} into {exact destination port}.
   It must not connect to {plausible but wrong object}.
3. Highlight only {row, column, range}; all other cells keep their base colors.

Preserve {known-correct formulas, dimensions, stages, and visual style}.
Before returning, check {small set of critical invariants}.
Return the complete image, not a cropped patch.
```

## 将方法迁移到其他课程

- **量化：** 同一向量依次经过共享尺度、有限刻度编码、解码；用数轴和相同元素高亮表现舍入与裁剪，权重读取和即时恢复另用块的移动表示。
- **投机解码：** 保持 token 位置不变，用候选、接受前缀、拒绝后缀和替补 token 的状态变化表示提交；KV 有效边界与已采样长度分开画。
- **集合通信：** 每个 rank 固定颜色或位置，追踪同一数据分片如何传输和求和；输出是拼接还是部分和要由实际数值体现。
- **流水线：** 用同一时间轴上的阶段泳道表现等待、重叠与空泡；不要把依赖工作画成同时执行。

每种题材都重新选择例子与图形，不套用 Attention 的 Q/K/V 标签或四条横带。
