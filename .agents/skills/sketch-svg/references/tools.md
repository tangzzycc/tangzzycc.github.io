# 工具用法

## 运行环境与资源

- 字体打包使用 Python 3 和 `fonttools[woff]`（含 WOFF2 的 Brotli 支持）。用当前可用且已安装这些依赖的解释器；需要时在工作用虚拟环境安装 `python -m pip install 'fonttools[woff]'`。
- 绘图辅助库使用 Node.js 18+，仅依赖内置模块；Rough.js 4.6.6 随技能提供，不需要 `npm install`。若 `node` 不在 PATH，可用调用者指定的 Node 可执行路径，不能把当前临时环境路径写入长期脚本。
- 英文数字字体为技能自带 `assets/fonts/ComicShanns-Regular.woff2`；中文字体默认读取仓库 `assets/fonts/LXGWWenKaiGBScreen.ttf`，不重复复制整份字体。移植到其他仓库时用 `--wenkai /path/to/LXGWWenKaiGBScreen.ttf` 显式指定。

所有命令在仓库根目录运行，临时例子写入工作目录；最终只把目标 SVG 放入发布图片目录。

## 字体包

先将全部图内标签写入 UTF-8 文件，例如 `labels.txt`，再运行：

```bash
python .agents/skills/sketch-svg/scripts/prepare_fonts.py \
  --text-file /path/to/work/labels.txt \
  --output /path/to/work/fonts.json
```

也可从已有 SVG 收集 `<text>`（含嵌套 `<tspan>`）、`title`、`desc`；有新标签时以更新后的 SVG 或完整标签文件重新打包：

```bash
python .agents/skills/sketch-svg/scripts/prepare_fonts.py \
  --svg /path/to/work/draft.svg \
  --output /path/to/work/fonts.json
```

输出 JSON 包含内嵌字体 CSS、完整许可证、字符覆盖和 Comic Shanns 的 glyph ink 度量。`text()` 会拒绝缺失的字符，提示重新生成字体包，防止悄悄回退为系统中文字体。此工具不自动改变既有 SVG；由生成器使用新的字体包写出最终图。

## 一个小型生成器

以下例子画输入、计算和输出关系，并演示 `Q_i` 与 `K_j^T`。作为 API 使用示例，布局应按实际笔记内容重写。

为例子准备的 `labels.txt`：

```text
输入分块
Query
转置 Key
矩阵乘法
输出得分
共享同一特征维度 d
逻辑范围示意
```

把以下代码保存为工作目录中的 `draw.cjs`；第三个参数显式指定技能的辅助库，因此脚本可放在仓库以外。

```js
const fs = require('node:fs');
const {SketchSvg} = require(process.argv[4]);
const fonts = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const svg = new SketchSvg({width: 1000, height: 290, fonts, seed: 801});

svg.text(30, 31, '输入分块', {size: 20});
svg.rectangle(35, 65, 130, 135, {fill: '#6584b0', stroke: '#5878a3'});
svg.text(100, 93, 'Query', {color: '#ffffff', anchor: 'middle'});
svg.math(100, 144, 'Q', {sub: 'i', color: '#ffffff', size: 20});
svg.rectangle(230, 65, 130, 135, {fill: '#83a86c', stroke: '#769c5f'});
svg.text(295, 93, '转置 Key', {color: '#ffffff', anchor: 'middle'});
svg.math(295, 144, 'K', {sub: 'j', sup: 'T', color: '#ffffff', size: 20});

svg.rectangle(465, 102, 145, 62, {stroke: '#929da7'});
svg.text(537.5, 138, '矩阵乘法', {anchor: 'middle', size: 17});
// Two independent operands enter an operation; Q does not produce K.
svg.arrow([[100, 204], [100, 218], [403, 218], [403, 148], [461, 148]]);
svg.arrow([[363, 118], [461, 118]]);
svg.arrow([[614, 133], [714, 133]]);
svg.rectangle(718, 90, 190, 90, {fill: '#c95d65', stroke: '#b7555d'});
svg.text(813, 117, '输出得分', {color: '#ffffff', anchor: 'middle'});
svg.math(813, 151, 'S', {sub: 'ij', color: '#ffffff', size: 20});
svg.rectangle(704, 75, 218, 120,
  {stroke: '#d1b275', strokeLineDash: [7, 7], disableMultiStroke: true});
svg.text(813, 224, '逻辑范围示意', {size: 14, color: '#737b83', anchor: 'middle'});
svg.text(35, 257, '共享同一特征维度 d', {size: 14, color: '#737b83'});

fs.writeFileSync(process.argv[3], svg.render({
  title: '输入分块的矩阵乘法',
  description: '两个独立输入分块进入矩阵乘法并产生得分块。虚线只示意逻辑范围。',
}));
```

```bash
node /path/to/work/draw.cjs /path/to/work/fonts.json /path/to/work/example.svg \
  "$PWD/.agents/skills/sketch-svg/scripts/sketch_svg.cjs"
```

## API 约定

- `rectangle(x, y, width, height, options, attrs)`、`ellipse(cx, cy, width, height, options, attrs)`、`line(x1, y1, x2, y2, options, attrs)`、`polyline(points, options, attrs)` 接受 Rough.js 参数；`attrs` 用于 `id`、`data-*` 等 SVG 属性。
- `arrow(points, options, attrs)` 沿给定折线绘制，末端加稳定箭头；`options.headSize` 默认为 7。避让与起止位置由布局决定。
- `text(x, y, value, {size, color, anchor, attrs})` 的 `y` 是 baseline。手动拆分多行，使用相同 x 与合适行距。
- `math(x, y, base, {sub, sup, suffix, size, color, anchor, gap, scriptRatio, suffixGap, attrs})` 以 ink bounds 定位整个组合，默认居中。所有参数字符必须存在于 Comic Shanns 的度量中；中文后缀另用 `text()` 留足间隔。
- `render({title, description, background})` 返回完整 SVG，包含两种字体和许可证。相同字体包、种子及绘图调用顺序产生相同输出。

不要给文字属性另设字体、粗体或 `textLength` 压缩；这些设置会破坏指定字体和 glyph 度量关系。输出是普通 SVG，可以继续编辑；修改标签后要更新内嵌字体子集。

## 渲染检查

用 Chromium / Playwright 在最终显示宽度查看最终 SVG；等待图片加载以及 `document.fonts.ready`。必须检查实际字形使用情况，单看 CSS 的 `font-family` 声明不足以证明字体正确。

对独立 SVG 文档或内联 SVG，浏览器开发者工具的 Rendered Fonts，或 Playwright CDP 的 `CSS.getPlatformFontsForNode` 可确认：英文和数字来自内嵌 Comic Shanns（font 文件内部名称可能显示 `Comic Shanns Regular`），中文来自内嵌 WenKaiSketch；字体应标为 custom font。图以 `<img>` 嵌入时，另打开 SVG 文档检查字体，仍在真实页面查看最终大小与布局。

检查 `Q_i`、`K_j^T`、`B_r`、`B_c` 的视觉间距和上下标对齐；`getBBox()` 的文本盒可能含 advance，不能用它代替实际 ink 度量去把下标推远。定位大范围越界可以使用 `getBBox()`，同时目视检查对象内部留白、相邻标签和线条。

如果修改了笔记引用或布局，使用项目已有 MkDocs 构建方式，并在真实页面核对加载、通栏宽度和窄屏溢出。仅修改技能时无需全站重建。

## 资源来源与授权

- Comic Shanns Regular：Excalidraw 0.18.0 的 ASCII 字体文件，源自 `https://unpkg.com/@excalidraw/excalidraw@0.18.0/dist/prod/fonts/ComicShanns/ComicShanns-Regular-279a7b317d12eb88de06167bd672b4b4.woff2`。打包完整 ASCII 集，不复用单张图内的裁剪版本。MIT 授权含原作者和 Excalidraw 衍生版本的五位版权人。
- 霞鹜文楷：仓库现有 `assets/fonts/LXGWWenKaiGBScreen.ttf`，按图的中文与符号裁剪为 WOFF2。SIL OFL 1.1；子集内部家族名为 WenKaiSketch，字形仍为霞鹜文楷。
- 两种字体的完整授权在 [assets/fonts/LICENSES.txt](../assets/fonts/LICENSES.txt)，生成时写入 SVG 的 `font-license` metadata。
- Rough.js 4.6.6：`https://unpkg.com/roughjs@4.6.6/bundled/rough.js`，保存在 `scripts/vendor/rough-4.6.6.js`；完整 MIT 授权在 [rough-LICENSE.txt](../scripts/vendor/rough-LICENSE.txt)。
