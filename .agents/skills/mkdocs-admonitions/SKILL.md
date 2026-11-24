---
name: mkdocs-admonitions
description: 为本仓库的 MkDocs Material 笔记添加、优化和规范原生提示框（!!! note / tip / warning / example / question / info / abstract）与折叠块（???），以清晰的信息层级美化排版。适用于新建笔记润色、重构长文视觉层次，或把密集成段的约定、推导、排错与踩坑提炼为可扫读的提示框。
---

# MkDocs Material 提示框与排版美化

沿用本仓库 `mkdocs.yml` 已启用的原生扩展（`admonition`、`pymdownx.details`、`pymdownx.tabbed`、`pymdownx.arithmatex`）与 `docs/stylesheets/extra.css` 的正文等字号样式，把笔记中不同性质的信息放进语义匹配的提示框，让读者一眼分辨“背景前提、核心约定、手算推导、工程建议与高危陷阱”。

用户当前任务的明确要求优先；忠实翻译笔记默认保留原文结构，仅在原文自带注记或用户要求美化时引入提示框。

## 提示框类型与语义分工

按信息功能选型，不把所有提示框都写成 `!!! note`。**特别注意：MkDocs Material 的标准关键词是单数 `tip`，不要写成 `tips`（`tips` 不会命中内置样式，会退回默认样式）。**

| 语法关键词 | 适用场景 | 标题写法建议 |
| --- | --- | --- |
| `??? info` / `!!! info` | 开篇的源码版本、实测平台、验证环境等元信息（**默认优先用 `??? info` 折叠**，减少开篇噪音） | `"实测平台与源码依据"` |
| `!!! abstract` | 章节开头的全局心智模型、核心概念速览、多步架构边界 | `"进入代码前先对齐两组基础概念"` |
| `!!! note` | 核心定义、不变量、硬件 ABI 约定、必须牢记的规则 | `"核心约定：一维坐标始终按列主序展开"` |
| `!!! tip` | 实战经验、语法速查、调试工作流、阅读源码的切入点 | `"用 FLYDSL_DUMP_IR=1 按三步顺藤摸瓜"` |
| `!!! example` | 闭合手算验算、真实 IR/ISA 片段对照、小规模数值追踪 | `"手算闭合：验证偏移 = 741"` |
| `!!! question` | 初学者最高频的“为什么这里要这样写？”疑问与直答 | `"为什么输入已经是 64×32，还要用 make_view？"` |
| `!!! warning` | 默认行为陷阱、静默算错风险、边界条件、硬约束与常见误区 | `"RDNA 上 make_buffer_tensor 默认不开启越界检查"` |
| `??? note` / `???+ example` | 篇幅较长的辅助推导、完整对照或可选细节（`???` 默认折叠，`???+` 默认展开） | `"展开查看完整推导"` |

## 选材与节制原则

- **服务于阅读节奏，不把正文全装进框里。** 连贯的因果论述、主流程代码和核心图解保持为普通正文；只把“值得单独拎出来扫读、对照或警惕”的模块放进提示框。
- **避免连续堆叠。** 相邻位置最多放 1–2 个不同功能的提示框（例如 `warning` + `tip`）；避免把 5–6 条并列清单拆成 6 个独立提示框，长清单优先用编号列表。
- **始终写具体的中文自定义标题。** 如 `!!! tip "切片语法速查：fx.slice vs 方括号 []"`，不留默认英文标题（`Note`、`Tip`、`Warning`），标题本身应传达核心结论或问题。
- **不改写技术事实。** 美化排版只重组视觉层级，不删减推导前提，也不替未核实的内容增加确定性结论。

## 语法与嵌套铁律

Python-Markdown 对提示框的缩进与空行非常敏感，编写时必须遵守以下四条规则：

1. **前后留空行**：`!!! <type> "<标题>"` 之前和整个提示框结束之后，必须各有一个空行；切忌把 `!!!` 直接接在上一段句号后面。
2. **严格 4 空格缩进**：框内全部内容（段落、列表、公式、表格、代码块）统一缩进 4 个空格，标题行后留一个空行再写正文。
3. **框内块级元素前后留空行**：在提示框里嵌套表格（`| --- |`）、围栏代码块（`` ``` ``）或行间公式（`$$...$$`）时，元素前后都要保留空行并保持 4 空格缩进，否则 Markdown 解析器会退化为普通文本。
4. **反引号与代码注释**：标题中可以使用行内代码反引号（如 `` `retile` ``），代码块保持正确的语言标签（`python`、`mlir`、`asm`、`bash`）。

具体模板与本仓库示范写法见 [提示框示范样例](references/examples.md)，完整长文示范见 `docs/rocm/FlyDSL：RDNA3的布局与矩阵计算.md`。

## 渲染与交付检查

编辑完成后，用仓库现有的严格模式构建站点，并检查生成的 HTML 中没有裸露语法或未解析的嵌套元素：

```bash
NO_MKDOCS_2_WARNING=1 .venv/bin/mkdocs build --strict
```

检查目标页面 HTML（将路径替换为本次修改的笔记生成路径）：

```bash
python3 - <<'EOF'
import pathlib, re

html = pathlib.Path("site/rocm/FlyDSL：RDNA3的布局与矩阵计算/index.html").read_text(encoding="utf-8")
raw = re.findall(r"(?:!!!|\?\?\?)\s*\w+", html)
admons = re.findall(r'<div class="admonition ([^"]+)">\s*<p class="admonition-title">(.*?)</p>', html, re.S)
blocks = re.findall(r'<div class="admonition [^"]+">.*?</div>', html, re.S)
bad_pipes = [b for b in blocks if "| --- |" in b]
bad_fences = [b for b in blocks if "```" in b]

print("Rendered admonitions:", len(admons))
print("Raw syntax leaked:", len(raw), "| Unparsed tables:", len(bad_pipes), "| Unparsed fences:", len(bad_fences))
assert not raw and not bad_pipes and not bad_fences
EOF
```

如果本地正在运行 `mkdocs serve`，也刷新对应页面确认深浅色模式下的视觉留白、公式换行与窄屏排版。
