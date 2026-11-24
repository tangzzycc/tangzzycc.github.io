# MkDocs Material 提示框示范样例

以下样例均提炼自仓库示范笔记 `docs/rocm/FlyDSL：RDNA3的布局与矩阵计算.md`，已在 MkDocs Material + `extra.css`（正文等字号样式）下验证通过。

## 1. 开篇环境与版本锚点（`??? info`，默认折叠）

用于文章开头交代源码版本、实测平台与验证环境，默认用 `??? info` 折叠，既保留可追溯依据又不打断读者快速进入正文。

```markdown
??? info "源码版本与验证环境"

    本文源码锚点全部固定到 **FlyDSL v0.5.0**，目标为 `gfx11` 的 Wave32 路径（覆盖 RDNA 3 与 RDNA 3.5 的 `gfx1151`），文中的 IR、ISA 与运行验证均基于本机 AMD Radeon 8060S（`gfx1151`）实测结果。
```

## 2. 章节前置心智模型（`abstract`）

用于 M0 或导言处集中对齐术语和整体分层，方便读者在看代码前建立地图。

```markdown
!!! abstract "进入代码前先对齐两组基础概念"

    - **硬件与执行分工**：Block / Workgroup 共享同一块 LDS；`gfx11` 下 Wave32 满足 `wave_id = tid // 32`、`lane_id = tid % 32`。
    - **三层编译视角**：Python Tracing → 显式 MLIR IR（`!fly.layout`、`!fly.memref`）→ ROCDL / LLVM `gfx11` ISA。
```

## 3. 核心约定与公式（`note` + MathJax）

用于贯穿后文的硬规则或硬件 ABI。框内公式 `$$...$$` 前后均保留空行并缩进 4 空格。

```markdown
!!! note "核心约定：一维坐标始终按列主序（colexicographic）展开"

    当把一个一维整数 $i \in [0, \text{size}(S))$ 直接喂给多维 Layout $L = ((S_0, S_1), (D_0, D_1))$ 时，FlyDSL / CuTe **始终按 Shape 的列主序**把 $i$ 拆成坐标 $(i \bmod S_0,\; \lfloor i / S_0 \rfloor)$，再与 Stride 做点积：

    $$
    L(i) = (i \bmod S_0)\cdot D_0 + \lfloor i / S_0 \rfloor \cdot D_1.
    $$

    后文的 `Composition`、`Right Inverse` 和 `slice(..., (None, bid))` 的全部结果都直接由这条规则决定。
```

## 4. 闭合手算验算或真实 IR/ISA 对照（`example` + 代码块）

把手算过程或实测 IR/ISA 放进 `example` 框，读者既可以顺着主线速读，也可以停下来核对每一个数字。

````markdown
!!! example "手算闭合：验证 `(t=2, v=5, i=1, j=2)` 的全局偏移 = `741`"

    设 Block 0 内线程 `t = 2`，逻辑值 `v = 5`（第 2 次向量 `rest_v = 1` 的第 `atom_v = 1` 个元素），重复坐标 `i = 1, j = 2`：

    - **按全局矩阵算**：`m = 2 + 4 = 6`，`n = 5 + 16 = 21`，偏移为 $6 \times 120 + 21 = 741$。
    - **按 `src` 的 Layout 算**：基址 $2 \times 120 = 240$，相对偏移 $1 + 4 + 480 + 16 = 501$，合计 `741`。
````

## 5. 带着表格的默认行为警告（`warning` + 表格）

用于高亮静默算错、越界检查默认关闭、单位混淆等高危点。框内表格前后保留空行并缩进 4 空格。

```markdown
!!! warning "RDNA 上 `make_buffer_tensor` 默认不开启越界检查"

    **v0.5.0 在 RDNA 上默认关闭资源范围越界检查**，须按需显式指定：

    | 写法 | Descriptor 范围 | RDNA 越界检查 |
    | --- | --- | --- |
    | `make_buffer_tensor(A)` | `0xFFFFFFFF` | 关闭，`OOB_SELECT=2` |
    | `make_buffer_tensor(A, max_size=False)` | `cosize(A.layout) × 元素字节数` | 开启，`OOB_SELECT=3` |
```

## 6. 实战写法或排错工作流（`tip` + 代码块）

用于给出可复制的习惯写法、调试命令与排查步骤。

````markdown
!!! tip "用 `FLYDSL_DUMP_IR=1` 按三步顺藤摸瓜"

    1. **Shape/Stride 或分片不对 → 看 `00_origin.mlir` 与 `04_fly_layout_lowering.mlir`**。
    2. **寄存器未消除 → 看 `08_fly_promote_regmem_to_vectorssa.mlir`**。
    3. **目标指令不对 → 看 `09_convert_fly_to_rocdl.mlir` 与 `22_final_isa.s`**。
````

## 7. 高频疑问直答（`question`）

用于把初学者最容易卡壳的“为什么要多写这一行？”直接拎成问答框。

```markdown
!!! question "明明传入的 `torch.Tensor` 已经是 `64×32`，为什么还要用 `make_view` 重包一层？"

    JIT 从普通 `torch.Tensor` 接收到的视图默认带动态维度 `?`；用 `make_view` 换上编译期常量组成的静态 Layout，既不复制数据，又让后续分片与寄存器数组大小在编译期确定。
```

## 8. 折叠块（`???` / `???+`）

适合篇幅较长但不必打断主线的完整 IR dump 或扩展推导：

```markdown
??? example "展开查看完整中间 IR（`04_fly_layout_lowering.mlir`）"

    这里放较长的辅助片段，默认折叠；若希望默认展开可改为 `???+ example`。
```

## 常见错误对照

| 错误写法 | 问题 | 正确写法 |
| --- | --- | --- |
| `!!! tips "小技巧"` | `tips` 不是内置类型名，丢失 `tip` 样式 | `!!! tip "小技巧"` |
| `上一段句号。!!! note "标题"` | 缺少前置空行，整段退化为裸文本 | 前置空行后再起 `!!! note "标题"` |
| 框内表格只缩进 2 空格或无空行 | 表格不会被解析为 `<table>` | 统一 4 空格缩进，表格上下各留空行 |
| 每段正文都包一层 `!!! note` | 失去重点区分，视觉疲劳 | 只把高信号约定、推导、技巧与陷阱装框 |
