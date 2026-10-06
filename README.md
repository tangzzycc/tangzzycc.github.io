# Notebook

个人技术笔记与本地博客。笔记、图片和站点资源统一保存在 `docs/` 中。

## 本地预览

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
NO_MKDOCS_2_WARNING=1 .venv/bin/mkdocs serve
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000) 即可查看。开发服务会监听文件变化并自动刷新页面。
`NO_MKDOCS_2_WARNING=1` 只关闭 Material 关于未来 MkDocs 2.0 的公告，不影响构建和预览。

## 内容结构

- `docs/ai-infra/`：AI Infra 学习索引
- `docs/vllm/`：vLLM 专题笔记
- `docs/gpu-kernel/`：GPU Kernel 源码精读与优化笔记
- `docs/handcoding/`：动手编程练习，目前包含 LeetGPU
- `docs/pr-tracking/`：tangzzycc 的 Open 与 Merged PR 追踪
- `docs/open-source/`：旧 PR 索引与重复 Kernel 文档归档（不参与站点构建）
- `docs/life/`：生活记录，其中 `career/` 保存秋招进度
- `assets/`：站点 Logo、笔记图片和本地字体
- `mkdocs.yml`：导航与站点配置

## 更新 PR 追踪

```bash
python3 scripts/update_pr_tracker.py
```

从 GitHub 查询 `tangzzycc` 在所有仓库提交的 Open 与 Merged PR，排除关闭但未合并的记录，在同一页按项目生成正方形卡片。每张卡片包含标题、简短说明、状态与日期，日期按北京时间显示。已有 PR 的中文说明保存在脚本的 `SUMMARIES` 中，新 PR 默认提取描述的首段，可随后补充中文概括。页面记录最近一次同步的状态；实时状态可通过页面上的 GitHub 链接查看。脚本支持通过 `GH_TOKEN` 或 `GITHUB_TOKEN` 使用已有 GitHub 凭据。
