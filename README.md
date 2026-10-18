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

- `docs/ai-infra/`：推理架构、并行策略、vLLM 执行机制与 GEMM 优化笔记
- `docs/vllm/`：搬运栏目，收录 vLLM 框架基础与 Attention 进阶笔记
- `docs/handcoding/`：HandCoding 入口，目前仅保留空的 `index.md`
- `docs/life/`：生活记录，其中 `career/` 按 `fundamentals/`、`projects/`、`development/`、`resume/`、`applications/` 整理面试基础、项目经历、开发实践、简历与秋招进度；FlashRT RDNA 3.5 与 FlyDSL 源码笔记归入各自的 `development/` 目录
- `assets/`：站点 Logo、笔记图片和本地字体
- `mkdocs.yml`：导航与站点配置
