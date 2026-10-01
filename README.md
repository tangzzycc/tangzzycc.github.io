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
- `docs/gpu-kernel/`：GPU Kernel 源码精读、优化与 LeetGPU 练习笔记
- `docs/open-source/`：开源 PR 记录
- `docs/career/`：秋招进度
- `assets/`：站点 Logo、笔记图片和本地字体
- `mkdocs.yml`：导航与站点配置
