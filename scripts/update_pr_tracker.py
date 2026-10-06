#!/usr/bin/env python3
"""查询 GitHub，更新 tangzzycc 的 Open / Merged PR 追踪页。"""

import json
import os
import re
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "tangzzycc"
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
PROJECTS = {
    "vllm-project/vllm": ("vllm", "vLLM", "推理框架中的 ROCm 算子、量化、Attention 与投机解码。"),
    "ROCm/FlyDSL": ("flydsl", "FlyDSL", "AMD GPU Kernel 的 Tile 选择、RoPE 实现与编译接口。"),
    "flashrt-project/FlashRT": ("flashrt", "FlashRT", "Pi0.5 的 RDNA 3.5 推理后端与 Temporal K/V 复用。"),
    "ROCm/aiter": ("aiter", "AITER", "面向 AMD GPU 的算子实现与 Triton GEMM。"),
    "vllm-project/vllm-gguf-plugin": ("vllm-gguf", "vLLM GGUF Plugin", "GGUF 模型接入与 Qwen-VL 多模态支持。"),
}
SUMMARIES = {
    "vllm-project/vllm#60049": "区分有效 token 与 PIECEWISE 填充，避免自适应投机解码启动时超出编译范围。",
    "vllm-project/vllm#60040": "为 Gemma3 等模型补齐图像 token 双向注意力，串起范围元数据、分页 KV 与图执行。",
    "vllm-project/vllm#59600": "在投机轮次间恢复 ShortConv 历史，并修正 Chunked Prefill 的下一步输入。",
    "vllm-project/vllm#59550": "将窗口左侧跨度换算为完整窗口长度，修复 ROCm Attention 漏读最早有效 token。",
    "vllm-project/vllm#58513": "修正按通道 GPTQ MoE 的权重分配和 scale 索引，让 Triton fallback 正确处理整通道量化。",
    "vllm-project/vllm#53283": "让单输出 GEMM 进入已有 wvSplitK 路径，加速 Qwen Shared Expert 的门控计算。",
    "vllm-project/vllm#52637": "清理对称 GPTQ 的冗余零点，复用布局重排，减少 RDNA INT4 权重加载开销。",
    "vllm-project/vllm#52619": "对齐 Python 与 HIP 的分发边界，让 medium-skinny 形状使用已有 INT4 kernel。",
    "ROCm/FlyDSL#1113": "调整向量宽度与 NeoX lane 配对，使融合 RoPE 支持 head dimension 96。",
    "ROCm/FlyDSL#1108": "将 Softmax、RMSNorm 的旧编译调用迁至 compiler API，恢复缺失的计时结果。",
    "ROCm/FlyDSL#1057": "为 gfx1151 选择合适的 GEMM Tile，并统一设备架构、JIT target 与缓存标识。",
    "flashrt-project/FlashRT#215": "连续帧复用视觉语言 Prefix K/V，中间帧只执行 Decoder；修改提示词后刷新缓存。",
    "flashrt-project/FlashRT#213": "为 Pi0.5 增加 RDNA 3.5 BF16 后端，接通共享 Pipeline、HIP 算子与 GEMM。",
    "ROCm/aiter#4814": "补齐 gfx1151 的 A16W16 默认配置，按 M 范围选择 Tile，启用已有 Triton GEMM。",
    "vllm-project/vllm-gguf-plugin#140": "支持 Qwen2／2.5／3-VL 的 GGUF 权重加载，处理视觉、投影层与融合 QKV 布局。",
}
NOTES = {
    "vllm-project/vllm#53283": "vllm/开发实践/单输出GEMM与Shared-Expert门控的ROCm优化.md",
    "vllm-project/vllm#52619": "vllm/开发实践/RDNA-W4A16的形状分发与LDS边界.md",
    "vllm-project/vllm#52637": "vllm/开发实践/GPTQ权重加载与RDNA-INT4布局重排.md",
    "vllm-project/vllm#58513": "vllm/开发实践/Channelwise-GPTQ-MoE的加载与Triton执行.md",
    "vllm-project/vllm#59550": "vllm/开发实践/ROCm-Attention滑动窗口的参数与边界.md",
    "vllm-project/vllm#60040": "vllm/开发实践/多模态Prefix-LM在ROCm-Attention中的实现.md",
    "vllm-project/vllm#59600": "vllm/开发实践/MRV2投机解码中的ShortConv历史恢复.md",
    "vllm-project/vllm#60049": "vllm/开发实践/MRV2的PIECEWISE填充与投机解码启动Profiling.md",
    "ROCm/FlyDSL#1057": "gpu-kernel/flydsl/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md",
    "ROCm/FlyDSL#1113": "gpu-kernel/flydsl/FlyDSL-1113-Fused-RoPE-Head-Dim-96.md",
    "flashrt-project/FlashRT#213": "gpu-kernel/flashrt-rdna35/index.md",
    "flashrt-project/FlashRT#215": "gpu-kernel/flashrt-rdna35/06-Temporal-KV跨帧复用.md",
}


def search(query):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "notebook-pr-tracker",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    items = []
    page = 1
    while True:
        params = urlencode({
            "q": query, "per_page": 100, "page": page,
            "sort": "created", "order": "desc",
        })
        request = Request(f"https://api.github.com/search/issues?{params}", headers=headers)
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
        if data.get("incomplete_results") or data["total_count"] > 1000:
            raise RuntimeError("GitHub 搜索结果不完整，保留现有追踪页。")
        items.extend(data["items"])
        if len(items) >= data["total_count"]:
            break
        if not data["items"]:
            raise RuntimeError("GitHub 分页结果不完整，保留现有追踪页。")
        page += 1
    if len(items) != data["total_count"] or len({x["id"] for x in items}) != len(items):
        raise RuntimeError("GitHub 搜索数量不一致，保留现有追踪页。")
    if any(x["user"]["login"].lower() != ACCOUNT.lower() or "pull_request" not in x for x in items):
        raise RuntimeError("GitHub 返回了不符合作者或 PR 条件的记录。")
    return items


def repository(item):
    return item["repository_url"].removeprefix("https://api.github.com/repos/")


def date(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d")


def card(item, status):
    repo = repository(item)
    name = project_info(repo)[1]
    url = escape(item["html_url"], quote=True)
    original_title = escape(item["title"], quote=True)
    # 将标题前缀中的标签移到项目名、PR 编号旁边。
    prefix = re.match(r"^(?:\[[^\]]+\]\s*)+", item["title"])
    keywords = re.findall(r"\[([^\]]+)\]", prefix.group() if prefix else "")
    tags = "".join(f'<span class="pr-card-tag">{escape(keyword)}</span>' for keyword in keywords)
    display_title = escape(re.sub(r"^(?:\[[^\]]+\]\s*)+", "", item["title"]))
    summary = SUMMARIES.get(f'{repo}#{item["number"]}')
    if not summary:
        body = re.sub(r"<!--.*?-->", "", item.get("body") or "", flags=re.S)
        paragraphs = [p.strip() for p in body.replace("\r", "").split("\n\n") if p.strip() and not p.lstrip().startswith(("#", "```", "|"))]
        summary = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", paragraphs[0]) if paragraphs else re.sub(r"^(?:\[[^\]]+\]\s*)+", "", item["title"])
        summary = " ".join(summary.split())
        if len(summary) > 100:
            summary = summary[:99].rstrip() + "…"
    note = NOTES.get(f'{repository(item)}#{item["number"]}')
    action = '<span class="pr-card-arrow" aria-hidden="true">↗</span>'
    if note and (ROOT / "docs" / note).is_file():
        # 原生 HTML 链接使用站点 URL，MkDocs 不会转换这里的 .md 路径。
        note_url = quote(note.removesuffix("index.md").removesuffix(".md").rstrip("/"), safe="/") + "/"
        action = f'<a class="pr-card-note" href="../{note_url}">学习笔记 <span aria-hidden="true">→</span></a>'
    timestamp = item["created_at"] if status == "open" else item["pull_request"]["merged_at"]
    label = "提交" if status == "open" else "合并"
    state = "Open" if status == "open" else "Merged"
    return "\n".join([
        f'<article class="pr-card pr-card--{status}" aria-label="{escape(repo)} #{item["number"]}">',
        '  <div class="pr-card-top">',
        '    <span class="pr-card-identity">',
        f'      <a class="pr-card-repo" href="https://github.com/{escape(repo)}" title="{escape(repo)}">{escape(name)}</a>',
        f'      <a class="pr-card-number" href="{url}">#{item["number"]}</a>',
        f'      {tags}',
        '    </span>',
        f'    <span class="pr-card-status"><span class="pr-status-dot" aria-hidden="true"></span>{state}</span>',
        '  </div>',
        f'  <h3 class="pr-card-title"><a href="{url}" title="{original_title}">{display_title}</a></h3>',
        f'  <p class="pr-card-summary">{escape(summary)}</p>',
        '  <div class="pr-card-footer">',
        f'    <span class="pr-card-date">{label}于 <time datetime="{date(timestamp)}">{date(timestamp)}</time></span>',
        f'    {action}',
        '  </div>',
        '</article>',
    ])


def project_info(repo):
    return PROJECTS.get(repo, (repo.lower().replace("/", "--"), repo.split("/")[-1], "开源贡献与实现改进。"))


def page_header(title, today, github_url):
    return [
        "---", f"title: {json.dumps(title, ensure_ascii=False)}", "hide:", "  - navigation", "  - toc", "---", "",
        '<div class="pr-tracker" markdown="1">', "",
        f"# {title}", "",
        '<div class="pr-tracker-meta">',
        f'  <a href="https://github.com/{ACCOUNT}">@{ACCOUNT}</a>',
        f'  <span>更新于 {today}</span>',
        f'  <a class="pr-tracker-github" href="{escape(github_url, quote=True)}">GitHub <span aria-hidden="true">↗</span></a>',
        '</div>', "",
    ]


def main():
    open_items = search(f"is:pr author:{ACCOUNT} is:open")
    merged_items = search(f"is:pr author:{ACCOUNT} is:merged")
    # 独立查询全部 PR，以实际 state / merged_at 再核对一次收录范围。
    all_items = search(f"is:pr author:{ACCOUNT}")
    expected = {
        x["id"] for x in all_items
        if x["state"] == "open" or x["pull_request"].get("merged_at")
    }
    actual = {x["id"] for x in open_items + merged_items}
    if expected != actual or len(actual) != len(open_items) + len(merged_items):
        raise RuntimeError("分类查询与全部 PR 查询不一致，请重新同步；现有页面未修改。")
    if any(x["state"] != "open" for x in open_items) or any(not x["pull_request"].get("merged_at") for x in merged_items):
        raise RuntimeError("PR 状态不符合筛选条件，现有页面未修改。")

    items = sorted(open_items + merged_items, key=lambda x: x["created_at"], reverse=True)
    repos = sorted({repository(x) for x in items}, key=lambda repo: (list(PROJECTS).index(repo) if repo in PROJECTS else len(PROJECTS), repo))
    today = datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d")
    groups = {repo: [x for x in items if repository(x) == repo] for repo in repos}
    lines = page_header("开源 PR 追踪", today, f"https://github.com/{ACCOUNT}")
    lines.extend([
        f"按项目记录 {len(items)} 个 Open 或 Merged PR，覆盖推理框架、GPU Kernel 与模型部署。", "",
    ])
    for repo, project_items in groups.items():
        slug, name, _ = project_info(repo)
        lines.extend([
            f'## {name} <span class="pr-section-count">{len(project_items)}</span> {{ #{slug} }}', "",
            f'<div class="pr-card-grid pr-project--{slug}" markdown="0">',
            *[card(x, "open" if x["state"] == "open" else "merged") for x in project_items],
            '</div>', "",
        ])
    lines.append('</div>')
    output = ROOT / "docs" / "pr-tracking"
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已同步 {len(actual)} 个 PR：Open {len(open_items)}，Merged {len(merged_items)}；覆盖 {len(repos)} 个仓库。")


if __name__ == "__main__":
    main()
