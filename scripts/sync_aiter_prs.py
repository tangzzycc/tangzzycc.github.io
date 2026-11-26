#!/usr/bin/env python3
"""Sync public Pull Requests for a GitHub user/repo into a static JSON snapshot."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "ROCm/aiter"
DEFAULT_AUTHOR = "tangzzycc"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "docs/experiments/data/aiter-prs.json"


def _classify_state(item: dict) -> str:
    pr_meta = item.get("pull_request") or {}
    if pr_meta.get("merged_at"):
        return "merged"
    if item.get("state") == "open":
        return "draft" if item.get("draft") else "open"
    return "closed"


def _extract_fallback_summary(body: str) -> str:
    for line in (body or "").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith(("#", "```", "|", "- ", "* ")):
            continue
        cleaned = " ".join(cleaned.replace("\u200b", " ").split())
        return cleaned[:160] + ("…" if len(cleaned) > 160 else "")
    return ""


def fetch_prs(
    repo: str,
    author: str,
    timeout: int = 15,
    existing_by_num: dict[int, dict] | None = None,
) -> dict:
    existing_by_num = existing_by_num or {}
    query = urllib.parse.quote(f"repo:{repo} is:pr author:{author}")
    url = (
        f"https://api.github.com/search/issues?q={query}"
        "&sort=created&order=desc&per_page=100"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "tangzzycc-notebook-pr-sync",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    items = []
    counts = {"total": 0, "open": 0, "merged": 0, "closed": 0}
    for raw in payload.get("items", []):
        state = _classify_state(raw)
        pr_meta = raw.get("pull_request") or {}
        title = " ".join(raw.get("title", "").replace("\u200b", " ").split())
        prev = existing_by_num.get(raw["number"], {})
        summary = prev.get("summary") or _extract_fallback_summary(raw.get("body", ""))
        entry = {
            "number": raw["number"],
            "title": title,
            "summary": summary,
            "state": state,
            "is_draft": bool(raw.get("draft")),
            "created_at": (raw.get("created_at") or "")[:10],
            "updated_at": (raw.get("updated_at") or "")[:10],
            "merged_at": (pr_meta.get("merged_at") or "")[:10] or None,
            "closed_at": (raw.get("closed_at") or "")[:10] or None,
            "url": raw["html_url"],
            "labels": [
                {"name": lbl["name"], "color": lbl.get("color", "")}
                for lbl in raw.get("labels", [])
            ],
            "comments": raw.get("comments", 0),
        }
        if prev.get("note_url"):
            entry["note_url"] = prev["note_url"]
        items.append(entry)
        counts["total"] += 1
        if state in ("open", "draft"):
            counts["open"] += 1
        elif state == "merged":
            counts["merged"] += 1
        else:
            counts["closed"] += 1

    return {
        "repo": repo,
        "author": author,
        "repo_url": f"https://github.com/{repo}",
        "author_prs_url": f"https://github.com/{repo}/pulls/{author}",
        "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "counts": counts,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    existing_by_num = {}
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        existing_by_num = {it["number"]: it for it in old.get("items", [])}

    data = fetch_prs(args.repo, args.author, existing_by_num=existing_by_num)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    c = data["counts"]
    print(
        f"Synced {args.repo} PRs by @{args.author} -> {args.output} "
        f"(total={c['total']}, open={c['open']}, merged={c['merged']}, closed={c['closed']})"
    )


if __name__ == "__main__":
    main()
