"""Inject static Pull Request contribution data into MkDocs page contexts."""

from __future__ import annotations

import json
from pathlib import Path

_cache: dict[str, dict] = {}


def _load_pr_data(docs_dir: Path, rel_path: str) -> dict | None:
    full_path = (docs_dir / rel_path).resolve()
    key = str(full_path)
    if key in _cache:
        return _cache[key]
    if not full_path.exists():
        return None
    data = json.loads(full_path.read_text(encoding="utf-8"))
    _cache[key] = data
    return data


def on_files(files, *, config):
    _cache.clear()
    return files


def on_page_context(context, *, page, config, **kwargs):
    docs_dir = Path(config["docs_dir"])
    meta = page.meta or {}

    raw_projects = meta.get("pr_projects")
    if isinstance(raw_projects, list):
        enriched = []
        for proj in raw_projects:
            data = _load_pr_data(docs_dir, proj["data"]) if proj.get("data") else None
            if not data:
                continue
            enriched.append({**proj, "pr_data": data})
        context["pr_projects"] = enriched

    pr_data_path = meta.get("pr_data")
    if isinstance(pr_data_path, str):
        data = _load_pr_data(docs_dir, pr_data_path)
        if data:
            context["pr_project"] = data

    return context
