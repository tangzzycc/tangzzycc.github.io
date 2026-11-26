"""Build topic archives and dated article lists from note front matter."""

import math
import re
from datetime import date as Date
from itertools import groupby

import yaml
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File, InclusionLevel
from mkdocs.utils.meta import get_data
from pymdownx.slugs import slugify

_notes = {}
_private_notes = {}
_topics = {}
_archives = {}
_sections = {"ai-infra": "AI Infra", "cuda": "CUDA", "rocm": "ROCm"}
_private_sections = {"experiments": "Experiments"}


def _date(value):
    if value is None:
        return None
    if isinstance(value, Date):
        return value
    try:
        return Date.fromisoformat(str(value))
    except ValueError as error:
        raise PluginError(f"Invalid note date: {value!r}; use YYYY-MM-DD") from error


def _groups(notes):
    ordered = sorted(notes, key=lambda note: (
        -(note["date"] or Date.min).toordinal(), note["title"].casefold()
    ))
    return [
        {"year": year, "entries": list(entries)}
        for year, entries in groupby(
            ordered, key=lambda note: note["date"].year if note["date"] else None
        )
    ]


def on_files(files, *, config):
    _notes.clear()
    _private_notes.clear()
    _topics.clear()
    _archives.clear()
    for file in list(files):
        section = file.src_uri.split("/", 1)[0]
        if (not file.is_documentation_page()
            or file.src_uri.endswith("/index.md")
            or file.inclusion.is_excluded()):
            continue
        if section in _private_sections:
            _, meta = get_data(file.content_string)
            _private_notes[file.src_uri] = {
                "url": file.url,
                "title": meta.get("title", file.name),
                "description": meta.get("description", ""),
                "date": _date(meta.get("date")),
                "section": _private_sections[section],
                "status": meta.get("status", "Design"),
                "target": meta.get("target", ""),
                "cover": meta.get("cover", ""),
                "group": str(meta.get("group", "") or "").strip(),
                "order": meta.get("order", 0),
                "tags": list(dict.fromkeys(meta.get("tags", []))),
            }
            continue
        if section not in _sections:
            continue
        _, meta = get_data(file.content_string)
        if meta.get("exclude_from_index"):
            continue
        labels = meta.get("tags", [])
        note = {
            "url": file.url,
            "title": meta.get("title", file.name),
            "description": meta.get("description", ""),
            "date": _date(meta.get("date")),
            "section": _sections[section],
        }
        _notes[file.src_uri] = note
        for label in dict.fromkeys(labels):
            _topics.setdefault(label, []).append(note)

    make_slug = slugify(case="lower")
    for label in sorted(_topics, key=str.casefold):
        slug = make_slug(label, "-")
        path = f"tags/{slug}/index.md"
        if not slug or files.get_file_from_path(path):
            raise PluginError(f"Tag archive path is empty or already exists: {label!r}")
        meta = {
            "title": f"# {label}",
            "template": "tag-archive.html",
            "tag_archive": label,
            "hide": ["navigation", "toc", "footer"],
        }
        content = "---\n" + yaml.safe_dump(meta, allow_unicode=True) + "---\n"
        archive = File.generated(config, path, content=content,
                                 inclusion=InclusionLevel.NOT_IN_NAV)
        files.append(archive)
        _archives[label] = archive
    return files


def on_page_content(content, *, page, **kwargs):
    note = _notes.get(page.file.src_uri)
    note_date = note["date"] if note else _date(page.meta.get("date"))
    if not note_date:
        return content
    day = note_date.isoformat()
    stamp = (f'<p class="note-date"><time datetime="{day}" '
             f'aria-label="记录日期">{day}</time></p>')
    title = re.search(r"<h1\b[^>]*>.*?</h1>", content, re.S)
    if title:
        return content[:title.end()] + "\n" + stamp + content[title.end():]
    # Pages without a Markdown h1 get their title from the content template.
    return stamp + "\n" + content


def on_env(env, **kwargs):
    env.globals["notebook_note_count"] = len(_notes)
    env.globals["notebook_tag_count"] = len(_topics)
    return env


def on_page_context(context, *, page, **kwargs):
    context["tag_archive_urls"] = {
        label: archive.url for label, archive in _archives.items()
    }
    if page.file.src_uri == "tags/index.md":
        cloud = [
            {"name": label, "url": archive.url, "count": len(_topics[label])}
            for label, archive in _archives.items()
        ]
        if cloud:
            minimum = math.log1p(min(tag["count"] for tag in cloud))
            maximum = math.log1p(max(tag["count"] for tag in cloud))
            for tag in cloud:
                tag["weight"] = round(
                    (math.log1p(tag["count"]) - minimum) / (maximum - minimum)
                    if maximum > minimum else 0.5, 4,
                )
        context["tag_cloud"] = cloud
        context["tag_article_count"] = len({
            note["url"] for notes in _topics.values() for note in notes
        })
    elif page.meta.get("tag_archive"):
        entries = _topics[page.meta["tag_archive"]]
        context["article_groups"] = _groups(entries)
        context["archive_article_count"] = len(entries)
    elif page.meta.get("note_section"):
        sec = page.meta["note_section"]
        source_map = _private_notes if sec in _private_sections else _notes
        entries = [
            note for source, note in source_map.items()
            if source.startswith(sec + "/")
        ]
        context["article_groups"] = _groups(entries)
        if sec in _private_sections:
            def _entry_key(note):
                return (
                    note.get("order", 0),
                    -(note["date"] or Date.min).toordinal(),
                    note["title"].casefold(),
                )

            ordered_entries = sorted(entries, key=_entry_key)
            context["experiment_cards"] = ordered_entries

            group_defs = {}
            group_order = []
            for raw in page.meta.get("groups", []) or []:
                if isinstance(raw, dict) and raw.get("id"):
                    gid = str(raw["id"]).strip()
                    group_defs[gid] = raw
                    group_order.append(gid)

            grouped = {}
            items = []
            for note in ordered_entries:
                gid = note.get("group", "")
                if gid:
                    grouped.setdefault(gid, []).append(note)
                else:
                    items.append({
                        "kind": "single",
                        "date": note["date"],
                        "order": note.get("order", 0),
                        "title": note["title"],
                        "entry": note,
                    })

            seen_groups = set()
            for gid in group_order + [g for g in grouped if g not in group_defs]:
                if gid in seen_groups:
                    continue
                seen_groups.add(gid)
                meta_g = group_defs.get(gid, {})
                members = grouped.get(gid, [])
                if not members and not meta_g.get("show_empty"):
                    continue
                latest = max(
                    (m["date"] for m in members if m["date"]),
                    default=_date(meta_g.get("date")),
                )
                items.append({
                    "kind": "group",
                    "id": gid,
                    "title": meta_g.get("title", gid),
                    "description": meta_g.get("description", ""),
                    "target": meta_g.get("target", ""),
                    "status": meta_g.get("status", ""),
                    "date": latest,
                    "order": meta_g.get("order", 0),
                    "entries": members,
                })

            items.sort(key=lambda item: (
                item.get("order", 0),
                -(item["date"] or Date.min).toordinal(),
                item["title"].casefold(),
            ))
            context["experiment_items"] = items
    return context
