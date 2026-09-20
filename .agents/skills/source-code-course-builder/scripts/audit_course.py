#!/usr/bin/env python3
"""Audit Markdown course files and versioned GitHub source links."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote


GITHUB_RE = re.compile(
    r"https://github\.com/(?P<repo>[^/]+/[^/]+)/(?:blob|tree)/"
    r"(?P<revision>[^/]+)/(?P<path>[^\s)#>。，]+)"
    r"(?:#L(?P<start>\d+)(?:-L(?P<end>\d+))?)?"
)
LOCAL_LINK_RE = re.compile(r"\[[^]]*]\((?P<angle><)?(?P<target>[^)>]+)(?(angle)>)\)")


@dataclass
class Finding:
    severity: str
    file: str
    line: int
    message: str


def run_git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def selected_files(course_dir: Path, patterns: list[str]) -> list[Path]:
    files = []
    for path in course_dir.rglob("*.md"):
        relative = str(path.relative_to(course_dir))
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            files.append(path)
    return sorted(files)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def inside_fence(text: str, offset: int) -> bool:
    """Return whether offset is inside a Markdown fenced code block."""
    prefix = text[:offset]
    return sum(line.lstrip().startswith("```") for line in prefix.splitlines()) % 2 == 1


def audit_markdown(path: Path, course_dir: Path, require_visual: bool) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(course_dir))
    findings: list[Finding] = []
    first = next((line for line in text.splitlines() if line.strip()), "")

    if not first.startswith("# "):
        findings.append(Finding("error", rel, 1, "first content line is not an H1"))
    if text.count("```") % 2:
        findings.append(Finding("error", rel, 1, "unbalanced fenced code blocks"))

    if require_visual:
        has_visual = "```mermaid" in text or re.search(r"!\[[^]]*]\([^)]+\)", text)
        if not has_visual:
            findings.append(
                Finding("error", rel, 1, "no Mermaid or image visual in document")
            )

    for match in LOCAL_LINK_RE.finditer(text):
        target = unquote(match.group("target").strip())
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if target and not (path.parent / target).exists():
            findings.append(
                Finding(
                    "error",
                    rel,
                    line_number(text, match.start()),
                    f"missing local link target: {target}",
                )
            )
    return findings


def audit_source_links(
    path: Path,
    course_dir: Path,
    repo_dir: Path,
    expected_repo: str | None,
    revision: str,
    cache: dict[tuple[str, str], list[str] | None],
) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(course_dir))
    findings: list[Finding] = []

    for match in GITHUB_RE.finditer(text):
        if inside_fence(text, match.start()):
            continue
        source_repo = match.group("repo")
        source_revision = match.group("revision")
        source_path = match.group("path")
        source_line = line_number(text, match.start())

        if source_revision in {"main", "master"}:
            findings.append(
                Finding("error", rel, source_line, "mutable main/master source link")
            )
        if source_revision != revision:
            findings.append(
                Finding(
                    "error",
                    rel,
                    source_line,
                    f"source revision {source_revision!r} != expected {revision!r}",
                )
            )
            continue
        if expected_repo and source_repo != expected_repo:
            continue

        key = (revision, source_path)
        if key not in cache:
            result = run_git(repo_dir, "show", f"{revision}:{source_path}")
            cache[key] = result.stdout.splitlines() if result.returncode == 0 else None
        source_lines = cache[key]
        if source_lines is None:
            findings.append(
                Finding(
                    "error", rel, source_line, f"path absent at {revision}: {source_path}"
                )
            )
            continue

        start = int(match.group("start")) if match.group("start") else None
        end = int(match.group("end")) if match.group("end") else start
        if end is not None and end > len(source_lines):
            findings.append(
                Finding(
                    "error",
                    rel,
                    source_line,
                    f"anchor L{end} exceeds {source_path} ({len(source_lines)} lines)",
                )
            )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--github-repo", help="Expected GitHub owner/repository")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Relative Markdown glob; repeatable (default: *.md)",
    )
    parser.add_argument(
        "--require-visual",
        action="store_true",
        help="Require a Mermaid or image visual somewhere in each document",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patterns = args.include or ["*.md"]
    files = selected_files(args.course_dir, patterns)
    findings: list[Finding] = []
    cache: dict[tuple[str, str], list[str] | None] = {}

    if not files:
        findings.append(Finding("error", ".", 1, "no Markdown files selected"))
    for path in files:
        findings.extend(audit_markdown(path, args.course_dir, args.require_visual))
        findings.extend(
            audit_source_links(
                path,
                args.course_dir,
                args.repo_dir,
                args.github_repo,
                args.revision,
                cache,
            )
        )

    if args.json_output:
        print(json.dumps([asdict(item) for item in findings], ensure_ascii=False, indent=2))
    elif findings:
        for item in findings:
            print(f"{item.severity.upper()} {item.file}:{item.line} {item.message}")
    else:
        print(
            f"OK: checked {len(files)} Markdown files and "
            f"{len(cache)} versioned source paths at {args.revision}"
        )
    return 1 if any(item.severity == "error" for item in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
