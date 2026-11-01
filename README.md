# Notebook

Personal technical notes and a local blog. Notes are stored in `docs/`, while images and fonts are stored in `assets/`.

## Local Preview

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
NO_MKDOCS_2_WARNING=1 .venv/bin/mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to view the site. The development server watches for file changes and automatically refreshes the page.
`NO_MKDOCS_2_WARNING=1` suppresses Material's announcement about the upcoming MkDocs 2.0 release without affecting builds or previews.

## Search Engine Visibility

Every HTML page includes `noindex, nofollow, noimageindex` directives. Sitemap generation is disabled, and stale sitemap files are removed during builds. `robots.txt` permits crawling so search engines can read these directives; blocking crawling would prevent them from seeing `noindex`.

Direct links, site navigation, and the built-in search remain available. Rebuild and deploy the site to apply these settings online. These directives apply to compliant search engines, not access control; previously indexed pages may remain in results until they are crawled again. Raw assets would need an HTTP `X-Robots-Tag` header from the hosting server to receive their own indexing directive.

## GitHub Pages

The site is published at `https://tangzzycc.github.io/`. In the repository's **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.

The workflow in `.github/workflows/pages.yml` builds the site with MkDocs and deploys the generated `site/` directory whenever changes are pushed to `master`. It can also be run manually from the Actions tab. GitHub's default Jekyll build cannot render this MkDocs project.

## Repository Structure

- `docs/ai-infra/`: Notes on inference architecture and vLLM execution.
- `docs/cuda/`: Notes on CUDA programming, matrix multiplication, FlashAttention, and performance analysis.
- `docs/rocm/`: Notes on GPU programming and engineering with the ROCm platform.
- `docs/tags/`: Automatically generated topic index.
- `assets/ai-infra/`: Images organized by topic: prefill/decode disaggregation (PD), Multi-LoRA, and Model Runner V2.
- `assets/cuda/`: Images organized by note topic, including fundamentals, reduction, matrix multiplication, memory access, and scheduling.
- `assets/fonts/`: Local fonts; `assets/logo.jpg`: Site logo.
- `mkdocs.yml`: Site navigation and configuration.

Use descriptive filenames for images and store them in the appropriate topic directory. Notes reference images using relative paths; links that open images at full size use the same image path.

## Tags

Add a record date, summary, and topic tags to a note's YAML front matter. The module indexes and tag archives use this metadata to display articles by year, with the newest entries first. Each tag has its own archive page, generated automatically during the build.

```yaml
date: 2026-10-18
description: A summary of the note.
tags:
  - vLLM
  - KV Cache
```
