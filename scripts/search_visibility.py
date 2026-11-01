"""Keep search engine indexing disabled without affecting site navigation."""

from pathlib import Path


def on_config(config):
    config.theme.static_templates.discard("sitemap.xml")
    return config


def on_post_build(*, config):
    # Remove old sitemap output even when MkDocs runs a dirty build.
    for filename in ("sitemap.xml", "sitemap.xml.gz"):
        (Path(config.site_dir) / filename).unlink(missing_ok=True)
