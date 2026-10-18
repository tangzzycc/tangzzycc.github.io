"""Generate the normal MkDocs TOC for the translated article's HTML headings."""

from html import escape
from html.parser import HTMLParser

from mkdocs.structure.toc import AnchorLink, TableOfContents


class ArticleHeadings(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headings = []
        self.heading = None
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"h2", "h3"}:
            self.heading = [int(tag[1]), attrs.get("id"), []]
        elif self.heading is not None:
            if self.skip_depth or tag == "span" and "sidenote" in attrs.get("class", "").split():
                self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"h2", "h3"} and self.heading is not None:
            level, anchor, text = self.heading
            if anchor:
                self.headings.append((level, anchor, "".join(text).strip()))
            self.heading = None
            self.skip_depth = 0
        elif self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.heading is not None and not self.skip_depth:
            self.heading[2].append(data)


def on_page_content(html, page, config, files):
    if page.file.src_uri != "ai-infra/CUDA矩阵乘法优化：从Naive到cuBLAS.md":
        return html

    parser = ArticleHeadings()
    parser.feed(html)
    items = []
    for level, anchor, title in parser.headings:
        item = AnchorLink(escape(title), anchor, level)
        if level == 3 and items:
            items[-1].children.append(item)
        else:
            items.append(item)
    if items:
        items[0].active = True
    page.toc = TableOfContents(items)
    return html
