#!/usr/bin/env python3
"""Build embedded WOFF2 fonts and ink metrics for a sketch SVG.

Requires fonttools[woff]; reads the repository's WenKai without copying it.
"""

import argparse
import base64
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SKILL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WENKAI = SKILL_DIR.parents[2] / "assets/fonts/LXGWWenKaiGBScreen.ttf"


def glyph_metrics(font):
    from fontTools.pens.boundsPen import BoundsPen

    glyph_set = font.getGlyphSet()
    metrics = {}
    for codepoint, name in font.getBestCmap().items():
        pen = BoundsPen(glyph_set)
        glyph_set[name].draw(pen)
        bounds = pen.bounds
        metrics[chr(codepoint)] = {
            "advance": font["hmtx"][name][0],
            "left": bounds[0] if bounds else 0,
            "right": bounds[2] if bounds else 0,
            "hasInk": bounds is not None,
        }
    return {"em": font["head"].unitsPerEm, "glyphs": metrics}


def prepare(text, wenkai_path):
    from fontTools import subset
    from fontTools.ttLib import TTFont

    with TTFont(SKILL_DIR / "assets/fonts/ComicShanns-Regular.woff2") as comic:
        comic_points = set(comic.getBestCmap())
        metrics = glyph_metrics(comic)
    with TTFont(wenkai_path, recalcTimestamp=False) as wenkai:
        wenkai_points = set(wenkai.getBestCmap())
        requested = {ord(char) for char in text if not char.isspace()}
        missing = requested - comic_points - wenkai_points
        if missing:
            chars = ", ".join(f"{chr(cp)!r} (U+{cp:04X})" for cp in sorted(missing))
            raise ValueError(f"Neither bundled font covers these labels: {chars}")
        # Comic Shanns takes precedence; WenKai supplies Chinese and symbols.
        fallback = (requested - comic_points) | {32}
        options = subset.Options()
        options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 17]
        options.name_languages = [0x409]
        # AAT tables are unused in this cmap-based Chinese subset; drop explicitly.
        options.drop_tables += ["feat", "morx"]
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes=sorted(fallback))
        subsetter.subset(wenkai)
        # Give this modified subset its own family name, keeping license names.
        for record in wenkai["name"].names:
            value = {
                1: "WenKaiSketch", 2: "Regular", 3: "WenKaiSketch-Regular",
                4: "WenKaiSketch Regular", 6: "WenKaiSketch-Regular",
                16: "WenKaiSketch", 17: "Regular",
            }.get(record.nameID)
            if value:
                record.string = value.encode(record.getEncoding())
        wenkai.flavor = "woff2"
        stream = io.BytesIO()
        wenkai.save(stream)
    comic_bytes = (SKILL_DIR / "assets/fonts/ComicShanns-Regular.woff2").read_bytes()
    chinese_bytes = stream.getvalue()
    css = "\n".join(
        f"@font-face{{font-family:'{family}';"
        f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode('ascii')}) format('woff2');"
        "font-weight:400;font-style:normal;}"
        for family, data in [("Comic Shanns", comic_bytes), ("WenKaiSketch", chinese_bytes)]
    )
    css += "\ntext{font-family:'Comic Shanns','WenKaiSketch','LXGW WenKai GB Screen',cursive;font-weight:400;}"
    return {
        "schemaVersion": 1,
        "css": css,
        "license": (SKILL_DIR / "assets/fonts/LICENSES.txt").read_text(encoding="utf-8"),
        "metrics": metrics,
        "coveredCodepoints": sorted(comic_points | fallback),
        "fontBytes": {"comic": len(comic_bytes), "wenkai": len(chinese_bytes)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", type=Path, help="UTF-8 file containing all planned labels")
    source.add_argument("--svg", type=Path, help="Existing SVG; collect text, tspan, title, and desc")
    parser.add_argument("--wenkai", type=Path, default=DEFAULT_WENKAI)
    parser.add_argument("--output", type=Path, required=True, help="Temporary font bundle JSON")
    args = parser.parse_args()
    try:
        if args.svg:
            tree = ET.parse(args.svg)
            text = "\n".join(
                "".join(element.itertext()) for element in tree.iter()
                if element.tag.rsplit("}", 1)[-1] in {"text", "title", "desc"}
            )
        else:
            text = args.text_file.read_text(encoding="utf-8")
        bundle = prepare(text, args.wenkai)
        args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, ET.ParseError) as error:
        parser.exit(1, f"prepare_fonts: {error}\n")
    except ImportError:
        parser.exit(1, "prepare_fonts: install dependencies with: python -m pip install 'fonttools[woff]'\n")
    print(f"Font bundle: {args.output} (Comic {bundle['fontBytes']['comic']} B, WenKai subset {bundle['fontBytes']['wenkai']} B)")


if __name__ == "__main__":
    main()
