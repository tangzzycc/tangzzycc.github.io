#!/usr/bin/env python3
"""Recreate seven supplied inference-parallelism figures as editable SVG/PNG.

Run with .venv/bin/python scripts/reproduce_parallel_diagrams.py.
The source figures were supplied in the conversation, not downloaded from Zhihu.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import html
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/assets/vllm/parallelism"
FONT = ROOT / "assets/fonts/LXGWWenKaiGBScreen.ttf"
W = 1100
COLORS = dict(bg="#191b2d", text="#e3e7f1", muted="#a9aec5",
              blue="#75b9ff", orange="#f3a56e", gold="#ffd05a",
              green="#70df80", purple="#c397f6", cyan="#74d8ec")
FILLS = dict(blue="#213c60", orange="#302013", gold="#30200e",
             green="#0c281b", purple="#251e3d", cyan="#102735")
RANKS = ["#ef545b", "#50c15d", "#6365ec", "#e5c84b"]
EXPERTS = ["#cf494e", "#c6743c", "#b1a541", "#b7c93b", "#85c841", "#3bc74f",
           "#2cbe78", "#35b9ac", "#3d9ec7", "#5166dd", "#7c4ed5", "#a445cf",
           "#c948b4", "#cf477f", "#c44368", "#dd4657"]
FONT_CACHE = {}


def esc(value):
    return html.escape(str(value), quote=True)


def font(size):
    if size not in FONT_CACHE:
        FONT_CACHE[size] = ImageFont.truetype(str(FONT), size)
    return FONT_CACHE[size]


def width(value, size=16):
    return font(size).getlength(str(value))


def wrap(value, maximum, size):
    lines = []
    for paragraph in str(value).split("\n"):
        line = ""
        for ch in paragraph:
            if line and width(line + ch, size) > maximum:
                lines.append(line)
                line = ""
            line += ch
        lines.append(line)
    return lines


class Figure:
    def __init__(self, slug, title, description, config, banner, accent="orange"):
        self.slug, self.title, self.description = slug, title, description
        self.parts, self.strings, self.bounds = [], [], []
        self.regions = []
        self.y = 18
        self.rect(46, self.y, 1008, 60, COLORS["blue"], "#1b2540", rx=6)
        self.text(550, self.y + 20, "模型配置和输入", 17, "blue", "middle", True)
        n = len(config)
        for i, item in enumerate(config):
            self.text(64 + (i + .5) * 972 / n, self.y + 45, item, 16, "text", "middle")
        self.y = 98
        self.rect(46, self.y, 1008, 42, "none", FILLS[accent], rx=4)
        self.text(550, self.y + 27, banner, 19, "text", "middle", True)
        self.y = 162

    def rect(self, x, y, w, h, stroke="none", fill="none", rx=3, sw=1.5, dash=False):
        self.bounds.append((x, y, x + w, y + h))
        d = ' stroke-dasharray="6 4"' if dash else ""
        alpha = ""
        if fill.startswith("#") and len(fill) == 9:
            alpha = f' fill-opacity="{int(fill[7:9], 16) / 255:.4f}"'
            fill = fill[:7]
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
                          f'rx="{rx:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{alpha}/>')

    def text(self, x, y, value, size=16, color="text", anchor="start", bold=False):
        value = str(value)
        self.strings.append(value)
        tw = width(value, size)
        left = x if anchor == "start" else x - tw if anchor == "end" else x - tw / 2
        self.bounds.append((left, y - size, left + tw, y + size * .25))
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" '
                          f'fill="{COLORS.get(color, color)}" text-anchor="{anchor}" '
                          f'font-weight="{700 if bold else 400}">{esc(value)}</text>')

    def paragraph(self, x, y, value, maxw=960, size=16, color="muted", lineheight=None):
        for line in wrap(value, maxw, size):
            self.text(x, y, line, size, color)
            y += lineheight or size * 1.5
        return y

    def line(self, x1, y1, x2, y2, color="muted", sw=1.2, arrow=False, opacity=1, dash=False):
        c = COLORS.get(color, color)
        marker = f' marker-end="url(#arrow-{color})"' if arrow else ""
        dashed = ' stroke-dasharray="5 5"' if dash else ""
        self.parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
                          f'stroke="{c}" stroke-width="{sw}" opacity="{opacity}"{marker}{dashed}/>')

    def panel(self, title, height, color="orange", subtitle=None, dash=False):
        y = self.y
        self.regions.append(dict(title=title, top=y, bottom=y + height))
        self.rect(46, y, 1008, height, COLORS[color], FILLS[color], rx=5, dash=dash)
        self.text(66, y + 25, title, 18, color, bold=True)
        if subtitle:
            self.text(1033, y + 24, subtitle, 13, "muted", "end")
        self.y += height
        return y

    def gap(self, color="muted", size=26):
        self.line(550, self.y + 2, 550, self.y + size - 3, color, 2, True)
        self.y += size

    def bar(self, value, color="blue", height=42, size=17):
        y = self.y
        self.regions.append(dict(title=value, top=y, bottom=y + height))
        self.rect(70, y, 960, height, COLORS[color], FILLS[color], rx=4)
        self.text(550, y + height / 2 + size * .34, value, size, color, "middle")
        self.y += height
        return y

    def field(self, x, y, w, value, color="orange", size=15, height=25):
        self.rect(x, y, w, height, COLORS[color], "#ffffff06", sw=.8)
        self.text(x + w / 2, y + height / 2 + size * .34, value, size, "text", "middle")

    def fields(self, y, rows, color="orange", x=70, w=960, size=15, step=29):
        for i, row in enumerate(rows):
            if isinstance(row, tuple):
                self.field(x, y + i * step, w * .48, row[0], color, size)
                self.field(x + w * .52, y + i * step, w * .48, row[1], color, size)
            else:
                self.field(x, y + i * step, w, row, color, size)

    def stripes(self, x, y, w, h, labels, colors=None, vertical=False, labelsize=12):
        colors = colors or RANKS
        n = len(labels)
        for i, label in enumerate(labels):
            xx, yy = (x + w * i / n, y) if vertical else (x, y + h * i / n)
            ww, hh = (w / n, h) if vertical else (w, h / n)
            c = colors[i % len(colors)]
            self.rect(xx, yy, ww, hh, c, c + "35", rx=0, sw=1.5)
            if label:
                self.text(xx + ww / 2, yy + hh / 2 + labelsize * .33, label,
                          labelsize, "text", "middle")

    def summary(self, rows, title=None):
        height = 64 + len(rows) * 29
        y = self.panel(title or self.title + " · 关键要点", height, "blue")
        for i, (key, value) in enumerate(rows):
            yy = y + 54 + i * 29
            self.text(76, yy, key, 16, "orange", bold=True)
            self.text(222, yy, value, 15, "text")

    def topology(self, y, dp=4, cp=1, tp=4, highlight="tp", x=96, w=900):
        """Use the supplied figure's CP-major rank numbering explicitly."""
        left = x + 64
        cellw = (w - 154) / tp
        for t in range(tp):
            self.text(left + (t + .5) * cellw, y + 15, f"tp={t}", 13, "blue", "middle")
        for c in range(cp):
            for d in range(dp):
                r = c * dp + d
                yy = y + 25 + r * 31
                self.text(left - 10, yy + 19, f"dp{d}" + (f",cp{c}" if cp > 1 else ""),
                          12, "muted", "end")
                for t in range(tp):
                    rank = (c * dp + d) * tp + t
                    color = RANKS[d % 4]
                    self.rect(left + t * cellw, yy, cellw - 5, 25, color,
                              color + "22", rx=3, sw=1)
                    self.text(left + (t + .5) * cellw - 2, yy + 17,
                              f"GPU {rank} · t{t},d{d}" + (f",c{c}" if cp > 1 else ""),
                              12, "text", "middle")
                self.text(left + tp * cellw + 8, yy + 18,
                          f"{highlight} 组 {r}" if highlight == "tp" else f"cp 配对行 {d}",
                          12, "gold" if highlight == "tp" else "purple")
        end = y + 25 + dp * cp * 31
        if highlight == "cp":
            self.text(x + w / 2, end + 18, "同 dp、同 tp 的 CP 配对跨两行；颜色标识请求所属 DP 组", 14,
                      "purple", "middle")
            end += 25
        return end

    def collective(self, kind, title, shape_in, shape_out, color="gold", topology=None, note=None):
        height = 190 + (34 if note else 0) + (190 if topology else 0)
        y = self.panel(title, height, color)
        xs = [110, 360, 610, 860]
        w = 132
        # Connections are placed behind objects and carry contributions or slices.
        for i, xx in enumerate(xs):
            for j, out in enumerate(xs):
                self.line(xx + w / 2, y + 81, out + w / 2, y + 126,
                          RANKS[i], .85, opacity=.52)
        for i, xx in enumerate(xs):
            self.text(xx + w / 2, y + 47, f"R{i} ({'attn_tp' if topology else 'TP'})", 12, "muted", "middle")
            if kind == "rs":
                self.stripes(xx, y + 54, w, 42, [f"p{i}[t{j}]" for j in range(4)], labelsize=9)
                self.rect(xx, y + 127, w, 29, RANKS[i], RANKS[i] + "35")
                self.text(xx + w / 2, y + 147, f"Σ pⱼ[t{i}]", 13, "text", "middle")
            elif kind == "ag":
                self.rect(xx, y + 54, w, 25, RANKS[i], RANKS[i] + "35")
                self.text(xx + w / 2, y + 72, f"t{i}" if "vocab" not in title.lower() else f"v{i}",
                          13, "text", "middle")
                self.stripes(xx, y + 127, w, 42, [f"t{j}" for j in range(4)], labelsize=10)
            else:
                self.rect(xx, y + 54, w, 25, RANKS[i], RANKS[i] + "35")
                self.text(xx + w / 2, y + 72, f"partial p{i}", 13, "text", "middle")
                self.rect(xx, y + 127, w, 29, COLORS[color], "#624120")
                self.text(xx + w / 2, y + 147, "sum = Σ pⱼ", 13, "text", "middle")
        self.text(550, y + 183, f"输入 {shape_in}  →  输出 {shape_out}", 15, color, "middle")
        yy = y + 205
        if note:
            self.text(550, yy, note, 14, "muted", "middle")
            yy += 28
        if topology:
            self.topology(yy, **topology)

    def matrix_column(self, y, mlp=False):
        self.line(77, y, 1023, y, "muted", .8, dash=True)
        self.text(550, y + 25, "列切分矩阵视图 · 每卡计算输出维的 1/4", 16, "purple", "middle", True)
        self.rect(94, y + 44, 91, 76, COLORS["blue"], FILLS["blue"], rx=0)
        self.text(139, y + 87, "x [S,H]", 16, "text", "middle")
        self.text(211, y + 88, "×", 30, "text", "middle")
        self.stripes(239, y + 44, 160, 76, ["", "", "", ""], vertical=True)
        self.text(319, y + 141, "W [H,M] · 按列分 4 份", 13, "muted", "middle")
        self.text(433, y + 88, "=", 28, "text", "middle")
        self.stripes(468, y + 44, 160, 76, ["", "", "", ""], vertical=True)
        self.text(548, y + 141, "Y [S,M] · 每卡 M/4", 13, "muted", "middle")
        for i, value in enumerate(["每卡输入：完整 x", "每卡权重：W 的 1/4 列", "每卡输出：Y 的 1/4 列", "✓ 无需通信"]):
            self.text(690, y + 57 + i * 27, value, 16, "green" if i == 3 else "text")
        self.text(550, y + 164, "M = 4H（MLP）" if mlp else "M = H（Q / K / V）；图中采用标准 MHA", 14,
                  "muted", "middle")

    def matrix_row(self, y, sp=False):
        self.line(77, y, 1023, y, "muted", .8, dash=True)
        self.text(550, y + 25, "行切分矩阵视图 · 输出形状完整，数值是 partial sum", 16, "purple", "middle", True)
        self.rect(90, y + 43, 46, 76, RANKS[0], RANKS[0] + "25", rx=0)
        self.text(113, y + 87, "xᵢ", 18, "text", "middle")
        self.text(163, y + 87, "×", 28, "text", "middle")
        self.rect(190, y + 66, 87, 26, RANKS[0], RANKS[0] + "25", rx=0)
        self.text(234, y + 85, "Wᵢ", 16, "text", "middle")
        self.text(304, y + 87, "=", 27, "text", "middle")
        self.rect(332, y + 43, 93, 76, RANKS[0], RANKS[0] + "25", rx=0)
        self.text(378, y + 75, "partialᵢ", 15, "text", "middle")
        self.text(378, y + 100, "[S,H]", 15, "text", "middle")
        for i in range(4):
            self.rect(464 + i * 8, y + 42 + i * 8, 56, 69, RANKS[i], FILLS["blue"], rx=0)
        self.text(504, y + 93, "4 份", 15, "text", "middle")
        self.text(554, y + 91, "Σ", 28, "gold", "middle")
        self.line(576, y + 83, 610, y + 83, "gold", 3, True)
        self.rect(624, y + 46, 66, 73, COLORS["text"], "#3b3d46", rx=0)
        self.text(657, y + 77, "Y", 19, "text", "middle")
        self.text(657, y + 101, "[S/4,H]" if sp else "[S,H]", 12, "text", "middle")
        for i, value in enumerate(["每卡输出：partialᵢ [S,H]", "4 个 partial 形状相同，逐元素求和", "ReduceScatter → 保留 1/4 tokens" if sp else "AllReduce → 每卡获得完整 Y"]):
            self.text(717, y + 58 + i * 27, value, 14, "gold" if i == 2 else "text")
        self.text(112, y + 141, "[S,M/4]", 13, "muted", "middle")
        self.text(233, y + 141, "[M/4,H]", 13, "muted", "middle")

    def qkv(self, b=32, s=2000, heads=16, tp=4, full=False, matrix=False, cp=False):
        h = 346 if matrix else 155
        y = self.panel("QKV Projection — " + ("完整权重，本地 batch 独立计算" if full else "列切分（Column Parallel）"), h)
        wshape = "[8192,8192]（完整）" if full else f"[8192,{8192 // tp}]（1/{tp}）"
        shape = f"[{b},{s},{heads},128]" if b is not None else f"[{s},{heads},128]"
        rows = [(f"Wq：{wshape}", f"Qᵢ：{shape}"),
                (f"Wk：{wshape}", f"Kᵢ：{shape}"),
                (f"Wv：{wshape}", f"Vᵢ：{shape}")]
        self.fields(y + 41, rows)
        trans = f"[{b},{heads},{s},128]" if b is not None else f"[{s},{heads},128]"
        self.text(550, y + 143, f"Attention 布局 Qᵢ / Kᵢ / Vᵢ：{trans} · " +
                  ("本 CP 段的 token；接下来收集 K/V" if cp else "无跨卡通信"), 15, "text", "middle")
        if matrix:
            self.matrix_column(y + 163)

    def attention(self, b=32, s=2000, heads=16, full=False):
        y = self.panel(f"Attention 计算 — 每卡独立计算自己的 {heads} heads，无跨卡通信", 143)
        self.fields(y + 42, [f"scoresᵢ = Qᵢ @ Kᵢᵀ / √128 ：[{b},{heads},{s},{s}]（逻辑形状）",
                             f"attn_outᵢ = softmax(scoresᵢ) @ Vᵢ ：[{b},{heads},{s},128]",
                             f"transpose + concat ：[{b},{s},{heads * 128}]（{'完整 H' if full else '每卡 H/4'}）"])
        self.text(1027, y + 134, "Flash Attention 不物化完整 scores", 13, "cyan", "end")

    def outproj(self, b=32, s=2000, tp=4, full=False, matrix=False, sp=False, flatten=False):
        y = self.panel("Output Projection — " + ("完整 Wo，直接得到本地 batch 输出" if full else "行切分（Row Parallel），各卡产生 partial sum"), 269 if matrix else 113)
        shape = f"[{b},{s},8192]" if b is not None and not flatten else f"[{b * s if b else s},8192]"
        local = 8192 if full else 8192 // tp
        self.fields(y + 42, [(f"Wo：[{local},8192]" + ("（完整权重）" if full else f"（输入维切 1/{tp}）"),
                             f"输入 attn_outᵢ：[{b},{s},{local}]" if b else f"输入 attn_outᵢ：[{s},{local}]"),
                            f"{'outᵢ' if full else 'partialᵢ'}：{shape}" + ("（完整结果，无需通信）" if full else "（形状完整，需跨 head 分片求和）")])
        if matrix:
            self.matrix_row(y + 108, sp)

    def mlp(self, b=32, s=2000, tp=4, full=False, down=False):
        y = self.panel("MLP Down（W2）— " if down else "MLP Up（W1）— ", 115)
        # Extend the heading without relying on a second overlapping text label.
        self.text(286 if down else 258, y + 25,
                  "完整权重，无跨卡通信" if full else "行切分，产生 partial sum" if down else "列切分，无跨卡通信",
                  18, "orange", bold=True)
        m = 32768 if full else 32768 // tp
        weight = f"[{m},8192]" if down else f"[8192,{m}]"
        output = 8192 if down else m
        self.fields(y + 43, [(f"{'W2' if down else 'W1'}：{weight}",
                              f"{'partialᵢ' if down and not full else 'hᵢ'}：[{b},{s},{output}]"),
                             f"{'hᵢ @ W2ᵢ' if down else 'GeLU(x @ W1ᵢ)'} → " +
                             ("完整本地输出" if full and down else "partial [B,S,H]，下一步归约" if down else "本地 intermediate 激活")])

    def moe_exchange(self, combine=False, ep=4, tokens=4000, topology_note=None):
        h = 283 if ep == 4 else 290
        y = self.panel(("反向 EP A2A（combine）— 按来源 rank 返回并加权合并" if combine else
                        f"MoE EP A2A（dispatch）— 按 Router 结果发送到 {ep} 个 expert rank"), h, "green")
        xs = [94 + i * 250 for i in range(4)] if ep == 4 else [77 + i * 60 for i in range(16)]
        boxw = 128 if ep == 4 else 37
        top, bottom = y + 64, y + 168
        for i, x in enumerate(xs):
            for j, xx in enumerate(xs):
                c = EXPERTS[i if combine else j] if ep == 16 else RANKS[i if combine else j]
                # A token packet is a slice: lines meet the relevant colored slice.
                sy = top + 27 if combine else top + (j + .5) * 58 / ep
                ty = bottom + (i + .5) * 58 / ep if combine else bottom + 27
                self.line(x + boxw, sy, xx, ty, c, .75 if ep == 4 else .55,
                          opacity=.32 if ep == 4 else .18)
        for i, x in enumerate(xs):
            self.text(x + boxw / 2, y + 52, f"R{i}" + ("（expert）" if combine and ep == 4 else ""),
                      12 if ep == 4 else 11, "muted", "middle")
            expert_label = f"E{i * (256 // ep)}–{(i + 1) * (256 // ep) - 1}"
            colors = RANKS if ep == 4 else EXPERTS
            if combine:
                self.rect(x, top, boxw, 58, colors[i], colors[i] + "44", rx=0)
                self.text(x + boxw / 2, top + 34, expert_label if ep == 4 else f"{256 // ep} experts",
                          12 if ep == 4 else 8, "text", "middle")
                self.stripes(x, bottom, boxw, 58,
                             [f"E{j * 64}–{j * 64 + 63}" for j in range(4)] if ep == 4 else [""] * ep,
                             colors, labelsize=9)
            else:
                self.stripes(x, top, boxw, 58,
                             [f"E{j * 64}–{j * 64 + 63}" for j in range(4)] if ep == 4 else [""] * ep,
                             colors, labelsize=9)
                self.rect(x, bottom, boxw, 58, colors[i], colors[i] + "44", rx=0)
                self.text(x + boxw / 2, bottom + 34, expert_label if ep == 4 else f"{256 // ep} experts",
                          12 if ep == 4 else 8, "text", "middle")
            self.text(x + boxw / 2, y + 244, f"R{i}", 11, "muted", "middle")
        self.text(550, y + h - 13,
                  topology_note or (f"combine 后每 rank [{tokens},8192] · 每个 token 的 top-k expert 输出按路由权重求和" if combine else
                                    f"dispatch 前每 rank [{tokens},8192] · top-k 路由可使一个 token 发往多个 expert"),
                  14, "green", "middle")

    def ffn(self, ep=4):
        self.bar(f"Expert FFN：每 rank 本地持有 {256 // ep} 个完整 expert · moe_tp=1 · 无 MOE_TP AllReduce", "green", 42, 16)

    def head(self, b=8, s=2000, topo=None, cp=False):
        shape = f"[{b},{s},8192]" if b else f"[{s},8192]"
        self.bar(f"Final LayerNorm（每 token 独立归一化，形状不变 {shape}）")
        self.gap()
        y = self.panel("LM Head — 本图选择 attn_tp=4 的 vocab 切分", 102)
        self.fields(y + 40, [("W_lm_headᵢ：[8192,32000]（全 vocab = 128000）",
                              f"logitsᵢ：{'[' + str(b) + ',' if b else '['}{s},32000]")])
        self.text(550, y + 91, "只对需采样的 token 计算 logits；此处保留原图的全 token 逻辑形状", 14, "muted", "middle")
        self.gap("gold")
        local = f"[{b},{s},32000]" if b else f"[{s},32000]"
        complete = f"[{b},{s},128000]" if b else f"[{s},128000]"
        self.collective("ag", "AllGather（vocab 维）— 每个 attn_tp 小组拼回完整 logits", local, complete,
                        topology=topo)
        self.gap("purple")
        y = self.panel("Sampling — 每个请求得到 next_token_id，进入下一 decode step", 117, "purple")
        for i in range(4):
            x = 90 + i * 251
            self.rect(x, y + 44, 167, 42, COLORS["purple"], "#352344")
            self.text(x + 83.5, y + 62, f"R{i}：softmax / top-k / argmax", 12, "text", "middle")
            self.text(x + 83.5, y + 80, "next_token_id", 13, "gold", "middle")
        self.text(550, y + 105, "确定性示意；随机采样须同步 RNG 或结果" + ("；CP prefill 只采样每条序列末 token" if cp else ""),
                  14, "muted", "middle")

    def finish(self):
        self.height = math.ceil(self.y + 24)
        for x1, y1, x2, y2 in self.bounds:
            if x1 < 0 or x2 > W or y1 < 0 or y2 > self.height:
                raise ValueError(f"{self.slug}: out of bounds {(x1, y1, x2, y2)}")
        return self

    def svg(self, embedded_font):
        markers = "".join(f'<marker id="arrow-{c}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
                          f'orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{v}"/></marker>'
                          for c, v in COLORS.items())
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.height}" '
                f'viewBox="0 0 {W} {self.height}" role="img" aria-labelledby="title desc">\n'
                f'<title id="title">{esc(self.title)}</title><desc id="desc">{esc(self.description)}</desc>\n'
                f'<defs>{markers}</defs>\n<style>@font-face{{font-family:"ParallelDiagram";'
                f'src:url("data:font/ttf;base64,{embedded_font}") format("truetype");font-weight:100 900;}}'
                'text{font-family:"ParallelDiagram","LXGW WenKai GB Screen",sans-serif;}</style>\n'
                f'<rect width="{W}" height="{self.height}" fill="{COLORS["bg"]}"/>\n'
                + "\n".join(self.parts) + "\n</svg>\n")


CONFIG = ["H=8192", "n_heads=64", "d_head=128", "S=2000", "B=32"]


def tp_figure():
    f = Figure("01-tp", "TP · 张量并行", "TP=4 的标准 Dense MHA 单层。列切/行切配对，Attention 与 MLP 出口各一次 AllReduce。", CONFIG,
               "TP=4（单机 4 卡 NVLink，标准 Dense MHA）")
    f.bar("输入 x：[32,2000,8192]（每卡完整副本）")
    f.gap()
    f.qkv(matrix=True)
    f.gap("gold")
    f.attention()
    f.gap("gold")
    f.outproj(matrix=True)
    f.gap("gold")
    f.collective("ar", "★ AllReduce #1（TP group=4）— Output Projection 之后", "4 × partial [32,2000,8192]", "每 rank [32,2000,8192]")
    f.gap()
    f.bar("Residual Add + LayerNorm：[32,2000,8192]（TP 区域外的完整激活副本）", height=45, size=16)
    f.gap()
    f.mlp()
    f.gap("gold")
    f.mlp(down=True)
    f.gap("gold")
    f.collective("ar", "★ AllReduce #2（TP group=4）— MLP Down 之后", "4 × partial [32,2000,8192]", "每 rank [32,2000,8192]")
    f.gap()
    f.bar("Residual Add → 层输出 x：[32,2000,8192]（每卡完整副本，进入下一层）", height=44, size=16)
    f.gap(size=23)
    f.summary([
        ("切分方式", "Q/K/V/W1 按输出维列切；Wo/W2 按输入维行切"),
        ("权重显存", "单层 1.5 GiB（fp16，4H 非 gated FFN）→ 每卡 384 MiB（÷4）"),
        ("通信", "每层 2 × AllReduce；Ring 单次每 rank 传输约 1.5 GiB（本配置）"),
        ("逻辑 scores", "fp16：15.26 GiB → 每卡 3.81 GiB；Flash Attention 不物化"),
        ("拓扑建议", "优先机内高速互连；跨机效果需结合链路与 batch 实测"),
        ("激活冗余", "LN / Residual 仍持完整 [B,S,H]；SP 可进一步按 token 切"),
        ("GQA / MQA", "TP 大于 KV head 数时，常见实现复制 KV heads，缓存不再线性缩小"),
        ("Decode", "S=1：单次每 rank 约 768 KiB；小消息的同步延迟更突出"),
    ])
    return f.finish()


def pp_figure():
    f = Figure("02-pp", "PP · 流水线并行", "32 层分成 2 个 stage，每 stage 16 层。四个 micro-batch 的纯 forward 流水线及气泡。",
               ["H=8192", "n_heads=64", "n_layers=32", "S=2000", "B=32", "M=4"],
               "PP=2（2 GPU，按 layer 切分，stage 间 P2P 传递激活）", "purple")
    f.bar("全局输入 x：[32,2000,8192]（B,S,H）→ Stage 0；M=4 时每 micro-batch Bμ=8")
    f.gap("orange")
    y = f.panel("层切分（Layer Partition）— 32 层分到 2 个 stage，每段 16 层", 269, "purple")
    for stage in range(2):
        yy = y + 44 + stage * 41
        color = RANKS[stage]
        f.text(76, yy + 21, f"Stage {stage}（GPU {stage}）", 16, color, bold=True)
        for i in range(16):
            x = 283 + i * 45.6
            f.rect(x, yy, 40, 28, color, color + "24")
            f.text(x + 20, yy + 19, f"L{i + stage * 16}", 12, "text", "middle")
    f.line(76, y + 135, 1024, y + 135, "muted", dash=True)
    f.text(550, y + 162, "权重分布对比：层内完整保留，切的是模型深度", 16, "purple", "middle")
    for stage in range(2):
        x = 77 + stage * 493
        f.rect(x, y + 176, 455, 72, RANKS[stage], FILLS["green" if stage else "orange"])
        f.text(x + 227.5, y + 200, f"Stage {stage}：L{stage * 16}–L{stage * 16 + 15} 全部权重", 16, "text", "middle", True)
        f.text(x + 227.5, y + 227, "1.5 GiB / 层 × 16 = 24 GiB（不含 embedding / lm_head）", 14, "text", "middle")
    f.gap("orange")

    def stagepanel(stage):
        color = "orange" if stage == 0 else "green"
        y = f.panel(f"Stage {stage}（GPU {stage}）前向 — 顺序计算 16 层完整 Dense MHA + MLP", 245, color)
        f.field(71, y + 41, 958, "一个 micro-batch 输入：[8,2000,8192]；全局 batch 为 4 个 micro-batch", "blue", 15)
        start = stage * 16
        for i, layer in enumerate([start, start + 1, start + 2, start + 13, start + 14, start + 15]):
            xx = 90 + i * 142
            f.rect(xx, y + 83, 67, 37, COLORS[color], FILLS[color])
            f.text(xx + 33.5, y + 108, f"L{layer}", 18, "text", "middle")
            if i < 5:
                f.line(xx + 68, y + 101, xx + 137, y + 101, color, 1.8, True)
        f.text(482, y + 105, "⋯", 22, "text", "middle")
        f.fields(y + 142, [("Wq / Wk / Wv / Wo：[8192,8192]（完整）", "Q / K / V：[8,64,2000,128]"),
                           ("W1：[8192,32768]；W2：[32768,8192]", "MLP intermediate：[8,2000,32768]"),
                          f"L{start + 15} 输出：[8,2000,8192]（层内无 AllReduce / A2A / AllGather）"], color)

    stagepanel(0)
    f.gap("cyan")
    y = f.panel("★ Point-to-Point Send / Recv — Stage 0 → Stage 1", 159, "cyan")
    f.rect(115, y + 50, 247, 57, RANKS[0], FILLS["orange"])
    f.text(238, y + 74, "GPU 0：L15 output", 17, "text", "middle")
    f.text(238, y + 97, "[8,2000,8192] fp16", 15, "text", "middle")
    f.line(365, y + 79, 730, y + 79, "cyan", 3, True)
    f.text(550, y + 63, "send = 256 MiB / micro-batch", 15, "cyan", "middle")
    f.rect(740, y + 50, 247, 57, RANKS[1], FILLS["green"])
    f.text(863, y + 74, "GPU 1：L16 input", 17, "text", "middle")
    f.text(863, y + 97, "[8,2000,8192] fp16", 15, "text", "middle")
    f.text(550, y + 138, "4 个 micro-batch 总计 1 GiB；每 micro-batch 在每个 stage 边界传一次激活", 16, "cyan", "middle")
    f.gap("green")
    stagepanel(1)
    f.gap()
    f.bar("全局输出 x：[32,2000,8192]（4 个 micro-batch 的汇总，仅 Stage 1 持有）", size=16)
    f.gap("gold")
    y = f.panel("★ 流水气泡（Pipeline Bubble）与 Micro-batch 调度", 364, "gold")
    f.text(550, y + 49, "示意假设：各 stage 等时；不计通信与调度开销；每个格子时长为 T_stage(Bμ)", 15, "muted", "middle")
    for case, m, top in [(0, 1, y + 85), (1, 4, y + 187)]:
        f.text(78, top, f"({'a' if case == 0 else 'b'}) M={m}：" + ("整 batch 串行，气泡 50%" if m == 1 else "4 个 micro-batch 流水，气泡 20%"), 17, "orange" if case == 0 else "green", bold=True)
        start, total, cw = 171, m + 1, 797 / (m + 1)
        for stage in range(2):
            yy = top + 17 + stage * 34
            f.text(156, yy + 21, f"Stage {stage}", 14, "muted", "end")
            for t in range(total):
                active = stage <= t < stage + m
                f.rect(start + t * cw, yy, cw - 3, 27, RANKS[stage] if active else "#767580",
                       RANKS[stage] + "55" if active else "#353440", dash=not active)
                f.text(start + (t + .5) * cw - 1.5, yy + 19,
                       f"m{t - stage}" if active and m > 1 else "FWD（B=32）" if active else "idle",
                       14, "text" if active else "muted", "middle")
    f.field(83, y + 293, 934, "bubble_ratio = (P − 1) / (M + P − 1)；P=2,M=4 → 1/5 = 20%", "gold", 18, 31)
    f.text(550, y + 346, "T_stage(B=32) 与 T_stage(Bμ=8) 不同；两张时间表的格子不能直接作绝对时延比较", 15, "muted", "middle")
    f.gap()
    f.summary([
        ("切分方式", "按 layer 切；层内完整；可叠加机内 TP"),
        ("权重显存", "总层权重 48 GiB → 每 stage 24 GiB（本图简化 Dense 模型）"),
        ("通信", "每 stage 边界 P2P；本配置 256 MiB / micro-batch，合计 1 GiB"),
        ("拓扑", "边界通信频率较低，常用于跨机；实际带宽需满足目标吞吐"),
        ("流水气泡", "M 越大越易填满；stage 工作量不均衡会降低流水效率"),
        ("Decode", "单请求 token 串行穿过 stages；需并发请求帮助填流水"),
        ("KV cache", "每 stage 只保存其本地层的 KV cache"),
        ("组合", "TP（机内）× PP（跨机）× DP（外层请求副本）"),
    ])
    return f.finish()


def dp_figure():
    f = Figure("03-dp", "DP · 数据并行", "DP=4 的四个独立完整模型副本，每个副本服务 B/4 个请求。", CONFIG,
               "DP=4（每卡完整模型副本，推理前向无跨 DP 通信）", "green")
    f.bar("全局输入 x：[32,2000,8192]（B,S,H）")
    f.gap("green")
    y = f.panel("输入切分 — Batch 维切分：按请求分配到 4 个独立副本", 316, "green")
    f.fields(y + 42, [("Rank 0：x[0:8] → [8,2000,8192]", "Wq / Wk / Wv / Wo：[8192,8192] 完整"),
                     ("Rank 1：x[8:16] → [8,2000,8192]", "W1：[8192,32768]；W2：[32768,8192]"),
                     "Rank 2：x[16:24] → [8,2000,8192]",
                     "Rank 3：x[24:32] → [8,2000,8192]（64 heads 完整，H 不变）"], "green")
    f.line(77, y + 170, 1023, y + 170, "muted", dash=True)
    f.text(550, y + 194, "Batch 切分矩阵视图 · 各样本独立前向", 16, "blue", "middle", True)
    f.stripes(93, y + 212, 96, 72, ["x₀", "x₁", "x₂", "x₃"], labelsize=13)
    f.line(200, y + 249, 257, y + 249, "green", 2, True)
    f.stripes(272, y + 212, 97, 72, ["x₀", "x₁", "x₂", "x₃"], labelsize=13)
    f.text(396, y + 251, "×", 29, "text", "middle")
    for i in range(4):
        f.rect(435 + i * 9, y + 208 + i * 7, 83, 60, COLORS["orange"], "#442d1b", rx=0)
    f.text(490, y + 257, "W [H,H]", 13, "text", "middle")
    f.text(564, y + 251, "=", 27, "text", "middle")
    f.stripes(598, y + 212, 97, 72, ["y₀", "y₁", "y₂", "y₃"], labelsize=13)
    f.text(743, y + 229, "每卡持完整 W，服务 1/4 batch", 16)
    f.text(743, y + 258, "✓ 推理前向无需通信", 16, "green", bold=True)
    f.text(743, y + 287, "切 batch，不切 heads 或 hidden", 14, "muted")
    f.gap("green")
    f.qkv(b=8, heads=64, tp=1, full=True)
    f.gap("green")
    f.attention(b=8, heads=64, full=True)
    f.gap("green")
    f.outproj(b=8, tp=1, full=True)
    f.gap()
    f.bar("Residual Add + LayerNorm：每卡 [8,2000,8192]（只持有本地 batch 的激活）", size=16)
    f.gap()
    f.mlp(b=8, tp=1, full=True)
    f.gap("green")
    f.mlp(b=8, tp=1, full=True, down=True)
    f.gap()
    f.bar("Residual Add → 层输出 xᵢ：[8,2000,8192]（各自进入下一层）", size=16)
    f.gap(size=22)
    y = f.panel("训练对照（仅作旁注）", 73, "gold", dash=True)
    f.text(77, y + 54, "训练需归约梯度；本图推理前向不涉及反向或 gradient AllReduce。", 16, "gold")
    f.gap(size=22)
    f.summary([
        ("切分方式", "Batch / 请求维度；每卡 B/4，权重与 heads 不切"),
        ("权重显存", "每卡 1 份完整模型；4 卡总权重是单份的 4 倍"),
        ("推理通信", "纯 DP 前向无跨副本通信；请求分发属于服务调度"),
        ("KV cache", "只保存本地请求的 KV；固定 global B 时，总 KV 不随 DP 倍增"),
        ("吞吐", "请求充足且负载均衡时可接近线性扩展"),
        ("限制", "DP 本身不帮助单个模型装入显存；可叠 TP / PP / EP"),
        ("请求绑定", "已有 KV 的请求通常留在所属副本；迁移需另做 KV 传输"),
    ])
    return f.finish()


def ep_figure():
    f = Figure("04-ep", "EP · 专家并行", "SGLang 风格的 4 个独立 TP4+EP4 副本。Attention 出口按 token ReduceScatter，MoE 后 AllGather。",
               ["H=8192", "n_heads=64", "d_head=128", "n_experts=256", "S=2000", "B=32"],
               "SGLang：--tp 4 --dp 4 --ep-size 4 --moe-a2a-backend deepep（不开 DP Attention）")
    y = f.panel("DP=4：4 个独立模型副本（replica-level DP），总卡数 16", 128, "purple", dash=True)
    for i in range(4):
        x = 78 + i * 251
        f.rect(x, y + 46, 190, 52, COLORS["purple"], "#2a203c")
        f.text(x + 95, y + 67, f"副本 {i}：GPU {i * 4}–{i * 4 + 3}", 15, "purple", "middle", True)
        f.text(x + 95, y + 87, "TP=4 + EP=4；64 experts/rank", 12, "text", "middle")
    f.text(550, y + 116, "副本间无前向通信；每副本服务 8 条请求；下图展开一个副本", 15, "muted", "middle")
    f.gap()
    f.bar("副本内输入 x：[8,2000,8192]（4 张 TP 卡共享本副本的 8 条请求）", size=16)
    f.gap()
    f.qkv(b=8)
    f.gap("gold")
    f.attention(b=8)
    f.gap("gold")
    f.outproj(b=8, flatten=True)
    f.gap("gold")
    f.collective("rs", "★ ReduceScatter（attn_tp=4）— 对 head partial 求和，并按 token 分片", "每 rank partial [16000,8192]", "每 rank [4000,8192]",
                 note="本图选用 A2A backend 的 token 分片路径；none backend 可采用不同通信路径")
    f.gap()
    y = f.panel("RS → Router → EP A2A 的衔接", 98, "blue", dash=True)
    f.text(77, y + 52, "RS 后每个 token 只有一个来源 rank；避免 4 个 TP rank 重复 dispatch 同一 token。", 16)
    f.text(77, y + 79, "Router 在本地 token 上选 top-k experts；Expert FFN 后由 combine 加权合并。", 16)
    f.gap("green")
    f.moe_exchange(ep=4)
    f.gap("green")
    f.ffn(ep=4)
    f.gap("green")
    f.moe_exchange(combine=True, ep=4)
    f.gap("gold")
    f.collective("ag", "★ AllGather（attn_tp=4，token 维）— 拼回本副本完整 token 输出", "每 rank [4000,8192]", "每 rank [16000,8192] = [8,2000,8192]",
                 note="RS 在 MoE 前完成 attention partial 求和；AG 拼接 MoE 结果，不把 MoE 当成线性归约")
    f.gap()
    f.bar("以上是一个 sparse 层的并行主线；重复 N 层后进入 Final LN + LM Head", height=42, size=16)
    f.gap()
    f.head()
    f.gap()
    f.summary([
        ("副本结构", "4 个独立副本 × 4 卡；Attention TP4 + MoE EP4"),
        ("专家权重", "256 experts / 4 ranks = 64 完整 experts / rank；moe_tp=1"),
        ("每层通信", "RS(attn_tp) → dispatch A2A(EP) → combine A2A(EP) → AG(attn_tp)"),
        ("副本间通信", "无；每个 EP group 只覆盖本副本的 4 张卡"),
        ("token 数", "每副本 16000 tokens → RS 后每 rank 4000 tokens"),
        ("图示范围", "展示 routed-expert 并行主线；层内 LN / Residual / shared expert 简化"),
        ("采样", "示意 deterministic sampling；通用采样须保持 TP ranks 一致"),
    ])
    return f.finish()


def dpa_figure():
    f = Figure("05-dp-attention", "DP Attention · 小 Attention TP + 大 MoE EP", "world=16，DP=4，attn_tp=4，EP=16。局部 Attention 组与全域 MoE 组具有不同通信范围。",
               ["H=8192", "n_heads=64", "d_head=128", "n_experts=256", "S=2000", "B=32"],
               "attn_tp=4，DP=4，DP Attention，EP=16，world=16")
    f.bar("每 DP 组输入 x：[8,2000,8192]；SGLang 启动参数 --tp 16 --dp 4 --enable-dp-attention", size=16)
    f.gap()
    f.qkv(b=8)
    f.gap("gold")
    f.attention(b=8)
    f.gap("gold")
    f.outproj(b=8, flatten=True)
    f.gap("gold")
    topo = dict(dp=4, cp=1, tp=4)
    f.collective("rs", "★ ReduceScatter（attn_tp=4）— 仅在每个 DP 组内对 partial 求和", "每 rank partial [16000,8192]", "每 rank [4000,8192]",
                 topology=topo, note="4 个局部 TP 小组并行 RS；随后 16 个 rank 一起参与 EP dispatch")
    f.gap("green")
    f.moe_exchange(ep=16)
    f.gap("green")
    f.ffn(ep=16)
    f.gap("green")
    f.moe_exchange(combine=True, ep=16)
    f.gap("gold")
    f.collective("ag", "★ AllGather（attn_tp=4，token 维）— 拼回本 DP 组 token 输出", "每 rank [4000,8192]", "每 rank [16000,8192] = [8,2000,8192]",
                 note="Attention 组内拼接，无需 world=16 的 attention AllReduce")
    f.gap()
    f.head(topo=topo)
    f.gap()
    f.summary([
        ("world 与 TP", "SGLang --tp=16 是副本内 world；attention 实际 attn_tp=16/4=4"),
        ("请求布局", "global B=32 → 每 DP 组 8 seq；每组 4 卡按 heads 切 attention"),
        ("通信范围", "Attention RS/AG：局部 4 卡；MoE dispatch/combine：全 16 卡"),
        ("专家权重", "256/16=16 完整 experts / rank，比 EP4 的 64 个少 4 倍"),
        ("KV cache", "MHA 下每 rank [8,16,S,128]；请求绑定 DP 组"),
        ("LM Head", "本图选择 --enable-dp-lm-head；vocab AG 在 attn_tp 小组内"),
        ("适用", "MoE 部署中独立选择 attention 小 TP 与大 EP；不要求 expert TP"),
        ("省略", "模型细节与调度同步简化；此图不作为特定版本运行 trace"),
    ])
    return f.finish()


def sp_figure():
    f = Figure("06-sp", "SP · 序列并行", "TP=4 + SP：LN/Residual 保持 token 分片，进入 TP 区前 AG，行切输出后 RS。", CONFIG,
               "TP=4 + SP（单机 4 卡 NVLink，标准 Dense MHA，S/4=500）")
    f.bar("输入 xᵢ：[32,500,8192]（SP 区域：每卡持有 1/4 序列）")
    f.gap()
    f.bar("LayerNorm（SP 区）：每卡独立归一化本地 token，输出 [32,500,8192]", height=47, size=16)
    f.gap("gold")
    f.collective("ag", "★ AllGather #1（TP group=4，token 维）— 进入 Attention TP 区", "每 rank [32,500,8192]", "每 rank [32,2000,8192]")
    f.gap("gold")
    f.qkv(matrix=True)
    f.gap("gold")
    f.attention()
    f.gap("gold")
    f.outproj(matrix=True, sp=True)
    f.gap("gold")
    f.collective("rs", "★ ReduceScatter #1（token 维）— Attention TP 区出口", "每 rank partial [32,2000,8192]", "每 rank [32,500,8192]")
    f.gap()
    f.bar("Residual Add（SP 区）：xᵢ + outᵢ → [32,500,8192]（本地 token 分片）", size=16)
    f.gap()
    f.bar("LayerNorm（SP 区，MLP 前）：[32,500,8192]，无跨卡通信", size=16)
    f.gap("gold")
    f.collective("ag", "★ AllGather #2（TP group=4，token 维）— 进入 MLP TP 区", "每 rank [32,500,8192]", "每 rank [32,2000,8192]")
    f.gap("gold")
    f.mlp()
    f.gap("gold")
    f.mlp(down=True)
    f.gap("gold")
    f.collective("rs", "★ ReduceScatter #2（token 维）— MLP TP 区出口", "每 rank partial [32,2000,8192]", "每 rank [32,500,8192]")
    f.gap()
    f.bar("Residual Add（SP 区）：层输出 xᵢ [32,500,8192] → 下一层 LayerNorm", size=16)
    f.gap()
    f.summary([
        ("切分方式", "SP 区按 token 切 LN / Residual；TP 区仍按 heads / intermediate 切"),
        ("AllGather", "每个 TP 入口 1 次：把本地 [B,S/4,H] 拼回 [B,S,H]"),
        ("ReduceScatter", "每个 TP 出口 1 次：对 partial 求和并保留 [B,S/4,H]"),
        ("通信", "每层 2 AG + 2 RS；理想 Ring 字节量与 2 AR 相同，调用数 2→4"),
        ("激活显存", "两个 [B,S,H] fp16 缓冲：2 GiB → 512 MiB（只计这两个缓冲）"),
        ("推理峰值", "不能把每层节省量直接乘层数；取决于同时存活的张量和融合"),
        ("Attention", "每 rank 的 Q 仍覆盖完整序列；KV cache 不因 SP 按序列切分"),
        ("用途", "训练和大 token 数 prefill 更有价值；decode 可按 batch 的 token 分片"),
    ])
    return f.finish()


def cp_figure():
    f = Figure("07-cp", "CP · 上下文并行", "参考图选择 AllGather KV 的 zigzag prefill CP，CP=2、DP=2、attn_tp=4、EP=16。每 DP 组一条序列。",
               ["H=8192", "n_heads=64", "d_head=128", "n_experts=256", "S=4000", "B=2（global）"],
               "attn_tp=4，DP=2，CP=2，EP=16，world=16；zigzag prefill，每 DP 组 B=1")
    f.bar("输入 xᵢ：[2000,8192]（每 CP rank：早 chunk 1000 + 晚 chunk 1000）", size=16)
    f.gap()
    f.qkv(b=None, s=2000, cp=True)
    f.gap("purple")
    y = f.panel("★ CP K/V AllGather（attn_cp=2，token 维）— 本图采用全量 K/V 收集", 405, "purple")
    xx = [197, 696]
    for i in range(2):
        for j in range(2):
            f.line(xx[i] + 100, y + 79, xx[j] + 100, y + 132, RANKS[i], 1.6, opacity=.55)
    for i, x in enumerate(xx):
        f.text(x + 100, y + 49, f"attn_cp={i}：本地 K/V 2000 tokens", 15, "orange", "middle")
        f.rect(x, y + 56, 200, 30, RANKS[i], RANKS[i] + "35")
        f.text(x + 100, y + 77, "[2000,16,128]", 15, "text", "middle")
        f.text(x + 100, y + 121, f"CP rank {i} 完整 K/V", 15, "green", "middle")
        f.stripes(x, y + 130, 200, 49, ["chunk 0", "chunk 1", "chunk 2", "chunk 3"],
                  [RANKS[0], RANKS[1], RANKS[1], RANKS[0]], labelsize=10)
    f.text(550, y + 200, "AG 后恢复原序列顺序：[4000,16,128]；每 CP rank 保存全量 K/V 供两次 FA 使用", 15, "purple", "middle")
    f.topology(y + 215, dp=2, cp=2, tp=4, highlight="cp")
    f.gap("orange")
    y = f.panel("Attention Zigzag 本地计算 — Q 分早晚两段；分别调用 causal Flash Attention", 284)
    f.text(550, y + 48, "CP0：chunk 0 + chunk 3；CP1：chunk 1 + chunk 2；全局 S=4000，每 chunk=1000", 15, "muted", "middle")
    f.rect(390, y + 63, 320, 33, COLORS["green"], FILLS["green"])
    f.text(550, y + 86, "Shared full K/V：[4000,16,128]", 16, "green", "middle")
    for i in range(2):
        x = 94 + i * 490
        entry_x = x - 13 if i == 0 else x + 400
        source_x = 390 if i == 0 else 710
        f.line(source_x, y + 79, entry_x, y + 79, "green", 1.8)
        f.line(entry_x, y + 79, entry_x, y + 197, "green", 1.8)
        f.line(entry_x, y + 197, x if i == 0 else x + 387, y + 197, "green", 1.8, True)
        f.rect(x, y + 108, 387, 45, RANKS[0], FILLS["orange"])
        f.text(x + 193.5, y + 130, "Q_prev：[1000,16,128]（早 chunk）" if i == 0 else "Q_next：[1000,16,128]（晚 chunk）",
               16, "text", "middle")
        f.text(x + 193.5, y + 147, "CP0 原位置 0:1000" if i == 0 else "CP0 原位置 3000:4000", 13, "muted", "middle")
        f.line(x + 193.5, y + 154, x + 193.5, y + 171, "orange", 1.8, True)
        f.rect(x, y + 172, 387, 49, COLORS["purple"], FILLS["purple"])
        f.text(x + 193.5, y + 193, f"FA #{i + 1}：" + ("kv_len_prev=1000" if i == 0 else "kv_len_next=4000"), 16, "purple", "middle")
        f.text(x + 193.5, y + 212, "out：[1000,16,128]；本地 causal mask", 14, "text", "middle")
        f.line(x + 193.5, y + 223, 550, y + 244, "gold", 1.3)
    f.field(245, y + 242, 610, "concat(out_prev, out_next) → [2000,16,128]；CP1 的 KV 长度为 2000 / 3000", "gold", 14)
    f.gap("gold")
    f.outproj(b=None, s=2000, flatten=True)
    f.gap("gold")
    topo = dict(dp=2, cp=2, tp=4)
    f.collective("rs", "★ ReduceScatter（attn_tp=4，token 维）— 本 (DP,CP) 段的 head partial 求和", "每 rank partial [2000,8192]", "每 rank [500,8192]",
                 topology=topo, note="示意 rank 编号：(cp × DP + dp) × attn_tp + tp；同 dp、同 tp 的 cp 两 rank 配对")
    f.gap("purple")
    y = f.panel("MOE_DP = ATTN_CP — MoE 入口 gather / 出口 scatter 的概念接口", 116, "purple")
    f.text(77, y + 55, "每 CP rank [500,8192] → CP pair gather → [1000,8192] → EP dispatch", 16)
    f.text(77, y + 83, "本段按参考图保留 CP/MoE 衔接；实际 buffer、去重与通信后端需核对相应代码版本。", 15, "muted")
    f.gap("green")
    f.moe_exchange(ep=16, tokens=1000,
                   topology_note="示意 dispatch 输入 [1000,8192]；图中的 CP gather 不等同于独立 MoE 权重副本")
    f.gap("green")
    f.ffn(ep=16)
    f.gap("green")
    f.moe_exchange(combine=True, ep=16, tokens=1000,
                   topology_note="combine 后 [1000,8192] → MOE_DP scatter（CP pair）→ 每 rank [500,8192]")
    f.gap("gold")
    f.collective("ag", "★ AllGather（attn_tp=4，token 维）— 拼回本 (DP,CP) 段", "每 rank [500,8192]", "每 rank [2000,8192]",
                 note="下一层仍持本 CP 段 Q；K/V 在下一 attention 层再次按本图路径收集")
    f.gap()
    f.head(b=None, s=2000, topo=topo, cp=True)
    f.gap()
    f.summary([
        ("配置", "SGLang world=16，DP=2，CP=2 → attn_tp=16/(2×2)=4"),
        ("请求布局", "global B=2 → 每 DP 组 1 seq → 每 CP rank 2000 tokens"),
        ("Zigzag", "4 chunks：CP0 拿 0+3，CP1 拿 1+2；两次 FA 的 KV 范围不同"),
        ("CP 通信", "本图是 K/V AllGather + 两次本地 FA，属于 prefill CP 示意"),
        ("内存限制", "每 rank attention 工作区持全量 K/V；不能据此声称该工作区按 CP 缩小"),
        ("Ring 区别", "Ring 逐块交换 K/V 并合并 online softmax 状态；本图未画 Ring"),
        ("MoE 衔接", "按原图保留 MOE_DP/ATTN_CP 接口；具体数据去重需版本级验证"),
        ("采样", "只需序列末 token logits；采样结果须返回该请求的下一步执行路径"),
    ])
    return f.finish()


CORRECTIONS = [
    ("TP 权重", "原图底部单层约 384 MB、每卡约 96 MB，与其矩阵尺寸不一致。", "按 fp16、MHA、intermediate=4H、非 gated FFN：单层 1.5 GiB，TP4 每卡 384 MiB。"),
    ("PP 权重", "原图按完整层 384 MB 计算每 stage 6.1 GB。", "同一 Dense 配置的完整层为 1.5 GiB；16 层每 stage 24 GiB；未计 embedding、LM Head、KV。"),
    ("PP micro-batch", "原图 M=4 但 stage/P2P 主线仍使用 B=32 的张量。", "主线使用每 micro-batch Bμ=8，传输 256 MiB/次；四次累计 1 GiB。全局输入/输出保留 B=32。"),
    ("Attention scores", "正文将 [32,64,2000,2000] 标为约 32 GB，dtype 未明确。", "用 fp16 逻辑 scores 计为 15.26 GiB（TP4 每卡 3.81 GiB）。fp32 是两倍；Flash Attention 不物化这些 scores。"),
    ("DP 的 KV", "原图总 KV 容量 N×单卡容易与固定 global batch 下的实际用量混淆。", "固定 global B 时，请求 KV 分配到各副本，总 KV 不因 DP 倍增；模型权重复制 N 份。"),
    ("DP Attention 配置", "原图标题 TP=4、DP=4、EP=16，但 SGLang --tp 指的是副本内 world。", "明确 attn_tp=4、world=16；SGLang 风格启动 --tp 16 --dp 4 --enable-dp-attention --ep-size 16。"),
    ("DP Attention 的 LM Head", "原图按局部 4 卡拼 vocab，但正文又说默认全 world 拼接。", "显式选择 --enable-dp-lm-head 的图示条件；不将此配置写成所有部署的默认行为。"),
    ("RS / MoE / AG", "将跨非线性 MoE 的 RS→MoE→AG 直接等同于一个 AllReduce 容易误导。", "RS 完成 Attention partial 归约，MoE 对 token 独立路由并计算，AG 拼接 MoE 结果；不声称 MoE 可与求和交换。"),
    ("专家大小", "正文使用每 expert 约 1.5 GB 的无明确层数口径估算。", "图中只保留 256/EP 个 experts/rank，不复用缺乏 intermediate、层数与 dtype 口径的显存估算。"),
    ("SP 显存", "正文把每层激活节省直接乘层数，并将其说成单层峰值÷N。", "只计算两个指定的 [B,S,H] 缓冲：2 GiB→512 MiB。推理峰值取决于张量生命周期，不能直接乘层数。"),
    ("CP 实现", "正文称 Ring，但原图画的是 K/V AllGather→两次本地 FA。", "按实际参考图画 AllGather CP，明确每 rank 工作区持全量 K/V；不声称它是 Ring Attention 或通用 decode CP。"),
    ("Sampling", "原图写每 rank 独立采样且无需同步，随机采样未说明 RNG。", "保留每 rank sampling 布局，标记为确定性示意；随机采样须同步 RNG 或采样结果。"),
]


def embed_font(figures):
    from fontTools import subset
    from fontTools.ttLib import TTFont
    options = subset.Options()
    options.name_IDs = [1, 2, 4, 6]
    options.layout_features = ["*"]
    chars = "".join(value for fig in figures for value in fig.strings)
    f = TTFont(str(FONT))
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=chars)
    subsetter.subset(f)
    out = io.BytesIO()
    f.save(out)
    return base64.b64encode(out.getvalue()).decode()


def register_font():
    # CairoSVG uses fontconfig rather than CSS @font-face. Register for this
    # process only; standalone SVGs carry the subset font in their own CSS.
    lib = ctypes.CDLL("libfontconfig.so.1")
    lib.FcConfigGetCurrent.restype = ctypes.c_void_p
    lib.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.FcConfigAppFontAddFile.restype = ctypes.c_int
    if not lib.FcConfigAppFontAddFile(lib.FcConfigGetCurrent(), str(FONT).encode()):
        raise RuntimeError("Could not register Chinese font with fontconfig")


def write_gallery(out, figures):
    cards = "\n".join(f'''<article id="{fig.slug}"><header><h2>{esc(fig.title)}</h2>
<div><button data-open="{fig.slug}.svg">放大查看</button> <a href="{fig.slug}.svg" download>SVG</a> ·
<a href="{fig.slug}.png" download>PNG</a></div></header><p>{esc(fig.description)}</p>
<button class="preview" data-open="{fig.slug}.svg" aria-label="放大查看 {esc(fig.title)}"><img src="{fig.slug}.png" loading="lazy" alt="{esc(fig.description)}"></button></article>'''
                      for fig in figures)
    navigation = "".join(f'<a href="#{fig.slug}">{esc(fig.title.split(" · ")[0])}</a>' for fig in figures)
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>7 种推理并行 · 本地流程图复现</title>
<style>
@font-face{font-family:Notebook;src:url("../../fonts/LXGWWenKaiGBScreen.ttf") format("truetype")}
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111423;color:#e5e9f5;font:16px/1.6 Notebook,system-ui,sans-serif}
main{max-width:1580px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 8px}p{color:#adb8d2}a{color:#86c4ff;text-underline-offset:3px}
nav{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0 26px}nav a{padding:7px 15px;border:1px solid #385575;border-radius:7px;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px;align-items:start}
article{border:1px solid #35415c;border-radius:10px;overflow:hidden;background:#191b2d;scroll-margin-top:20px}article header{padding:17px 18px 0}
h2{font-size:20px;margin:0 0 8px}article p{padding:0 18px;font-size:14px;min-height:46px}button{font:inherit;color:#f0d16c;border:1px solid #6d6038;background:#302719;border-radius:5px;padding:3px 9px;cursor:pointer}
.preview{border:0;border-radius:0;padding:0;width:100%;display:block;background:transparent}.preview img{width:100%;height:auto;display:block}
dialog{padding:0;width:min(1280px,98vw);height:94vh;max-width:98vw;max-height:96vh;background:#14182a;border:1px solid #556482;color:#e5e9f5;border-radius:8px}
dialog::backdrop{background:#000b}.toolbar{position:sticky;top:0;z-index:2;display:flex;gap:12px;align-items:center;padding:10px;background:#1f2940;flex-wrap:wrap}
.viewport{overflow:auto;height:calc(100% - 60px)}#detail{display:block;margin:auto;max-width:none}small{color:#abb8d2}
</style><main><h1>7 种推理并行 · 本地流程图复现</h1><p>按用户提供的知乎 @科英流程图重绘：保留深色配色、中文标签、纵向主线、矩阵分片与跨 rank 通信。
SVG 可编辑且内嵌中文字体，PNG 可直接查看。点击图片可放大、滚动和调整宽度。</p>
<p><a href="README.md">复现范围与修正记录</a> · <a href="manifest.json">文件清单</a> · <a href="contact-sheet.png">七图总览</a></p><nav>__NAV__</nav><div class="grid">__CARDS__</div>
<p>来源：<a href="https://zhuanlan.zhihu.com/p/2035106376138085860">知乎原文</a>。这是参考图的本地重绘与明确冲突修正，未进行框架版本级源码验证。</p></main>
<dialog><div class="toolbar"><button id="close">关闭</button><button id="fit">适应窗口</button><button id="actual">原始宽度</button>
<label>宽度 <input id="zoom" type="range" min="500" max="2200" step="100" value="1100"></label><a id="direct" target="_blank">打开 SVG</a><small>Esc 关闭；滚动查看完整流程</small></div>
<div class="viewport"><img id="detail" alt="放大的并行策略流程图"></div></dialog>
<script>
const dlg=document.querySelector('dialog'),im=document.querySelector('#detail'),zoom=document.querySelector('#zoom');
function resize(w){im.style.width=w+'px';zoom.value=w}
document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',()=>{im.src=b.dataset.open;document.querySelector('#direct').href=im.src;resize(1100);dlg.showModal();document.querySelector('.viewport').scrollTop=0}));
zoom.addEventListener('input',()=>resize(+zoom.value));document.querySelector('#close').onclick=()=>dlg.close();
document.querySelector('#fit').onclick=()=>resize(Math.max(500,dlg.clientWidth-25));document.querySelector('#actual').onclick=()=>resize(1100);
dlg.addEventListener('click',e=>{if(e.target===dlg)dlg.close()});
</script></html>'''
    (out / "gallery.html").write_text(content.replace("__NAV__", navigation).replace("__CARDS__", cards))


def write_readme(out, figures, scale):
    rows = "\n".join(f"| {fig.title} | [{fig.slug}.svg]({fig.slug}.svg) | [{fig.slug}.png]({fig.slug}.png) | {W} × {fig.height} |" for fig in figures)
    corrections = "\n".join(f"| {name} | {before} | {after} |" for name, before, after in CORRECTIONS)
    content = f'''# 7 种推理并行流程图：本地复现

来源：[知乎 @科英的原文](https://zhuanlan.zhihu.com/p/2035106376138085860)。重绘依据是用户在会话中提供的七张参考图和正文。
没有从知乎下载原始图片，也没有将参考图里的实现断言当作已经验证的代码事实。

保留参考图的深色配色、中文标注、纵向步骤、矩阵分片、四 rank 集合通信、十六 rank 专家交换和拓扑网格。
这是可编辑的矢量重绘，非逐像素复制。原图中的明显数字/配置冲突按用户选择修正，下面有完整记录。

## 查看与修改

- 浏览器打开 [gallery.html](gallery.html)：七图导航、放大、宽度缩放及 SVG/PNG 下载。也可用本地 HTTP 服务打开。
- SVG 中的文本、矩阵、框、箭头是独立元素，中文字体经过子集化后内嵌，无外部网络依赖。
- 修改 `scripts/reproduce_parallel_diagrams.py` 的 `tp_figure` / `pp_figure` 等函数，可批量再生成。
- 高清 PNG 每张宽度 `{round(W * scale)}` 像素，与 vLLM 笔记的 3840px 宽图采用相同默认规格；SVG 逻辑宽度 1100，放大不失真。
- 正文单栏阅读，`details/` 的 22 张局部图放在对应机制段落；[完整流程图附件](../../../ai-infra/推理并行策略-完整流程图.md)只收七张长图。
- 局部图按完整流程块切分，上下边缘移除外部残余箭头；保留块内以及多块之间的连线。PNG 宽度与完整图相同，另提供可编辑的 SVG。
- 正文与附件均使用普通 Markdown 图片，沿用 vLLM 笔记的页面排版，没有专用工具栏或全屏弹窗。
- [contact-sheet.png](contact-sheet.png) 供快速浏览整体结构，详细文字请看单张图。

| 策略 | SVG | PNG | SVG 尺寸 |
| --- | --- | --- | --- |
{rows}

## 重新生成

在仓库根目录运行：

```bash
.venv/bin/python scripts/reproduce_parallel_diagrams.py
```

默认输出目录为 `docs/assets/vllm/parallelism/`。可加 `--out` 指定目录、`--png-width 3840` 指定 PNG 宽度，或 `--scale 2` 指定比例，
或 `--svg-only` 只生成 SVG。依赖 `Pillow`、`fonttools`、`CairoSVG`（已存在于当前 `.venv`）。
PNG 渲染使用项目内的 `assets/fonts/LXGWWenKaiGBScreen.ttf`，仅在当前进程注册 fontconfig 字体，不修改系统字体配置。

用 HTTP 预览：

```bash
python3 -m http.server 8765 --directory docs/assets/vllm/parallelism
```

打开 `http://127.0.0.1:8765/gallery.html`。整理后的笔记位于 `docs/ai-infra/推理并行策略.md`，已加入 AI Infra 导航。

## 口径与简化

- 图 1/2/3/6：教学用 Dense MHA，H=8192，64 heads，head_dim=128，非 gated FFN intermediate=32768，权重/示例缓冲 fp16。省略 bias、LN 参数、embedding、LM Head 权重的层权重估算。
- 图 4/5/7：教学用 256 routed experts，`moe_tp=1`；每层 LN、Residual、shared expert、融合及调度元数据简化。top-k 路由会产生多目的 token 传输，图中彩条为按目的 expert 组分类的 token 包示意，非等量实测。
- 箭头表示数据来源/去向。集合通信的全连接图是逻辑数据关系，不意味着 NCCL 使用直接全连接链路；AllReduce 不代表实际 Ring 调度图。
- 图 5/7 的 LM Head 显式选择在 attention TP 小组内切 vocab 的配置，不能视为默认配置；实际推理通常只对需采样位置计算 logits。
- 图 7 按参考图画 **AllGather KV 的 prefill CP**。其每 rank attention 工作区持全量 K/V，不提供 Ring Attention 的分段状态合并，也不证明 decode KV cache 按 CP 分布。
- 图 7 按参考图保留 `MOE_DP = ATTN_CP` 的 gather/scatter 接口；实际 dispatch 去重、数据缓冲和 group 定义必须结合框架版本验证。rank 编号遵循参考图的 CP-major 示例，非所有版本的固定编号规则。
- 图 7 的输入 B=2、DP=2，所以每 DP 组只有一条序列；不使用正文中与 zigzag batch=1 约束冲突的 B=32 例子。
- Sampling 为确定性示意；随机采样需要一致的 RNG 或同步采样结果。图中没有模拟某个框架的随机采样实现。
- 这些图未进行 SGLang/vLLM 截至某 commit 的源码核实；原文行号未被当作本地证据。

## 明确修正记录

| 项目 | 参考图/正文的问题 | 本地复现的处理 |
| --- | --- | --- |
{corrections}

## 验证

生成时检查文本/图形画布边界、SVG XML 能否解析，以及每个 PNG 的尺寸。
本次交付另用 Chromium 检查了 7 张 SVG：中文内嵌字体加载成功，文字无越界或互相重叠，图底参考小字已删除。
正文使用 22 张局部图，附件保留 7 张完整图；两页使用普通 Markdown 图片，与 vLLM 笔记一致。
这验证的是本地资产的完整性与可读性，不是对原文所有框架实现断言的验证。
'''
    (out / "README.md").write_text(content)


def contact_sheet(out, figures):
    columns, cw, gap = 4, 310, 15
    thumb_heights = [round(fig.height * cw / W) for fig in figures]
    row_heights = [max(thumb_heights[:4]) + 44, max(thumb_heights[4:]) + 44]
    img = Image.new("RGB", (columns * (cw + gap) + gap, sum(row_heights) + gap * 3), COLORS["bg"])
    draw = ImageDraw.Draw(img)
    for i, fig in enumerate(figures):
        row, col = divmod(i, columns)
        x, y = gap + col * (cw + gap), gap + (row_heights[0] + gap if row else 0)
        for j, line in enumerate(wrap(fig.title, cw, 15)):
            draw.text((x, y + j * 18), line, font=font(15), fill=COLORS["blue"])
        with Image.open(out / (fig.slug + ".png")) as source:
            source = source.convert("RGB").resize((cw, thumb_heights[i]), Image.Resampling.LANCZOS)
            img.paste(source, (x, y + 34))
    img.save(out / "contact-sheet.png")


DETAIL_SLICES = [
    ("01-tp", "tp-column", 1, 1, "TP：QKV 列切分与矩阵视图"),
    ("01-tp", "tp-row", 3, 3, "TP：输出投影行切分产生 partial"),
    ("01-tp", "tp-allreduce", 4, 4, "TP：AllReduce 合并 head partial"),
    ("01-tp", "tp-mlp", 6, 8, "TP：MLP 列切、行切与第二次归约"),
    ("02-pp", "pp-layers", 1, 1, "PP：32 层分成两个 stages"),
    ("02-pp", "pp-transfer", 3, 3, "PP：micro-batch 的 stage 边界 P2P"),
    ("02-pp", "pp-bubble", 6, 6, "PP：micro-batch 流水与气泡"),
    ("03-dp", "dp-batch", 1, 1, "DP：请求 batch 分片与完整权重副本"),
    ("03-dp", "dp-forward", 2, 4, "DP：本地请求的完整 Attention 前向"),
    ("04-ep", "ep-replicas", 0, 1, "EP：四个独立 TP4 + EP4 副本"),
    ("04-ep", "ep-reduce-scatter", 5, 6, "EP：Attention 归约并按 token 分片"),
    ("04-ep", "ep-dispatch", 7, 8, "EP：按 expert 路由 token 并本地计算 FFN"),
    ("04-ep", "ep-combine", 9, 10, "EP：返回 token 输出并拼回副本序列"),
    ("05-dp-attention", "dpa-attention", 1, 3, "DP Attention：小 TP 组内的 Attention"),
    ("05-dp-attention", "dpa-topology", 4, 4, "DP Attention：四个局部 TP 通信组"),
    ("05-dp-attention", "dpa-experts", 5, 6, "DP Attention：全域 EP16 的 expert 分布"),
    ("06-sp", "sp-gather", 0, 2, "SP：本地 LN 与进入 TP 区前的 AllGather"),
    ("06-sp", "sp-scatter", 5, 7, "SP：行切输出经 ReduceScatter 返回本地 tokens"),
    ("07-cp", "cp-kv-gather", 2, 2, "CP：跨 CP rank 收集完整 K/V"),
    ("07-cp", "cp-zigzag", 3, 3, "CP：早晚两段 Query 的独立 causal Attention"),
    ("07-cp", "cp-topology", 4, 5, "CP：本地段输出与 Attention TP 分组"),
    ("07-cp", "cp-moe-entry", 6, 6, "CP：MOE_DP / ATTN_CP gather 与 scatter 接口"),
]


def write_details(out, figures):
    """Crop semantic panels, removing incoming/outgoing arrows at both edges."""
    details = out / "details"
    details.mkdir(exist_ok=True)
    indexed = {f.slug: f for f in figures}
    manifest = []
    for source, name, first, last, title in DETAIL_SLICES:
        fig = indexed[source]
        content_top = fig.regions[first]["top"]
        content_bottom = fig.regions[last]["bottom"]
        top = max(0, content_top - 12)
        bottom = min(fig.height, content_bottom + 12)
        with Image.open(out / (source + ".png")) as full:
            ratio = full.height / fig.height
            pixel_top = round(top * ratio)
            crop = full.crop((0, pixel_top, full.width, round(bottom * ratio)))
            # Keep the complete panel stroke (0.75 logical px) and erase only
            # outside padding. Inter-panel arrows inside the crop stay intact.
            draw = ImageDraw.Draw(crop)
            clean_top = math.floor((content_top - 1) * ratio) - pixel_top
            clean_bottom = math.ceil((content_bottom + 1) * ratio) - pixel_top
            if clean_top > 0:
                draw.rectangle((0, 0, crop.width - 1, clean_top - 1), fill=COLORS["bg"])
            if clean_bottom < crop.height:
                draw.rectangle((0, clean_bottom, crop.width - 1, crop.height - 1), fill=COLORS["bg"])
            crop.save(details / (name + ".png"))
            size = list(crop.size)
        tree = ET.parse(out / (source + ".svg"))
        svg = tree.getroot()
        svg.set("height", str(bottom - top))
        svg.set("viewBox", f"0 {top} {W} {bottom - top}")
        ns = "{http://www.w3.org/2000/svg}"
        svg.find(ns + "title").text = title
        # Remove boundary connectors from the vector too, rather than hiding
        # half an arrow through the viewBox clipping rectangle.
        for element in list(svg):
            if element.tag == ns + "line":
                ys = [float(element.get("y1")), float(element.get("y2"))]
                if min(ys) < content_top or max(ys) > content_bottom:
                    svg.remove(element)
        # The original background must still fill the translated crop viewport.
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree.write(details / (name + ".svg"), encoding="unicode")
        manifest.append(dict(name=name, title=title, source=source,
                             source_svg_y=[top, bottom], png_size=size,
                             content_svg_y=[content_top, content_bottom],
                             panels=[r["title"] for r in fig.regions[first:last + 1]]))
    (details / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Details: {len(manifest)} semantic crops in {details}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png-width", type=int, default=3840, help="Full PNG width in pixels (default: 3840)")
    parser.add_argument("--scale", type=float, default=None, help="Override --png-width with an explicit SVG-to-PNG scale")
    parser.add_argument("--svg-only", action="store_true")
    args = parser.parse_args()
    scale = args.scale if args.scale is not None else args.png_width / W
    if scale <= 0:
        parser.error("PNG width/scale must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    figures = [fn() for fn in [tp_figure, pp_figure, dp_figure, ep_figure, dpa_figure, sp_figure, cp_figure]]
    embedded_font = embed_font(figures)
    if not args.svg_only:
        import cairosvg
        register_font()
    manifest = []
    for fig in figures:
        svg = fig.svg(embedded_font)
        ET.fromstring(svg)
        path = args.out / (fig.slug + ".svg")
        path.write_text(svg)
        if not args.svg_only:
            # Match the registered local family for Cairo, which ignores CSS font URLs.
            render_svg = svg.replace('font-family:"ParallelDiagram","LXGW WenKai GB Screen",sans-serif',
                                     'font-family:"LXGW WenKai GB Screen",sans-serif')
            png = args.out / (fig.slug + ".png")
            pw, ph = round(W * scale), round(fig.height * scale)
            cairosvg.svg2png(bytestring=render_svg.encode(), write_to=str(png),
                            output_width=pw, output_height=ph)
            with Image.open(png) as image:
                assert image.size == (pw, ph)
        manifest.append(dict(slug=fig.slug, title=fig.title, svg=path.name,
                             png=None if args.svg_only else fig.slug + ".png",
                             svg_size=[W, fig.height],
                             png_size=None if args.svg_only else [round(W * scale), round(fig.height * scale)],
                             description=fig.description))
        print(f"{fig.slug}: SVG {W}×{fig.height}" + ("" if args.svg_only else f"; PNG {pw}×{ph}"))
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_gallery(args.out, figures)
    write_readme(args.out, figures, scale)
    if not args.svg_only:
        contact_sheet(args.out, figures)
        write_details(args.out, figures)
    print(f"Gallery: {args.out / 'gallery.html'}")


if __name__ == "__main__":
    main()
