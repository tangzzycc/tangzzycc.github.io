"""Reproduce five reference figures from the saved MRV2 articles.

Geometry and editable text are native SVG; subset fonts are embedded so Chinese
labels do not depend on the viewer's installed fonts. Reference images and the
provenance manifest live in .cache/model-runner-v2-sources/.
"""

from pathlib import Path
from io import BytesIO
import base64
import html
import json

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/ai-infra/model-runner-v2"
FONT = ROOT / "assets/fonts/LXGWWenKaiGBScreen.ttf"
INK = "#263342"
GREEN = "#438958"
BLUE = "#347398"


class Diagram:
    def __init__(self, width, height, description):
        self.width, self.height = width, height
        self.description = description
        self.parts = []
        self.labels = []

    def text(self, x, y, content, size=21, color=INK, anchor="middle"):
        self.labels.append(content)
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{html.escape(content)}</text>'
        )

    def rect(self, x, y, width, height, fill="white", stroke=INK, radius=12, dashed=False):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"'
            + (' stroke-dasharray="6 6"' if dashed else '') + '/>'
        )

    def arrow(self, data, color=INK, dashed=False, width=1.8):
        self.parts.append(
            f'<path d="{data}" stroke="{color}" stroke-width="{width}" fill="none" '
            f'marker-end="url(#{"green" if color == GREEN else "blue" if color == BLUE else "arrow"})"'
            + (' stroke-dasharray="8 7"' if dashed else '') + '/>'
        )

    def save(self, filename):
        font = TTFont(FONT)
        options = subset.Options()
        options.flavor = "woff"
        options.drop_tables += ["feat", "morx"]
        sub = subset.Subsetter(options=options)
        sub.populate(text="".join(self.labels))
        sub.subset(font)
        font.flavor = "woff"
        buffer = BytesIO()
        font.save(buffer)
        data = base64.b64encode(buffer.getvalue()).decode()
        markers = ''.join(
            f'<marker id="{name}" viewBox="0 0 12 12" refX="11" refY="6" '
            f'markerWidth="9" markerHeight="9" orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
            f'<path d="M1 1 L11 6 L1 11" fill="none" stroke="{color}" stroke-width="1.6"/></marker>'
            for name, color in [('arrow', INK), ('green', GREEN), ('blue', BLUE)]
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-describedby="desc">'
            f'<desc id="desc">{html.escape(self.description)}</desc><defs>{markers}</defs>'
            f'<style>@font-face{{font-family:MRV2Figure;src:url(data:font/woff;base64,{data}) format("woff");}}'
            'text{font-family:MRV2Figure,"LXGW WenKai GB Screen",sans-serif;}</style>'
            f'<rect width="{self.width}" height="{self.height}" fill="white"/>'
            + '\n'.join(self.parts) + '</svg>\n'
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / filename).write_text(svg)
        return {"file": filename, "size": [self.width, self.height], "labels": self.labels}


def request_token_lifecycle():
    d = Diagram(1000, 250, "RequestState token flow, matching the original two-step layout.")
    d.text(20, 64, "CPU", anchor="start")
    d.text(20, 155, "GPU", anchor="start")
    d.text(224, 26, "StagedWriteTensor", 20, GREEN)
    d.text(215, 64, "all_token_ids", 23)
    d.text(405, 64, "all_token_ids", 23)
    d.text(781, 64, "all_token_ids", 23)
    d.text(438, 153, "last_sampled_tokens", 23)
    d.text(873, 153, "last_sampled_tokens", 23)
    d.text(438, 181, "Tensor", 18, GREEN)
    d.rect(195, 199, 126, 43)
    d.text(258, 220, "Sample", 23)
    d.rect(614, 199, 126, 43)
    d.text(677, 220, "Sample", 23)
    d.arrow("M216 84 L244 191")
    d.arrow("M286 195 L350 86", dashed=True)
    d.arrow("M318 206 L325 173", dashed=True)
    d.arrow("M584 165 L635 196")
    d.arrow("M734 195 L764 87", dashed=True)
    d.arrow("M740 210 L763 174", dashed=True)
    return d.save("mrv2-request-state-token-lifecycle.svg")


def synchronous_timeline():
    d = Diagram(1200, 210, "Original synchronous CPU/GPU timeline: Scheduler, Prepare Input, Forward, Sample.")
    d.text(15, 99, "CPU", 21, anchor="start")
    d.text(15, 163, "GPU", 21, anchor="start")
    d.text(347, 31, "Step N-1", 21)
    d.text(854, 31, "Step N", 21, "#7d858b")
    d.parts.extend([
        '<path d="M61 12 V62 M698 12 V62" fill="none" stroke="#66747c" stroke-width="1.6"/>',
        '<path d="M75 31 H287 M408 31 H684 M713 31 H794 M914 31 H1128" fill="none" stroke="#66747c" stroke-width="1.6" marker-start="url(#arrow)" marker-end="url(#arrow)"/>',
    ])
    for x, y, w, label in [(72, 78, 104, "Scheduler"), (199, 78, 150, "Prepare Input"), (373, 142, 172, "Forward"), (568, 142, 172, "Sample")]:
        d.rect(x, y, w, 42, "#b7ecc0", radius=7)
        d.text(x+w/2, y+21, label, 20)
    for x, w, label in [(717, 144, "Scheduler"), (884, 170, "Prepare Input")]:
        d.rect(x, 78, w, 42, "#f5f7f8", "#a4abb0", radius=7)
        d.text(x+w/2, 99, label, 20, "#8a9298")
    for data in ["M177 99 H194", "M349 100 L369 152", "M545 151 Q548 113 557 115 Q565 118 568 143", "M740 148 L716 109", "M862 99 H880"]:
        d.arrow(data)
    d.arrow("M1055 103 L1074 140", "#8a9298")
    d.text(1160, 32, "…", 26, "#8a9298")
    return d.save("mrv1-synchronous-cpu-gpu-timeline.svg")


def async_output():
    d = Diagram(1120, 430, "Two sample_tokens steps continue on the main stream, with asynchronous D2H and deferred synchronization.")
    for x in [15, 395]:
        d.rect(x, 332, 270, 55)
        d.text(x+135, 359, "sample_tokens", 23)
        d.text(x+135, 410, "Step N" if x == 15 else "Step N+1", 23)
    for x in [154, 544]:
        d.rect(x, 225, 274, 56, "#b7ecc0")
        d.text(x+137, 253, "AsyncOutput", 23)
        d.rect(x+140, 117, 274, 56)
        d.text(x+277, 145, "ModelExecutor", 23)
        d.rect(x+272, 13, 274, 56)
        d.text(x+409, 41, "update_from_output", 22)
        d.arrow(f"M{x-47} 327 L{x+24} 287", dashed=True)
        d.arrow(f"M{x+159} 222 L{x+231} 177", dashed=True)
        d.arrow(f"M{x+300} 115 L{x+367} 72", dashed=True)
        d.text(x+146, 308, "Async D2H", 21)
        d.text(x+303, 201, "synchronize", 21)
    d.arrow("M305 359 H376", width=2.8)
    return d.save("mrv2-async-output-d2h-lifecycle.svg")


def placeholders():
    d = Diagram(1010, 410, "The same Request placeholder counter is incremented by two scheduling steps and decremented by ModelRunnerOutput.")
    d.text(24, 38, "Request", 24, anchor="start")
    d.text(15, 224, "Scheduler", 24, anchor="start")
    d.text(12, 367, "ModelRunner", 23, anchor="start")
    d.rect(208, 12, 788, 50, radius=10)
    d.text(602, 37, "num_output_placeholders", 24)
    d.rect(210, 188, 170, 81, radius=15)
    d.text(295, 215, "Schedule", 24)
    d.text(295, 246, "Step N", 24)
    d.rect(437, 188, 170, 81, radius=15)
    d.text(522, 215, "Schedule", 24)
    d.text(522, 246, "Step N+1", 24)
    d.rect(460, 319, 308, 82, radius=15)
    d.text(614, 345, "ModelRunnerOutput", 24)
    d.text(614, 378, "Step N", 24)
    d.arrow("M320 186 L465 65")
    d.arrow("M610 211 L742 65")
    d.arrow("M755 320 L979 65")
    # White backing matches the original labels placed over diagonal arrows.
    for x,y,w in [(265,91,280),(555,103,280),(708,169,285)]:
        d.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="52" fill="white"/>')
    d.text(405, 104, "+=scheduled_spec_dec", 20)
    d.text(405, 129, "ode_tokens+1", 20)
    d.text(695, 116, "+=scheduled_spec_dec", 20)
    d.text(695, 141, "ode_tokens+1", 20)
    d.text(850, 182, "-=len(new_token_ids)", 20)
    d.text(850, 208, "-= num_rejected", 20)
    return d.save("mrv2-output-placeholder-updates.svg")


def spec_components():
    d = Diagram(1200, 570, "Speculator, Scheduler, execute_model and RejectionSampler, preserving the original labels and cross-step loop.")
    d.rect(220, 12, 255, 82, stroke="#8b9298", radius=17)
    d.text(347, 53, "Scheduler", 23, "#858e94")
    d.rect(12, 242, 250, 82, "#a7d8f3", radius=19)
    d.text(137, 283, "Speculator", 23)
    d.rect(787, 242, 270, 86, "#b7ecc0", radius=19)
    d.text(922, 285, "execute_model", 23)
    d.rect(480, 477, 363, 83, "#a7d8f3", radius=20)
    d.text(661, 518, "RejectionSampler", 23)
    d.arrow("M142 241 Q138 180 155 143 L213 56", "#858e94", True)
    d.arrow("M324 96 L317 171 L610 284", "#858e94", True)
    d.parts.append('<path d="M613 215 V461" fill="none" stroke="#879096" stroke-width="1.6" stroke-dasharray="2 5"/>')
    d.arrow("M263 285 H781", BLUE)
    d.arrow("M949 333 V502 Q949 522 924 522 H850", GREEN)
    d.arrow("M478 520 H165 Q148 520 148 500 V327", BLUE)
    for x,y,w,h,label,color in [(19,191,103,38,"Draft",BLUE),(789,191,103,38,"Verify",GREEN),(479,428,103,38,"Accept",BLUE)]:
        d.rect(x,y,w,h,stroke=color,radius=6)
        d.text(x+w/2,y+h/2,label,23,color)
    for x,y,w,h in [(65,110,170,58),(226,175,246,61),(744,349,456,87)]:
        d.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white"/>')
    d.text(150,125,"拷贝输出token",21,"#858e94")
    d.text(150,151,"（异步）",21,"#858e94")
    d.text(349,187,"分配kv cache block",21,"#858e94")
    d.text(349,213,"（逻辑索引层面）",21,"#858e94")
    d.text(624,227,"新的Step开始",21,"#858e94",anchor="start")
    d.text(277,260,"1. draft token",23,BLUE,anchor="start")
    d.text(277,311,"4. repeat draft token",23,BLUE,anchor="start")
    d.text(277,360,"为下一轮Step(Decode)准备",22,BLUE,anchor="start")
    d.text(180,490,"3. Reject/Accept token",22,BLUE,anchor="start")
    d.text(180,552,"拒绝采样，接收合理的token",22,BLUE,anchor="start")
    d.text(752,365,"2. forward",22,GREEN,anchor="start")
    d.text(752,394,"将接收的token与draft token一并进行forward",20,GREEN,anchor="start")
    d.text(752,422,"生成 hidden_state、logits等...",21,GREEN,anchor="start")
    return d.save("mrv2-spec-decoding-components.svg")


if __name__ == "__main__":
    records = [request_token_lifecycle(), synchronous_timeline(), async_output(), placeholders(), spec_components()]
    cache = ROOT / ".cache/model-runner-v2-sources"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "svg-reproduction-record.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    for r in records:
        print(r["file"], r["size"])
