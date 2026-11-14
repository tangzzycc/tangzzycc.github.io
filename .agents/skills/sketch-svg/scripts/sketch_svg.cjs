'use strict';

// Reusable editable SVG helpers. Node built-ins only; Rough.js is bundled.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const rough = vm.runInNewContext(
  fs.readFileSync(path.join(__dirname, 'vendor/rough-4.6.6.js'), 'utf8') + ';rough;'
);

function escapeXml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;');
}

function attributes(values = {}) {
  return Object.entries(values).filter(([, value]) => value != null)
    .map(([key, value]) => `${key}="${escapeXml(value)}"`).join(' ');
}

class SketchSvg {
  constructor({width = 1000, height = 600, fonts, seed = 801} = {}) {
    if (!fonts || fonts.schemaVersion !== 1 || !fonts.metrics || !fonts.css || !fonts.license) {
      throw new Error('Supply a font bundle made by prepare_fonts.py');
    }
    if (!(width > 0 && height > 0)) throw new Error('Canvas dimensions must be positive');
    if (!Number.isInteger(seed) || seed < 1 || seed >= 2147483647) {
      throw new Error('Use a nonzero deterministic seed below 2147483647');
    }
    this.width = width;
    this.height = height;
    this.fonts = fonts;
    this.covered = new Set(fonts.coveredCodepoints);
    this.seed = seed;
    this.generator = rough.generator();
    this.parts = [];
  }

  options(extra = {}) {
    const seed = this.seed;
    this.seed = seed === 2147483646 ? 1 : seed + 1;
    return {seed, roughness: 0.60, bowing: 0.5, stroke: '#343a40',
      strokeWidth: 1.15, fillStyle: 'solid', ...extra};
  }

  shape(drawable, attrs = {}) {
    const dash = drawable.options.strokeLineDash;
    const dashOffset = drawable.options.strokeLineDashOffset;
    this.parts.push(`<g ${attributes(attrs)}>`);
    for (const item of this.generator.toPaths(drawable)) {
      // toPaths omits line dashes; apply them to outlines, never to fill paths.
      this.parts.push(`<path ${attributes({d: item.d, stroke: item.stroke,
        'stroke-width': item.strokeWidth, fill: item.fill,
        'stroke-linecap': 'round', 'stroke-linejoin': 'round',
        'stroke-dasharray': item.stroke !== 'none' && dash ? dash.join(' ') : null,
        'stroke-dashoffset': item.stroke !== 'none' && dash ? dashOffset : null})}/>`);
    }
    this.parts.push('</g>');
    return this;
  }

  rectangle(x, y, width, height, options = {}, attrs = {}) {
    return this.shape(this.generator.rectangle(x, y, width, height, this.options(options)), attrs);
  }

  ellipse(x, y, width, height, options = {}, attrs = {}) {
    return this.shape(this.generator.ellipse(x, y, width, height, this.options(options)), attrs);
  }

  line(x1, y1, x2, y2, options = {}, attrs = {}) {
    return this.shape(this.generator.line(x1, y1, x2, y2, this.options(options)), attrs);
  }

  polyline(points, options = {}, attrs = {}) {
    return this.shape(this.generator.linearPath(points, this.options(options)), attrs);
  }

  arrow(points, options = {}, attrs = {}) {
    if (points.length < 2) throw new Error('An arrow needs at least two points');
    const {headSize = 7, ...lineOptions} = options;
    const [x, y] = points.at(-1);
    let previous = points.length - 2;
    while (previous >= 0 && points[previous][0] === x && points[previous][1] === y) previous--;
    if (previous < 0) throw new Error('An arrow needs distinct start and end points');
    const [px, py] = points[previous];
    const angle = Math.atan2(y - py, x - px);
    this.polyline(points, lineOptions, attrs);
    // Arrowheads stay relatively stable and solid, even on dashed connectors.
    const headOptions = {...lineOptions, roughness: 0.25, disableMultiStroke: true,
      strokeLineDash: undefined, strokeLineDashOffset: undefined};
    for (const delta of [-0.45, 0.45]) {
      this.line(x, y, x - headSize * Math.cos(angle + delta),
        y - headSize * Math.sin(angle + delta), headOptions);
    }
    return this;
  }

  checkCoverage(value) {
    for (const char of String(value)) {
      if (!/\s/u.test(char) && !this.covered.has(char.codePointAt(0))) {
        throw new Error(`Font bundle lacks ${JSON.stringify(char)}; rebuild it with the updated labels`);
      }
    }
  }

  text(x, y, value, {size = 16, color = '#343a40', anchor = 'start', attrs = {}} = {}) {
    this.checkCoverage(value);
    this.parts.push(`<text ${attributes({...attrs, x, y, 'font-size': size,
      fill: color, 'text-anchor': anchor})}>${escapeXml(value)}</text>`);
    return this;
  }

  measure(value, size, spacing = 0) {
    const {em, glyphs} = this.fonts.metrics;
    let advance = 0, left = Infinity, right = -Infinity;
    for (const char of String(value)) {
      const metric = glyphs[char];
      if (!metric) throw new Error(`Math label ${JSON.stringify(char)} has no Comic Shanns metrics; use text() for Chinese`);
      if (metric.hasInk) {
        left = Math.min(left, advance + metric.left * size / em);
        right = Math.max(right, advance + metric.right * size / em);
      }
      advance += metric.advance * size / em + spacing;
    }
    return {left: Number.isFinite(left) ? left : 0, right: Number.isFinite(right) ? right : 0};
  }

  math(x, y, base, {sub = '', sup = '', suffix = '', size = 17, color = '#343a40',
    anchor = 'middle', gap = 0.35, scriptRatio = 0.66, suffixGap = 3.5, attrs = {}} = {}) {
    if (!base) throw new Error('A math label needs a base');
    if (!['start', 'middle', 'end'].includes(anchor)) throw new Error('Invalid math anchor');
    // measure() intentionally rejects unmeasured glyphs rather than guessing.
    const main = this.measure(base, size);
    const scriptSize = size * scriptRatio;
    const spacing = -0.35;
    const scriptLeft = main.right + gap;
    const subInk = sub ? this.measure(sub, scriptSize, spacing) : null;
    const supInk = sup ? this.measure(sup, scriptSize, spacing) : null;
    const subX = subInk ? scriptLeft - subInk.left : 0;
    const supX = supInk ? scriptLeft - supInk.left : 0;
    let right = Math.max(main.right, subInk ? subX + subInk.right : main.right,
      supInk ? supX + supInk.right : main.right);
    let suffixX = 0;
    if (suffix) {
      const suffixInk = this.measure(suffix, size);
      suffixX = right + suffixGap - suffixInk.left;
      right = suffixX + suffixInk.right;
    }
    const origin = anchor === 'middle' ? x - (main.left + right) / 2 :
      anchor === 'end' ? x - right : x - main.left;
    this.parts.push(`<g ${attributes({...attrs, 'data-role': 'math-label', 'data-base': base,
      'data-sub': sub, 'data-sup': sup,
      'aria-label': base + (sub ? '_' + sub : '') + (sup ? '^' + sup : '') + suffix})}>`);
    this.text(origin, y, base, {size, color, attrs: {'data-role': 'math-base'}});
    if (sub) this.text(origin + subX, y + size * 0.20, sub,
      {size: scriptSize, color, attrs: {'data-role': 'math-sub', 'letter-spacing': spacing}});
    if (sup) this.text(origin + supX, y - size * 0.35, sup,
      {size: scriptSize, color, attrs: {'data-role': 'math-sup', 'letter-spacing': spacing}});
    if (suffix) this.text(origin + suffixX, y, suffix, {size, color});
    this.parts.push('</g>');
    return this;
  }

  render({title, description, background = '#ffffff'} = {}) {
    if (!title || !description) throw new Error('Provide a meaningful title and description');
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}" ` +
      `viewBox="0 0 ${this.width} ${this.height}" role="img" aria-labelledby="title desc">\n` +
      `<title id="title">${escapeXml(title)}</title><desc id="desc">${escapeXml(description)}</desc>\n` +
      `<metadata id="font-license">${escapeXml(this.fonts.license)}</metadata>\n` +
      `<defs><style>${escapeXml(this.fonts.css)}</style></defs>\n` +
      `<rect width="${this.width}" height="${this.height}" fill="${escapeXml(background)}"/>\n` +
      this.parts.join('\n') + '\n</svg>\n';
  }
}

module.exports = {SketchSvg};
