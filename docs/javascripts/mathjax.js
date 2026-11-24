window.MathJax = {
  // 由 Material 的页面事件统一触发，避免首次加载重复排版。
  startup: {
    typeset: false
  },
  // 霞鹜文楷的 x-height 高于 TeX 数学字体，关闭自动按 x-height 放大，
  // 改按大写字高比例（0.711em / 0.683em ≈ 1.04）微调至 1.05，使公式与正文齐平。
  chtml: {
    matchFontHeight: false,
    scale: 1.05
  },
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});

