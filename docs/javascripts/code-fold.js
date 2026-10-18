(() => {
  const foldAfterLines = 20;
  const previewLines = 14;

  const setupCodeFolds = () => {
    if (document.querySelector('meta[name="notebook-code-fold"][content="false"]')) return;

    document.querySelectorAll(".md-content .highlight").forEach((block) => {
      if (block.closest(".gemm-translation")) return;
      if (block.dataset.codeFoldReady === "true") return;

      const code = block.querySelector(":scope > pre > code");
      if (!code) return;

      const lineCount = code.textContent.replace(/\n$/, "").split("\n").length;
      block.dataset.codeFoldReady = "true";

      if (lineCount <= foldAfterLines) return;

      block.classList.add("code-fold");
      block.style.setProperty(
        "--code-preview-height",
        `${previewLines * 1.4 + 1.55}em`
      );

      const button = document.createElement("button");
      button.type = "button";
      button.className = "code-fold__toggle";
      button.setAttribute("aria-expanded", "false");
      button.textContent = `展开剩余 ${lineCount - previewLines} 行 ↓`;

      button.addEventListener("click", () => {
        const expanded = block.classList.toggle("code-fold--expanded");
        button.setAttribute("aria-expanded", String(expanded));
        button.textContent = expanded
          ? "收起代码 ↑"
          : `展开剩余 ${lineCount - previewLines} 行 ↓`;
      });

      block.appendChild(button);
    });
  };

  document$.subscribe(setupCodeFolds);
})();
