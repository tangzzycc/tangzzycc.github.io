(() => {
  const setupCodeMarks = () => {
    document.querySelectorAll(".md-content pre > code").forEach((code) => {
      if (code.dataset.codeMarkReady === "true") return;
      code.dataset.codeMarkReady = "true";

      const walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);

      nodes.forEach((node) => {
        if (!node.nodeValue.includes("==")) return;

        const fragment = document.createDocumentFragment();
        const parts = node.nodeValue.split(/(==[^=\n]+==)/g);
        parts.forEach((part) => {
          const match = part.match(/^==([^=\n]+)==$/);
          if (!match) {
            fragment.appendChild(document.createTextNode(part));
            return;
          }

          const mark = document.createElement("mark");
          mark.className = "code-mark";
          mark.textContent = match[1];
          fragment.appendChild(mark);
        });
        node.replaceWith(fragment);
      });
    });
  };

  document$.subscribe(setupCodeMarks);
})();
