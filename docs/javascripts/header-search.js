/* 复用 Material 搜索；面板显示后再聚焦，支持空格激活图标。 */
document$.subscribe(() => {
  const button = document.querySelector(".notebook-search-toggle");
  const toggle = document.getElementById("__search");
  const input = document.querySelector(".md-search__input");
  if (!button || !toggle || !input || button.dataset.keyboardReady) return;
  button.dataset.keyboardReady = "true";
  toggle.addEventListener("change", () => {
    button.setAttribute("aria-expanded", String(toggle.checked));
    if (toggle.checked) {
      requestAnimationFrame(() => {
        if (toggle.checked) input.focus();
      });
    }
  });
  button.addEventListener("keydown", (event) => {
    // Enter 已由 Material 的全局键盘处理器支持。
    if (event.key !== " ") return;
    event.preventDefault();
    button.click();
  });
  // 收起后的输入框不可聚焦，先打开面板再复用原有搜索快捷键。
  document.addEventListener("keydown", (event) => {
    if (toggle.checked || event.ctrlKey || event.metaKey || event.altKey) return;
    if (!["/", "s", "f"].includes(event.key)) return;
    const target = event.target;
    if (target instanceof HTMLElement && (
      target.closest("input, textarea, select") || target.isContentEditable
    )) return;
    event.preventDefault();
    event.stopPropagation();
    button.click();
  });
});
