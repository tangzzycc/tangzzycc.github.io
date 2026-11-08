(() => {
  const tocPreference = "notebook.toc.collapsed";

  const readPreference = () => {
    try {
      return window.localStorage.getItem(tocPreference) === "true";
    } catch (_error) {
      return false;
    }
  };

  const writePreference = (collapsed) => {
    try {
      window.localStorage.setItem(tocPreference, String(collapsed));
    } catch (_error) {
      // The current-page control also works when storage is unavailable.
    }
  };

  const setupTocControl = () => {
    const button = document.querySelector(".notebook-toc-toggle");
    const list = document.getElementById("notebook-page-toc");
    if (!button || !list) return;

    const setCollapsed = (collapsed) => {
      list.hidden = collapsed;
      button.setAttribute("aria-expanded", String(!collapsed));
      const label = `${collapsed ? "展开" : "收起"}目录`;
      button.setAttribute("aria-label", label);
      button.title = label;
    };

    setCollapsed(readPreference());
    button.hidden = false;
    if (button.dataset.tocKeybinding) return;
    button.dataset.tocKeybinding = "true";
    button.addEventListener("click", () => {
      setCollapsed(!list.hidden);
      writePreference(list.hidden);
    });
  };

  const setupSiteNavigation = () => {
    // Discard layout state left by the former sidebar rail controls.
    document.body.classList.remove("sidebar-primary-collapsed", "sidebar-secondary-collapsed");
    document.querySelectorAll(".sidebar-rail-toggle").forEach(button => button.remove());

    const drawer = document.getElementById("__drawer");
    if (!drawer) return;
    document.querySelectorAll(".notebook-drawer-toggle, .site-card-close").forEach(label => {
      label.tabIndex = 0;
      label.setAttribute("role", "button");
      label.setAttribute("aria-expanded", String(drawer.checked));
      if (label.dataset.drawerKeybinding) return;
      label.dataset.drawerKeybinding = "true";
      label.addEventListener("keydown", event => {
        // Material activates focused labels on Enter; add Space support.
        if (event.key !== " ") return;
        event.preventDefault();
        drawer.click();
      });
      drawer.addEventListener("change", () => {
        label.setAttribute("aria-expanded", String(drawer.checked));
      });
    });

    const toc = document.getElementById("notebook-page-toc");
    if (!toc || toc.dataset.drawerNavigation) return;
    toc.dataset.drawerNavigation = "true";
    toc.addEventListener("click", event => {
      if (!event.target.closest("a[href^='#']") || !drawer.checked) return;
      drawer.checked = false;
      drawer.dispatchEvent(new Event("change", { bubbles: true }));
    });
  };

  document$.subscribe(() => {
    setupSiteNavigation();
    setupTocControl();
  });
})();
