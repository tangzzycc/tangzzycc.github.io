(() => {
  const sidebars = {
    secondary: {
      selector: ".md-sidebar--secondary",
      label: "右侧目录",
      storageKey: "notebook.sidebar.secondary.collapsed",
      iconPath: "M6 7.5h12M6 12h9.5M6 16.5h12",
    },
  };

  const readPreference = (key) => {
    try {
      return window.localStorage.getItem(key) === "true";
    } catch (_error) {
      return false;
    }
  };

  const writePreference = (key, value) => {
    try {
      window.localStorage.setItem(key, String(value));
    } catch (_error) {
      // Keep the current-page interaction working when storage is unavailable.
    }
  };

  const isCollapsed = (side) =>
    document.body.classList.contains(`sidebar-${side}-collapsed`);

  const updateButton = (button, side) => {
    const collapsed = isCollapsed(side);
    const action = collapsed ? "展开" : "收起";

    button.classList.toggle("sidebar-rail-toggle--collapsed", collapsed);
    button.setAttribute("aria-expanded", String(!collapsed));
    button.setAttribute("aria-label", `${action}${sidebars[side].label}`);
    button.title = `${action}${sidebars[side].label}`;
  };

  const setCollapsed = (side, collapsed, persist = true) => {
    document.body.classList.toggle(`sidebar-${side}-collapsed`, collapsed);

    const button = document.querySelector(
      `.sidebar-rail-toggle[data-sidebar="${side}"]`
    );
    if (button) updateButton(button, side);
    if (persist) writePreference(sidebars[side].storageKey, collapsed);
  };

  const createButton = (side) => {
    const config = sidebars[side];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sidebar-rail-toggle sidebar-rail-toggle--${side}`;
    button.dataset.sidebar = side;
    button.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="${config.iconPath}"></path>
      </svg>
    `;
    button.addEventListener("click", () => {
      setCollapsed(side, !isCollapsed(side));
    });
    return button;
  };

  const setupSidebarControls = () => {
    Object.entries(sidebars).forEach(([side, config]) => {
      const sidebar = document.querySelector(config.selector);
      if (!sidebar || sidebar.hidden || !sidebar.querySelector('[data-md-component="toc"]')) return;

      const scrollwrap = sidebar.querySelector(":scope > .md-sidebar__scrollwrap");
      if (!scrollwrap) return;

      let button = scrollwrap.querySelector(
        `:scope > .sidebar-rail-toggle[data-sidebar="${side}"]`
      );
      if (!button) {
        button = createButton(side);
        scrollwrap.prepend(button);
      }

      setCollapsed(side, readPreference(config.storageKey), false);
    });
  };

  const setupSiteNavigation = () => {
    document.body.classList.remove("sidebar-primary-collapsed");
    document.querySelectorAll(".sidebar-rail-toggle--primary").forEach(button => button.remove());
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
  };

  document$.subscribe(setupSiteNavigation);
  document$.subscribe(setupSidebarControls);
})();
