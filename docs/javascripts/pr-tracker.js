(() => {
  const CACHE_TTL_MS = 10 * 60 * 1000;

  const ICONS = {
    open: '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"/></svg>',
    draft: '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"/></svg>',
    merged: '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M5.45 5.154A4.25 4.25 0 0 0 9.25 7.5h1.378a2.251 2.251 0 1 1 0 1.5H9.25A5.734 5.734 0 0 1 5 7.123v3.505a2.25 2.25 0 1 1-1.5 0V5.372a2.25 2.25 0 1 1 1.95-.218ZM4.25 13.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm8.5-4.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z"/></svg>',
    closed: '<svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M3.25 1A2.25 2.25 0 0 1 4 5.372v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.251 2.251 0 0 1 3.25 1Zm9.5 5.5a.75.75 0 0 1 .75.75v3.378a2.251 2.251 0 1 1-1.5 0V7.25a.75.75 0 0 1 .75-.75Zm-2.03-5.273a.75.75 0 0 1 1.06 0l.97.97.97-.97a.748.748 0 0 1 1.265.332.75.75 0 0 1-.205.729l-.97.97.97.97a.751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018l-.97-.97-.97.97a.749.749 0 0 1-1.275-.326.749.749 0 0 1 .215-.734l.97-.97-.97-.97a.75.75 0 0 1 0-1.06ZM2.5 3.25a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0ZM3.25 12a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm9.5 0a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Z"/></svg>',
  };

  const BADGE_TEXT = { open: "Open", draft: "Draft", merged: "Merged", closed: "Closed" };

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function classifyState(item) {
    if (item.pull_request && item.pull_request.merged_at) return "merged";
    if (item.state === "open") return item.draft ? "draft" : "open";
    return "closed";
  }

  function extractFallbackSummary(body) {
    for (const line of String(body || "").split(/\r?\n/)) {
      const cleaned = line.trim();
      if (!cleaned || /^(#|```|\||-\s|\*\s)/.test(cleaned)) continue;
      const flat = cleaned.replace(/\u200b/g, " ").replace(/\s+/g, " ");
      return flat.length > 160 ? `${flat.slice(0, 160)}…` : flat;
    }
    return "";
  }

  function normalizePayload(repo, author, payload) {
    const counts = { total: 0, open: 0, merged: 0, closed: 0 };
    const items = (payload.items || []).map((raw) => {
      const state = classifyState(raw);
      const mergedAt = (raw.pull_request && raw.pull_request.merged_at)
        ? raw.pull_request.merged_at.slice(0, 10)
        : null;
      const closedAt = raw.closed_at ? raw.closed_at.slice(0, 10) : null;
      const title = String(raw.title || "").replace(/\u200b/g, " ").replace(/\s+/g, " ").trim();
      counts.total += 1;
      if (state === "open" || state === "draft") counts.open += 1;
      else if (state === "merged") counts.merged += 1;
      else counts.closed += 1;
      return {
        number: raw.number,
        title,
        summary: extractFallbackSummary(raw.body),
        state,
        created_at: (raw.created_at || "").slice(0, 10),
        merged_at: mergedAt,
        closed_at: closedAt,
        url: raw.html_url,
        labels: (raw.labels || []).map((l) => ({ name: l.name })),
      };
    });
    return {
      repo,
      author,
      counts,
      items,
      fetchedAt: Date.now(),
    };
  }

  async function fetchLivePRs(repo, author, force) {
    const cacheKey = `notebook:pr-tracker:${repo}:${author}`;
    if (!force) {
      try {
        const cached = JSON.parse(sessionStorage.getItem(cacheKey) || "null");
        if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
          return cached;
        }
      } catch (_) {
        // Ignore storage errors
      }
    }
    const q = encodeURIComponent(`repo:${repo} is:pr author:${author}`);
    const url = `https://api.github.com/search/issues?q=${q}&sort=created&order=desc&per_page=100`;
    const resp = await fetch(url, {
      headers: { Accept: "application/vnd.github+json" },
    });
    if (!resp.ok) throw new Error(`GitHub API ${resp.status}`);
    const payload = await resp.json();
    const data = normalizePayload(repo, author, payload);
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify(data));
    } catch (_) {
      // Ignore storage quota errors
    }
    return data;
  }

  function applyCounts(root, counts) {
    for (const key of ["total", "open", "merged", "closed"]) {
      root.querySelectorAll(`[data-pr-count="${key}"]`).forEach((el) => {
        el.textContent = String(counts[key]);
      });
    }
    const filterMap = { all: counts.total, open: counts.open, merged: counts.merged, closed: counts.closed };
    for (const [k, val] of Object.entries(filterMap)) {
      root.querySelectorAll(`[data-pr-filter-count="${k}"]`).forEach((el) => {
        el.textContent = String(val);
      });
    }
  }

  function renderPRRow(pr, savedMeta) {
    const bucket = (pr.state === "open" || pr.state === "draft") ? "open" : pr.state;
    const icon = ICONS[pr.state] || ICONS.open;
    const badge = BADGE_TEXT[pr.state] || "Open";
    const meta = (savedMeta && savedMeta[String(pr.number)]) || {};
    const summaryText = meta.summary || pr.summary || "";
    const noteUrl = meta.noteUrl || "";
    const noteHtml = noteUrl
      ? `<a class="pr-note-link" href="${escapeHtml(noteUrl)}">笔记 →</a>`
      : "";
    const summaryHtml = summaryText
      ? `<p class="pr-row__summary">${escapeHtml(summaryText)}</p>`
      : "";
    const dateParts = [`create ${escapeHtml(pr.created_at)}`];
    if (pr.merged_at) dateParts.push(`merged ${escapeHtml(pr.merged_at)}`);
    else if (pr.closed_at) dateParts.push(`closed ${escapeHtml(pr.closed_at)}`);
    const labelsHtml = pr.labels && pr.labels.length
      ? `<span class="pr-row__labels">${pr.labels.map((l) => `<span class="pr-label">${escapeHtml(l.name)}</span>`).join("")}</span>`
      : "";
    return `
      <article class="pr-row" data-pr-state="${bucket}" data-pr-number="${pr.number}"${noteUrl ? ` data-pr-note-url="${escapeHtml(noteUrl)}"` : ""}>
        <span class="pr-row__icon pr-row__icon--${pr.state}" title="${badge}" aria-label="${pr.state}">${icon}</span>
        <div class="pr-row__body">
          <div class="pr-row__headline">
            <a class="pr-row__title" href="${escapeHtml(pr.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(pr.title)}</a>
            <span class="pr-row__badges">${noteHtml}<span class="pr-status-badge pr-status-badge--${pr.state}">${badge}</span></span>
          </div>
          ${summaryHtml}
          <div class="pr-row__meta">
            <span class="pr-row__number">#${pr.number}</span>
            <span class="pr-row__date">${dateParts.join(" · ")}</span>
            ${labelsHtml}
          </div>
        </div>
      </article>
    `;
  }

  function collectSavedMeta(pageEl) {
    const map = {};
    pageEl.querySelectorAll(".pr-row[data-pr-number]").forEach((row) => {
      const s = row.querySelector(".pr-row__summary");
      map[row.dataset.prNumber] = {
        summary: s ? s.textContent.trim() : "",
        noteUrl: row.dataset.prNoteUrl || "",
      };
    });
    return map;
  }

  function applyFilter(pageEl, filter) {
    pageEl.dataset.activeFilter = filter;
    pageEl.querySelectorAll("[data-pr-filter]").forEach((btn) => {
      const active = btn.dataset.prFilter === filter;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    let visibleCount = 0;
    pageEl.querySelectorAll(".pr-row").forEach((row) => {
      const match = filter === "all" || row.dataset.prState === filter;
      row.hidden = !match;
      if (match) visibleCount += 1;
    });
    const emptyEl = pageEl.querySelector("[data-pr-empty]");
    if (emptyEl) emptyEl.hidden = visibleCount > 0;
  }

  function updatePage(pageEl, data, savedSummaries) {
    applyCounts(pageEl, data.counts);
    const listEl = pageEl.querySelector("[data-pr-list]");
    if (listEl) {
      const rowsHtml = data.items.map((pr) => renderPRRow(pr, savedSummaries)).join("");
      listEl.innerHTML = `${rowsHtml}<p class="pr-empty-state" data-pr-empty hidden>当前筛选状态下暂无 PR。</p>`;
    }
    applyFilter(pageEl, pageEl.dataset.activeFilter || "all");
  }

  function initCards() {
    document.querySelectorAll("[data-pr-card]").forEach((card) => {
      const { repo, author } = card.dataset;
      if (!repo || !author) return;
      fetchLivePRs(repo, author, false)
        .then((data) => applyCounts(card, data.counts))
        .catch(() => {
          // Silently fall back to pre-rendered static snapshot
        });
    });
  }

  function initPage() {
    const pageEl = document.querySelector("[data-pr-page]");
    if (!pageEl) return;
    const { repo, author } = pageEl.dataset;
    const savedSummaries = collectSavedMeta(pageEl);

    pageEl.querySelectorAll("[data-pr-filter]").forEach((btn) => {
      btn.addEventListener("click", () => applyFilter(pageEl, btn.dataset.prFilter));
    });
    applyFilter(pageEl, "all");

    const refreshBtn = pageEl.querySelector("[data-pr-refresh]");
    if (refreshBtn && repo && author) {
      refreshBtn.addEventListener("click", async () => {
        refreshBtn.disabled = true;
        const orig = refreshBtn.textContent;
        refreshBtn.textContent = "同步中…";
        try {
          const data = await fetchLivePRs(repo, author, true);
          updatePage(pageEl, data, savedSummaries);
        } catch (_) {
          refreshBtn.textContent = "刷新失败（已保留快照）";
          setTimeout(() => {
            refreshBtn.textContent = orig;
            refreshBtn.disabled = false;
          }, 2200);
          return;
        }
        refreshBtn.textContent = orig;
        refreshBtn.disabled = false;
      });
    }

    if (repo && author) {
      fetchLivePRs(repo, author, false)
        .then((data) => updatePage(pageEl, data, savedSummaries))
        .catch(() => {
          // Keep pre-rendered static snapshot
        });
    }
  }

  function boot() {
    initCards();
    initPage();
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
