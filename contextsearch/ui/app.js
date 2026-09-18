/* contextsearch UI. Vanilla JS, talks only to the public API under /api/v1.
   Sections: utilities, settings, api client, router, views, components. */

(() => {
  "use strict";

  // ------------------------------------------------------------------ utilities

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  const icon = (name, cls = "") => `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

  const fmt = {
    int: (n) => (n == null ? "" : Number(n).toLocaleString()),
    ms: (n) => (n == null ? "" : n < 10 ? `${n.toFixed(1)} ms` : `${Math.round(n)} ms`),
    bytes: (b) => {
      if (b == null) return "";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let i = 0;
      let v = Number(b);
      while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
      return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
    },
    date: (ts) => (ts ? new Date(ts * 1000).toLocaleString() : ""),
    ago: (ts) => {
      if (!ts) return "never";
      const s = Math.max(0, (Date.now() / 1000) - ts);
      if (s < 60) return "just now";
      if (s < 3600) return `${Math.floor(s / 60)} min ago`;
      if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
      return `${Math.floor(s / 86400)} d ago`;
    },
    pct: (x) => `${Math.round((x || 0) * 100)}%`,
    modelShort: (m) => (m || "").split("/").pop(),
  };

  const debounce = (fn, wait) => {
    let t = null;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
  };

  function toast(message, kind = "info", ms = 3200) {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.innerHTML = `${icon(kind === "error" ? "alert" : kind === "success" ? "check" : "info", "sm")}<span>${esc(message)}</span>`;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  /** Render text with [start,end] highlight spans as escaped HTML with <mark>. */
  function highlighted(text, spans) {
    if (!spans || !spans.length) return esc(text);
    let out = "";
    let pos = 0;
    for (const [a, b] of spans) {
      if (a < pos || b > text.length || a >= b) continue;
      out += esc(text.slice(pos, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>";
      pos = b;
    }
    return out + esc(text.slice(pos));
  }

  // ------------------------------------------------------------------ settings (per browser)

  const DEFAULTS = {
    apiKey: "",
    theme: "system",
    instant: true,
    strategy: "hybrid",
    topK: 10,
    showEvidence: false,
    rerank: null, // null = server default
    fusion: "",
    alpha: 0.5,
    models: null, // null = all
  };
  const prefs = { ...DEFAULTS };
  try {
    Object.assign(prefs, JSON.parse(localStorage.getItem("contextsearch.prefs") || "{}"));
  } catch (e) { /* private mode or blocked storage: defaults apply */ }
  function savePrefs() {
    try { localStorage.setItem("contextsearch.prefs", JSON.stringify(prefs)); } catch (e) { /* ignore */ }
  }

  function applyTheme() {
    const root = document.documentElement;
    if (prefs.theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", prefs.theme);
    const dark = prefs.theme === "dark" || (prefs.theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
    $("#theme-toggle").innerHTML = icon(dark ? "sun" : "moon");
  }

  // ------------------------------------------------------------------ api client

  class ApiError extends Error {
    constructor(message, status, code, requestId, details) {
      super(message);
      this.status = status; this.code = code; this.requestId = requestId; this.details = details || {};
    }
  }

  const api = {
    headers(extra = {}) {
      const h = { ...extra };
      if (prefs.apiKey) h["X-API-Key"] = prefs.apiKey;
      return h;
    },
    async request(method, path, { params, body, signal, raw } = {}) {
      const url = new URL(`/api/v1${path}`, location.origin);
      if (params) for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
      const init = { method, headers: this.headers(body instanceof FormData ? {} : { "Content-Type": "application/json" }), signal };
      if (body !== undefined) init.body = body instanceof FormData ? body : JSON.stringify(body);
      const res = await fetch(url, init);
      if (raw) return res;
      let data = null;
      try { data = await res.json(); } catch (e) { data = null; }
      if (!res.ok) {
        const err = new ApiError(data?.error || res.statusText || "request failed", res.status, data?.code || "http_error", data?.request_id || res.headers.get("x-request-id"), data?.details);
        if (res.status === 401) { openSettings("A valid API key is required. Enter it below."); }
        throw err;
      }
      return data;
    },
    get(path, params, opts) { return this.request("GET", path, { params, ...opts }); },
    post(path, body, opts) { return this.request("POST", path, { body, ...opts }); },
    patch(path, body) { return this.request("PATCH", path, { body }); },
    del(path) { return this.request("DELETE", path); },
    fileUrl(docId, download = false) {
      const u = new URL(`/api/v1/documents/${docId}/file`, location.origin);
      if (download) u.searchParams.set("download", "true");
      if (prefs.apiKey) u.searchParams.set("api_key", prefs.apiKey);
      return u.toString();
    },
    upload(files, onProgress) {
      return new Promise((resolve, reject) => {
        const form = new FormData();
        for (const f of files) form.append("files", f, f.name);
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/v1/documents/upload?wait=true");
        if (prefs.apiKey) xhr.setRequestHeader("X-API-Key", prefs.apiKey);
        xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
        xhr.onload = () => {
          let data = null;
          try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
          if (xhr.status >= 200 && xhr.status < 300) resolve(data);
          else reject(new ApiError(data?.error || (data?.errors?.[0]?.error) || "upload failed", xhr.status, data?.code, data?.request_id));
        };
        xhr.onerror = () => reject(new ApiError("network error during upload", 0));
        xhr.send(form);
      });
    },
  };

  function describeError(err) {
    if (err instanceof ApiError) return err.requestId ? `${err.message} (request ${err.requestId})` : err.message;
    if (err?.name === "AbortError") return "";
    return err?.message || String(err);
  }

  // ------------------------------------------------------------------ shared state and polling

  const state = {
    models: null,
    sources: null,
    stats: null,
    health: null,
    jobsActive: false,
    search: { query: "", response: null, controller: null, evidenceOpen: new Set() },
    viewRoute: "",
  };

  async function loadModels(force = false) {
    if (state.models && !force) return state.models;
    state.models = await api.get("/models");
    return state.models;
  }
  async function loadSources(force = false) {
    if (state.sources && !force) return state.sources;
    state.sources = (await api.get("/sources")).sources;
    return state.sources;
  }
  async function loadStats() {
    state.stats = await api.get("/system/stats");
    return state.stats;
  }

  let pollTimer = null;
  async function pollJobs() {
    try {
      const data = await api.get("/jobs", { limit: 20 });
      const active = data.jobs.some((j) => j.status === "running" || j.status === "queued") || data.queued > 0;
      const wasActive = state.jobsActive;
      state.jobsActive = active;
      $("#job-indicator").hidden = !active;
      const running = data.jobs.find((j) => j.status === "running");
      if (running) $("#job-indicator .label").textContent = running.kind === "sync" ? "Scanning" : running.kind === "build_model" ? "Embedding" : "Indexing";
      document.dispatchEvent(new CustomEvent("jobs", { detail: data }));
      if (wasActive && !active) {
        state.sources = null; state.models = null;
        document.dispatchEvent(new CustomEvent("jobs-idle"));
      }
    } catch (e) { /* the indicator is best effort */ }
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollJobs, state.jobsActive ? 1500 : 8000);
  }

  // ------------------------------------------------------------------ router

  const routes = { search: renderSearch, compare: renderCompare, sources: renderSources, documents: renderDocuments, activity: renderActivity };

  function parseHash() {
    const raw = location.hash.replace(/^#\/?/, "");
    const [path, qs] = raw.split("?");
    const params = Object.fromEntries(new URLSearchParams(qs || ""));
    return { route: path || "search", params };
  }

  function navigate(route, params = {}, replace = false) {
    const qs = new URLSearchParams(params).toString();
    const hash = `#/${route}${qs ? "?" + qs : ""}`;
    if (replace) history.replaceState(null, "", hash);
    else location.hash = hash;
    if (replace) render();
  }

  async function render() {
    const { route, params } = parseHash();
    const fn = routes[route] || renderSearch;
    state.viewRoute = route;
    $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
    const view = $("#view");
    view.className = "view" + (route === "compare" || route === "documents" ? " wide" : "");
    try {
      await fn(view, params);
    } catch (err) {
      const msg = describeError(err);
      view.innerHTML = `<div class="empty">${icon("alert")}<h3>Could not load this page</h3><p>${esc(msg)}</p></div>`;
    }
  }

  // ------------------------------------------------------------------ modals and drawer

  function modal({ title, body, footer = "", narrow = false, onClose = null }) {
    closeModal();
    const root = $("#modal-root");
    root.innerHTML = `
      <div class="modal-backdrop" data-close>
        <div class="modal ${narrow ? "narrow" : ""}" role="dialog" aria-modal="true" aria-label="${esc(title)}">
          <div class="modal-head"><h2 class="grow truncate">${title}</h2><button class="icon-btn" data-close aria-label="Close">${icon("x")}</button></div>
          <div class="modal-body">${body}</div>
          ${footer ? `<div class="modal-foot">${footer}</div>` : ""}
        </div>
      </div>`;
    root.querySelector(".modal-backdrop").addEventListener("click", (e) => { if (e.target.dataset.close !== undefined) closeModal(); });
    root.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeModal));
    root._onClose = onClose;
    return root.querySelector(".modal");
  }
  function closeModal() {
    const root = $("#modal-root");
    if (root._onClose) { try { root._onClose(); } catch (e) { /* ignore */ } }
    root._onClose = null;
    root.innerHTML = "";
  }

  function confirmDialog(title, text, confirmLabel = "Confirm", danger = true) {
    return new Promise((resolve) => {
      const m = modal({
        title,
        narrow: true,
        body: `<p>${esc(text)}</p>`,
        footer: `<button class="btn" data-close>Cancel</button><button class="btn ${danger ? "danger" : "primary"}" id="confirm-ok">${esc(confirmLabel)}</button>`,
        onClose: () => resolve(false),
      });
      m.querySelector("#confirm-ok").addEventListener("click", () => { $("#modal-root")._onClose = null; closeModal(); resolve(true); });
    });
  }

  function openSettings(notice = "") {
    const drawer = $("#settings-drawer");
    drawer.innerHTML = `
      <div class="drawer-head"><h2 class="grow">Settings</h2><button class="icon-btn" id="settings-close" aria-label="Close">${icon("x")}</button></div>
      <div class="drawer-body">
        ${notice ? `<div class="warn">${icon("alert", "sm")}<span>${esc(notice)}</span></div>` : ""}
        <section class="drawer-section">
          <h3>Access</h3>
          <div class="stack">
            <div class="field">
              <label for="pref-key">API key</label>
              <div class="row"><input class="input grow" id="pref-key" type="password" value="${esc(prefs.apiKey)}" placeholder="Only needed when the server has CSE_API_KEY set" autocomplete="off"><button class="icon-btn" id="pref-key-eye" title="Show">${icon("eye")}</button></div>
              <span class="hint">Stored in this browser only and sent as X-API-Key.</span>
            </div>
          </div>
        </section>
        <section class="drawer-section">
          <h3>Appearance</h3>
          <div class="segmented block" id="pref-theme">
            <button data-v="system" class="${prefs.theme === "system" ? "active" : ""}">System</button>
            <button data-v="light" class="${prefs.theme === "light" ? "active" : ""}">${icon("sun", "sm")} Light</button>
            <button data-v="dark" class="${prefs.theme === "dark" ? "active" : ""}">${icon("moon", "sm")} Dark</button>
          </div>
        </section>
        <section class="drawer-section">
          <h3>Searching</h3>
          <div class="stack">
            <label class="toggle"><input type="checkbox" id="pref-instant" ${prefs.instant ? "checked" : ""}><span class="track"></span><span>Search as you type</span></label>
            <label class="toggle"><input type="checkbox" id="pref-evidence" ${prefs.showEvidence ? "checked" : ""}><span class="track"></span><span>Open the evidence panel on every result</span></label>
            <div class="field"><label for="pref-topk">Results per search</label><select class="select" id="pref-topk">${[5, 10, 20, 30, 50].map((n) => `<option value="${n}" ${prefs.topK === n ? "selected" : ""}>${n}</option>`).join("")}</select></div>
          </div>
        </section>
        <section class="drawer-section">
          <h3>About</h3>
          <p class="small muted">contextsearch ${esc(state.health?.version || "")}. <a href="/docs" target="_blank" rel="noopener">API reference</a> is served by this server.</p>
        </section>
      </div>`;
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    $("#drawer-backdrop").hidden = false;
    $("#settings-close").addEventListener("click", closeSettings);
    $("#pref-key").addEventListener("change", (e) => { prefs.apiKey = e.target.value.trim(); savePrefs(); state.models = null; state.sources = null; toast("API key saved", "success"); render(); });
    $("#pref-key-eye").addEventListener("click", () => { const i = $("#pref-key"); i.type = i.type === "password" ? "text" : "password"; });
    $("#pref-theme").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; prefs.theme = b.dataset.v; savePrefs(); applyTheme(); $$("#pref-theme button").forEach((x) => x.classList.toggle("active", x === b)); });
    $("#pref-instant").addEventListener("change", (e) => { prefs.instant = e.target.checked; savePrefs(); });
    $("#pref-evidence").addEventListener("change", (e) => { prefs.showEvidence = e.target.checked; savePrefs(); });
    $("#pref-topk").addEventListener("change", (e) => { prefs.topK = Number(e.target.value); savePrefs(); });
    setTimeout(() => $("#pref-key")?.focus(), 50);
  }
  function closeSettings() {
    $("#settings-drawer").classList.remove("open");
    $("#settings-drawer").setAttribute("aria-hidden", "true");
    $("#drawer-backdrop").hidden = true;
  }

  // ------------------------------------------------------------------ search view

  function searchBoxHtml(value, compact = false) {
    return `
      <div class="searchbox ${compact ? "compact" : ""}">
        <span class="lead">${icon("search")}</span>
        <input class="input" id="q" type="search" value="${esc(value)}" placeholder="Search your documents. Quote a phrase to require it." autocomplete="off" spellcheck="false" aria-label="Search">
        <span class="trail">
          <button class="icon-btn sm" id="q-clear" title="Clear" ${value ? "" : "hidden"}>${icon("x")}</button>
          <span class="kbd">/</span>
        </span>
      </div>`;
  }

  async function renderSearch(view, params) {
    const query = (params.q || "").trim();
    if (params.strategy) prefs.strategy = params.strategy;
    if (!query) {
      state.search.response = null;
      let stats = null;
      try { stats = await loadStats(); } catch (e) { stats = null; }
      let recent = [];
      try { recent = (await api.get("/search/recent", { limit: 30 })).queries; } catch (e) { recent = []; }
      const seen = new Set();
      const suggestions = recent.filter((q) => q.results > 0 && !seen.has(q.query) && seen.add(q.query)).slice(0, 6);
      const empty = stats && stats.documents === 0;
      view.innerHTML = `
        <section class="hero">
          <h1>Find it in your documents</h1>
          <p>Keyword and semantic search combined, with the page, the passage and the reasons behind every result.</p>
          ${searchBoxHtml("")}
          ${empty ? `
            <div class="empty" style="padding-top:8px">
              <h3>Nothing indexed yet</h3>
              <p>Add a folder to watch, or upload a few files to try it out.</p>
              <div class="actions"><a class="btn primary" href="#/sources">${icon("folder")} Add a folder</a><a class="btn" href="#/documents?upload=1">${icon("upload")} Upload files</a></div>
            </div>` : `
            ${suggestions.length ? `<div class="suggestions">${suggestions.map((s) => `<a class="chip clickable outline" href="#/search?q=${encodeURIComponent(s.query)}">${icon("clock", "sm")} ${esc(s.query)}</a>`).join("")}</div>` : ""}
            <div class="corpus-line">
              ${stats ? `<span>${fmt.int(stats.documents)} documents</span><span>${fmt.int(stats.chunks)} passages</span><span>${stats.models.length} embedding ${stats.models.length === 1 ? "model" : "models"}</span>` : ""}
            </div>`}
        </section>`;
      bindSearchBox(view, "");
      return;
    }

    view.innerHTML = `
      <div class="search-top">${searchBoxHtml(query, true)}</div>
      <div class="search-layout">
        <div>
          <div class="results-head" id="results-head"><span>Searching...</span></div>
          <div class="results" id="results">${skeletonResults()}</div>
        </div>
        <aside class="side" id="side"></aside>
      </div>`;
    bindSearchBox(view, query);
    renderSidePanel();
    await runSearch(query, { prefix_last: false });
  }

  function skeletonResults() {
    return Array.from({ length: 4 }, () => `<div class="result"><div class="skeleton" style="height:18px;width:45%"></div><div class="skeleton" style="height:12px;width:30%;margin-top:8px"></div><div class="skeleton" style="height:56px;margin-top:12px"></div></div>`).join("");
  }

  function bindSearchBox(view, initial) {
    const input = $("#q", view);
    const clear = $("#q-clear", view);
    const instant = debounce(() => {
      const q = input.value.trim();
      if (q.length >= 2 && prefs.instant) {
        if (state.viewRoute !== "search" || !state.search.response) { navigate("search", { q }, true); return; }
        history.replaceState(null, "", `#/search?q=${encodeURIComponent(q)}`);
        runSearch(q, { prefix_last: true });
      }
    }, 350);
    input.addEventListener("input", () => {
      clear.hidden = !input.value;
      if (!input.value.trim()) return;
      if ($("#results")) instant();
      else if (prefs.instant && input.value.trim().length >= 2) instant();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); const q = input.value.trim(); if (q) navigate("search", { q }); }
      if (e.key === "Escape") { input.blur(); }
    });
    clear.addEventListener("click", () => { input.value = ""; clear.hidden = true; navigate("search"); });
    if (!initial) setTimeout(() => input.focus(), 30);
  }

  function searchParams(query, extra = {}) {
    const p = {
      q: query,
      strategy: prefs.strategy,
      top_k: prefs.topK,
      ...extra,
    };
    if (prefs.models && prefs.models.length) p.models = prefs.models.join(",");
    if (prefs.rerank !== null) p.rerank = prefs.rerank;
    if (prefs.fusion) p.fusion = prefs.fusion;
    if (prefs.fusion === "weighted") p.alpha = prefs.alpha;
    if (prefs.source) p.source = prefs.source;
    if (prefs.ext) p.ext = prefs.ext;
    if (prefs.group === false) p.group = false;
    return p;
  }

  async function runSearch(query, extra = {}) {
    state.search.query = query;
    if (state.search.controller) state.search.controller.abort();
    const controller = new AbortController();
    state.search.controller = controller;
    const started = performance.now();
    try {
      const data = await api.get("/search", searchParams(query, extra), { signal: controller.signal });
      if (controller.signal.aborted) return;
      data._clientMs = performance.now() - started;
      state.search.response = data;
      renderResults(data);
      renderSidePanel(data);
    } catch (err) {
      if (err?.name === "AbortError") return;
      const el = $("#results");
      if (el) el.innerHTML = `<div class="empty">${icon("alert")}<h3>Search failed</h3><p>${esc(describeError(err))}</p></div>`;
      const head = $("#results-head");
      if (head) head.innerHTML = "";
    }
  }

  function confPill(conf, label) {
    return `<span class="conf-pill ${label}" title="Confidence ${fmt.pct(conf)}"><span class="meter"><span style="width:${Math.round(conf * 100)}%"></span></span>${label} ${fmt.pct(conf)}</span>`;
  }

  function foundByChips(ev) {
    const chips = [];
    if (ev.lexical) chips.push(`<span class="chip" title="BM25 ${ev.lexical.bm25}">${icon("text", "sm")} keyword #${ev.lexical.rank}</span>`);
    if (ev.dense) for (const [m, d] of Object.entries(ev.dense)) chips.push(`<span class="chip" title="${esc(m)} cosine ${d.cosine}">${icon("sparkle", "sm")} ${esc(fmt.modelShort(m))} #${d.rank}</span>`);
    if (ev.rerank) chips.push(`<span class="chip accent" title="reranker score ${ev.rerank.score}">${icon("scale", "sm")} reranked ${ev.rerank.rank_before} &rarr; ${ev.rerank.rank_after}</span>`);
    if (ev.agreement) chips.push(`<span class="chip strong">${icon("check", "sm")} both agree</span>`);
    return chips.join("");
  }

  function evidenceHtml(r) {
    const ev = r.evidence;
    if (!ev) return "";
    const parts = ev.confidence_parts || {};
    return `
      <div class="evidence">
        <div class="grid">
          ${ev.lexical ? `<div><h4>Keyword (BM25)</h4><div>rank <b>#${ev.lexical.rank}</b>, score ${ev.lexical.bm25}</div><div class="muted">matched ${(ev.lexical.covered_terms || []).length} of ${ev.lexical.query_terms} query terms</div><div class="terms" style="margin-top:4px">${ev.lexical.matched_terms.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div></div>` : `<div><h4>Keyword (BM25)</h4><div class="muted">not in the keyword candidates</div></div>`}
          ${ev.dense ? Object.entries(ev.dense).map(([m, d]) => `<div><h4>Semantic</h4><div class="truncate" title="${esc(m)}">${esc(m)}</div><div>rank <b>#${d.rank}</b>, cosine ${d.cosine}</div></div>`).join("") : `<div><h4>Semantic</h4><div class="muted">not in the semantic candidates</div></div>`}
          ${ev.rerank ? `<div><h4>Reranker</h4><div>score <b>${ev.rerank.score}</b></div><div class="muted">moved from #${ev.rerank.rank_before} to #${ev.rerank.rank_after}</div></div>` : ""}
          <div><h4>Fusion</h4><div>${ev.fusion ? `${esc(ev.fusion)} score ${ev.fused_score}` : `single retriever, score ${ev.fused_score}`}</div><div class="muted">found by ${ev.found_by.map((f) => esc(f.replace("dense:", ""))).join(", ")}</div></div>
          <div><h4>Confidence</h4><div>${esc(parts.source || "")}</div><div class="muted">${parts.lexical_coverage != null ? `keyword coverage ${parts.lexical_coverage}` : ""}${parts.lexical_coverage != null && parts.dense_similarity != null ? ", " : ""}${parts.dense_similarity != null ? `semantic ${parts.dense_similarity}` : ""}</div></div>
          <div><h4>Location</h4><div>page ${ev.locator.page}, passage ${ev.locator.chunk_index + 1}</div><div class="muted">characters ${ev.locator.char_start} to ${ev.locator.char_end}</div></div>
        </div>
      </div>`;
  }

  function resultHtml(r) {
    const d = r.document;
    const open = state.search.evidenceOpen.has(r.chunk_id) || prefs.showEvidence;
    return `
      <article class="result" data-chunk="${r.chunk_id}" data-doc="${esc(d.id)}" data-page="${r.page}" data-start="${r.char_start}" data-end="${r.char_end}">
        <div class="result-top">
          <span class="result-rank">${r.rank}</span>
          <div class="grow">
            <div class="result-title"><a href="#" data-act="view" title="${esc(d.title || d.filename)}">${esc(d.filename)}</a></div>
            <div class="result-path truncate" title="${esc(d.relpath)}">${esc(d.relpath !== d.filename ? d.relpath : d.title || "")}</div>
          </div>
          ${confPill(r.confidence, r.confidence_label)}
        </div>
        <div class="result-meta">
          <span class="chip outline">${esc(d.ext.toUpperCase())}</span>
          <span class="chip outline">${d.pages > 1 ? `page ${r.page} of ${d.pages}` : `page ${r.page}`}</span>
          ${d.revision > 1 ? `<span class="chip outline" title="This file has been re-indexed ${d.revision - 1} times after a change">${icon("branch", "sm")} rev ${d.revision}</span>` : ""}
          ${foundByChips(r.evidence || {})}
        </div>
        <div class="result-snippet">${highlighted(r.snippet, r.highlights)}</div>
        <div class="result-actions">
          <button class="btn ghost sm" data-act="view">${icon("eye")} Open page</button>
          <a class="btn ghost sm" href="${api.fileUrl(d.id)}" target="_blank" rel="noopener">${icon("external")} Open file</a>
          <button class="btn ghost sm" data-act="why" aria-expanded="${open}">${icon("info")} Why this result</button>
          <button class="btn ghost sm" data-act="copy">${icon("copy")} Copy reference</button>
        </div>
        <div class="evidence-slot" ${open ? "" : "hidden"}>${evidenceHtml(r)}</div>
        ${r.also_matched && r.also_matched.length ? `
          <div class="also">
            <div class="small muted" style="margin-bottom:4px">Also matched in this document</div>
            ${r.also_matched.map((a) => `<div class="also-item"><span class="chip outline clickable" data-act="view-also" data-page="${a.page}" data-start="${a.char_start}" data-end="${a.char_end}">p. ${a.page}</span><span class="txt">${highlighted(a.snippet, a.highlights)}</span></div>`).join("")}
          </div>` : ""}
      </article>`;
  }

  function renderResults(data) {
    const el = $("#results");
    const head = $("#results-head");
    if (!el) return;
    const n = data.results.length;
    head.innerHTML = `
      <span>${n ? `${n} ${data.grouped ? "documents" : "passages"} from ${data.total_candidates} candidates` : "No results"} in ${fmt.ms(data.timings.total)}${data.rerank ? ", reranked" : ""}</span>
      <span class="row">${data.warnings.length ? `<span class="chip likely" title="${esc(data.warnings.join(" | "))}">${icon("alert", "sm")} ${data.warnings.length} note${data.warnings.length > 1 ? "s" : ""}</span>` : ""}<span class="faint mono">${esc(data.request_id)}</span></span>`;
    if (!n) {
      el.innerHTML = `<div class="empty">${icon("search")}<h3>No matches</h3><p>Try fewer or different words, switch to semantic search for paraphrases, or check that the folder has finished indexing.</p></div>`;
      return;
    }
    el.innerHTML = data.results.map(resultHtml).join("");
    el.onclick = (e) => {
      const btn = e.target.closest("[data-act]");
      if (!btn) return;
      e.preventDefault();
      const card = btn.closest(".result");
      const r = data.results.find((x) => String(x.chunk_id) === card.dataset.chunk);
      const act = btn.dataset.act;
      if (act === "view") openViewer(r.document.id, r.page, [r.char_start, r.char_end], queryTerms(data.query, data));
      else if (act === "view-also") openViewer(r.document.id, Number(btn.dataset.page), [Number(btn.dataset.start), Number(btn.dataset.end)], queryTerms(data.query, data));
      else if (act === "why") {
        const slot = card.querySelector(".evidence-slot");
        slot.hidden = !slot.hidden;
        btn.setAttribute("aria-expanded", String(!slot.hidden));
        if (slot.hidden) state.search.evidenceOpen.delete(r.chunk_id); else state.search.evidenceOpen.add(r.chunk_id);
      } else if (act === "copy") {
        const ref = `${r.document.relpath}, page ${r.page}, characters ${r.char_start}-${r.char_end} (${r.confidence_label}, ${fmt.pct(r.confidence)})`;
        navigator.clipboard?.writeText(ref).then(() => toast("Reference copied", "success")).catch(() => toast(ref));
      }
    };
  }

  async function renderSidePanel(data = null) {
    const side = $("#side");
    if (!side) return;
    let models = null;
    try { models = await loadModels(); } catch (e) { models = null; }
    let sources = null;
    try { sources = await loadSources(); } catch (e) { sources = null; }
    const indexed = (models?.indexed || []).filter((m) => m.vectors > 0 || m.building);
    const denseOk = indexed.length > 0;
    const stats = state.stats || (await loadStats().catch(() => null));
    const exts = Object.keys(stats?.documents_by_type || {});
    const timings = data ? Object.entries(data.timings).filter(([k]) => k !== "total") : [];
    const maxMs = Math.max(1, ...timings.map(([, v]) => v));
    const lexical = data?.retrievers?.find((r) => r.kind === "lexical");
    const pq = lexical?.details?.query;

    side.innerHTML = `
      <div class="panel">
        <div class="panel-head">${icon("filter")} Ranking</div>
        <div class="panel-body">
          <div>
            <div class="row-label" style="margin-bottom:6px">Strategy</div>
            <div class="segmented block" id="strategy">
              <button data-v="lexical" class="${prefs.strategy === "lexical" ? "active" : ""}" title="BM25 keyword search">${icon("text", "sm")} Keyword</button>
              <button data-v="dense" class="${prefs.strategy === "dense" ? "active" : ""}" title="Embedding similarity" ${denseOk ? "" : "disabled"}>${icon("sparkle", "sm")} Semantic</button>
              <button data-v="hybrid" class="${prefs.strategy === "hybrid" ? "active" : ""}" title="Both, fused" ${denseOk ? "" : "disabled"}>${icon("layers", "sm")} Hybrid</button>
            </div>
            ${denseOk ? "" : `<div class="hint small muted" style="margin-top:6px">No dense index yet. ${models?.dense_available === false ? "Install the dense extra to enable semantic search." : "Build one from the Activity page."}</div>`}
          </div>
          ${denseOk && prefs.strategy !== "lexical" ? `
          <div>
            <div class="row-label" style="margin-bottom:6px">Embedding models</div>
            <div class="model-list" id="model-list">
              ${indexed.map((m) => `<label class="model-item"><input type="checkbox" value="${esc(m.name)}" ${!prefs.models || prefs.models.includes(m.name) ? "checked" : ""}><span class="name" title="${esc(m.name)}">${esc(m.name)}</span><span class="faint small">${m.dimension}d</span></label>`).join("")}
            </div>
          </div>` : ""}
          ${prefs.strategy === "hybrid" || (prefs.strategy === "dense" && (prefs.models || indexed).length > 1) ? `
          <div>
            <div class="row-label" style="margin-bottom:6px">Fusion</div>
            <div class="segmented block" id="fusion">
              <button data-v="rrf" class="${prefs.fusion !== "weighted" ? "active" : ""}" title="Reciprocal rank fusion">RRF</button>
              <button data-v="weighted" class="${prefs.fusion === "weighted" ? "active" : ""}" title="Weighted score blend">Weighted</button>
            </div>
            ${prefs.fusion === "weighted" ? `<div class="row" style="margin-top:8px"><span class="small muted">keyword</span><input type="range" class="range" id="alpha" min="0" max="1" step="0.05" value="${prefs.alpha}"><span class="small muted">semantic</span></div><div class="small faint" style="text-align:center">alpha ${prefs.alpha}</div>` : ""}
          </div>` : ""}
          <label class="toggle"><input type="checkbox" id="rerank" ${prefs.rerank === null ? (models?.rerank_default ? "checked" : "") : prefs.rerank ? "checked" : ""} ${models?.dense_available ? "" : "disabled"}><span class="track"></span><span class="grow">Rerank with a cross-encoder</span></label>
          <label class="toggle"><input type="checkbox" id="group" ${prefs.group === false ? "" : "checked"}><span class="track"></span><span class="grow">Group passages by document</span></label>
          <div class="row">
            <select class="select" id="f-source" title="Source"><option value="">All sources</option>${(sources || []).map((s) => `<option value="${esc(s.id)}" ${prefs.source === s.id ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select>
            <select class="select" id="f-ext" title="File type"><option value="">All types</option>${exts.map((x) => `<option value="${esc(x)}" ${prefs.ext === x ? "selected" : ""}>${esc(x.toUpperCase())}</option>`).join("")}</select>
          </div>
        </div>
      </div>
      ${data ? `
      <div class="panel">
        <div class="panel-head">${icon("activity")} How this search ran <span class="faint small" style="margin-left:auto">${fmt.ms(data.timings.total)}</span></div>
        <div class="panel-body">
          <div class="timing-bars">
            ${timings.map(([k, v]) => `<div class="timing-bar"><span class="truncate" title="${esc(k)}">${esc(k.replace("dense:", "").split("/").pop())}</span><span class="bar"><span style="width:${Math.max(2, (v / maxMs) * 100)}%"></span></span><span class="ms">${fmt.ms(v)}</span></div>`).join("")}
          </div>
          ${data.retrievers ? `<div class="stack" style="gap:4px">${data.retrievers.map((r) => `<div class="row small"><span class="chip ${r.kind === "lexical" ? "" : "accent"}">${esc(r.kind === "lexical" ? "keyword" : fmt.modelShort(r.details?.model || r.name))}</span><span class="muted">${r.hits} candidates${r.details?.or_fallback ? ", widened with OR" : ""}${r.details?.query_embedding_cached ? ", cached embedding" : ""}</span></div>`).join("")}</div>` : ""}
          ${pq ? `<div><div class="row-label" style="margin-bottom:6px">Query terms</div><div class="terms">${pq.phrases.map((p) => `<span class="chip accent">"${esc(p)}"</span>`).join("")}${pq.terms.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}${pq.dropped.map((t) => `<span class="chip dropped" title="common word, ignored for keyword matching">${esc(t)}</span>`).join("")}</div></div>` : ""}
          ${data.warnings.length ? `<div class="warn-list">${data.warnings.map((w) => `<div class="warn">${icon("alert", "sm")}<span>${esc(w)}</span></div>`).join("")}</div>` : ""}
          <div class="small faint">request <span class="mono">${esc(data.request_id)}</span>, ${fmt.ms(data._clientMs)} round trip</div>
        </div>
      </div>` : ""}`;

    const rerun = () => { savePrefs(); if (state.search.query) runSearch(state.search.query); renderSidePanel(state.search.response); };
    $("#strategy", side).addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || b.disabled) return; prefs.strategy = b.dataset.v; rerun(); });
    $("#fusion", side)?.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; prefs.fusion = b.dataset.v === "rrf" ? "" : "weighted"; rerun(); });
    $("#alpha", side)?.addEventListener("change", (e) => { prefs.alpha = Number(e.target.value); rerun(); });
    $("#model-list", side)?.addEventListener("change", () => {
      const picked = $$("#model-list input:checked", side).map((i) => i.value);
      prefs.models = picked.length === indexed.length ? null : picked;
      if (!picked.length) { toast("Pick at least one model", "error"); prefs.models = null; }
      rerun();
    });
    $("#rerank", side).addEventListener("change", (e) => { prefs.rerank = e.target.checked; rerun(); });
    $("#group", side).addEventListener("change", (e) => { prefs.group = e.target.checked; rerun(); });
    $("#f-source", side).addEventListener("change", (e) => { prefs.source = e.target.value; rerun(); });
    $("#f-ext", side).addEventListener("change", (e) => { prefs.ext = e.target.value; rerun(); });
  }

  // ------------------------------------------------------------------ document viewer

  /** Terms worth highlighting in the viewer: the server's parsed terms when we have a response, else the query words minus common ones. */
  function queryTerms(query, response = null) {
    const parsed = response?.retrievers?.find((r) => r.kind === "lexical")?.details?.query;
    if (parsed) return [...parsed.phrases, ...parsed.terms];
    return (query || "").replace(/"/g, " ").split(/\s+/).filter((t) => t.length >= 3 && !STOP.has(t.toLowerCase()));
  }
  const STOP = new Set("the and for are but not you all any can had her was one our out has his how its may new now old see two way who did get let put say she too use with from that this have will your what when where which while about into over than then them they were been being both each does done just more most much only other some such very".split(" "));

  async function openViewer(docId, page, span = null, terms = []) {
    const m = modal({ title: "Loading...", body: `<div class="skeleton" style="height:240px"></div>` });
    let doc;
    try { doc = await api.get(`/documents/${docId}`); } catch (err) { toast(describeError(err), "error"); closeModal(); return; }
    let current = page || 1;

    const draw = async () => {
      const head = m.querySelector(".modal-head h2");
      head.innerHTML = `${esc(doc.filename)} <span class="faint small" style="font-weight:400">${doc.title && doc.title !== doc.filename ? esc(doc.title) : ""}</span>`;
      const body = m.querySelector(".modal-body");
      body.innerHTML = `<div class="skeleton" style="height:240px"></div>`;
      let pg;
      try { pg = await api.get(`/documents/${docId}/pages/${current}`); } catch (err) { body.innerHTML = `<div class="empty">${esc(describeError(err))}</div>`; return; }
      let html;
      if (span && current === page) {
        const [a, b] = span;
        html = markTerms(esc(pg.text.slice(0, a)), terms) + `<span class="span-target" id="span-target">` + markTerms(esc(pg.text.slice(a, b)), terms) + `</span>` + markTerms(esc(pg.text.slice(b)), terms);
      } else html = markTerms(esc(pg.text), terms);
      body.innerHTML = `
        <div class="viewer">
          <div class="viewer-toolbar">
            <button class="btn sm" id="pg-prev" ${current <= 1 ? "disabled" : ""}>${icon("arrow-left")} Previous</button>
            <span class="small muted">Page</span>
            <select class="select" id="pg-select" style="width:auto">${Array.from({ length: doc.pages }, (_, i) => `<option value="${i + 1}" ${i + 1 === current ? "selected" : ""}>${i + 1}</option>`).join("")}</select>
            <span class="small muted">of ${doc.pages}${pg.ocr ? " (OCR)" : ""}</span>
            <button class="btn sm" id="pg-next" ${current >= doc.pages ? "disabled" : ""}>Next ${icon("arrow-right")}</button>
            <span class="grow"></span>
            <span class="chip outline">${pg.chunks.length} passages</span>
            ${doc.revision > 1 ? `<span class="chip outline">${icon("branch", "sm")} revision ${doc.revision}</span>` : ""}
            ${doc.file_available ? `<a class="btn sm" href="${api.fileUrl(docId)}" target="_blank" rel="noopener">${icon("external")} Open file</a>` : ""}
            <button class="btn ghost sm" id="doc-info">${icon("info")} Details</button>
          </div>
          <div class="viewer-page">${html || `<span class="muted">This page has no text.</span>`}</div>
        </div>`;
      $("#pg-prev", body).addEventListener("click", () => { current -= 1; draw(); });
      $("#pg-next", body).addEventListener("click", () => { current += 1; draw(); });
      $("#pg-select", body).addEventListener("change", (e) => { current = Number(e.target.value); draw(); });
      $("#doc-info", body).addEventListener("click", () => openDocument(docId));
      const target = $("#span-target", body);
      if (target) setTimeout(() => target.scrollIntoView({ block: "center", behavior: "smooth" }), 60);
    };
    await draw();
  }

  function markTerms(escapedText, terms) {
    if (!terms.length) return escapedText;
    const pattern = new RegExp(`(?<![\\w&#;])(${terms.map((t) => esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w{0,4}`, "gi");
    return escapedText.replace(pattern, (m) => `<span class="term">${m}</span>`);
  }

  async function openDocument(docId) {
    let doc;
    try { doc = await api.get(`/documents/${docId}`); } catch (err) { toast(describeError(err), "error"); return; }
    const m = modal({
      title: esc(doc.filename),
      body: `
        <div class="stack" style="gap:16px">
          <dl class="kv">
            <dt>Title</dt><dd>${esc(doc.title || "")}</dd>
            <dt>Source</dt><dd>${esc(doc.source.name)} <span class="faint">(${esc(doc.source.kind)})</span></dd>
            ${doc.path ? `<dt>Path</dt><dd class="mono small">${esc(doc.path)}</dd>` : `<dt>Path</dt><dd class="mono small">${esc(doc.relpath)}</dd>`}
            <dt>Status</dt><dd><span class="status-dot ${esc(doc.status)}"></span>${esc(doc.status)}${doc.error ? ` <span class="muted">${esc(doc.error)}</span>` : ""}</dd>
            <dt>Size</dt><dd>${fmt.bytes(doc.size)}</dd>
            <dt>Pages</dt><dd>${doc.pages}${doc.ocr_pages ? ` (${doc.ocr_pages} through OCR)` : ""}</dd>
            <dt>Passages</dt><dd>${doc.chunks} chunks, ${fmt.int(doc.chars)} characters</dd>
            <dt>Reader</dt><dd>${esc(doc.reader || "")}</dd>
            <dt>Revision</dt><dd>${doc.revision}</dd>
            <dt>First seen</dt><dd>${fmt.date(doc.first_seen_at)}</dd>
            <dt>Indexed</dt><dd>${fmt.date(doc.indexed_at)} <span class="faint">(${fmt.ago(doc.indexed_at)})</span></dd>
            <dt>Content hash</dt><dd class="mono small">${esc(doc.content_hash)}</dd>
          </dl>
          ${doc.revisions.length ? `
          <div>
            <h3 style="margin-bottom:8px">Revision history</h3>
            <table class="table"><thead><tr><th>Rev</th><th>Indexed</th><th class="num">Pages</th><th class="num">Passages</th><th class="num">Size</th><th>Hash</th></tr></thead>
            <tbody>${doc.revisions.map((r) => `<tr><td>${r.revision}</td><td>${fmt.date(r.indexed_at)}</td><td class="num">${r.pages}</td><td class="num">${r.chunks}</td><td class="num">${fmt.bytes(r.size)}</td><td class="mono small">${esc(r.content_hash.slice(0, 12))}</td></tr>`).join("")}</tbody></table>
          </div>` : ""}
          ${doc.page_list.length ? `<div><h3 style="margin-bottom:8px">Pages</h3><div class="row wrap">${doc.page_list.map((p) => `<button class="chip clickable outline" data-page="${p.page}" title="${p.chars} characters${p.ocr ? ", OCR" : ""}">p. ${p.page}</button>`).join("")}</div></div>` : ""}
        </div>`,
      footer: `
        ${doc.source.kind === "uploads" ? `<button class="btn danger" id="doc-delete">${icon("trash")} Delete</button>` : ""}
        <span class="grow"></span>
        ${doc.file_available ? `<a class="btn" href="${api.fileUrl(docId, true)}">${icon("download")} Download</a><a class="btn" href="${api.fileUrl(docId)}" target="_blank" rel="noopener">${icon("external")} Open file</a>` : ""}
        <button class="btn primary" id="doc-open">${icon("eye")} Read</button>`,
    });
    m.querySelector("#doc-open").addEventListener("click", () => openViewer(docId, 1));
    m.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => openViewer(docId, Number(b.dataset.page))));
    m.querySelector("#doc-delete")?.addEventListener("click", async () => {
      if (!(await confirmDialog("Delete document", `Remove "${doc.filename}" from the index and delete the uploaded file?`, "Delete"))) return;
      try { await api.del(`/documents/${docId}`); toast("Document deleted", "success"); closeModal(); state.stats = null; render(); } catch (err) { toast(describeError(err), "error"); }
    });
  }

  // ------------------------------------------------------------------ compare view

  const compareState = { variants: [{ strategy: "lexical" }, { strategy: "hybrid" }], query: "" };

  async function renderCompare(view, params) {
    let models = null;
    try { models = await loadModels(); } catch (e) { models = null; }
    const indexed = (models?.indexed || []).filter((m) => m.vectors > 0);
    compareState.query = params.q || compareState.query;
    const variantCard = (v, i) => `
      <div class="card variant" data-i="${i}">
        <div class="head"><span>Variant ${String.fromCharCode(65 + i)}</span>${compareState.variants.length > 1 ? `<button class="icon-btn sm" data-remove title="Remove">${icon("x")}</button>` : ""}</div>
        <div class="stack">
          <div class="field"><label>Strategy</label><select class="select" data-k="strategy"><option value="lexical" ${v.strategy === "lexical" ? "selected" : ""}>Keyword (BM25)</option><option value="dense" ${v.strategy === "dense" ? "selected" : ""} ${indexed.length ? "" : "disabled"}>Semantic</option><option value="hybrid" ${v.strategy === "hybrid" ? "selected" : ""} ${indexed.length ? "" : "disabled"}>Hybrid</option></select></div>
          ${v.strategy !== "lexical" ? `<div class="field"><label>Model</label><select class="select" data-k="model"><option value="">All indexed</option>${indexed.map((m) => `<option value="${esc(m.name)}" ${v.model === m.name ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select></div>` : ""}
          ${v.strategy === "hybrid" ? `<div class="field"><label>Fusion</label><select class="select" data-k="fusion"><option value="rrf" ${v.fusion !== "weighted" ? "selected" : ""}>RRF</option><option value="weighted" ${v.fusion === "weighted" ? "selected" : ""}>Weighted</option></select></div>` : ""}
          <label class="toggle"><input type="checkbox" data-k="rerank" ${v.rerank ? "checked" : ""} ${models?.dense_available ? "" : "disabled"}><span class="track"></span><span>Rerank</span></label>
        </div>
      </div>`;
    view.innerHTML = `
      <div class="page-head"><div><h1>Compare strategies</h1><p>Run one query several ways and see where the rankings agree. Documents every variant returns are outlined in green.</p></div></div>
      <div class="search-top">${searchBoxHtml(compareState.query, true)}<button class="btn primary" id="cmp-run">${icon("play")} Run</button></div>
      <div class="compare-controls" id="cmp-controls">${compareState.variants.map(variantCard).join("")}${compareState.variants.length < 4 ? `<button class="btn ghost" id="cmp-add" style="align-self:center;justify-self:start">${icon("plus")} Add variant</button>` : ""}</div>
      <div id="cmp-results"></div>`;
    const input = $("#q", view);
    input.placeholder = "Query to compare";
    $("#q-clear", view).addEventListener("click", () => { input.value = ""; });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
    $("#cmp-run").addEventListener("click", run);
    $("#cmp-add")?.addEventListener("click", () => { compareState.variants.push({ strategy: indexed.length ? "hybrid" : "lexical", rerank: true }); renderCompare(view, { q: input.value }); });
    $("#cmp-controls").addEventListener("change", (e) => {
      const card = e.target.closest(".variant"); if (!card) return;
      const v = compareState.variants[Number(card.dataset.i)];
      const k = e.target.dataset.k;
      if (k === "rerank") v.rerank = e.target.checked; else v[k] = e.target.value;
      renderCompare(view, { q: input.value });
    });
    $("#cmp-controls").addEventListener("click", (e) => {
      const b = e.target.closest("[data-remove]"); if (!b) return;
      compareState.variants.splice(Number(b.closest(".variant").dataset.i), 1);
      renderCompare(view, { q: input.value });
    });
    if (compareState.query && params.q) run();

    async function run() {
      const q = input.value.trim();
      if (!q) { input.focus(); return; }
      compareState.query = q;
      history.replaceState(null, "", `#/compare?q=${encodeURIComponent(q)}`);
      const out = $("#cmp-results");
      out.innerHTML = `<div class="compare-grid">${compareState.variants.map(() => `<div class="skeleton" style="height:200px"></div>`).join("")}</div>`;
      const variants = compareState.variants.map((v) => ({ strategy: v.strategy, models: v.model ? [v.model] : null, fusion: v.fusion || "", rerank: !!v.rerank, top_k: prefs.topK }));
      let data;
      try { data = await api.post("/search/compare", { query: q, variants }); } catch (err) { out.innerHTML = `<div class="empty">${icon("alert")}<h3>Compare failed</h3><p>${esc(describeError(err))}</p></div>`; return; }
      const common = new Set(data.common_documents);
      out.innerHTML = `<div class="compare-grid">${data.variants.map((v, i) => `
        <div class="compare-col">
          <div class="col-head"><h3>${String.fromCharCode(65 + i)}. ${esc(v.label)}</h3><span class="small muted">${fmt.ms(v.timings.total)}</span></div>
          ${v.warnings.length ? `<div class="warn" style="margin-bottom:8px">${icon("alert", "sm")}<span>${esc(v.warnings.join(" "))}</span></div>` : ""}
          ${v.results.length ? v.results.map((r) => `
            <div class="mini ${common.has(r.document.id) ? "common" : ""}" data-doc="${esc(r.document.id)}" data-page="${r.page}" data-start="${r.char_start}" data-end="${r.char_end}">
              <div class="t"><span class="result-rank">${r.rank}</span><span class="truncate grow" title="${esc(r.document.relpath)}">${esc(r.document.filename)}</span>${common.has(r.document.id) ? `<span class="chip strong" title="returned by every variant">${icon("check", "sm")}</span>` : ""}</div>
              <div class="row small" style="margin-top:4px">${confPill(r.confidence, r.confidence_label)}<span class="chip outline">p. ${r.page}</span></div>
              <div class="s">${highlighted(r.snippet.slice(0, 220), (r.highlights || []).filter(([a, b]) => b <= 220))}${r.snippet.length > 220 ? " ..." : ""}</div>
            </div>`).join("") : `<div class="mini muted">No results</div>`}
        </div>`).join("")}</div>`;
      out.onclick = (e) => {
        const card = e.target.closest(".mini[data-doc]"); if (!card) return;
        openViewer(card.dataset.doc, Number(card.dataset.page), [Number(card.dataset.start), Number(card.dataset.end)], queryTerms(q, data.variants[0]));
      };
    }
  }

  // ------------------------------------------------------------------ sources view

  async function renderSources(view) {
    const [sources, jobs] = await Promise.all([loadSources(true), api.get("/jobs", { limit: 30 })]);
    const activeByKey = {};
    for (const j of jobs.jobs) if (!["done", "failed", "cancelled"].includes(j.status)) activeByKey[j.meta?.key] = j;
    const watched = new Set((await api.get("/sources")).watched);
    const card = (s) => {
      const job = activeByKey[s.id];
      const scan = s.last_scan || {};
      const isUploads = s.kind === "uploads";
      return `
        <div class="card source-card" data-id="${esc(s.id)}">
          <div class="card-body">
            <div class="row spread">
              <div class="row" style="min-width:0">${icon(isUploads ? "upload" : "folder")}<h3 class="truncate" title="${esc(s.name)}">${esc(s.name)}</h3></div>
              <div class="row">${!s.enabled ? `<span class="chip likely">paused</span>` : watched.has(s.id) ? `<span class="chip strong" title="File changes are picked up within seconds">${icon("eye", "sm")} watching</span>` : isUploads ? `<span class="chip">uploads</span>` : `<span class="chip" title="Scanned on the configured interval">periodic</span>`}</div>
            </div>
            <div class="path" style="margin-top:6px">${esc(s.path)}</div>
            <div class="source-stats"><div><b>${fmt.int(s.document_count)}</b><span>documents</span></div><div><b>${fmt.int(s.chunk_count)}</b><span>passages</span></div><div><b>${s.last_scan_at ? fmt.ago(s.last_scan_at) : "never"}</b><span>last scan</span></div></div>
            ${job ? `<div class="small muted">${job.status === "queued" ? "Queued" : `Scanning: ${esc(job.progress?.current || "")}`}</div><div class="progress ${job.progress?.total ? "" : "indeterminate"}" style="margin:6px 0 10px"><span style="width:${job.progress?.total ? Math.round((job.progress.done / job.progress.total) * 100) : 40}%"></span></div><div class="scan-summary small">${["added", "updated", "removed", "failed"].filter((k) => job.progress?.[k]).map((k) => `<span class="chip ${k === "failed" ? "danger" : ""}">${job.progress[k]} ${k}</span>`).join("")}</div>` :
              scan.seen != null ? `<div class="scan-summary">${[["added", "added"], ["updated", "updated"], ["removed", "removed"], ["unchanged", "unchanged"], ["failed", "failed"], ["too_large", "too large"]].filter(([k]) => scan[k]).map(([k, label]) => `<span class="chip ${k === "failed" ? "danger" : ""}">${scan[k]} ${label}</span>`).join("")}<span class="chip outline">${scan.seconds}s</span>${scan.root_missing ? `<span class="chip danger">folder missing</span>` : ""}</div>` : `<div class="small muted">Not scanned yet</div>`}
            <div class="row wrap" style="margin-top:12px">
              <button class="btn sm" data-act="scan" ${job ? "disabled" : ""}>${icon("refresh")} Scan now</button>
              ${!isUploads ? `<button class="btn sm" data-act="toggle">${s.enabled ? `${icon("pause")} Pause` : `${icon("play")} Resume`}</button>` : `<button class="btn sm" data-act="upload">${icon("upload")} Upload files</button>`}
              <a class="btn sm ghost" href="#/documents?source=${esc(s.id)}">${icon("file")} Documents</a>
              <button class="btn sm ghost" data-act="events">${icon("activity")} Events</button>
              ${!isUploads ? `<button class="btn sm ghost danger" data-act="remove" style="margin-left:auto">${icon("trash")} Remove</button>` : ""}
            </div>
          </div>
        </div>`;
    };
    view.innerHTML = `
      <div class="page-head"><div><h1>Sources</h1><p>Folders are scanned, watched for changes and re-scanned on a schedule. Edited files get a new revision, deleted files leave the index.</p></div><div class="row"><button class="btn" id="src-upload">${icon("upload")} Upload files</button><button class="btn primary" id="src-add">${icon("plus")} Add folder</button></div></div>
      <div class="sources-grid" id="sources">${sources.map(card).join("")}</div>`;
    $("#src-add").addEventListener("click", () => addFolderDialog());
    $("#src-upload").addEventListener("click", () => uploadDialog());
    $("#sources").addEventListener("click", async (e) => {
      const btn = e.target.closest("[data-act]"); if (!btn) return;
      const id = btn.closest(".source-card").dataset.id;
      const s = sources.find((x) => x.id === id);
      try {
        if (btn.dataset.act === "scan") { await api.post(`/sources/${id}/sync`); toast(`Scanning ${s.name}`); pollJobs(); setTimeout(() => render(), 400); }
        else if (btn.dataset.act === "toggle") { await api.patch(`/sources/${id}`, { enabled: !s.enabled }); toast(s.enabled ? "Source paused" : "Source resumed", "success"); render(); }
        else if (btn.dataset.act === "upload") uploadDialog();
        else if (btn.dataset.act === "events") {
          const ev = (await api.get(`/sources/${id}/events`, { limit: 200 })).events;
          modal({ title: `Events: ${esc(s.name)}`, body: eventsHtml(ev) });
        } else if (btn.dataset.act === "remove") {
          if (!(await confirmDialog("Remove source", `Remove "${s.name}" and everything indexed from it? Files on disk are not touched.`, "Remove"))) return;
          await api.del(`/sources/${id}`); toast("Source removed", "success"); state.stats = null; render();
        }
      } catch (err) { toast(describeError(err), "error"); }
    });
  }

  function addFolderDialog() {
    const m = modal({
      title: "Add a folder",
      narrow: true,
      body: `
        <div class="stack" style="gap:14px">
          <div class="field"><label for="af-path">Folder path on the server</label><input class="input" id="af-path" placeholder="/srv/docs or C:\\Users\\me\\Documents" autofocus><span class="hint">The path is resolved on the machine running contextsearch, not in your browser. In production it must sit inside CSE_ALLOWED_ROOTS.</span></div>
          <div class="field"><label for="af-name">Name (optional)</label><input class="input" id="af-name" placeholder="Defaults to the folder name"></div>
          <label class="check"><input type="checkbox" id="af-recursive" checked> Include subfolders</label>
          <details class="disclosure"><summary>${icon("chevron", "sm chev")} Advanced filters</summary>
            <div class="stack" style="margin-top:10px">
              <div class="field"><label for="af-include">Only these extensions</label><input class="input" id="af-include" placeholder="pdf,docx (empty uses the server default)"></div>
              <div class="field"><label for="af-exclude">Extra exclude patterns</label><input class="input" id="af-exclude" placeholder="**/archive/**,**/*.draft.*"></div>
            </div>
          </details>
        </div>`,
      footer: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="af-ok">${icon("folder")} Add and scan</button>`,
    });
    const submit = async () => {
      const path = $("#af-path").value.trim();
      if (!path) { $("#af-path").focus(); return; }
      const btn = $("#af-ok"); btn.disabled = true;
      try {
        await api.post("/sources", { path, name: $("#af-name").value.trim(), recursive: $("#af-recursive").checked, include: $("#af-include").value.trim(), exclude: $("#af-exclude").value.trim() });
        toast("Folder added, scanning started", "success"); closeModal(); state.sources = null; pollJobs(); render();
      } catch (err) { toast(describeError(err), "error", 6000); btn.disabled = false; }
    };
    $("#af-ok").addEventListener("click", submit);
    m.querySelectorAll("input").forEach((i) => i.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); }));
  }

  function uploadDialog() {
    let files = [];
    const m = modal({
      title: "Upload files",
      narrow: true,
      body: `
        <div class="dropzone" id="dz" tabindex="0" role="button" aria-label="Choose files">${icon("upload")}<div><b>Drop files here</b> or click to choose</div><div class="small" style="margin-top:4px">PDF, Word, PowerPoint, text, Markdown, HTML. Files land in the Uploads source and are searchable as soon as indexing finishes.</div><input type="file" id="dz-input" multiple hidden></div>
        <div class="file-list" id="dz-list"></div>
        <div class="progress" id="dz-progress" hidden style="margin-top:12px"><span style="width:0"></span></div>
        <div id="dz-result" class="stack" style="margin-top:12px"></div>`,
      footer: `<button class="btn" data-close>Close</button><button class="btn primary" id="dz-go" disabled>${icon("upload")} Upload and index</button>`,
    });
    const dz = $("#dz", m); const input = $("#dz-input", m); const list = $("#dz-list", m); const go = $("#dz-go", m);
    const show = () => { list.innerHTML = files.map((f, i) => `<div class="file-row">${icon("file", "sm")}<span class="grow truncate">${esc(f.name)}</span><span class="faint">${fmt.bytes(f.size)}</span><button class="icon-btn sm" data-i="${i}" title="Remove">${icon("x", "sm")}</button></div>`).join(""); go.disabled = !files.length; };
    const add = (fl) => { files = files.concat(Array.from(fl)); show(); };
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
    input.addEventListener("change", () => add(input.files));
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("over"));
    dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); add(e.dataTransfer.files); });
    list.addEventListener("click", (e) => { const b = e.target.closest("[data-i]"); if (!b) return; files.splice(Number(b.dataset.i), 1); show(); });
    go.addEventListener("click", async () => {
      go.disabled = true;
      const bar = $("#dz-progress", m); bar.hidden = false;
      try {
        const res = await api.upload(files, (p) => { bar.querySelector("span").style.width = `${Math.round(p * 100)}%`; });
        bar.hidden = true;
        $("#dz-result", m).innerHTML = `${res.documents.map((d) => `<div class="row small"><span class="chip strong">${icon("check", "sm")} indexed</span><span class="truncate grow">${esc(d.filename)}</span><span class="faint">${d.pages} pages, ${d.chunks} passages</span></div>`).join("")}${res.errors.map((e) => `<div class="row small"><span class="chip danger">failed</span><span class="grow">${esc(e.file)}: ${esc(e.error)}</span></div>`).join("")}`;
        files = []; show();
        state.stats = null; state.sources = null;
        if (res.documents.length) toast(`${res.documents.length} file${res.documents.length > 1 ? "s" : ""} indexed`, "success");
        if (state.viewRoute !== "search") render();
      } catch (err) { bar.hidden = true; toast(describeError(err), "error", 6000); go.disabled = false; }
    });
  }

  // ------------------------------------------------------------------ documents view

  async function renderDocuments(view, params) {
    const sources = await loadSources();
    const q = params.q || ""; const source = params.source || ""; const status = params.status || "indexed"; const ext = params.ext || "";
    const sort = params.sort || "indexed_at"; const order = params.order || "desc"; const offset = Number(params.offset || 0); const limit = 50;
    view.innerHTML = `
      <div class="page-head"><div><h1>Documents</h1><p>Everything the index knows about, with revision history and a page-by-page reader.</p></div><button class="btn primary" id="docs-upload">${icon("upload")} Upload files</button></div>
      <div class="filters">
        <input class="input" id="df-q" placeholder="Filter by name, title or path" value="${esc(q)}">
        <select class="select" id="df-source"><option value="">All sources</option>${sources.map((s) => `<option value="${esc(s.id)}" ${source === s.id ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select>
        <select class="select" id="df-status">${["indexed", "failed", "removed", "pending", "all"].map((s) => `<option value="${s}" ${status === s ? "selected" : ""}>${s}</option>`).join("")}</select>
        <select class="select" id="df-sort">${[["indexed_at", "Recently indexed"], ["updated_at", "Recently changed"], ["filename", "Name"], ["size", "Size"], ["pages", "Pages"], ["chunks", "Passages"], ["revision", "Revisions"]].map(([v, l]) => `<option value="${v}" ${sort === v ? "selected" : ""}>${l}</option>`).join("")}</select>
        <button class="btn ghost sm" id="df-order" title="Toggle order">${order === "desc" ? "Descending" : "Ascending"}</button>
      </div>
      <div class="card"><div id="docs-table"><div class="skeleton" style="height:200px;margin:12px"></div></div></div>`;
    const apply = (patch) => navigate("documents", { q: $("#df-q").value.trim(), source: $("#df-source").value, status: $("#df-status").value, ext, sort: $("#df-sort").value, order, ...patch }, true);
    $("#df-q").addEventListener("keydown", (e) => { if (e.key === "Enter") apply({ offset: 0 }); });
    $("#df-q").addEventListener("input", debounce(() => apply({ offset: 0 }), 400));
    $("#df-source").addEventListener("change", () => apply({ offset: 0 }));
    $("#df-status").addEventListener("change", () => apply({ offset: 0 }));
    $("#df-sort").addEventListener("change", () => apply({ offset: 0 }));
    $("#df-order").addEventListener("click", () => apply({ order: order === "desc" ? "asc" : "desc" }));
    $("#docs-upload").addEventListener("click", uploadDialog);
    if (params.upload) { history.replaceState(null, "", "#/documents"); uploadDialog(); }

    const data = await api.get("/documents", { q, source, status, ext, sort, order, limit, offset });
    const box = $("#docs-table");
    if (!data.documents.length) {
      box.innerHTML = `<div class="empty">${icon("file")}<h3>No documents${status !== "all" ? ` with status "${esc(status)}"` : ""}</h3><p>${q ? "Try a different filter." : "Add a folder or upload files to get started."}</p></div>`;
      return;
    }
    box.innerHTML = `
      <table class="table"><thead><tr><th>Name</th><th>Source</th><th>Type</th><th class="num">Pages</th><th class="num">Passages</th><th class="num">Rev</th><th>Indexed</th><th>Status</th></tr></thead>
      <tbody>${data.documents.map((d) => `<tr class="clickable" data-id="${esc(d.id)}"><td><div class="truncate" style="max-width:360px" title="${esc(d.relpath)}"><b>${esc(d.filename)}</b></div><div class="small faint truncate" style="max-width:360px">${esc(d.title && d.title !== d.filename ? d.title : d.relpath !== d.filename ? d.relpath : "")}</div></td><td>${esc(d.source_name)}</td><td><span class="chip outline">${esc(d.ext.toUpperCase())}</span></td><td class="num">${d.pages}</td><td class="num">${d.chunks}</td><td class="num">${d.revision}</td><td class="nowrap" title="${fmt.date(d.indexed_at)}">${fmt.ago(d.indexed_at)}</td><td><span class="status-dot ${esc(d.status)}"></span>${esc(d.status)}${d.error ? ` <span class="faint" title="${esc(d.error)}">${icon("info", "sm")}</span>` : ""}${d.metadata?.empty ? ` <span class="chip weak">no text</span>` : ""}</td></tr>`).join("")}</tbody></table>
      <div class="pager"><span>${offset + 1} to ${Math.min(offset + limit, data.total)} of ${fmt.int(data.total)}</span><span class="row"><button class="btn sm" id="pg-prev" ${offset === 0 ? "disabled" : ""}>${icon("arrow-left")} Previous</button><button class="btn sm" id="pg-next" ${offset + limit >= data.total ? "disabled" : ""}>Next ${icon("arrow-right")}</button></span></div>`;
    box.querySelector("tbody").addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) openDocument(tr.dataset.id); });
    $("#pg-prev", box).addEventListener("click", () => apply({ offset: Math.max(0, offset - limit) }));
    $("#pg-next", box).addEventListener("click", () => apply({ offset: offset + limit }));
  }

  // ------------------------------------------------------------------ activity view

  function eventsHtml(events) {
    if (!events.length) return `<div class="empty"><p>No events yet.</p></div>`;
    const kindChip = { added: "strong", updated: "accent", removed: "weak", failed: "danger", scan: "", model_built: "accent", model_added: "accent", source_added: "", source_removed: "weak", deleted: "weak" };
    return `<div class="event-list">${events.map((e) => {
      const d = e.detail || {};
      let text = e.path ? e.path.split(/[\\/]/).pop() : "";
      if (e.kind === "scan") text = `${e.path}: ${d.added || 0} added, ${d.updated || 0} updated, ${d.removed || 0} removed, ${d.unchanged || 0} unchanged${d.failed ? `, ${d.failed} failed` : ""}${d.root_missing ? " (folder missing)" : ""}`;
      else if (e.kind === "updated") text += ` (revision ${d.revision})`;
      else if (e.kind === "failed") text += `: ${d.error || ""}`;
      else if (e.kind.startsWith("model")) text = `${d.model}${d.vectors != null ? ` (${fmt.int(d.vectors)} vectors)` : ""}`;
      return `<div class="event"><span class="when" title="${fmt.date(e.ts)}">${fmt.ago(e.ts)}</span><span class="kind"><span class="chip ${kindChip[e.kind] || ""}">${esc(e.kind.replace("_", " "))}</span></span><span class="truncate" title="${esc(e.path || "")}">${esc(text)}</span></div>`;
    }).join("")}</div>`;
  }

  async function renderActivity(view) {
    const [stats, models, jobs, recent, events, health] = await Promise.all([
      loadStats(), loadModels(true), api.get("/jobs", { limit: 20 }), api.get("/search/recent", { limit: 25 }), api.get("/system/events", { limit: 40 }), api.get("/health"),
    ]);
    state.health = health;
    const jobHtml = (j) => `
      <div class="job">
        <div class="row spread"><span class="row"><span class="chip ${j.status === "running" ? "accent" : j.status === "failed" ? "danger" : j.status === "done" ? "strong" : ""}">${esc(j.status)}</span><b class="truncate" style="max-width:420px">${esc(j.label)}</b></span><span class="small muted nowrap">${j.duration_s != null ? `${j.duration_s}s` : fmt.ago(j.created_at)}${["running", "queued"].includes(j.status) ? ` <button class="btn ghost sm" data-cancel="${j.id}">Cancel</button>` : ""}</span></div>
        ${j.status === "running" && j.progress?.total ? `<div class="progress"><span style="width:${Math.round((j.progress.done / j.progress.total) * 100)}%"></span></div><div class="small muted" style="margin-top:4px">${j.progress.done} of ${j.progress.total}${j.progress.current ? `: ${esc(j.progress.current)}` : ""}</div>` : j.status === "running" ? `<div class="progress indeterminate"><span></span></div>` : ""}
        ${j.status === "done" && j.result && Object.keys(j.result).length ? `<div class="small muted" style="margin-top:4px">${esc(Object.entries(j.result).filter(([k, v]) => typeof v !== "object" && ["added", "updated", "removed", "unchanged", "failed", "vectors", "outcome", "seconds"].includes(k)).map(([k, v]) => `${k} ${v}`).join(", "))}</div>` : ""}
        ${j.error_count ? `<div class="small" style="margin-top:4px;color:var(--danger)">${j.error_count} error${j.error_count > 1 ? "s" : ""}: ${esc(j.errors[0]?.error || "")}</div>` : ""}
      </div>`;
    view.innerHTML = `
      <div class="page-head"><div><h1>Activity</h1><p>What the engine is doing, what it has, and how searches have been performing.</p></div><button class="btn" id="act-refresh">${icon("refresh")} Refresh</button></div>
      <div class="kpis" style="margin-bottom:16px">
        <div class="kpi"><div class="kpi-value">${fmt.int(stats.documents)}</div><div class="kpi-label">documents${stats.failed_documents ? ` <span style="color:var(--danger)">(${stats.failed_documents} failed)</span>` : ""}</div></div>
        <div class="kpi"><div class="kpi-value">${fmt.int(stats.chunks)}</div><div class="kpi-label">passages on ${fmt.int(stats.pages)} pages</div></div>
        <div class="kpi"><div class="kpi-value">${fmt.bytes(stats.database_bytes)}</div><div class="kpi-label">database on disk</div></div>
        <div class="kpi"><div class="kpi-value">${stats.memory_mb} MB</div><div class="kpi-label">vectors in memory</div></div>
        <div class="kpi"><div class="kpi-value">${fmt.int(stats.queries.count)}</div><div class="kpi-label">searches, avg ${fmt.ms(stats.queries.avg_ms)}</div></div>
      </div>
      <div class="two-col">
        <div class="stack" style="gap:16px">
          <div class="card">
            <div class="card-head">${icon("cpu")}<h3 class="grow">Embedding models</h3><span class="chip ${models.dense_available ? "strong" : "weak"}">${models.dense_available ? `dense on ${esc(models.device || "cpu")}` : "dense extra not installed"}</span></div>
            <div class="card-body stack" style="gap:12px">
              ${models.indexed.length ? `<table class="table"><thead><tr><th>Model</th><th class="num">Dim</th><th class="num">Vectors</th><th>State</th><th></th></tr></thead><tbody>${models.indexed.map((m) => `<tr><td style="max-width:280px"><div class="truncate" title="${esc(m.name)}">${esc(m.name)}</div>${m.embedder?.note ? `<div class="small faint">${esc(m.embedder.note)}</div>` : ""}</td><td class="num">${m.dimension}</td><td class="num">${fmt.int(m.vectors)}</td><td>${m.building ? `<span class="chip accent">building</span>` : m.missing_vectors ? `<span class="chip likely" title="chunks without a vector for this model">${fmt.int(m.missing_vectors)} missing</span>` : `<span class="chip strong">complete</span>`}${m.embedder?.loaded ? ` <span class="chip outline">${esc(m.embedder.device || "")}</span>` : ""}</td><td class="num nowrap">${m.missing_vectors && !m.building ? `<button class="btn sm" data-build="${esc(m.name)}">Fill in</button> ` : ""}<button class="btn sm ghost danger" data-drop="${esc(m.name)}" title="Drop this dense index">${icon("trash", "sm")}</button></td></tr>`).join("")}</tbody></table>` : `<p class="muted small">No dense index yet. Keyword search works without one; build a dense index below to enable semantic and hybrid search.</p>`}
              <details class="disclosure"><summary>${icon("chevron", "sm chev")} Build another model</summary>
                <div class="stack" style="margin-top:10px">
                  ${models.presets.filter((p) => !p.indexed).map((p) => `<div class="row"><div class="grow"><div class="truncate" title="${esc(p.name)}"><b>${esc(fmt.modelShort(p.name))}</b> <span class="faint small">${p.dimension}d, ${p.size_mb} MB${p.multilingual ? ", multilingual" : ""}</span></div><div class="small muted">${esc(p.note)}</div></div><button class="btn sm" data-build="${esc(p.name)}" ${models.dense_available ? "" : "disabled"}>Build</button></div>`).join("")}
                  <div class="row"><input class="input grow" id="custom-model" placeholder="Any sentence-transformers model id"><button class="btn sm" id="custom-build" ${models.dense_available ? "" : "disabled"}>Build</button></div>
                  <p class="small faint">Building embeds every passage once with the new model. On a CPU that is roughly 100 to 300 passages per second for small models.</p>
                </div>
              </details>
              <div class="row small muted">${icon("scale", "sm")} Reranker: <span class="mono">${esc(models.rerank_model)}</span>${models.rerank_default ? " (on by default)" : " (off by default, toggle per search)"}</div>
              <div class="row"><button class="btn sm" id="reindex" title="Re-chunk documents indexed with older chunking settings">${icon("refresh")} Reindex changed settings</button><span class="small faint">signature ${esc(health.version ? "" : "")}<span class="mono">${esc((await api.get("/system/config")).chunking_signature)}</span></span></div>
            </div>
          </div>
          <div class="card">
            <div class="card-head">${icon("clock")}<h3 class="grow">Jobs</h3><span class="small muted">${jobs.queued} queued</span></div>
            <div class="card-body" id="jobs">${jobs.jobs.length ? jobs.jobs.map(jobHtml).join("") : `<p class="muted small">No jobs yet.</p>`}</div>
          </div>
        </div>
        <div class="stack" style="gap:16px">
          <div class="card">
            <div class="card-head">${icon("search")}<h3 class="grow">Recent searches</h3><span class="small muted">max ${fmt.ms(recent.stats.max_ms)}</span></div>
            <div class="card-body scroll-x" style="padding:0">${recent.queries.length ? `<table class="table"><thead><tr><th>Query</th><th>Strategy</th><th class="num">Results</th><th class="num">Latency</th><th>When</th></tr></thead><tbody>${recent.queries.map((r) => `<tr class="clickable" data-q="${esc(r.query)}"><td class="truncate" style="max-width:260px">${esc(r.query)}</td><td><span class="chip">${esc(r.strategy)}${r.rerank ? " + rerank" : ""}</span></td><td class="num">${r.results}</td><td class="num">${fmt.ms(r.latency_ms)}</td><td class="nowrap">${fmt.ago(r.ts)}</td></tr>`).join("")}</tbody></table>` : `<p class="muted small" style="padding:16px">No searches recorded yet.</p>`}</div>
          </div>
          <div class="card">
            <div class="card-head">${icon("activity")}<h3 class="grow">Index events</h3></div>
            <div class="card-body" style="padding:4px 18px">${eventsHtml(events.events)}</div>
          </div>
          <div class="card">
            <div class="card-head">${icon("shield")}<h3 class="grow">Health</h3><span class="chip strong">${esc(health.status)}</span></div>
            <div class="card-body"><dl class="kv">
              <dt>Version</dt><dd>${esc(health.version)} (${esc(health.profile)})</dd>
              <dt>Uptime</dt><dd>${Math.round(health.uptime_s / 60)} min</dd>
              <dt>Watcher</dt><dd>${health.watcher.enabled ? `${health.watcher.running ? "running" : "stopped"}, ${health.watcher.watched.length} folder${health.watcher.watched.length === 1 ? "" : "s"}` : "disabled"}</dd>
              <dt>OCR</dt><dd>${health.ocr.enabled ? (health.ocr.available ? "on" : "on, but the ocr extra is missing") : "off"}</dd>
              <dt>Readers</dt><dd>${health.readers.map((r) => `<span class="chip outline" style="margin:1px">${esc(r)}</span>`).join(" ")}</dd>
              <dt>Database</dt><dd class="mono small">${esc(health.database.path)}</dd>
            </dl></div>
          </div>
        </div>
      </div>`;
    $("#act-refresh").addEventListener("click", () => render());
    view.addEventListener("click", async (e) => {
      const build = e.target.closest("[data-build]"); const drop = e.target.closest("[data-drop]"); const cancel = e.target.closest("[data-cancel]"); const row = e.target.closest("tr[data-q]");
      try {
        if (build) { await api.post("/models", { name: build.dataset.build }); toast(`Building dense index for ${fmt.modelShort(build.dataset.build)}`, "success"); pollJobs(); setTimeout(render, 500); }
        else if (drop) { if (await confirmDialog("Drop dense index", `Drop the vectors for ${drop.dataset.drop}? The model can be rebuilt later.`, "Drop")) { await api.del(`/models/${drop.dataset.drop}`); toast("Index dropped", "success"); render(); } }
        else if (cancel) { await api.post(`/jobs/${cancel.dataset.cancel}/cancel`); toast("Cancel requested"); setTimeout(render, 400); }
        else if (row) navigate("search", { q: row.dataset.q });
      } catch (err) { toast(describeError(err), "error", 6000); }
    });
    $("#custom-build").addEventListener("click", async () => {
      const name = $("#custom-model").value.trim(); if (!name) return;
      try { await api.post("/models", { name }); toast(`Building dense index for ${name}`, "success"); pollJobs(); setTimeout(render, 500); } catch (err) { toast(describeError(err), "error", 6000); }
    });
    $("#reindex").addEventListener("click", async () => {
      if (!(await confirmDialog("Reindex", "Re-chunk every document whose chunks were cut with different settings than the current ones. Documents already up to date are skipped.", "Reindex", false))) return;
      try { await api.post("/reindex", { force: false }); toast("Reindex started", "success"); pollJobs(); setTimeout(render, 500); } catch (err) { toast(describeError(err), "error"); }
    });
  }

  // ------------------------------------------------------------------ boot

  applyTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark" || (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    prefs.theme = dark ? "light" : "dark"; savePrefs(); applyTheme();
  });
  $("#settings-open").addEventListener("click", () => openSettings());
  $("#drawer-backdrop").addEventListener("click", closeSettings);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeModal(); closeSettings(); }
    if (e.key === "/" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      const q = $("#q"); if (q) { e.preventDefault(); q.focus(); q.select(); } else navigate("search");
    }
  });
  document.addEventListener("jobs-idle", () => { if (["sources", "activity"].includes(state.viewRoute)) render(); if (state.viewRoute === "search" && state.search.response) renderSidePanel(state.search.response); });
  document.addEventListener("jobs", () => { if (state.viewRoute === "sources" && state.jobsActive) render(); });
  window.addEventListener("hashchange", render);
  api.get("/health").then((h) => { state.health = h; }).catch(() => {});
  render();
  pollJobs();
})();
