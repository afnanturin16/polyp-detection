// ============================================================
// app.js — talks to the FastAPI backend via fetch(). No build step.
// ============================================================

const API = ""; // same-origin; set to e.g. "http://127.0.0.1:8000" if you
                 // run the frontend from a separate dev server.

const CLASSES = ["NORM", "HP", "TA.LG", "TA.HG", "TVA.LG", "TVA.HG"];

// ---------------- state ----------------
let currentImage = null;      // data URL, Classify tab
let compareImage = null;      // data URL, Compare tab
let selectedModel = "cascade";
let history = JSON.parse(sessionStorage.getItem("polyp_history") || "[]");

// ============================================================
// View routing
// ============================================================
function goto(view) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("is-active"));
  document.getElementById(`view-${view}`).classList.add("is-active");
  document.querySelectorAll(".rail-link").forEach(l => l.classList.remove("is-active"));
  document.querySelector(`.rail-link[data-view="${view}"]`)?.classList.add("is-active");
  if (view === "history") renderHistory();
}

document.querySelectorAll(".rail-link").forEach(btn => {
  btn.addEventListener("click", () => goto(btn.dataset.view));
});
document.querySelectorAll("[data-goto]").forEach(btn => {
  btn.addEventListener("click", () => goto(btn.dataset.goto));
});

// ============================================================
// File -> data URL helper
// ============================================================
function fileToDataURL(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.readAsDataURL(file);
  });
}

// ============================================================
// Input sanity check — runs entirely in the browser, before any
// request is sent to the backend. This is a cheap heuristic, not a
// model: it flags images that are very unlikely to be H&E tissue
// patches (wrong shape, or a colour profile with little pink/purple
// in it), but it never blocks the user — it's a warning, not a gate.
// ============================================================
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  return [h, s, max];
}

function checkImageSanity(dataUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const warnings = [];
      const ratio = Math.max(img.width, img.height) / Math.max(1, Math.min(img.width, img.height));
      if (ratio > 2.5) {
        warnings.push("Unusual aspect ratio for a tissue patch — UniToPatho patches are roughly square.");
      }
      if (Math.min(img.width, img.height) < 50) {
        warnings.push("Image is very small — check it's a tissue patch, not a thumbnail or icon.");
      }

      // Downsample to a small canvas for a fast colour-profile check.
      const SZ = 48;
      const canvas = document.createElement("canvas");
      canvas.width = SZ; canvas.height = SZ;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, SZ, SZ);
      const data = ctx.getImageData(0, 0, SZ, SZ).data;

      // Thresholds are deliberately forgiving — stain intensity varies a
      // lot (see Section 5.8.2 on our own normalisation issues), so this
      // should only flag images that are clearly outside the H&E family,
      // not penalise pale-but-genuine patches.
      let saturated = 0, inHeRange = 0;
      for (let i = 0; i < data.length; i += 4) {
        const [h, s] = rgbToHsv(data[i], data[i + 1], data[i + 2]);
        if (s > 0.06) {
          saturated++;
          // H&E tissue sits in the pink/magenta/purple band (eosin pink
          // through haematoxylin purple). Greens, blues, and yellows
          // fall outside this and are unlikely to be real tissue.
          if (h >= 250 || h <= 25) inHeRange++;
        }
      }
      const satFraction = saturated / (SZ * SZ);
      const heFraction = saturated > 0 ? inHeRange / saturated : 0;

      if (satFraction < 0.015) {
        warnings.push("Very little colour detected — H&E tissue usually has visible pink/purple staining.");
      } else if (heFraction < 0.3) {
        warnings.push("Colour profile doesn't look like typical H&E staining (expected pink/purple tones).");
      }

      resolve({ ok: warnings.length === 0, warnings });
    };
    img.onerror = () => resolve({ ok: true, warnings: [] }); // fail open — never block on a decode error
    img.src = dataUrl;
  });
}

function renderSanityWarning(elId, result) {
  const el = document.getElementById(elId);
  if (result.ok) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = `<b>This may not be a tissue patch</b>` +
    result.warnings.map(w => `<div>${w}</div>`).join("");
}

// ============================================================
// Dropzone wiring (shared between Classify and Compare tabs)
// ============================================================
function wireDropzone(zoneId, innerId, onImageReady, onClear) {
  const zone = document.getElementById(zoneId);
  const inner = document.getElementById(innerId);
  const input = zone.querySelector("input[type=file]");
  const emptyHTML = inner.innerHTML;          // remember the "Drop a patch here" prompt

  // Clicking the empty zone (or the preview) opens the file picker -
  // but never when the click was on the remove (x) button.
  zone.addEventListener("click", (e) => {
    if (e.target.closest(".dz-remove")) return;
    input.click();
  });

  input.addEventListener("change", async () => {
    if (input.files[0]) await loadFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("is-drag"); })
  );
  ["dragleave", "drop"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove("is-drag"); })
  );
  zone.addEventListener("drop", async e => {
    const file = e.dataTransfer.files[0];
    if (file) await loadFile(file);
  });

  function clearPreview() {
    zone.querySelectorAll("img, .dz-remove").forEach(n => n.remove());
  }

  function showPreview(dataUrl) {
    clearPreview();                           // replace, never stack, previews
    inner.innerHTML = "";
    const img = document.createElement("img");
    img.src = dataUrl;
    img.alt = "Uploaded patch";
    zone.appendChild(img);

    const x = document.createElement("button");
    x.type = "button";
    x.className = "dz-remove";
    x.setAttribute("aria-label", "Remove patch");
    x.title = "Remove patch";
    x.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
    x.addEventListener("click", (e) => { e.stopPropagation(); removePatch(); });
    zone.appendChild(x);
  }

  function removePatch() {
    clearPreview();
    inner.innerHTML = emptyHTML;              // bring back the empty prompt
    input.value = "";                         // so choosing the same file again fires "change"
    if (onClear) onClear();
  }

  async function loadFile(file) {
    const dataUrl = await fileToDataURL(file);
    showPreview(dataUrl);
    onImageReady(dataUrl);
  }
}

wireDropzone("dropzone", "dz-inner", async (dataUrl) => {
  currentImage = dataUrl;
  document.getElementById("classify-btn").disabled = false;
  renderSanityWarning("sanity-warning", await checkImageSanity(dataUrl));
}, () => {
  // patch removed on the Classify tab: reset everything that depended on it
  currentImage = null;
  document.getElementById("classify-btn").disabled = true;
  renderSanityWarning("sanity-warning", { ok: true, warnings: [] });
  lastClassify = null;                                         // no report for a removed patch
  document.getElementById("report-btn").disabled = true;
  document.getElementById("classify-results").innerHTML =
    '<div class="empty-state"><div class="empty-glyph">○</div>' +
    '<p>Results will appear here once you run a classification.</p></div>';
});

wireDropzone("dropzone-cmp", "dz-inner-cmp", async (dataUrl) => {
  compareImage = dataUrl;
  document.getElementById("compare-btn").disabled = false;
  renderSanityWarning("sanity-warning-cmp", await checkImageSanity(dataUrl));
}, () => {
  // patch removed on the Compare tab
  compareImage = null;
  document.getElementById("compare-btn").disabled = true;
  renderSanityWarning("sanity-warning-cmp", { ok: true, warnings: [] });
  const sum = document.getElementById("compare-summary");
  sum.className = "compare-summary-empty";
  sum.innerHTML = "<p>Comparison summary will appear here.</p>";
  document.getElementById("compare-pair").innerHTML = "";
});

// ============================================================
// Model toggle (Classify tab)
// ============================================================
document.querySelectorAll(".mt-opt").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mt-opt").forEach(b => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    selectedModel = btn.dataset.model;
  });
});

// ---------------- input gate (server-side H&E check) ----------------
function notHeMessage(reasons) {
  const li = (reasons || []).map(r => `<div>${String(r).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("");
  return `<b>This doesn't look like an H&amp;E-stained tissue patch, so it was not analysed.</b>${li}`;
}

function showProblem(containerId, res, body) {    // any non-OK reply -> a clear message, never a blank panel
  if (body && body.error === "not_he_tissue") return showNotHE(containerId, body.reasons);
  const msg = body && body.error === "unreadable_image"
    ? "This file could not be read as an image, so it was not analysed."
    : `The server could not process this image (HTTP ${res.status}).`;
  document.getElementById(containerId).innerHTML = `<div class="sanity-warning"><b>${msg}</b></div>`;
}

function showNotHE(containerId, reasons) {
  document.getElementById(containerId).innerHTML = `<div class="sanity-warning">${notHeMessage(reasons)}</div>`;
}

function addGateWarning(elId, reasons) {      // appended to the existing warning box
  if (!reasons || !reasons.length) return;
  const el = document.getElementById(elId);
  el.hidden = false;
  el.querySelectorAll("[data-gate]").forEach(n => n.remove());      // don't stack duplicates on re-runs
  el.insertAdjacentHTML("beforeend", `<div data-gate><b>The input check flagged this image — results may be unreliable</b>` +
    reasons.map(r => `<div>${String(r).replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("") + `</div>`);
}

// ============================================================
// Classify
// ============================================================
document.getElementById("classify-btn").addEventListener("click", async () => {
  if (!currentImage) return;
  const btn = document.getElementById("classify-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  try {
    const res = await fetch(`${API}/api/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel, image: currentImage }),
    });
    const result = await res.json();
    if (!res.ok) {
      showProblem("classify-results", res, result);
      lastClassify = null;
      document.getElementById("report-btn").disabled = true;
      return;
    }
    renderClassifyResult(result, selectedModel);
    addGateWarning("sanity-warning", result.gate_warning);
    logPrediction(selectedModel, result);
    lastClassify = { result, model: selectedModel, image: currentImage };
    document.getElementById("report-btn").disabled = false;
  } catch (err) {
    document.getElementById("classify-results").innerHTML =
      `<div class="empty-state"><p>Request failed — is the backend running?<br>${err}</p></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run classification";
  }
});

function renderClassifyResult(result, modelName) {
  const container = document.getElementById("classify-results");
  container.innerHTML = "";
  container.appendChild(buildResultCard(result, modelName));
  container.appendChild(buildRouteCard(result));
  container.appendChild(buildProbsCard(result));
  container.appendChild(buildOverlays(result));
}

// One plain-language sentence built only from the existing response.
function explainResult(result, modelName) {
  const ran = result.stages.filter(s => s.ran).length;
  const pct = Math.round(result.confidence * 100);
  const unit = modelName === "caslite" ? "CasLite heads" : "cascade stages";
  const path = ran === result.stages.length
    ? `after passing through all three ${unit}`
    : `after stage 1 (type) alone, so architecture and grade were not evaluated`;
  return `This patch was classified as ${result.pred_class} with ${pct}% confidence, ${path}.`;
}

function buildResultCard(result, modelName) {
  const el = document.createElement("div");
  el.className = "result-card";
  el.innerHTML = `
    <div class="result-head"><span>RESULT</span><span>${modelName.toUpperCase()}</span></div>
    <div class="result-body">
      <div class="result-row"><span class="result-label">Predicted class</span>
        <span class="result-value hero">${result.pred_class}</span></div>
      <div class="result-row"><span class="result-label">Confidence</span>
        <span class="result-value">${result.confidence.toFixed(3)}</span></div>
      <div class="result-row"><span class="result-label">Inference time</span>
        <span class="result-value">${result.latency_ms.toFixed(1)} ms</span></div>
      <p class="about-note">${explainResult(result, modelName)}</p>
    </div>`;
  if (result.banner) {
    // Reuses the sanity-warning style. <strong>/<em> (not <b>) so the
    // ".sanity-warning b" block rule doesn't break inline emphasis.
    const note = document.createElement("div");
    note.className = "sanity-warning";
    note.innerHTML = result.banner
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");
    el.appendChild(note);
  }
  return el;
}

function buildRouteCard(result) {
  const el = document.createElement("div");
  el.className = "route";
  let rows = "";
  result.stages.forEach((s, i) => {
    if (s.ran) {
      rows += `<div class="route-stage">
        <span class="rs-num">${i + 1}</span>
        <span>${s.name}: <b>${s.pred}</b></span>
        <span class="rs-val">${s.conf !== null ? s.conf.toFixed(3) : "—"}</span>
      </div>`;
    } else {
      rows += `<div class="route-stage is-skipped">
        <span class="rs-num">${i + 1}</span>
        <span>${s.name} — not evaluated</span>
      </div>`;
    }
  });
  el.innerHTML = `<div class="route-title">Cascade routing</div>${rows}`;
  return el;
}

function buildProbsCard(result) {
  const el = document.createElement("div");
  el.className = "probs";
  const sorted = Object.entries(result.probs).sort((a, b) => b[1] - a[1]);
  const top = result.pred_class; // highlight the routed prediction, not the probs argmax
  let rows = `<div class="route-title">Class probabilities</div>`;
  sorted.forEach(([cls, p]) => {
    const isTop = cls === top;
    rows += `
      <div class="prob-row">
        <span class="prob-cls ${isTop ? "is-top" : ""}">${cls}</span>
        <div class="prob-track"><div class="prob-fill ${isTop ? "is-top" : ""}" data-w="${p * 100}"></div></div>
        <span class="prob-pct">${(p * 100).toFixed(1)}%</span>
      </div>`;
  });
  el.innerHTML = rows;
  // animate bars in on next frame
  requestAnimationFrame(() => {
    el.querySelectorAll(".prob-fill").forEach(f => { f.style.width = f.dataset.w + "%"; });
  });
  return el;
}

function buildOverlays(result) {
  const el = document.createElement("div");
  el.className = "overlays";
  const gland = result.gland_mask || currentImage;
  const attn = result.attention_map || currentImage;
  el.innerHTML = `
    <div class="overlay-card">
      <div class="oc-label">Gland segmentation</div>
      <img src="${gland}">
    </div>
    <div class="overlay-card">
      <div class="oc-label">Grad-CAM attention</div>
      <img src="${attn}">
    </div>
    <p class="overlay-note">Gland segmentation marks glandular structure, not
      disease. The attention map shows what influenced the prediction, not
      where pathology is. See Chapter 5.5.</p>`;
  return el;
}

// ============================================================
// Compare
// ============================================================
document.getElementById("compare-btn").addEventListener("click", async () => {
  if (!compareImage) return;
  const btn = document.getElementById("compare-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  try {
    const res = await fetch(`${API}/api/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: compareImage }),
    });
    const data = await res.json();
    if (!res.ok) {
      showProblem("compare-summary", res, data);
      document.getElementById("compare-pair").innerHTML = "";
      return;
    }
    renderCompare(data.cascade, data.caslite);
    addGateWarning("sanity-warning-cmp", data.gate_warning);
    logPrediction("cascade", data.cascade);
    logPrediction("caslite", data.caslite);
  } catch (err) {
    document.getElementById("compare-summary").innerHTML = `<p>Request failed — is the backend running?<br>${err}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Compare both models";
  }
});

function renderCompare(cascadeResult, caslitetResult) {
  const agree = cascadeResult.pred_class === caslitetResult.pred_class;
  const speedup = (cascadeResult.latency_ms / Math.max(caslitetResult.latency_ms, 0.01)).toFixed(1);

  const summary = document.getElementById("compare-summary");
  summary.className = "compare-summary";
  summary.innerHTML = `
    <div class="cs-pill ${agree ? "agree" : "disagree"}">
      ${agree ? "Models agree" : "Models disagree — see Section 5.2.7"}
    </div>
    <div class="result-row"><span class="result-label">CasLite speed-up</span>
      <span class="result-value hero">${speedup}×</span></div>
    <div class="result-row"><span class="result-label">Parameter ratio</span>
      <span class="result-value">1 : 22</span></div>`;

  const pair = document.getElementById("compare-pair");
  pair.innerHTML = "";
  pair.appendChild(buildResultCard(cascadeResult, "cascade"));
  pair.appendChild(buildResultCard(caslitetResult, "caslite"));
}

// ============================================================
// Single-result report — filled into #report and printed; the print
// stylesheet hides the app, so "Save as PDF" in the print dialog gives the PDF.
// Name/age stay in the page only (not sent anywhere, not stored).
// ============================================================
let lastClassify = null;

function buildReport(last, name, age) {
  const { result, model, image } = last;
  const rep = document.getElementById("report");
  rep.innerHTML = "";
  const add = (tag, text, parent = rep) => {
    const e = document.createElement(tag);
    e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  add("h1", "Colorectal Polyp Detection report");
  add("p", `Generated ${new Date().toLocaleString()} · ${model === "cascade" ? "Cascade (33.5M params)" : "CasLite (1.53M params)"}`);

  const img = document.createElement("img");
  img.src = image;
  img.style.margin = "12pt 0";
  rep.appendChild(img);

  const table = document.createElement("table");
  const row = (k, v) => {
    const tr = document.createElement("tr");
    add("th", k, tr); add("td", v, tr);
    table.appendChild(tr);
  };
  row("Name", name || "Not provided");
  row("Age", age || "Not provided");
  row("Predicted class", result.pred_class);
  row("Confidence", result.confidence.toFixed(3));
  result.stages.forEach((s, i) => {
    row(`Stage ${i + 1}: ${s.name}`, s.ran ? `${s.pred} (${s.conf.toFixed(3)})` : "not evaluated");
  });
  rep.appendChild(table);

  if (result.banner) {
    const b = document.createElement("div");
    b.className = "rp-note";
    b.innerHTML = result.banner
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");
    rep.appendChild(b);
  }
  add("p", "Research use only. This is not a diagnostic tool and must not be used for clinical decisions.").className = "rp-note";
}

// Downscale for the report so the PDF isn't carrying a multi-MB full-resolution patch.
function makeThumb(dataUrl, px = 440) {
  return new Promise(resolve => {
    const im = new Image();
    im.onload = () => {
      const s = Math.min(1, px / Math.max(im.width, im.height));
      const c = document.createElement("canvas");
      c.width = Math.round(im.width * s); c.height = Math.round(im.height * s);
      c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
      resolve(c.toDataURL("image/jpeg", 0.9));
    };
    im.onerror = () => resolve(dataUrl);
    im.src = dataUrl;
  });
}

document.getElementById("report-btn").addEventListener("click", async () => {
  if (!lastClassify) return;
  const name = document.getElementById("patient-name").value.trim();
  const age = document.getElementById("patient-age").value.trim();
  buildReport({ ...lastClassify, image: await makeThumb(lastClassify.image) }, name, age);
  window.print();
});
window.addEventListener("afterprint", () => { document.getElementById("report").innerHTML = ""; });

// ============================================================
// Batch upload — up to MAX_BATCH files, one request, looped server-side
// ============================================================
const MAX_BATCH = 10;
let batchFiles = [];

(function wireBatch() {
  const zone = document.getElementById("dropzone-batch");
  const input = zone.querySelector("input[type=file]");
  const info = document.getElementById("batch-files");
  const btn = document.getElementById("batch-btn");

  function setFiles(list) {
    const files = [...list].filter(f => f.type.startsWith("image/"));
    batchFiles = files.slice(0, MAX_BATCH);
    btn.disabled = batchFiles.length === 0;
    info.textContent = batchFiles.length === 0
      ? "No files selected."
      : `${batchFiles.length} file${batchFiles.length > 1 ? "s" : ""} selected` +
        (files.length > MAX_BATCH ? ` (only the first ${MAX_BATCH} of ${files.length} will be used)` : "") + ".";
  }

  zone.addEventListener("click", () => input.click());
  input.addEventListener("click", e => e.stopPropagation());
  input.addEventListener("change", () => setFiles(input.files));
  ["dragenter", "dragover"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove("is-drag"); }));
  zone.addEventListener("drop", e => setFiles(e.dataTransfer.files));

  btn.addEventListener("click", async () => {
    if (batchFiles.length === 0) return;
    const out = document.getElementById("batch-results");
    const model = document.querySelector("input[name=batch-model]:checked").value;
    btn.disabled = true;
    btn.textContent = `Running ${batchFiles.length}…`;
    try {
      const images = [];
      for (const f of batchFiles) images.push({ name: f.name, image: await fileToDataURL(f) });
      const res = await fetch(`${API}/api/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, images }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const results = (await res.json()).results;
      results.filter(r => !r.error).forEach(r => logPrediction(model, r));   // each image gets its own session-log entry
      renderBatchResults(results, model);
    } catch (err) {
      out.innerHTML = `<div class="empty-state"><p>Batch failed — ${String(err).replace(/</g, "&lt;")}</p></div>`;
    } finally {
      btn.disabled = batchFiles.length === 0;
      btn.textContent = "Run batch";
    }
  });
})();

function renderBatchResults(results, model) {
  const out = document.getElementById("batch-results");
  out.innerHTML = "";
  const table = document.createElement("table");
  table.className = "log-table";
  table.innerHTML = `<thead><tr><th></th><th>File</th><th>Prediction</th><th>Confidence</th></tr></thead>`;
  const tbody = document.createElement("tbody");
  const banners = new Set();
  results.forEach(r => {
    const tr = document.createElement("tr");
    const thumbTd = document.createElement("td");
    if (r.thumbnail) {
      const img = document.createElement("img");
      img.src = r.thumbnail;
      img.style.cssText = "width:48px;height:48px;object-fit:cover;display:block;";
      thumbTd.appendChild(img);
    }
    const nameTd = document.createElement("td");
    nameTd.textContent = r.name;
    const predTd = document.createElement("td");
    const confTd = document.createElement("td");
    if (r.error) {
      predTd.textContent = r.error;
      confTd.textContent = "—";
    } else {
      predTd.textContent = r.pred_class;
      confTd.textContent = r.confidence.toFixed(3);
      if (r.banner) banners.add(r.banner);
    }
    tr.append(thumbTd, nameTd, predTd, confTd);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  out.appendChild(table);

  const flagged = results.filter(r => r.gate_warning && r.gate_warning.length);
  if (flagged.length) {
    const box = document.createElement("div");
    box.className = "sanity-warning";
    box.innerHTML = `<b>The input check flagged some images — results may be unreliable</b>` +
      flagged.map(r => `<div>${String(r.name).replace(/&/g, "&amp;").replace(/</g, "&lt;")}: ` +
        r.gate_warning.map(w => String(w).replace(/&/g, "&amp;").replace(/</g, "&lt;")).join("; ") + `</div>`).join("");
    out.appendChild(box);
  }

  const ok = results.filter(r => !r.error);
  const summary = document.createElement("p");
  summary.className = "about-note";
  summary.textContent = `${model === "cascade" ? "Cascade" : "CasLite"}: ${ok.length} of ${results.length} classified.`;
  out.appendChild(summary);

  banners.forEach(text => {   // same approved high-grade notice as the single-image view
    const note = document.createElement("div");
    note.className = "sanity-warning";
    note.innerHTML = text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");
    out.appendChild(note);
  });
}

// ============================================================
// History (sessionStorage-backed, survives reloads this session)
// ============================================================
function logPrediction(model, result) {
  history.unshift({
    time: new Date().toLocaleTimeString(),
    model: model === "cascade" ? "Cascade" : "CasLite",
    pred: result.pred_class,
    conf: result.confidence.toFixed(3),
    latency: result.latency_ms.toFixed(1) + " ms",
  });
  history = history.slice(0, 200);
  sessionStorage.setItem("polyp_history", JSON.stringify(history));
}

// Session analytics: aggregated from the same `history` array the table uses.
function renderLogStats() {
  const el = document.getElementById("log-stats");
  const n = history.length;
  if (n === 0) { el.innerHTML = ""; return; }

  const byClass = Object.fromEntries(CLASSES.map(c => [c, 0]));
  let cascade = 0;
  let confSum = 0;
  history.forEach(r => {
    byClass[r.pred] = (byClass[r.pred] || 0) + 1;
    if (r.model === "Cascade") cascade++;
    confSum += parseFloat(r.conf);
  });
  const caslite = n - cascade;
  const maxCount = Math.max(...Object.values(byClass));

  const bars = Object.entries(byClass).map(([cls, k]) => `
      <div class="prob-row">
        <span class="prob-cls ${k === maxCount && k > 0 ? "is-top" : ""}">${cls}</span>
        <div class="prob-track"><div class="prob-fill ${k === maxCount && k > 0 ? "is-top" : ""}"
          style="width:${maxCount ? (k / maxCount) * 100 : 0}%"></div></div>
        <span class="prob-pct">${k}</span>
      </div>`).join("");

  el.innerHTML = `
    <div class="stat-row">
      <div class="stat"><div class="stat-value">${n}</div><div class="stat-label">predictions<br>this session</div></div>
      <div class="stat-div"></div>
      <div class="stat"><div class="stat-value">${cascade} / ${caslite}</div><div class="stat-label">Cascade / CasLite<br>(${Math.round(cascade / n * 100)}% / ${Math.round(caslite / n * 100)}%)</div></div>
      <div class="stat-div"></div>
      <div class="stat"><div class="stat-value">${(confSum / n).toFixed(3)}</div><div class="stat-label">average confidence</div></div>
    </div>
    <div class="probs"><div class="route-title">Predictions by class</div>${bars}</div>`;
}

function renderHistory() {
  const body = document.getElementById("log-body");
  const empty = document.getElementById("log-empty");
  renderLogStats();
  body.innerHTML = "";
  if (history.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  history.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${row.time}</td><td>${row.model}</td><td>${row.pred}</td><td>${row.conf}</td><td>${row.latency}</td>`;
    body.appendChild(tr);
  });
}

document.getElementById("clear-log").addEventListener("click", () => {
  history = [];
  sessionStorage.removeItem("polyp_history");
  renderHistory();
});

// ============================================================
// Theme switch (icon toggle) — smooth colour transition.
// The saved theme is applied before first paint by the <head> script in index.html.
// ============================================================
(function () {
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let timer;

  function setTheme(theme, animate) {
    if (animate && !reduced()) {
      root.classList.add("theme-anim");
      clearTimeout(timer);
      timer = setTimeout(() => root.classList.remove("theme-anim"), 450);
    }
    root.setAttribute("data-theme", theme);
    btn.setAttribute("aria-checked", String(theme === "dark"));
    btn.setAttribute("title", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
    try { localStorage.setItem("polyp_theme", theme); } catch (e) {}
  }

  // Sync the switch with the saved theme without animating the knob on page load.
  const parts = [btn, ...btn.querySelectorAll("*")];
  parts.forEach(e => { e.style.transition = "none"; });
  setTheme(root.getAttribute("data-theme") || "light", false);
  void btn.offsetWidth;
  parts.forEach(e => { e.style.transition = ""; });
  btn.addEventListener("click", () =>
    setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true));
})();

// ============================================================
// Page changes: cross-fade through the View Transitions API
// (falls back to the plain CSS fade where unsupported)
// ============================================================
(function () {
  if (!document.startViewTransition) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  document.documentElement.classList.add("has-vt");
  const original = goto;
  goto = function (view) { document.startViewTransition(() => original(view)); };
})();
