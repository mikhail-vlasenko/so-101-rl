"use strict";

// ---------- shared helpers ----------

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  const text = await resp.text();
  if (text) data = JSON.parse(text);
  return { ok: resp.ok, status: resp.status, data };
}

function fmtUptime(seconds) {
  if (seconds == null) return "";
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${s % 60}s`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}

// ---------- form-field persistence ----------
//
// Every container with [data-persist="<scope>"] has its named fields saved to
// the server on change and restored on load. The server stores only overrides;
// the original defaults stay in the rendered HTML, so "Reset defaults" wipes
// the overrides and a reload shows the code defaults again.

function fieldValue(el) {
  return el.type === "checkbox" ? el.checked : el.value;
}

function applyValue(el, value) {
  if (el.type === "checkbox") el.checked = Boolean(value);
  else el.value = value;
}

async function initSettings() {
  const containers = document.querySelectorAll("[data-persist]");
  const resetBtn = document.getElementById("reset-settings");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      await postJSON("/api/settings/reset", {});
      window.location.reload();
    });
  }
  if (!containers.length) return;
  const resp = await fetch("/api/settings");
  const saved = resp.ok ? await resp.json() : {};
  for (const container of containers) {
    const scope = container.dataset.persist;
    const scopeSaved = saved[scope] || {};
    for (const el of container.querySelectorAll("[name]")) {
      if (el.name in scopeSaved) applyValue(el, scopeSaved[el.name]);
      el.addEventListener("change", () => {
        postJSON("/api/settings", { scope, field: el.name, value: fieldValue(el) });
      });
    }
  }
}

// ---------- script launch cards ----------

function collectValues(card) {
  const values = {};
  for (const el of card.querySelectorAll("form [name]")) {
    if (el.name === "__stream__") continue;
    values[el.name] = el.type === "checkbox" ? (el.checked ? "on" : "") : el.value;
  }
  return values;
}

async function launchScript(card) {
  const specId = card.dataset.specId;
  const streamBox = card.querySelector('[name="__stream__"]');
  const body = {
    values: collectValues(card),
    stream: streamBox ? streamBox.checked : false,
  };
  const resp = await postJSON(`/api/run/${specId}`, body);
  if (!resp.ok) {
    alert(`Launch failed (${resp.status}): ${resp.data.detail || "unknown error"}`);
    return;
  }
  window.location.href = `/run/${resp.data.run_id}`;
}

function initScriptCards() {
  for (const card of document.querySelectorAll(".script-card")) {
    card.querySelector(".launch-btn").addEventListener("click", () => launchScript(card));
    for (const input of card.querySelectorAll(".ckpt-input")) {
      input.addEventListener("focus", () => fillCheckpointList(card, input), { once: false });
    }
  }
}

async function fillCheckpointList(card, input) {
  // Resolve {field} placeholders in the log-dir template from the card's
  // current form values, falling back to each field's placeholder default.
  let logDir = input.dataset.logdirTemplate;
  for (const m of logDir.matchAll(/\{(\w+)\}/g)) {
    const field = card.querySelector(`[name="${m[1]}"]`);
    const value = field ? (field.value || field.dataset.default || field.placeholder) : "";
    logDir = logDir.replace(m[0], value || "");
  }
  if (!logDir || logDir.includes("{")) return;
  const resp = await fetch(`/api/checkpoints/${logDir}`);
  if (!resp.ok) return;
  const data = await resp.json();
  const list = document.getElementById(input.getAttribute("list"));
  list.replaceChildren(...data.options.map((o) => {
    const opt = document.createElement("option");
    opt.value = o;
    return opt;
  }));
}

// ---------- dashboard ----------

function initDashboard() {
  const table = document.getElementById("runs-table");
  if (!table) return;
  const tbody = table.querySelector("tbody");

  async function refresh() {
    const resp = await fetch("/api/status");
    if (!resp.ok) return;
    const { runs } = await resp.json();
    if (!runs.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="muted">nothing has run yet</td></tr>';
      return;
    }
    tbody.replaceChildren(...runs.map((r) => {
      const tr = document.createElement("tr");
      const status = r.running
        ? (r.stop_requested ? "stopping…" : "running")
        : `exit ${r.returncode}`;
      tr.innerHTML = `
        <td><a href="/run/${r.run_id}">${r.run_id}</a> <span class="muted">${r.title}</span></td>
        <td><span class="badge ${r.running ? "running" : "done"}">${status}</span></td>
        <td>${fmtUptime(r.uptime_s)}</td>
        <td>${r.running && r.stream_port ? `<a href="http://${location.hostname}:${r.stream_port}/stream" target="_blank">:${r.stream_port}</a>` : ""}</td>
        <td></td>`;
      if (r.running) {
        const btn = document.createElement("button");
        btn.className = "danger";
        btn.textContent = "Stop";
        btn.addEventListener("click", async () => {
          btn.disabled = true;
          await postJSON(`/api/stop/${r.run_id}`, {});
          refresh();
        });
        tr.lastElementChild.appendChild(btn);
      }
      return tr;
    }));
  }

  refresh();
  setInterval(refresh, 2000);
}

// ---------- run page ----------

function initRunPage() {
  const view = document.getElementById("run-view");
  if (!view) return;
  const runId = view.dataset.runId;
  const log = document.getElementById("log");
  const statusBadge = document.getElementById("run-status");
  const stopBtn = document.getElementById("stop-btn");
  const restartBtn = document.getElementById("restart-btn");

  // Only rendered while the run is alive (see run.html); stop retrying and
  // hide it the moment the run finishes — the script's streamer dies with it.
  const img = document.getElementById("sim-stream");
  let streamActive = img !== null;
  if (img) {
    const src = `http://${location.hostname}:${view.dataset.streamPort}/stream`;
    // The script 503s until its first frame; retry until the stream is up.
    const tryLoad = () => {
      img.src = `${src}?t=${Date.now()}`;
    };
    img.addEventListener("error", () => {
      if (streamActive) setTimeout(tryLoad, 1000);
    });
    tryLoad();
  }

  function endStream() {
    streamActive = false;
    const wrap = document.getElementById("stream-wrap");
    if (wrap) wrap.classList.add("hidden");
  }

  const source = new EventSource(`/api/logs/${runId}?start=0`);
  source.onmessage = (e) => {
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    log.textContent += JSON.parse(e.data) + "\n";
    if (atBottom) log.scrollTop = log.scrollHeight;
  };
  source.addEventListener("done", (e) => {
    source.close();
    endStream();
    statusBadge.textContent = `exit ${JSON.parse(e.data)}`;
    statusBadge.classList.remove("running");
    statusBadge.classList.add("done");
    stopBtn.disabled = true;
    restartBtn.disabled = false;
  });

  stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    await postJSON(`/api/stop/${runId}`, {});
  });

  // Only enabled once the run has exited (server 409s otherwise).
  restartBtn.addEventListener("click", async () => {
    restartBtn.disabled = true;
    const resp = await postJSON(`/api/restart/${runId}`, {});
    if (!resp.ok) {
      alert(`Restart failed (${resp.status}): ${resp.data.detail || "unknown error"}`);
      restartBtn.disabled = false;
      return;
    }
    window.location.href = `/run/${resp.data.run_id}`;
  });
}

// ---------- camera page ----------

function initCameraPage() {
  const view = document.getElementById("camera-view");
  if (!view) return;
  const wrap = document.getElementById("camera-stream-wrap");
  const imgs = wrap.querySelectorAll("img.camera-stream");
  const startBtn = document.getElementById("camera-start");
  const stopBtn = document.getElementById("camera-stop");

  function showStream(on) {
    wrap.classList.toggle("hidden", !on);
    const t = Date.now();
    for (const img of imgs) {
      img.src = on ? `/camera/stream/${img.dataset.camera}?t=${t}` : "";
    }
    startBtn.disabled = on;
    stopBtn.disabled = !on;
  }

  startBtn.addEventListener("click", async () => {
    const body = {
      family: document.getElementById("camera-family").value,
      exposure: document.getElementById("camera-exposure").value,
      gain: document.getElementById("camera-gain").value,
    };
    const resp = await postJSON("/api/camera/start", body);
    if (!resp.ok) {
      alert(`Camera start failed (${resp.status}): ${resp.data.detail || "unknown error"}`);
      return;
    }
    showStream(true);
  });

  stopBtn.addEventListener("click", async () => {
    await postJSON("/api/camera/stop", {});
    showStream(false);
  });

  for (const ctrl of ["exposure", "gain"]) {
    document.getElementById(`camera-${ctrl}`).addEventListener("change", async (e) => {
      if (stopBtn.disabled) return; // not streaming: value is picked up at start
      await postJSON(`/api/camera/${ctrl}`, { value: e.target.value });
    });
  }

  showStream(view.dataset.streaming === "1");
}

initSettings();
initScriptCards();
initDashboard();
initRunPage();
initCameraPage();
