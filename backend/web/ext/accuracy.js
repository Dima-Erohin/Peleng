// Расширение «Точность»: проверки по доске и по рулетке, отчёт по камерам, кольцо σ вокруг людей/объектов.
// Сервер: peleng/ext/accuracy.py, описание: docs/accuracy.md.
const { THREE, scene, log } = window.peleng;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

const box = window.peleng.addSection("Точность");
box.insertAdjacentHTML("beforeend", `
  <div class="small" id="accLive">σ: нет данных (нужен человек/объект, видимый ≥ 2 камерами)</div>
  <label class="chk small">кольцо в 3D:
    <select id="accScale" style="font: inherit; background:#0d1117; color: var(--text); border:1px solid var(--line); border-radius:6px">
      <option value="0">скрыть</option><option value="1">1σ</option><option value="3" selected>3σ</option><option value="10">×10 σ (наглядно)</option>
    </select></label>
  <div class="btns" style="margin-top:6px">
    <button id="accBoard" title="Положите доску в любое место зоны, чтобы её видели ≥ 2 камеры">Проверка по доске</button>
  </div>
  <div class="small" id="accBoardRes" style="margin:4px 0 8px"></div>
  <div class="btns">
    <button id="accA" title="Встаньте в точку A и стойте ~1 с">Точка A</button>
    <button id="accB" title="Перейдите в точку B и стойте ~1 с">Точка B</button>
  </div>
  <div class="btns" style="margin-top:6px; align-items:center">
    по рулетке, м <input type="number" id="accTape" step="0.01" min="0" style="width:80px">
    <button id="accCmp" class="primary">Сравнить</button>
  </div>
  <div class="small" id="accTapeRes" style="margin:4px 0 8px"></div>
  <div class="btns"><button id="accRep">Отчёт по камерам</button><button id="accClear" title="Очистить историю проверок">Очистить историю</button></div>
  <div class="small" id="accReport" style="margin-top:6px"></div>
  <div class="small" id="accHist" style="margin-top:6px"></div>
`);
const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const r = await fetch(`/api/accuracy/${path}`, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

async function busy(btn, fn) {
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = old + " …";
  try { await fn(); } catch (e) { log(`точность: ${e.message}`); return e; } finally { btn.disabled = false; btn.textContent = old; }
}

// ---------------------------------------------------------------- проверка по доске
$("accBoard").onclick = () => busy($("accBoard"), async () => {
  $("accBoardRes").textContent = "ищу доску в кадрах (полное разрешение)…";
  const r = await api("board", {});
  if (!r.ok) {
    const per = Object.entries(r.per_cam || {}).map(([c, s]) => `${esc(c)}: ${esc(s)}`).join("; ");
    $("accBoardRes").innerHTML = `<span style="color:var(--bad)">${esc(r.message)}</span>${per ? "<br>" + per : ""}`;
    return;
  }
  $("accBoardRes").innerHTML =
    `<b>клетка ${fmt(r.square_mm)} мм: ошибка медиана ${fmt(r.median_err_mm, 2)} мм, макс ${fmt(r.max_err_mm, 2)} мм</b> (${fmt(r.median_err_pct, 2)} %)<br>` +
    `длина по доске ${fmt(r.span_m * 1000, 0)} мм: ${r.span_err_mm > 0 ? "+" : ""}${fmt(r.span_err_mm, 1)} мм; плоскость: ${fmt(r.plane_rms_mm, 2)} мм (макс ${fmt(r.plane_max_mm, 2)})<br>` +
    `углов ${r.corners}, пар ${r.pairs}, камеры ${r.cams.map(esc).join(", ")}, наклон ${fmt(r.tilt_deg)}°, репроекция ${fmt(r.reproj_px, 2)} px` +
    (r.expected_err_mm != null ? `, ожидалось ≈ ${fmt(r.expected_err_mm, 2)} мм` : "");
  loadHistory();
});

// ---------------------------------------------------------------- рулетка
function pointText(name, p) {
  const xy = p.floor || p.pos;
  return `${name}: ${esc(p.key)} (${xy.map((v) => fmt(v, 2)).join(", ")}) по ${p.n} кадрам` + (p.warning ? ` <span style="color:var(--warn)">${esc(p.warning)}</span>` : "");
}
let pts = { a: null, b: null };
function showPoints(extra = "") {
  $("accTapeRes").innerHTML = [pts.a && pointText("A", pts.a), pts.b && pointText("B", pts.b), extra].filter(Boolean).join("<br>");
}
for (const w of ["a", "b"]) {
  const btn = $(`acc${w.toUpperCase()}`);
  btn.onclick = () => busy(btn, async () => {
    pts[w] = await api(`point/${w}`, {});
    if (w === "a") pts.b = null;
    showPoints();
  }).then((e) => e && showPoints(`<span style="color:var(--bad)">${esc(e.message)}</span>`));
}
$("accCmp").onclick = () => busy($("accCmp"), async () => {
  const m = parseFloat($("accTape").value);
  if (!(m > 0)) throw new Error("введите расстояние по рулетке, м");
  const r = await api("tape", { measured_m: m });
  const col = Math.abs(r.err_cm) <= 5 ? "var(--acc)" : Math.abs(r.err_cm) <= 15 ? "var(--warn)" : "var(--bad)";
  showPoints(`<b style="color:${col}">система ${fmt(r.system_m, 3)} м, рулетка ${fmt(r.measured_m, 3)} м → ошибка ${r.err_cm > 0 ? "+" : ""}${fmt(r.err_cm)} см (${fmt(r.err_pct, 2)} %)</b>` +
    (r.expected_cm != null ? ` · ожидаемая ±${fmt(r.expected_cm)} см` : "") + (r.horizontal ? " · по полу" : " · в 3D"));
  loadHistory();
}).then((e) => e && showPoints(`<span style="color:var(--bad)">${esc(e.message)}</span>`));

// ---------------------------------------------------------------- история и отчёт
async function loadHistory() {
  try {
    const h = await api("history");
    if (!h.length) { $("accHist").innerHTML = ""; return; }
    const rows = h.slice(-12).reverse().map((r) => {
      const t = new Date(r.time * 1000).toLocaleTimeString();
      if (r.type === "tape") return `<tr><td>${t}</td><td>рулетка</td><td>${fmt(r.measured_m, 2)} м</td><td>${r.err_cm > 0 ? "+" : ""}${fmt(r.err_cm)} см (${fmt(r.err_pct, 1)} %)</td></tr>`;
      return `<tr><td>${t}</td><td>доска</td><td>${(r.cams || []).length} кам.</td><td>${fmt(r.median_err_mm, 2)} / ${fmt(r.max_err_mm, 2)} мм</td></tr>`;
    }).join("");
    $("accHist").innerHTML = `<table style="width:100%; border-collapse:collapse; font-size:12px"><tr style="color:var(--muted)"><td>время</td><td>проверка</td><td></td><td>ошибка</td></tr>${rows}</table>`;
  } catch (e) { /* сервер без расширения */ }
}

async function loadReport() {
  const r = await api("report");
  const cams = r.cameras.map((c) =>
    `<tr><td>${esc(c.id)}</td><td>${c.calibrated ? fmt(c.extrinsics_reproj_px, 2) + " px" : "—"}</td><td>${esc(c.intrinsics)}${c.intrinsics_rms != null ? " " + fmt(c.intrinsics_rms, 2) + " px" : ""}</td><td>${c.dist_to_zone_m != null ? fmt(c.dist_to_zone_m) + " м" : ""}</td></tr>`).join("");
  const pairs = r.pairs.map((p) =>
    `<tr style="color:${p.ok ? "inherit" : "var(--warn)"}"><td>${esc(p.a)}–${esc(p.b)}</td><td>${fmt(p.effective_deg, 0)}°</td><td colspan="2">σ ${p.sigma_cm ? p.sigma_cm.map((v) => fmt(v)).join(" / ") + " см" : "–"}</td></tr>`).join("");
  const ch = r.checks || {};
  const summary = [
    r.zone_sigma_cm && `σ в центре зоны (все камеры, ${fmt(r.sigma_px)} px): <b>±${fmt(r.zone_sigma_h_cm)} см</b> по горизонтали, ±${fmt(r.zone_sigma_cm[2])} см по высоте`,
    r.live && `сейчас: σ по горизонтали медиана ±${fmt(r.live.sigma_h_cm_median)} см, макс ±${fmt(r.live.sigma_h_cm_max)} см`,
    ch.board_checks && `доска: ${ch.board_checks} пров., медиана ${fmt(ch.board_median_err_mm, 2)} мм`,
    ch.tape_checks && `рулетка: ${ch.tape_checks} пров., медиана ${fmt(ch.tape_median_err_cm)} см (${fmt(ch.tape_median_err_pct, 1)} %), макс ${fmt(ch.tape_max_err_cm)} см`,
  ].filter(Boolean).join("<br>");
  const warn = (r.warnings || []).map((w) => `<div style="color:var(--warn)">⚠ ${esc(w)}</div>`).join("");
  $("accReport").innerHTML =
    `<table style="width:100%; border-collapse:collapse; font-size:12px"><tr style="color:var(--muted)"><td>камера</td><td>положение</td><td>оптика</td><td>до центра</td></tr>${cams}` +
    `<tr style="color:var(--muted)"><td>пара</td><td>угол</td><td colspan="2">σ x / y / z в центре</td></tr>${pairs}</table>` +
    `<div style="margin-top:4px">${summary}</div>${warn}`;
}
$("accRep").onclick = () => busy($("accRep"), loadReport);
$("accClear").onclick = async () => { await api("history/clear", {}); loadHistory(); };
loadHistory();
window.peleng.on("hello", () => setTimeout(() => loadReport().catch(() => {}), 500));
window.peleng.on("cameras", () => loadReport().catch(() => {}));

// ---------------------------------------------------------------- кольцо σ в 3D
// Эллипс с полуосями k·σx, k·σy (по горизонтали) у ног человека / под объектом.
const CIRCLE = (() => {
  const g = new THREE.BufferGeometry();
  const a = [];
  for (let i = 0; i < 64; i++) { const t = (i / 64) * Math.PI * 2; a.push(Math.cos(t), Math.sin(t), 0); }
  g.setAttribute("position", new THREE.Float32BufferAttribute(a, 3));
  return g;
})();
const rings = {}; // "person:3" -> LineLoop
function ringFor(key, id) {
  if (rings[key]) return rings[key];
  const r = new THREE.LineLoop(CIRCLE, new THREE.LineBasicMaterial({ color: window.peleng.colorFor(id), transparent: true, opacity: 0.9, depthTest: false }));
  scene.add(r);
  return rings[key] = r;
}
window.peleng.on("frame", (m) => {
  const k = parseFloat($("accScale").value);
  const seen = new Set();
  const live = [];
  for (const [kind, list] of [["person", m.persons], ["object", m.objects]]) {
    for (const p of list || []) {
      if (!p.pos_std_m) continue;
      const key = `${p.kind || kind}:${p.id}`;
      const [sx, sy] = p.pos_std_m;
      live.push(`${kind === "person" ? "ID" : esc(p.cls || "объект")} ${p.id}: ±${fmt(Math.hypot(sx, sy) * 100)} см`);
      if (!k) continue;
      seen.add(key);
      const r = ringFor(key, p.id);
      const base = p.floor || p.pos;
      r.position.set(base[0], base[1], (p.floor ? 0 : base[2]) + 0.01);
      r.scale.set(Math.max(k * sx, 0.003), Math.max(k * sy, 0.003), 1);
      r.material.opacity = p.predicted ? 0.35 : 0.9;
      r.visible = true;
    }
  }
  for (const [key, r] of Object.entries(rings)) if (!seen.has(key)) r.visible = false;
  $("accLive").innerHTML = live.length ? `σ по горизонтали (${k ? "кольцо " + k + "σ" : "кольцо скрыто"}): ` + live.join(", ") : "σ: нет точек, видимых ≥ 2 камерами";
});
