// Расширение «Зоны и события»: прямоугольники и линии подсчёта на полу, события вход/выход/пересечение, тепловая карта.
// Сервер: peleng/ext/zones.py (REST /api/zones/…, поле "zones" в сообщении frame), описание: docs/zones.md.
const P = window.peleng;
const { THREE, scene, log } = P;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const Z_HEAT = 0.006, Z_ZONE = 0.008, Z_LINE = 0.01;  // над покрытием (0.004), ниже людей

const box = P.addSection("Зоны и события");
box.insertAdjacentHTML("beforeend", `
  <style>
    .zn-row { display: flex; gap: 6px; align-items: center; margin: 4px 0; flex-wrap: wrap; }
    .zn-row input[type=text], .zn-row select { flex: 1; min-width: 0; font: inherit; background: #0d1117; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px; }
    .zn-item { display: flex; gap: 6px; align-items: center; padding: 3px 0; border-bottom: 1px solid var(--line); font-size: 12px; }
    .zn-item .zn-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .zn-item button { padding: 0 6px; }
    .zn-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; }
    #znFeed { font: 12px ui-monospace, monospace; background: #0d1117; border-radius: 6px; padding: 6px; max-height: 150px; overflow-y: auto; white-space: pre-wrap; margin-top: 6px; }
    #znFeed div { color: var(--muted); } #znFeed b { color: var(--text); font-weight: 600; }
    .zn-table { width: 100%; border-collapse: collapse; font: 12px ui-monospace, monospace; }
    .zn-table th, .zn-table td { padding: 2px 4px; text-align: right; border-bottom: 1px solid var(--line); }
    .zn-table th:first-child, .zn-table td:first-child { text-align: left; }
  </style>
  <div class="zn-row"><input type="text" id="znName" placeholder="имя (необязательно)" maxlength="60"></div>
  <div class="btns">
    <button id="znDrawZone" title="Два клика по полу в 3D — противоположные углы прямоугольника">Нарисовать зону</button>
    <button id="znDrawLine" title="Два клика по полу: начало и конец отрезка; стрелка показывает направление A→B">Нарисовать линию</button>
  </div>
  <div class="small" id="znHint" style="margin:4px 0"></div>
  <div id="znList"></div>
  <label class="chk"><input type="checkbox" id="znShow" checked> зоны и линии в 3D</label>
  <label class="chk"><input type="checkbox" id="znHeat"> тепловая карта на полу</label>
  <div class="btns"><button id="znHeatReset" title="Обнулить накопленное время присутствия">Сбросить карту</button><button id="znResetStats" title="Обнулить счётчики и ленту событий">Сбросить счётчики</button></div>
  <div class="small" id="znHeatInfo" style="margin-top:4px"></div>
  <div id="znFeed"></div>
  <div class="small" style="margin-top:4px"><a href="/api/zones/events.csv" style="color:var(--text)">события CSV</a></div>
  <h2 style="margin-top:12px">Анализ записи</h2>
  <div class="zn-row"><select id="znSess"><option value="">— нет записей —</option></select><button id="znSessRefresh" title="Обновить список">⟳</button></div>
  <div class="btns"><button id="znAnalyze" class="primary">Анализировать</button><button id="znShowRec" title="Показать тепловую карту записи на полу">Карта записи на пол</button></div>
  <div id="znAnalysis" class="small" style="margin-top:6px"></div>
`);
const $ = (id) => box.querySelector("#" + id);

async function api(path, opt) {
  const r = await fetch(`/api/zones/${path}`, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
const send = (path, method, body) => api(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });

// ---------------------------------------------------------------- состояние
let zones = [], lines = [];
let live = { inside: {}, lines: {} };  // из frame.zones
let stats = null;  // из /stats: посещения, время
const group = new THREE.Group(); scene.add(group);
const objs = { zones: new Map(), lines: new Map() };  // id -> {fill, outline, label, ...}

function disposeObj(o) {
  o.traverse((c) => { c.geometry?.dispose(); c.material?.map?.dispose(); c.material?.dispose(); });
}

// ---------------------------------------------------------------- 3D: зоны и линии
function makeLabel(text, color) {
  const s = P.textSprite(text, color);
  s.scale.set(0.9, 0.225, 1);
  return s;
}

function buildZone(z) {
  const g = new THREE.Group();
  const shape = new THREE.Shape(z.points.map((p) => new THREE.Vector2(p[0], p[1])));
  const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: z.color, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
  fill.position.z = Z_ZONE; fill.renderOrder = -0.4;
  const pts = z.points.map((p) => new THREE.Vector3(p[0], p[1], Z_ZONE + 0.001));
  const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: z.color, depthWrite: false }));
  outline.renderOrder = -0.3;
  g.add(fill, outline);
  const cx = z.points.reduce((a, p) => a + p[0], 0) / z.points.length, cy = z.points.reduce((a, p) => a + p[1], 0) / z.points.length;
  g.userData = { fill, cx, cy, labelText: null, label: null };
  scene.add(g);
  return g;
}

function buildLine(l) {
  const g = new THREE.Group();
  const dx = l.b[0] - l.a[0], dy = l.b[1] - l.a[1], L = Math.hypot(dx, dy);
  const ribbon = new THREE.Mesh(new THREE.PlaneGeometry(L, 0.06), new THREE.MeshBasicMaterial({ color: l.color, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }));
  ribbon.position.set((l.a[0] + l.b[0]) / 2, (l.a[1] + l.b[1]) / 2, Z_LINE);
  ribbon.rotation.z = Math.atan2(dy, dx); ribbon.renderOrder = -0.2;
  // стрелка A→B: от центра к правой стороне вектора A→B
  const dir = new THREE.Vector3(dy / L, -dx / L, 0);
  const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3((l.a[0] + l.b[0]) / 2, (l.a[1] + l.b[1]) / 2, Z_LINE + 0.01), 0.45, new THREE.Color(l.color).getHex(), 0.14, 0.09);
  arrow.line.material.depthWrite = false; arrow.cone.material.depthWrite = false;
  g.add(ribbon, arrow);
  g.userData = { mx: (l.a[0] + l.b[0]) / 2, my: (l.a[1] + l.b[1]) / 2, dir, label: null, labelText: null };
  scene.add(g);
  return g;
}

function setLabel(g, text, color, x, y, z) {
  const u = g.userData;
  if (u.labelText === text) return;
  if (u.label) { g.remove(u.label); disposeObj(u.label); }
  u.label = makeLabel(text, color); u.labelText = text;
  u.label.position.set(x, y, z);
  g.add(u.label);
}

function syncScene() {
  for (const [m, list, build] of [[objs.zones, zones, buildZone], [objs.lines, lines, buildLine]]) {
    const ids = new Set(list.map((x) => x.id));
    for (const [id, g] of m) if (!ids.has(id)) { scene.remove(g); disposeObj(g); m.delete(id); }
    for (const it of list) {
      const key = JSON.stringify(it);
      const cur = m.get(it.id);
      if (cur && cur.userData.key === key) continue;
      if (cur) { scene.remove(cur); disposeObj(cur); }
      const g = build(it);
      g.userData.key = key;
      m.set(it.id, g);
    }
  }
  updateLive();
}

function updateLive() {
  const show = $("znShow").checked;
  for (const z of zones) {
    const g = objs.zones.get(z.id);
    if (!g) continue;
    g.visible = show;
    const n = (live.inside[z.id] || []).length;
    g.userData.fill.material.opacity = n ? 0.38 : 0.16;
    setLabel(g, n ? `${z.name}: ${n}` : z.name, z.color, g.userData.cx, g.userData.cy, 0.25);
  }
  for (const l of lines) {
    const g = objs.lines.get(l.id);
    if (!g) continue;
    g.visible = show;
    const c = live.lines[l.id] || { a_to_b: 0, b_to_a: 0 };
    setLabel(g, `${l.name} A→B ${c.a_to_b} · B→A ${c.b_to_a}`, l.color, g.userData.mx + g.userData.dir.x * 0.75, g.userData.my + g.userData.dir.y * 0.75, 0.25);
  }
}
$("znShow").onchange = updateLive;

// ---------------------------------------------------------------- список и счётчики
function renderList() {
  const st = new Map((stats?.zones || []).map((s) => [s.id, s]));
  const rows = zones.map((z) => {
    const s = st.get(z.id) || {};
    const inside = (live.inside[z.id] || []).length;
    return `<div class="zn-item"><span class="zn-sw" style="background:${esc(z.color)}"></span><span class="zn-name" title="${esc(z.name)}">${esc(z.name)}</span>
      <span title="сейчас внутри / всего входов / среднее время">${inside} / ${s.visits ?? 0} / ${s.mean_dwell_s != null ? fmt(s.mean_dwell_s) + " с" : "–"}</span>
      <button data-del-zone="${esc(z.id)}" title="Удалить зону">✕</button></div>`;
  });
  const lrows = lines.map((l) => {
    const c = live.lines[l.id] || { a_to_b: 0, b_to_a: 0 };
    return `<div class="zn-item"><span class="zn-sw" style="background:${esc(l.color)}"></span><span class="zn-name" title="${esc(l.name)}">${esc(l.name)}</span>
      <span title="пересечений A→B / B→A">A→B ${c.a_to_b} · B→A ${c.b_to_a}</span>
      <button data-del-line="${esc(l.id)}" title="Удалить линию">✕</button></div>`;
  });
  $("znList").innerHTML = rows.length || lrows.length
    ? `<div class="small" style="margin-top:6px">зона: внутри / входов / среднее время</div>` + rows.join("") + lrows.join("")
    : `<div class="small" style="margin-top:6px">зон и линий нет</div>`;
}
$("znList").onclick = async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  try {
    if (b.dataset.delZone) await api(`zone/${b.dataset.delZone}`, { method: "DELETE" });
    else if (b.dataset.delLine) await api(`line/${b.dataset.delLine}`, { method: "DELETE" });
  } catch (err) { log(`зоны: ${err.message}`); }
};

async function loadAll() {
  try {
    const d = await api("");
    zones = d.zones; lines = d.lines;
    syncScene(); renderList();
  } catch (e) { log(`зоны: ${e.message}`); }
}

async function loadStats() {
  try { stats = await api("stats"); renderList(); } catch { /* сервер без расширения или недоступен */ }
}

// ---------------------------------------------------------------- лента событий
const feed = [];
const who = (e) => (e.kind === "person" ? `ID ${e.id}` : `${e.kind} ${e.id}`);
const clock = (t) => new Date(t * 1000).toLocaleTimeString("ru-RU");
function eventText(e) {
  const name = `«${esc(e.name)}»`;
  if (e.type === "enter") return `<b>${who(e)}</b> вошёл в ${name}`;
  if (e.type === "exit") return `<b>${who(e)}</b> вышел из ${name}, пробыл ${fmt(e.dwell_s)} с${e.reason === "lost" ? " (трек пропал)" : e.reason ? ` (${esc(e.reason)})` : ""}`;
  return `<b>${who(e)}</b> пересёк ${name} ${esc(e.direction)}`;
}
function pushEvents(evs) {
  if (!evs.length) return;
  for (const e of evs) feed.push(`<div>${clock(e.t)} ${eventText(e)}</div>`);
  while (feed.length > 40) feed.shift();
  const el = $("znFeed");
  const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
  el.innerHTML = feed.join("");
  if (atEnd) el.scrollTop = el.scrollHeight;
  loadStats();
}

P.on("frame", (m) => {
  if (!m.zones) { if (Object.keys(live.inside).length || Object.keys(live.lines).length) { live = { inside: {}, lines: {} }; updateLive(); renderList(); } return; }
  const prev = JSON.stringify(live);
  live = { inside: m.zones.inside || {}, lines: m.zones.lines || {} };
  if (JSON.stringify(live) !== prev) { updateLive(); renderList(); }
  pushEvents(m.zones.events || []);
});
P.on("zones", (m) => { zones = m.zones; lines = m.lines; syncScene(); renderList(); loadStats(); });
P.on("hello", () => loadAll());
$("znResetStats").onclick = async () => {
  await send("reset", "POST");
  feed.length = 0; $("znFeed").innerHTML = ""; loadStats();
};

// ---------------------------------------------------------------- рисование мышью
const canvas = document.querySelector("#view canvas");
const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const ray = new THREE.Raycaster();
let mode = null, p1 = null, preview = null;

function floorPoint(ev) {
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, P.camera);
  const out = new THREE.Vector3();
  return ray.ray.intersectPlane(plane, out) ? [Math.round(out.x * 100) / 100, Math.round(out.y * 100) / 100] : null;
}

function clearPreview() {
  if (preview) { scene.remove(preview); disposeObj(preview); preview = null; }
}
function setMode(m) {
  mode = m; p1 = null; clearPreview();
  $("znDrawZone").classList.toggle("primary", m === "zone");
  $("znDrawLine").classList.toggle("primary", m === "line");
  $("znDrawZone").textContent = m === "zone" ? "Отмена" : "Нарисовать зону";
  $("znDrawLine").textContent = m === "line" ? "Отмена" : "Нарисовать линию";
  canvas.style.cursor = m ? "crosshair" : "";
  $("znHint").textContent = m ? (m === "zone" ? "Кликните по полу: первый угол прямоугольника (Esc — отмена)" : "Кликните по полу: начало линии (Esc — отмена)") : "";
}
$("znDrawZone").onclick = () => setMode(mode === "zone" ? null : "zone");
$("znDrawLine").onclick = () => setMode(mode === "line" ? null : "line");
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && mode) setMode(null); });

function drawPreview(a, b) {
  clearPreview();
  const pts = mode === "zone"
    ? [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]], [a[0], a[1]]]
    : [a, b];
  preview = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(p[0], p[1], Z_LINE + 0.02))),
    new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
  preview.renderOrder = 10;
  scene.add(preview);
}

let down = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener("pointermove", (e) => {
  if (!mode || !p1) return;
  const q = floorPoint(e);
  if (q) drawPreview(p1, q);
});
canvas.addEventListener("pointerup", async (e) => {
  if (!mode || !down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;  // перетаскивание = вращение камеры, не клик
  const q = floorPoint(e);
  if (!q) return;
  if (!p1) {
    p1 = q;
    $("znHint").textContent = mode === "zone" ? "Теперь противоположный угол" : "Теперь конец линии";
    return;
  }
  const a = p1, kind = mode, name = $("znName").value.trim();
  setMode(null);
  try {
    if (kind === "zone") await send("zone", "POST", { rect: [a[0], a[1], q[0], q[1]], ...(name && { name }) });
    else await send("line", "POST", { a, b: q, ...(name && { name }) });
    $("znName").value = "";
  } catch (err) { log(`зоны: ${err.message}`); $("znHint").textContent = err.message; }
});

// ---------------------------------------------------------------- тепловая карта
let heatMesh = null, heatSource = "live", recHeat = null, heatTimer = null;

function heatColor(t) {  // t 0..1 -> RGBA: прозрачный → синий → жёлтый → красный
  if (t <= 0) return [0, 0, 0, 0];
  const lerp = (a, b, k) => a + (b - a) * k;
  if (t < 0.4) { const k = t / 0.4; return [40, 90, 255, Math.round(lerp(0, 170, k))]; }
  if (t < 0.75) { const k = (t - 0.4) / 0.35; return [Math.round(lerp(40, 255, k)), Math.round(lerp(90, 220, k)), Math.round(lerp(255, 40, k)), Math.round(lerp(170, 200, k))]; }
  const k = (t - 0.75) / 0.25;
  return [255, Math.round(lerp(220, 40, k)), Math.round(lerp(40, 20, k)), Math.round(lerp(200, 225, k))];
}

function clearHeat() {
  if (!heatMesh) return;
  scene.remove(heatMesh); disposeObj(heatMesh); heatMesh = null;
}

function drawHeat(h) {
  clearHeat();
  if (!h.nx || !h.max) { $("znHeatInfo").textContent = "карта пока пуста"; return; }
  const data = new Uint8Array(h.nx * h.ny * 4);
  for (let i = 0; i < h.nx * h.ny; i++) data.set(heatColor(Math.pow(h.values[i] / h.max, 0.6)), i * 4);  // степень <1 — малые значения заметнее
  const tex = new THREE.DataTexture(data, h.nx, h.ny, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter; tex.needsUpdate = true;
  heatMesh = new THREE.Mesh(new THREE.PlaneGeometry(h.nx * h.cell_m, h.ny * h.cell_m),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  heatMesh.position.set(h.origin[0] + (h.nx * h.cell_m) / 2, h.origin[1] + (h.ny * h.cell_m) / 2, Z_HEAT);
  heatMesh.renderOrder = -0.5;
  scene.add(heatMesh);
  $("znHeatInfo").textContent = `${heatSource === "live" ? "карта: онлайн" : "карта: запись"}, шаг ${h.cell_m} м, максимум ${fmt(h.max)} с в клетке, всего ${fmt(h.total_s, 0)} с`;
}

async function refreshHeat() {
  if (!$("znHeat").checked) return;
  try {
    if (heatSource === "live") drawHeat(await api("heatmap"));
    else if (recHeat) drawHeat(recHeat);
  } catch (e) { log(`зоны: ${e.message}`); }
}
function setHeat(on) {
  clearInterval(heatTimer); heatTimer = null;
  if (on) { refreshHeat(); heatTimer = setInterval(refreshHeat, 1500); } else { clearHeat(); $("znHeatInfo").textContent = ""; }
}
$("znHeat").onchange = () => { heatSource = "live"; setHeat($("znHeat").checked); };
$("znHeatReset").onclick = async () => { await send("heatmap/reset", "POST"); heatSource = "live"; refreshHeat(); };

// ---------------------------------------------------------------- анализ записи
async function loadSessions() {
  try {
    const list = await api("sessions");
    const sel = $("znSess"), cur = sel.value;
    sel.innerHTML = list.length ? list.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} (${fmt(s.size / 1048576, 1)} МБ)</option>`).join("") : `<option value="">— нет записей —</option>`;
    if (cur) sel.value = cur;
  } catch (e) { log(`зоны: ${e.message}`); }
}
$("znSessRefresh").onclick = loadSessions;
$("znAnalyze").onclick = async () => {
  const name = $("znSess").value;
  if (!name) return;
  const btn = $("znAnalyze");
  btn.disabled = true; btn.textContent = "Анализ…";
  try {
    const r = await api(`analyze?session=${encodeURIComponent(name)}`);
    recHeat = r.heatmap;
    const zr = r.summary.zones.map((z) => `<tr><td><span class="zn-sw" style="background:${esc(z.color)}"></span>${esc(z.name)}</td><td>${z.visits}</td><td>${fmt(z.total_dwell_s)}</td><td>${z.mean_dwell_s != null ? fmt(z.mean_dwell_s) : "–"}</td></tr>`).join("");
    const lr = r.summary.lines.map((l) => `<tr><td><span class="zn-sw" style="background:${esc(l.color)}"></span>${esc(l.name)}</td><td colspan="3">A→B ${l.a_to_b} · B→A ${l.b_to_a}</td></tr>`).join("");
    $("znAnalysis").innerHTML = `${r.frames} кадров, ${fmt(r.duration_s, 0)} с, событий ${r.events_total}` +
      (zr || lr ? `<table class="zn-table"><tr><th></th><th>входов</th><th>Σ с</th><th>ср. с</th></tr>${zr}${lr}</table>` : "<br>зон и линий нет — нарисуйте их и повторите") +
      `<br><a href="/api/zones/analyze.csv?session=${encodeURIComponent(name)}" style="color:var(--text)">события CSV</a>`;
  } catch (e) { $("znAnalysis").innerHTML = `<span style="color:var(--bad)">${esc(e.message)}</span>`; }
  finally { btn.disabled = false; btn.textContent = "Анализировать"; }
};
$("znShowRec").onclick = () => {
  if (!recHeat) return;
  heatSource = "rec"; $("znHeat").checked = true; setHeat(true);
};

// ---------------------------------------------------------------- старт
loadAll(); loadSessions(); loadStats();
setInterval(loadStats, 3000);
api("events?limit=20").then((d) => { pushEvents(d.events); }).catch(() => {});
