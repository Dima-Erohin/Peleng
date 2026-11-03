// Расширение «Карта покрытия»: какие участки пола видят ≥ 2 / 1 / 0 камер — помощь при расстановке камер.
// Сервер: peleng/ext/coverage.py (GET /api/coverage), описание: docs/coverage.md.
const { THREE, scene, log } = window.peleng;

const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SEL = 'style="font: inherit; background:#0d1117; color: var(--text); border:1px solid var(--line); border-radius:6px"';

const box = window.peleng.addSection("Покрытие");
box.insertAdjacentHTML("beforeend", `
  <div class="btns" style="align-items:center">
    <button id="covBtn" title="Карта на полу: сколько откалиброванных камер видят каждый участок">Показать покрытие</button>
    <select id="covCell" ${SEL} title="Шаг сетки"><option value="0.1">0.1 м</option><option value="0.2" selected>0.2 м</option><option value="0.25">0.25 м</option><option value="0.5">0.5 м</option></select>
    <select id="covH" ${SEL} title="На какой высоте точка должна быть видна"><option value="0.9" selected>таз 0.9 м</option><option value="0">пол</option><option value="1.6">голова 1.6 м</option></select>
  </div>
  <div class="small" id="covLegend" style="margin-top:6px; display:none">
    <span style="display:inline-block;width:10px;height:10px;background:#2ea043;border-radius:2px"></span> ≥ 2 камер (ярче — точнее)
    <span style="display:inline-block;width:10px;height:10px;background:#f58c32;border-radius:2px;margin-left:6px"></span> 1
    <span style="display:inline-block;width:10px;height:10px;background:#963232;border-radius:2px;margin-left:6px"></span> 0
  </div>
  <div class="small" id="covInfo" style="margin-top:4px"></div>
`);
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- слой на полу
// Одна плоскость с текстурой nx×ny (клетка = тексель, без сглаживания), чуть выше пола, не пишет в буфер глубины.
let mesh = null;
let shown = false;

function colorOf(n, q) {
  if (n <= 0) return [150, 50, 50, 90];  // мёртвая зона — тусклый красный
  if (n === 1) return [245, 140, 50, 175];  // одна камера — только «ноги на полу»
  const t = Math.max(0, Math.min(1, q ?? 0));  // ≥ 2: от блёкло-оливкового (σ большая) к насыщенному зелёному
  return [Math.round(150 - 115 * t), Math.round(160 + 60 * t), Math.round(90 - 30 * t), Math.round(120 + 70 * t)];
}

function draw(r) {
  clear();
  const { nx, ny, cell_m: c, origin } = r;
  const data = new Uint8Array(nx * ny * 4);
  for (let i = 0; i < nx * ny; i++) data.set(colorOf(r.counts[i], r.quality[i]), i * 4);
  const tex = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat);  // строка 0 — клетки с наименьшим y (низ плоскости)
  tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  mesh = new THREE.Mesh(new THREE.PlaneGeometry(nx * c, ny * c), mat);  // PlaneGeometry лежит в XY — это и есть пол (Z вверх)
  mesh.position.set(origin[0] + nx * c / 2, origin[1] + ny * c / 2, 0.004);
  mesh.renderOrder = -1;  // раньше людей и следов: они рисуются поверх карты
  scene.add(mesh);
}

function clear() {
  if (!mesh) return;
  scene.remove(mesh);
  mesh.geometry.dispose(); mesh.material.map.dispose(); mesh.material.dispose();
  mesh = null;
}

function summary(r) {
  const s = r.summary;
  const pct = (a) => (s.area_m2 ? Math.round((100 * a) / s.area_m2) : 0);
  const cams = r.cameras.map((c) => `${esc(c.id)} ${fmt(c.area_m2)} м²`).join(", ");
  const warn = (r.warnings || []).map((w) => `<div style="color:var(--warn)">⚠ ${esc(w)}</div>`).join("");
  return `<b style="color:var(--acc)">≥ 2 камер: ${fmt(s.area_2plus_m2)} м² (${pct(s.area_2plus_m2)} %)</b>` +
    ` · 1: ${fmt(s.area_1_m2)} м² · 0: ${fmt(s.area_0_m2)} м²<br>` +
    (s.sigma_h_cm_median != null ? `σ по горизонтали (≥ 2 камер, ${fmt(r.params.sigma_px)} px): медиана ±${fmt(s.sigma_h_cm_median)} см, 90 % ≤ ±${fmt(s.sigma_h_cm_p90)} см; точных (σ ≤ ~3 см) ${fmt(s.area_good_m2)} м²<br>` : "") +
    `область ${fmt(r.nx * r.cell_m)}×${fmt(r.ny * r.cell_m)} м (${r.bounds_from === "room" ? "комната" : "по камерам"}), шаг ${r.cell_m} м, высота ${r.height_m} м<br>` +
    (cams ? `видят: ${cams}` : "") + warn;
}

async function refresh() {
  const btn = $("covBtn");
  btn.disabled = true;
  try {
    const q = new URLSearchParams({ cell_m: $("covCell").value, height_m: $("covH").value });
    const res = await fetch(`/api/coverage?${q}`);
    const r = await res.json();
    if (!res.ok) throw new Error(r.detail || res.statusText);
    if (!shown) return;  // пока считали, карту выключили
    draw(r);
    $("covInfo").innerHTML = summary(r);
  } catch (e) {
    log(`покрытие: ${e.message}`);
    $("covInfo").innerHTML = `<span style="color:var(--bad)">${esc(e.message)}</span>`;
  } finally { btn.disabled = false; }
}

function setShown(v) {
  shown = v;
  $("covBtn").textContent = v ? "Скрыть покрытие" : "Показать покрытие";
  $("covBtn").classList.toggle("primary", v);
  $("covLegend").style.display = v ? "" : "none";
  if (v) refresh(); else { clear(); $("covInfo").innerHTML = ""; }
}
$("covBtn").onclick = () => setShown(!shown);
$("covCell").onchange = $("covH").onchange = () => shown && refresh();

// калибровка изменилась → пересчитать (сообщение "cameras" приходит и при подключении/отключении телефона — сверяем положения)
let calibSig = "";
const sigOf = (list) => (list || []).filter((c) => c.calibrated).map((c) => `${c.id}:${(c.t || []).map((v) => v.toFixed(3))}:${c.hfov_deg}`).sort().join("|");
function onCameras(m) {
  const sig = sigOf(m.cameras);
  if (sig === calibSig) return;
  calibSig = sig;
  if (shown) refresh();
}
window.peleng.on("hello", onCameras);
window.peleng.on("cameras", onCameras);
