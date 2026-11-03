// Расширение «Калибровка цепочкой»: положение камер, когда доску не видят все камеры сразу.
// Сервер: peleng/ext/chain.py (REST /api/chain/...), описание: docs/chain-calibration.md.
const { THREE, scene, log } = window.peleng;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const TBL = 'style="width:100%; border-collapse:collapse; font-size:12px; margin-top:6px"';

const box = window.peleng.addSection("Калибровка цепочкой");
box.insertAdjacentHTML("beforeend", `
  <div class="small">Если доску на полу видят не все камеры: первый снимок — доска на полу, дальше переносите доску так,
    чтобы её одновременно видели ≥ 2 камеры (одна из них уже привязана).</div>
  <div class="btns" style="margin-top:6px">
    <button id="chStart" title="Начать заново: прежние снимки цепочки забываются, калибровка камер пока не меняется">Начать</button>
    <button id="chSnap" class="primary" title="Около 1 с кадров со всех камер; доску держать неподвижно">Снимок</button>
    <button id="chSolve" title="Посчитать без сохранения: ошибки и предупреждения">Рассчитать</button>
    <button id="chApply" title="Сохранить положения всех связанных камер">Применить</button>
    <button id="chCancel" title="Выйти, ничего не меняя">Отмена</button>
  </div>
  <label class="chk small"><input type="checkbox" id="chFloor"> доска лежит на полу (уточняет наклон пола; достаточно одной камеры)</label>
  <label class="chk small"><input type="checkbox" id="chAuto"> автоснимок (доску видят ≥ 2 камеры, она неподвижна и перенесена)</label>
  <label class="chk small"><input type="checkbox" id="chShow" checked> позы доски в 3D</label>
  <div class="small" id="chStatus" style="margin-top:4px"></div>
  <div class="small" id="chHints" style="margin-top:4px"></div>
  <div class="small" id="chCams"></div>
  <div class="small" id="chSnaps"></div>
  <div class="small" id="chResult" style="margin-top:6px"></div>
`);
const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const r = await fetch(`/api/chain${path}`, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
async function busy(btn, fn) {
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = old + " …";
  try { await fn(); } catch (e) { log(`цепочка: ${e.message}`); } finally { btn.textContent = old; render(state); }
}

let state = null;

// ---------------------------------------------------------------- отрисовка панели
function render(s) {
  if (!s) return;
  state = s;
  const g = s.graph || { cams: {}, pairs: [], hints: [] };
  const res = s.result;
  $("chSnap").disabled = !s.active || s.busy;
  $("chSolve").disabled = !s.active || !s.snaps.length;
  $("chApply").disabled = !s.active || !s.snaps.length;
  $("chCancel").disabled = !s.active;
  $("chAuto").disabled = !s.active || !s.snaps.length;
  $("chAuto").checked = !!s.auto;
  if (!s.snaps.length) $("chFloor").checked = false;
  $("chFloor").disabled = !s.active || !s.snaps.length;  // первый снимок и так «на полу»

  const nLinked = (g.linked || []).length, nAll = Object.keys(g.cams).length;
  $("chStatus").innerHTML = s.active
    ? `<b>идёт цепочка</b>: снимков ${s.snaps.length}, привязано камер ${nLinked} из ${nAll}${s.busy ? " · снимаю…" : ""}${s.auto ? " · автоснимок" : ""}`
    : s.applied
      ? `<span class="ok">применено ${new Date(s.applied.time * 1000).toLocaleTimeString()}: ${s.applied.cameras.map(esc).join(", ")}, ошибка ${fmt(s.applied.reproj_px, 2)} px${s.applied.saved ? "" : " (не сохранено на диск: демо/запись)"}</span>`
      : "не начато";
  $("chHints").innerHTML = (g.hints || []).map((h) => `<div>→ ${esc(h)}</div>`).join("");

  // камеры: привязка, снимки, итог расчёта
  const camRows = Object.entries(g.cams).map(([c, x]) => {
    const r = res && res.cams ? res.cams[c] : null;
    const link = x.linked ? `<span class="ok">✓ ${x.hops === 1 ? "к полу" : x.hops + "-е звено"}</span>` : `<span class="bad">не связана</span>`;
    const err = r ? `<span class="${r.reproj_px > 1 ? "warn" : ""}">${fmt(r.reproj_px, 2)} px</span>` : "";
    const std = r && r.center_std_norm_m != null ? `±${fmt(r.center_std_norm_m * 100)} см` : "";
    return `<tr><td>${esc(c)}</td><td>${link}</td><td>${x.snapshots}</td><td>${err}</td><td>${std}</td></tr>`;
  }).join("");
  const pairs = (g.pairs || []).map((p) => `<span style="white-space:nowrap; color:${p.n >= 2 ? "inherit" : "var(--warn)"}">${esc(p.a)}–${esc(p.b)}: ${p.n}</span>`).join(" · ");
  $("chCams").innerHTML = nAll
    ? `<table ${TBL}><tr style="color:var(--muted)"><td>камера</td><td>привязка</td><td>снимков</td><td>ошибка</td><td>σ центра</td></tr>${camRows}</table>` +
      (pairs ? `<div style="margin-top:4px">общих снимков у пар: ${pairs}</div>` : "")
    : "";

  // снимки
  const snaps = s.snaps.slice().reverse().map((sn) => {
    const cams = Object.entries(sn.cams).map(([c, x]) => `${esc(c)} (${x.corners})`).join(", ");
    const skipped = Object.entries(sn.skipped || {}).map(([c, m]) => `${esc(c)}: ${esc(m)}`).join("; ");
    const sr = res && res.snaps ? res.snaps[String(sn.k)] : null;
    const tag = sn.anchor ? "опорный, на полу" : sn.floor ? "на полу" : "";
    const del = sn.k !== 0 && s.active ? ` <a href="#" data-del="${sn.k}" title="Убрать снимок" style="color:var(--bad); text-decoration:none">✕</a>` : "";
    return `<div>#${sn.k}${tag ? ` <i>${tag}</i>` : ""}: ${cams}${sr ? ` · ${fmt(sr.reproj_px, 2)} px` : ""}${del}` +
      (skipped ? `<br><span style="color:var(--warn)">&nbsp;&nbsp;${skipped}</span>` : "") + `</div>`;
  }).join("");
  $("chSnaps").innerHTML = snaps ? `<div style="margin-top:6px">${snaps}</div>` : "";
  for (const a of $("chSnaps").querySelectorAll("[data-del]")) {
    a.onclick = async (ev) => { ev.preventDefault(); try { render(await api(`/snapshot/${a.dataset.del}/delete`, {})); } catch (e) { log(`цепочка: ${e.message}`); } };
  }

  // итог расчёта
  if (res) {
    const drop = (res.dropped || []).map((d) => `#${d.snap} ${esc(d.cam)}${d.reproj_px != null ? ` (${fmt(d.reproj_px, 1)} px)` : ""}: ${esc(d.reason)}`).join("<br>");
    const warn = (res.warnings || []).map((w) => `<div style="color:var(--warn)">⚠ ${esc(w)}</div>`).join("");
    $("chResult").innerHTML = res.ok
      ? `<b>ошибка репроекции ${fmt(res.reproj_px, 2)} px</b> (чистая цепочка ${fmt(res.chain_reproj_px, 2)} px)` +
        (drop ? `<br>отброшено:<br>${drop}` : "") + warn
      : `<span class="bad">${esc(res.message)}</span>${warn}`;
  } else {
    $("chResult").innerHTML = "";
  }
  drawBoards(res);
}

// ---------------------------------------------------------------- позы доски в 3D
const group = new THREE.Group();
scene.add(group);
function drawBoards(res) {
  while (group.children.length) {
    const o = group.children.pop();
    o.geometry?.dispose();
    o.material?.map?.dispose();
    o.material?.dispose();
  }
  if (!res || !res.ok || !$("chShow").checked) return;
  for (const [k, sn] of Object.entries(res.snaps || {})) {
    const pts = sn.corners_world.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const color = k === "0" ? 0x2ea043 : 0xf0a030;
    group.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 })));
    const label = window.peleng.textSprite(`#${k}`, k === "0" ? "#2ea043" : "#f0a030");
    const c = pts.reduce((a, p) => a.add(p), new THREE.Vector3()).multiplyScalar(0.25);
    label.position.set(c.x, c.y, c.z + 0.12);
    label.scale.set(0.44, 0.11, 1);
    group.add(label);
  }
}
$("chShow").onchange = () => drawBoards(state && state.result);

// ---------------------------------------------------------------- кнопки
$("chStart").onclick = () => busy($("chStart"), async () => {
  if (state && state.active && state.snaps.length && !confirm("Начать заново? Снимки текущей цепочки будут забыты.")) return;
  render(await api("/start", {}));
  log("цепочка: начато — положите доску на пол (её должна видеть хотя бы одна камера) и нажмите «Снимок»");
});
$("chSnap").onclick = () => busy($("chSnap"), async () => {
  const r = await api("/snapshot", { floor: $("chFloor").checked });
  if (r.ok) {
    log(`цепочка: ✓ ${r.message}`);
    $("chFloor").checked = false;
  } else {
    const sk = Object.entries(r.skipped || {}).map(([c, m]) => `${c}: ${m}`).join("; ");
    log(`цепочка: ✗ ${r.message}${sk ? " (" + sk + ")" : ""}`);
  }
});
$("chSolve").onclick = () => busy($("chSolve"), async () => {
  const r = await api("/solve", {});
  log(r.ok ? `цепочка: расчёт — ошибка ${fmt(r.reproj_px, 2)} px, камер ${Object.keys(r.cams).length}, предупреждений ${r.warnings.length}` : `цепочка: ✗ ${r.message}`);
});
$("chApply").onclick = () => busy($("chApply"), async () => {
  const r = await api("/apply", {});
  if (!r.ok) { log(`цепочка: ✗ ${r.message}`); return; }
  for (const [c, x] of Object.entries(r.cameras || {})) {
    log(x.ok ? `${c}: ✓ цепочкой: ошибка ${fmt(x.reproj_px, 2)} px, звено ${x.hops}, до начала координат ${x.dist_to_origin_m} м${x.center_std_norm_m != null ? `, ±${fmt(x.center_std_norm_m * 100)} см` : ""}` : `${c}: ✗ ${x.message}`);
  }
  for (const w of r.warnings || []) log(`цепочка: ⚠ ${w}`);
});
$("chCancel").onclick = () => busy($("chCancel"), async () => {
  render(await api("/cancel", {}));
  log("цепочка: отменено, калибровка камер не менялась");
});
$("chAuto").onchange = async () => {
  try { render(await api("/auto", { on: $("chAuto").checked })); } catch (e) { log(`цепочка: ${e.message}`); $("chAuto").checked = false; }
};

window.peleng.on("chain", render);
window.peleng.on("hello", () => api("").then(render).catch(() => {}));
api("").then(render).catch(() => {});
