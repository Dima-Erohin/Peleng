// Блок «Траектории»: записанная сессия → восстановление (сшивка, сглаживание вперёд-назад, заполнение пропусков)
// → гладкие линии в 3D поверх сырых, метрики по трекам, скачивание CSV/JSON. Сервер: peleng/ext/trajectory.py.
const P = window.peleng;
const THREE = P.THREE;

const box = P.addSection("Траектории");
box.insertAdjacentHTML("beforeend", `
  <style>
    .tj-row { display: flex; gap: 6px; align-items: center; margin: 4px 0; flex-wrap: wrap; }
    .tj-row select { flex: 1; min-width: 0; font: inherit; background: #0d1117; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px; }
    .tj-table { width: 100%; border-collapse: collapse; font: 12px ui-monospace, monospace; margin-top: 6px; }
    .tj-table th, .tj-table td { padding: 2px 4px; text-align: right; border-bottom: 1px solid var(--line); }
    .tj-table th:first-child, .tj-table td:first-child { text-align: left; }
    .tj-sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
    .tj-dl a { color: var(--text); }
  </style>
  <div class="tj-row"><select id="tjSel"><option value="">— нет записей —</option></select><button id="tjRefresh" title="Обновить список">⟳</button></div>
  <div class="tj-row small">
    пропуск ≤ <input type="number" id="tjGap" value="1.5" min="0" max="10" step="0.1"> с,
    сетка <input type="number" id="tjRate" value="30" min="1" max="240" step="1"> Гц
  </div>
  <div class="btns"><button class="primary" id="tjRun">Восстановить</button><button id="tjHide">Скрыть</button></div>
  <label class="chk"><input type="checkbox" id="tjRaw" checked> показывать сырые точки</label>
  <div id="tjInfo" class="small"></div>
  <div id="tjTable"></div>
  <div id="tjDl" class="small tj-dl"></div>
`);
const $ = (id) => box.querySelector("#" + id);

const group = new THREE.Group(); group.visible = false; P.scene.add(group);
const rawGroup = new THREE.Group(); group.add(rawGroup);
let current = null;  // {session, max_gap_s, rate_hz}

function clearGroup() {
  for (const g of [group, rawGroup]) {
    for (const o of [...g.children]) {
      if (o === rawGroup) continue;
      g.remove(o);
      o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose();
    }
  }
}

function fmtSize(n) { return n > 1e6 ? (n / 1e6).toFixed(1) + " МБ" : (n / 1e3).toFixed(0) + " КБ"; }

async function refresh() {
  const list = await (await fetch("/api/trajectory/sessions")).json();
  const sel = $("tjSel"), prev = sel.value;
  sel.innerHTML = list.length ? "" : `<option value="">— нет записей —</option>`;
  for (const s of list) {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = `${s.name.replace(/^session_|\.jsonl$/g, "")} · ${fmtSize(s.size)}${s.recording ? " · идёт запись" : ""}`;
    sel.appendChild(o);
  }
  if (list.some(s => s.name === prev)) sel.value = prev;
}

function line(points, colors, opts) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(points.flat(), 3));
  if (colors) g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return new THREE.Line(g, new THREE.LineBasicMaterial(opts));
}

// разрывы (пропуск больше max_gap_s) не соединяем линией
function splitByGap(t, pos, extra, gap) {
  const parts = []; let cur = null;
  for (let i = 0; i < t.length; i++) {
    if (!cur || t[i] - t[i - 1] > gap) { cur = { pos: [], extra: [] }; parts.push(cur); }
    cur.pos.push(pos[i]); cur.extra.push(extra ? extra[i] : null);
  }
  return parts;
}

function draw(r) {
  clearGroup();
  const white = new THREE.Color(0xffffff);
  for (const tr of r.raw) {
    const c = new THREE.Color(P.colorFor(tr.id));
    for (const part of splitByGap(tr.t, tr.pos, null, r.params.max_gap_s + 0.05)) {
      if (part.pos.length < 2) continue;
      rawGroup.add(line(part.pos, null, { color: c, transparent: true, opacity: 0.25, depthWrite: false }));
    }
  }
  for (const tr of r.restored) {
    const c = new THREE.Color(P.colorFor(tr.id));
    for (const part of splitByGap(tr.t, tr.pos, tr.source, 1.5 / r.params.rate_hz + 1e-3)) {
      if (part.pos.length < 2) continue;
      // заполненные пропуски — белым
      const cols = part.extra.flatMap(s => (s === "interpolated" ? white : c).toArray());
      group.add(line(part.pos, cols, { vertexColors: true }));
    }
    const last = tr.pos[tr.pos.length - 1];
    const label = P.textSprite(`${tr.kind === "person" ? "ID" : tr.kind} ${tr.id}`, "#" + new THREE.Color(P.colorFor(tr.id)).getHexString());
    label.position.set(last[0], last[1], last[2] + 0.25);
    group.add(label);
  }
  rawGroup.visible = $("tjRaw").checked;
  group.visible = true;
}

function table(r) {
  const rows = r.items.map(it => `
    <tr title="ID записи: ${it.merged_ids.join(", ")}; измерений ${it.points_measured}, прогнозных ${it.points_predicted}; путь по сырым точкам ${it.path_raw_m} м">
      <td><span class="tj-sw" style="background:#${new THREE.Color(P.colorFor(it.id)).getHexString()}"></span>${it.kind === "person" ? "" : it.kind + " "}${it.id}${it.merged_ids.length > 1 ? ` (+${it.merged_ids.length - 1})` : ""}</td>
      <td>${it.duration_s.toFixed(1)}</td><td>${it.path_m.toFixed(2)}</td><td>${it.speed_mean_mps.toFixed(2)}</td>
      <td>${it.speed_max_mps.toFixed(2)}</td><td>${it.gaps_filled}/${it.gaps_filled_s.toFixed(1)}</td><td>${(it.std_mean_m * 100).toFixed(1)}</td>
    </tr>`).join("");
  $("tjTable").innerHTML = r.items.length ? `
    <table class="tj-table">
      <tr><th>трек</th><th title="длительность, с">с</th><th title="путь по сглаженной траектории, м">м</th><th title="средняя скорость, м/с">ср.</th>
          <th title="максимальная скорость, м/с">макс</th><th title="заполненные пропуски: число / секунд">проп.</th><th title="σ положения, см">σ см</th></tr>
      ${rows}
    </table>` : `<div class="small">треков нет</div>`;
}

function downloads() {
  const q = (fmt, data) => `/api/trajectory/export?session=${encodeURIComponent(current.session)}&fmt=${fmt}&data=${data}&max_gap_s=${current.max_gap_s}&rate_hz=${current.rate_hz}`;
  $("tjDl").innerHTML = `скачать: восстановленные <a href="${q("csv", "restored")}">CSV</a> · <a href="${q("json", "restored")}">JSON</a>;
    сырые <a href="${q("csv", "raw")}">CSV</a> · <a href="${q("json", "raw")}">JSON</a>`;
}

$("tjRun").onclick = async () => {
  const session = $("tjSel").value;
  if (!session) { P.log("траектории: нет выбранной записи"); return; }
  current = { session, max_gap_s: +$("tjGap").value || 1.5, rate_hz: +$("tjRate").value || 30 };
  $("tjInfo").textContent = "восстанавливаю…";
  const resp = await fetch(`/api/trajectory/restore?session=${encodeURIComponent(session)}&max_gap_s=${current.max_gap_s}&rate_hz=${current.rate_hz}`);
  const r = await resp.json();
  if (!resp.ok) { $("tjInfo").textContent = `ошибка: ${r.detail || resp.status}`; return; }
  const dur = r.t0 != null ? (r.t1 - r.t0).toFixed(1) : "0";
  $("tjInfo").textContent = `${dur} с записи: треков в записи ${r.raw_tracks}, после сшивки ${r.tracks}. Белым — заполненные пропуски, бледные — сырые.`;
  draw(r); table(r); downloads();
  P.log(`траектории: ${session} — ${r.tracks} трек(ов)`);
};
$("tjHide").onclick = () => { group.visible = false; };
$("tjRaw").onchange = () => { rawGroup.visible = $("tjRaw").checked; };
$("tjRefresh").onclick = refresh;
// новые записи появляются по кнопке «Записать сессию» — список обновляется сам (если им сейчас не пользуются)
P.on("hello", refresh);
setInterval(() => { if (document.activeElement !== $("tjSel")) refresh().catch(() => {}); }, 5000);
refresh();
