// Расширение «Объекты»: режим детекции, классы, счётчик; в 3D — сфера (YOLO) или куб (метка) цвета ID,
// подпись класса и след траектории. Данные — поле "objects" сообщения frame (docs/objects.md).
const { THREE, scene, colorFor, textSprite, log } = window.peleng;

const TRAIL_MAX = 600;
const box = window.peleng.addSection("Объекты");
box.innerHTML += `
  <div class="btns" style="align-items:center">
    <select id="objMode" style="font:inherit;background:#0d1117;color:inherit;border:1px solid #2a313c;border-radius:6px;padding:5px">
      <option value="off">выкл</option><option value="yolo">YOLO (классы COCO)</option><option value="aruco">метки ArUco</option>
    </select>
    <button id="objReset" title="Сбросить треки и следы объектов">Сбросить</button>
  </div>
  <div style="margin:6px 0">
    <input id="objClasses" list="objClassList" placeholder="классы через запятую; пусто — все, кроме person"
      style="width:100%;font:inherit;background:#0d1117;color:inherit;border:1px solid #2a313c;border-radius:6px;padding:4px">
    <datalist id="objClassList"></datalist>
  </div>
  <div class="small" id="objStatus">…</div>
  <div class="small" id="objList" style="font-family:ui-monospace,monospace;white-space:pre"></div>`;
const $ = (id) => box.querySelector("#" + id);

let st = null;
function showState(s) {
  st = s;
  $("objMode").value = s.mode;
  if (document.activeElement !== $("objClasses")) $("objClasses").value = s.classes.join(", ");
  $("objClasses").disabled = s.mode === "aruco";
  if (!$("objClassList").children.length) $("objClassList").innerHTML = s.all_classes.map(c => `<option value="${c}">`).join("");
  $("objStatus").textContent = `детектор: ${s.detector} · объектов: ${s.count}${s.mode !== "off" ? ` · ${s.proc_ms} мс` : ""}`;
}
async function api(body) {
  const r = await fetch("/api/objects/", body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!r.ok) { log(`объекты: ${r.status} ${await r.text()}`); return; }
  showState(await r.json());
}
$("objMode").onchange = () => { api({ mode: $("objMode").value }); log(`объекты: режим ${$("objMode").value}`); };
$("objClasses").onchange = () => api({ classes: $("objClasses").value });
$("objReset").onclick = () => { fetch("/api/objects/reset", { method: "POST" }); clearTrails(); };
api();
setInterval(() => api(), 2000);  // статус загрузки модели

// ---------------------------------------------------------------- 3D
const objs = {};  // id -> {mesh, label, trail, pts[], last}
function setLabel(s, text, color) {
  const c = s.userData.canvas, g = c.getContext("2d");
  g.clearRect(0, 0, 256, 64); g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, 256, 64);
  g.font = "bold 30px system-ui"; g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 10, 32);
  s.material.map.needsUpdate = true;
}
function makeObj(o) {
  const color = colorFor(o.id);
  const isTag = o.cls.startsWith("tag:");
  const geo = isTag ? new THREE.BoxGeometry(0.12, 0.12, 0.12) : new THREE.SphereGeometry(0.08, 20, 14);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, transparent: true }));
  const label = textSprite("", "#fff");
  label.scale.set(0.48, 0.12, 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(TRAIL_MAX * 3), 3));
  g.setDrawRange(0, 0);
  const trail = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }));
  scene.add(mesh); scene.add(label); scene.add(trail);
  return objs[o.id] = { mesh, label, trail, pts: [], last: 0, text: "" };
}
function clearTrails() {
  for (const o of Object.values(objs)) { o.pts = []; o.trail.geometry.setDrawRange(0, 0); }
}
function update(list) {
  const now = performance.now();
  for (const p of list) {
    const o = objs[p.id] || makeObj(p);
    o.last = now;
    o.mesh.visible = o.label.visible = true;
    o.mesh.position.set(...p.pos);
    o.mesh.material.opacity = p.predicted ? 0.35 : 1;
    o.label.position.set(p.pos[0], p.pos[1], p.pos[2] + 0.25);
    const text = `${p.cls} #${p.id}${p.mode === "mono" ? " · 1 кам" : ""}`;
    if (text !== o.text) { o.text = text; setLabel(o.label, text, "#fff"); }
    // след в 3D (объект может лететь)
    const last = o.pts[o.pts.length - 1];
    if (!p.predicted && (!last || Math.hypot(last[0] - p.pos[0], last[1] - p.pos[1], last[2] - p.pos[2]) > 0.01)) {
      o.pts.push(p.pos.slice());
      if (o.pts.length > TRAIL_MAX) o.pts.shift();
      const arr = o.trail.geometry.attributes.position.array;
      o.pts.forEach((q, i) => arr.set(q, i * 3));
      o.trail.geometry.attributes.position.needsUpdate = true;
      o.trail.geometry.setDrawRange(0, o.pts.length);
      o.trail.geometry.computeBoundingSphere();
    }
  }
  for (const o of Object.values(objs)) if (now - o.last > 500) o.mesh.visible = o.label.visible = false;
  $("objList").textContent = list.map(p =>
    `#${p.id} ${p.cls}: (${p.pos.map(v => v.toFixed(2)).join(", ")}) ${p.speed.toFixed(2)} м/с ${p.mode}${p.reproj_err_px != null ? " ±" + p.reproj_err_px + "px" : ""}`).join("\n");
  if (st && st.mode !== "off") $("objStatus").textContent = `детектор: ${st.detector} · объектов: ${list.length}`;
}

window.peleng.on("frame", (m) => update(m.objects || []));
window.peleng.on("hello", () => clearTrails());
