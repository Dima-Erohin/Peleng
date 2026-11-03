import * as THREE from "/static/vendor/three.module.min.js";
import { OrbitControls } from "/static/vendor/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const log = (s) => window.dispatchEvent(new CustomEvent("peleng:log", { detail: s }));
const publish = (name, detail) => window.dispatchEvent(new CustomEvent(`peleng:${name}`, { detail }));

// ---------------------------------------------------------------- сцена (Z вверх)
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 200);
camera.position.set(-4, -5, 4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.8);
controls.update();
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(3, -2, 6); scene.add(sun);

const grid = new THREE.GridHelper(20, 40, 0x30363d, 0x1c2128);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
window.addEventListener("resize", resize);
requestAnimationFrame(resize);
window.setTimeout(resize, 100);

const COLORS = [0xff6b6b, 0x4dabf7, 0x51cf66, 0xfcc419, 0xcc5de8, 0x22b8cf, 0xff922b, 0xf06595];
const colorFor = (id) => COLORS[(id - 1) % COLORS.length];
const BONES = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];

function textSprite(text, color = "#fff") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 30px system-ui"; g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, 256, 64);
  g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 10, 32);
  const tex = new THREE.CanvasTexture(c);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  s.scale.set(0.6, 0.15, 1);
  s.userData.canvas = c;
  return s;
}
function setSpriteText(s, text, color = "#fff") {
  const c = s.userData.canvas, g = c.getContext("2d");
  g.clearRect(0, 0, 256, 64); g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, 256, 64);
  g.font = "bold 30px system-ui"; g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 10, 32);
  s.material.map.needsUpdate = true;
}

// ---------------------------------------------------------------- статичные объекты: доска, комната, камеры
const staticGroup = new THREE.Group(); scene.add(staticGroup);
let boardMesh = null, roomBox = null;
function drawBoard(b) {
  if (boardMesh) staticGroup.remove(boardMesh);
  const pts = b.corners_world.map(p => new THREE.Vector3(...p));
  const g = new THREE.BufferGeometry().setFromPoints([...pts, pts[0]]);
  boardMesh = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xff00ff }));
  staticGroup.add(boardMesh);
}
function drawRoom(room) {
  if (roomBox) staticGroup.remove(roomBox);
  roomBox = null;
  if (!room || !room.size_m) return;
  const [w, d, h] = room.size_m, off = room.board_offset_m || [0, 0];
  const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(w, d, h));
  roomBox = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.5 }));
  roomBox.position.set(-off[0], -off[1], h / 2);
  staticGroup.add(roomBox);
}

const camGroup = new THREE.Group(); scene.add(camGroup);
const camObjs = {};   // id -> {group, plane, tex}
let cameras = [];
let showTextures = true;
let showSkeletons = true;
function drawCameras(list) {
  cameras = list;
  for (const id of Object.keys(camObjs)) { camGroup.remove(camObjs[id].group); delete camObjs[id]; }
  for (const c of list) {
    if (!c.calibrated) continue;
    const R = c.R, t = c.t, K = c.K, [W, H] = c.size;
    const C = new THREE.Vector3(...c.center);
    // точка изображения (u,v) на глубине d -> мир: X = R^T (d*K^-1 [u v 1] - t)
    const toWorld = (u, v, d) => {
      const x = (u - K[0][2]) / K[0][0] * d, y = (v - K[1][2]) / K[1][1] * d, z = d;
      const p = [x - t[0], y - t[1], z - t[2]];
      return new THREE.Vector3(R[0][0]*p[0] + R[1][0]*p[1] + R[2][0]*p[2], R[0][1]*p[0] + R[1][1]*p[1] + R[2][1]*p[2], R[0][2]*p[0] + R[1][2]*p[1] + R[2][2]*p[2]);
    };
    const d = 0.35;
    const corners = [toWorld(0, 0, d), toWorld(W, 0, d), toWorld(W, H, d), toWorld(0, H, d)];
    const group = new THREE.Group();
    const segs = [];
    for (const p of corners) segs.push(C, p);
    for (let i = 0; i < 4; i++) segs.push(corners[i], corners[(i + 1) % 4]);
    group.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), new THREE.LineBasicMaterial({ color: 0xffd33d })));
    // плоскость изображения с текстурой кадра
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array([...corners[0].toArray(), ...corners[1].toArray(), ...corners[2].toArray(), ...corners[3].toArray()]);
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]), 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    const mat = new THREE.MeshBasicMaterial({ color: 0x333333, side: THREE.DoubleSide, transparent: true, opacity: 0.9 });
    const plane = new THREE.Mesh(geo, mat);
    group.add(plane);
    const label = textSprite(c.name || c.id, "#ffd33d");
    label.position.copy(C).add(new THREE.Vector3(0, 0, 0.15));
    group.add(label);
    camGroup.add(group);
    camObjs[c.id] = { group, plane };
  }
}
const texLoader = new THREE.TextureLoader();
setInterval(() => {
  if (!showTextures) return;
  for (const [id, o] of Object.entries(camObjs)) {
    texLoader.load(`/api/cameras/${encodeURIComponent(id)}/snapshot.jpg?overlay=0&width=320&_=${Date.now()}`, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      const old = o.plane.material.map;
      o.plane.material.map = tex; o.plane.material.color.set(0xffffff); o.plane.material.needsUpdate = true;
      if (old) old.dispose();
    }, undefined, () => {});
  }
}, 1500);

// ---------------------------------------------------------------- облако точек комнаты
let roomPoints = null;
function drawRoomPoints(m) {
  if (roomPoints) { scene.remove(roomPoints); roomPoints.geometry.dispose(); roomPoints = null; }
  if (!m.points || !m.points.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(m.points.flat(), 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(m.colors.flat().map(v => v / 255), 3));
  roomPoints = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.03, vertexColors: true }));
  scene.add(roomPoints);
}

// ---------------------------------------------------------------- люди и траектории
const people = {};   // id -> {group, joints[], bones, body, ring, label, trail, trailPts[]}
const TRAIL_MAX = 900;
function makePerson(id) {
  const color = colorFor(id);
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color, transparent: true });
  const joints = [];
  for (let i = 0; i < 17; i++) { const s = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 6), mat); group.add(s); joints.push(s); }
  const bones = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color, transparent: true }));
  bones.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(BONES.length * 6), 3));
  group.add(bones);
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.18, 1.1, 4, 12), new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.45 }));
  body.rotation.x = Math.PI / 2;
  group.add(body);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.22, 0.28, 32), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true }));
  group.add(ring);
  const label = textSprite(`ID ${id}`); group.add(label);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(TRAIL_MAX * 3), 3));
  trailGeo.setDrawRange(0, 0);
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color }));
  scene.add(group); scene.add(trail);
  return people[id] = { group, joints, bones, body, ring, label, trail, trailPts: [], last: 0, mat };
}
function updatePeople(list) {
  const now = performance.now();
  for (const p of list) {
    const o = people[p.id] || makePerson(p.id);
    o.last = now;
    o.group.visible = true;
    const opacity = p.predicted ? 0.35 : 1;
    const hasSkel = showSkeletons && p.kp3d && p.kp3d.filter(Boolean).length >= 4;
    o.joints.forEach((s, i) => { const k = hasSkel && p.kp3d[i]; s.visible = !!k; if (k) s.position.set(...k); });
    o.mat.opacity = opacity;
    o.bones.visible = !!hasSkel;
    if (hasSkel) {
      const arr = o.bones.geometry.attributes.position.array;
      BONES.forEach(([a, b], i) => {
        const A = p.kp3d[a], B = p.kp3d[b];
        const ok = A && B;
        arr.set(ok ? A : [0, 0, -100], i * 6); arr.set(ok ? B : [0, 0, -100], i * 6 + 3);
      });
      o.bones.geometry.attributes.position.needsUpdate = true;
      o.bones.geometry.computeBoundingSphere();
    }
    o.body.visible = !hasSkel;
    o.body.position.set(p.floor[0], p.floor[1], 0.8);
    o.body.material.opacity = 0.45 * opacity;
    o.ring.position.set(p.floor[0], p.floor[1], 0.005);
    o.label.position.set(p.floor[0], p.floor[1], 2.0);
    setSpriteText(o.label, `ID ${p.id} · ${p.speed.toFixed(2)} м/с${p.mode === "mono" ? " · 1 кам" : ""}${p.predicted ? " · прогноз" : ""}`);
    // траектория по полу
    const last = o.trailPts[o.trailPts.length - 1];
    if (!last || Math.hypot(last[0] - p.floor[0], last[1] - p.floor[1]) > 0.02) {
      o.trailPts.push([p.floor[0], p.floor[1], 0.01]);
      if (o.trailPts.length > TRAIL_MAX) o.trailPts.shift();
      const arr = o.trail.geometry.attributes.position.array;
      o.trailPts.forEach((q, i) => arr.set(q, i * 3));
      o.trail.geometry.attributes.position.needsUpdate = true;
      o.trail.geometry.setDrawRange(0, o.trailPts.length);
      o.trail.geometry.computeBoundingSphere();
    }
  }
  for (const o of Object.values(people)) if (now - o.last > 500) o.group.visible = false;
}
function clearTrails() {
  for (const o of Object.values(people)) { o.trailPts = []; o.trail.geometry.setDrawRange(0, 0); }
}

// ---------------------------------------------------------------- отметка и замер расстояния
let mark = null, lastPersons = [];
const markMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.02, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
markMesh.rotation.x = Math.PI / 2; markMesh.visible = false; scene.add(markMesh);
function setMark() {
  const p = lastPersons.find(q => !q.predicted) || lastPersons[0];
  if (!p) { log("нет человека для отметки"); return; }
  mark = { id: p.id, pos: p.floor.slice() };
  markMesh.position.set(mark.pos[0], mark.pos[1], 0.01); markMesh.visible = true;
  publish("measure", { visible: true, text: `Отметка ID ${p.id} · ${p.floor.map(v => v.toFixed(2)).join(", ")} м` });
  log(`отметка: ID ${p.id} в (${p.floor.map(v => v.toFixed(2)).join(", ")})`);
}
addEventListener("keydown", (e) => { if (e.key === "m" || e.key === "M" || e.key === "ь" || e.key === "Ь") setMark(); });

// ---------------------------------------------------------------- WebSocket
let hudInfo = { mode: "", detector: "" }, frames = 0, fpsView = 0;
setInterval(() => { fpsView = frames; frames = 0; }, 1000);
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/stream`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    for (const fn of peleng.hooks[m.type] || []) { try { fn(m); } catch (e) { console.error(e); } }
    publish("message", m);
    if (m.type === "hello") {
      hudInfo = { mode: m.mode, detector: m.detector };
      drawBoard(m.board); drawRoom(m.room); drawCameras(m.cameras);
      log(`подключено: режим ${m.mode}, детектор ${m.detector}`);
    } else if (m.type === "cameras") {
      drawCameras(m.cameras);
    } else if (m.type === "room_points") {
      drawRoomPoints(m);
    } else if (m.type === "frame") {
      frames++;
      lastPersons = m.persons;
      updatePeople(m.persons);
      const s = m.stats || {};
      publish("hud",
        `режим: ${hudInfo.mode}  детектор: ${hudInfo.detector}\n` +
        `обработка: ${s.fps ?? "–"} к/с  задержка: ${s.latency_ms} мс  рассинхрон: ${s.skew_ms} мс  YOLO: ${s.detect_ms} мс\n` +
        `камеры в 3D: ${(s.cams_used || []).join(", ") || "нет (нужна калибровка)"}\n` +
        m.persons.map(p => `ID ${p.id}: (${p.pos.map(v => v.toFixed(2)).join(", ")}) ${p.mode}${p.reproj_err_px != null ? " ±" + p.reproj_err_px + "px" : ""} [${p.cams.join(",")}]`).join("\n"));
      if (mark) {
        const p = m.persons.find(q => q.id === mark.id) || m.persons[0];
        if (p) publish("measure", { visible: true, text: `От отметки: ${Math.hypot(p.floor[0] - mark.pos[0], p.floor[1] - mark.pos[1]).toFixed(2)} м · ID ${p.id}` });
      }
    }
  };
  ws.onclose = () => { publish("status", "Нет связи с сервером · переподключение…"); setTimeout(connect, 1000); };
}
connect();
setInterval(async () => {  // обновлять fps/онлайн камер
  try {
    const s = await (await fetch("/api/state")).json();
    cameras = s.cameras;
    publish("cameras", s.cameras);
  } catch (e) { /* retry on next interval */ }
}, 2000);

renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// ---------------------------------------------------------------- расширения (web/ext/*.js)
// Скрипт расширения — ES-модуль; берёт всё нужное из window.peleng:
//   peleng.on(<type сообщения>, fn)          — "hello", "frame", "cameras", … и свои типы с сервера
//   peleng.addSection(title) -> <div>        — свой блок в правой панели
//   peleng.THREE, scene, camera, controls, log, colorFor, textSprite
window.peleng = {
  THREE, scene, camera, controls, log, colorFor, textSprite,
  hooks: {},
  clearTrails,
  clearRoomPoints() { drawRoomPoints({ points: [] }); },
  setShowSkeletons(value) { showSkeletons = value; },
  setShowTextures(value) { showTextures = value; },
  setMark,
  addSection(title) {
    const box = document.createElement("div");
    box.innerHTML = `<h2></h2>`;
    box.querySelector("h2").textContent = title;
    $("ext").appendChild(box);
    return box;
  },
};
window.peleng.on = (type, fn) => (window.peleng.hooks[type] ||= []).push(fn);
fetch("/api/ext").then(r => r.json()).then(async (files) => {
  for (const f of files) {
    try { await import(`/static/ext/${f}`); } catch (e) { log(`расширение ${f}: ${e.message}`); }
  }
});
