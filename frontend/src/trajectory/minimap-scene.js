import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { sampleTrack } from "./data.js";
import { minimapStyle } from "./minimap-style.js";

function dispose(root) {
  root.traverse((node) => {
    node.geometry?.dispose();
    node.material?.map?.dispose();
    node.material?.dispose();
  });
}

function cameraLabel(text, size) {
  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Не удалось создать подписи камер");
  context.fillStyle = "#122e3c";
  context.fillRect(0, 0, 160, 64);
  context.strokeStyle = "#48ccec";
  context.lineWidth = 3;
  context.strokeRect(2, 2, 156, 60);
  context.fillStyle = "#d7f7ff";
  context.font = "500 32px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 80, 34);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false }));
  sprite.renderOrder = 20;
  sprite.scale.set(size * 3.8, size * 1.52, 1);
  sprite.position.z = size * 2;
  return sprite;
}

export function overviewBounds(tracks) {
  const box = new THREE.Box3();
  for (const track of tracks) {
    box.expandByPoint(new THREE.Vector3(...track.bounds.min));
    box.expandByPoint(new THREE.Vector3(...track.bounds.max));
  }
  if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  return box;
}

export function routeGeometry(track) {
  const values = [], ends = [0];
  for (let i = 1; i < track.times.length; i++) {
    if (track.times[i] - track.times[i - 1] <= track.maxGap) {
      values.push(...track.positions.subarray((i - 1) * 3, (i + 1) * 3));
    }
    ends.push(values.length / 3);
  }
  return { geometry: new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(values, 3)), ends };
}

export class MinimapScene {
  constructor(container, onError) {
    this.container = container;
    this.active = true;
    this.objects = new Map();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#141b20");
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "low-power" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.setAttribute("aria-label", "Наклонный 3D-обзор маршрутов и камер");
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.minZoom = 0.5;
    this.controls.maxZoom = 8;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.addEventListener("change", () => this.render());
    this.scene.add(new THREE.AmbientLight(0xffffff, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(-100, -150, 300);
    this.scene.add(light);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.contextLost = (event) => {
      event.preventDefault();
      onError("3D-обзор: потерян WebGL-контекст. Перезагрузите страницу.");
    };
    this.renderer.domElement.addEventListener("webglcontextlost", this.contextLost);
  }

  render() {
    if (this.active && !this.disposed) this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.aspect = width / height;
    this.fit();
  }

  fit() {
    if (!this.bounds) return;
    const aspect = this.aspect || 1;
    this.camera.updateMatrixWorld(true);
    const target = this.controls.target.clone().applyMatrix4(this.camera.matrixWorldInverse);
    let halfHeight = 10;
    for (const x of [this.bounds.min.x, this.bounds.max.x]) {
      for (const y of [this.bounds.min.y, this.bounds.max.y]) {
        for (const z of [this.bounds.min.z, this.bounds.max.z]) {
          const point = new THREE.Vector3(x, y, z).applyMatrix4(this.camera.matrixWorldInverse).sub(target);
          halfHeight = Math.max(halfHeight, Math.abs(point.y) * 1.15, Math.abs(point.x) / aspect * 1.15);
        }
      }
    }
    this.camera.left = -halfHeight * aspect;
    this.camera.right = halfHeight * aspect;
    this.camera.top = halfHeight;
    this.camera.bottom = -halfHeight;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  focus() {
    if (!this.bounds) return;
    const center = this.bounds.getCenter(new THREE.Vector3());
    const distance = Math.max(100, this.bounds.getSize(new THREE.Vector3()).length() * 2);
    this.camera.zoom = 1;
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(new THREE.Vector3(0.65, -1, 1.65).normalize(), distance);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.fit();
  }

  setDataset(dataset) {
    this.scene.remove(this.content);
    dispose(this.content);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.objects.clear();
    this.liveId = null;
    this.dataset = dataset;
    this.bounds = overviewBounds([...dataset.tracks, ...dataset.cameras]);
    this.markerSize = Math.max(2, this.bounds.getSize(new THREE.Vector3()).length() / 38);
    for (const track of dataset.tracks) this.addTrack(track);
    this.cameras = dataset.cameras.map((track, index) => {
      const mesh = new THREE.Group();
      mesh.add(new THREE.Mesh(new THREE.OctahedronGeometry(this.markerSize * 0.65),
        new THREE.MeshBasicMaterial({ color: "#47d7f5" })));
      mesh.add(cameraLabel(`CAM ${String(index + 1).padStart(2, "0")}`, this.markerSize));
      this.content.add(mesh);
      return { track, mesh };
    });
    const size = this.bounds.getSize(new THREE.Vector3());
    const center = this.bounds.getCenter(new THREE.Vector3());
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(20, size.x + 30), Math.max(20, size.y + 30)),
      new THREE.MeshBasicMaterial({ color: "#172d38", side: THREE.DoubleSide }));
    plane.position.set(center.x, center.y, Math.min(0, this.bounds.min.z) - 2);
    this.content.add(plane);
    this.floor = plane.position.z;
    const gridSize = Math.max(20, size.x + 30, size.y + 30);
    const grid = new THREE.GridHelper(gridSize, 12, "#47728a", "#345063");
    grid.rotation.x = Math.PI / 2;
    grid.position.set(center.x, center.y, this.floor + 0.1);
    grid.material.transparent = true;
    grid.material.opacity = 0.25;
    this.content.add(grid);
    const border = new THREE.LineSegments(new THREE.EdgesGeometry(plane.geometry),
      new THREE.LineBasicMaterial({ color: "#638da3" }));
    border.position.copy(plane.position);
    this.content.add(border);
    this.focus();
  }

  addTrack(track) {
    const style = minimapStyle(track);
    const { geometry, ends } = routeGeometry(track);
    const positions = geometry.attributes.position.array;
    const makeLine = (width, opacity) => {
      const lineGeometry = new LineSegmentsGeometry();
      if (positions.length) lineGeometry.setPositions(positions);
      const material = new LineMaterial({
        color: style.color, linewidth: width, transparent: true, opacity,
        depthTest: false, depthWrite: false, dashed: style.dashed,
        dashSize: this.markerSize * 0.8, gapSize: this.markerSize * 0.65,
      });
      const line = new LineSegments2(lineGeometry, material);
      if (positions.length) line.computeLineDistances();
      return line;
    };
    const route = makeLine(style.width, 0.35);
    const trail = makeLine(style.width, 1);
    // Explicit layering avoids z-fighting without changing measured XYZ.
    route.renderOrder = style.order;
    trail.renderOrder = style.order + 1;
    geometry.dispose();
    const group = new THREE.Group();
    const marker = new THREE.Mesh(new THREE.ConeGeometry(this.markerSize * 0.65, this.markerSize * 2, style.dashed ? 3 : 4),
      new THREE.MeshBasicMaterial({ color: style.color, depthTest: false, depthWrite: false }));
    marker.renderOrder = style.order + 10;
    marker.rotation.z = -Math.PI / 2;
    const ring = new THREE.Mesh(new THREE.RingGeometry(this.markerSize, this.markerSize * 1.25, 32),
      new THREE.MeshBasicMaterial({ color: "#ffffff", side: THREE.DoubleSide, depthTest: false, depthWrite: false }));
    ring.renderOrder = 18;
    group.add(marker, ring);
    this.content.add(group, route, trail);
    this.objects.set(track.id, { track, group, ring, route, trail, ends, enabled: true });
  }

  setLiveTrack(track) {
    const old = this.liveId && this.objects.get(this.liveId);
    if (old) {
      for (const node of [old.group, old.route, old.trail]) { this.content.remove(node); dispose(node); }
      this.objects.delete(this.liveId);
    }
    this.liveId = track?.id;
    if (track) {
      this.addTrack(track);
      const expanded = overviewBounds([...this.dataset.tracks, ...this.dataset.cameras, track]);
      if (!this.bounds.containsBox(expanded)) { this.bounds.union(expanded); this.fit(); }
    }
  }

  setObjectVisible(id, visible) { this.objects.get(id).enabled = visible; }
  select(id) { this.selected = id; }

  seek(time) {
    for (const object of this.objects.values()) {
      const { track, group } = object;
      const last = track.times.at(-1);
      const sample = sampleTrack(track, track.live && time > last && time - last <= 1 ? last : time);
      group.visible = object.enabled && sample.visible;
      group.position.set(sample.x, sample.y, sample.z);
      group.rotation.z = sample.heading;
      object.ring.visible = track.id === this.selected;
      object.route.visible = object.enabled && !track.live;
      object.trail.visible = object.enabled;
      object.trail.geometry.instanceCount = (object.ends[sample.index] || 0) / 2;
    }
    for (const { track, mesh } of this.cameras) {
      const sample = sampleTrack(track, time);
      mesh.visible = sample.visible;
      mesh.position.set(sample.x, sample.y, sample.z);
    }
    this.render();
  }

  setActive(active) { this.active = active; this.render(); }
  dispose() {
    this.disposed = true;
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.domElement.removeEventListener("webglcontextlost", this.contextLost);
    dispose(this.scene);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
