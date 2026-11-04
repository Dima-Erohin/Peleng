import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { advancePlayback, sampleTrack } from "./data.js";
import { observationRays } from "./reconstruction.js";
import { createDroneModel } from "./drone-model.js";

function disposeTree(root) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  root.traverse((node) => {
    if (node.geometry) geometries.add(node.geometry);
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  geometries.forEach((value) => value.dispose());
  materials.forEach((value) => value.dispose());
  textures.forEach((value) => value.dispose());
}

function lineSegments(values, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
  const line = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  line.frustumCulled = false;
  return line;
}

function mergeModel(root) {
  root.updateMatrixWorld(true);
  const buckets = new Map();
  root.traverse((node) => {
    if (!node.isMesh) return;
    const geometry = node.geometry.index ? node.geometry.toNonIndexed() : node.geometry;
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    const groups = geometry.groups.length ? geometry.groups : [{ start: 0, count: geometry.attributes.position.count, materialIndex: 0 }];
    for (const group of groups) {
      const material = materials[group.materialIndex];
      let bucket = buckets.get(material.name);
      if (!bucket) {
        bucket = { material: material.clone(), geometries: [] };
        buckets.set(material.name, bucket);
      }
      const part = new THREE.BufferGeometry();
      for (const name of ["position", "normal"]) {
        const attribute = geometry.attributes[name];
        if (attribute) part.setAttribute(name, new THREE.BufferAttribute(attribute.array.slice(group.start * attribute.itemSize, (group.start + group.count) * attribute.itemSize), attribute.itemSize));
      }
      part.applyMatrix4(node.matrixWorld);
      if (!part.attributes.normal) part.computeVertexNormals();
      bucket.geometries.push(part);
    }
    if (geometry !== node.geometry) geometry.dispose();
  });
  const merged = new THREE.Group();
  for (const { material, geometries } of buckets.values()) {
    const geometry = mergeGeometries(geometries, false);
    if (!geometry) throw new Error("Не удалось объединить геометрию CargoTruck");
    geometry.computeBoundingSphere();
    merged.add(new THREE.Mesh(geometry, material));
    geometries.forEach((part) => part.dispose());
  }
  disposeTree(root);
  return merged;
}

function buildTrail(track) {
  const values = [], endIndices = [];
  for (let i = 1; i < track.times.length; i++) {
    if (track.times[i] - track.times[i - 1] <= track.maxGap) {
      const a = (i - 1) * 3, b = i * 3;
      values.push(...track.positions.subarray(a, a + 3), ...track.positions.subarray(b, b + 3));
    }
    endIndices[i] = values.length / 3;
  }
  const route = lineSegments(values, track.color, 0.18);
  const travelled = new THREE.LineSegments(route.geometry, new THREE.LineBasicMaterial({ color: track.color }));
  travelled.frustumCulled = false;
  travelled.geometry = route.geometry.clone();
  const tip = lineSegments(new Float32Array(6), track.color);
  return { route, travelled, tip, endIndices };
}

export class TrajectoryScene {
  constructor(container, callbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.disposed = false;
    this.active = true;
    this.playing = false;
    this.loop = true;
    this.rate = 1;
    this.follow = true;
    this.time = 0;
    this.objects = new Map();
    this.cameraObjects = [];
    this.lastTick = 0;
    this.lastUi = 0;
    this.lastStats = 0;
    this.frameTimes = [];
    this.dirtyPose = true;
    this.needsRender = true;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.5;
    this.renderer.domElement.setAttribute("aria-label", "3D-сцена траекторий объектов");
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#10161b");
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 5000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 2000;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.scene.add(new THREE.HemisphereLight(0xd8eaff, 0x343b47, 2.4));
    const sun = new THREE.DirectionalLight(0xffead5, 3);
    sun.position.set(-20, -35, 60);
    this.scene.add(sun);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.temp = new THREE.Vector3();
    this.previousTarget = new THREE.Vector3();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.visibility = () => this.updateLoop();
    document.addEventListener("visibilitychange", this.visibility);
    this.contextLost = (event) => {
      event.preventDefault();
      this.playing = false;
      callbacks.onError("WebGL-контекст потерян. Перезагрузите страницу для восстановления сцены.");
      callbacks.onPlayback(false);
    };
    this.renderer.domElement.addEventListener("webglcontextlost", this.contextLost);
    this.resize();
    this.updateLoop();
  }

  async loadModel() {
    const revision = this.revision;
    const source = await new FBXLoader().loadAsync(`${import.meta.env.BASE_URL}models/CargoTruck.fbx`);
    if (this.disposed || revision !== this.revision) { disposeTree(source); return; }
    const bounds = new THREE.Box3().setFromObject(source);
    const center = bounds.getCenter(new THREE.Vector3());
    // FBX is Y-up, +Z forward, in centimeters; the scene is Z-up, +X forward.
    const correction = new THREE.Group();
    source.position.sub(new THREE.Vector3(center.x, center.y, center.z));
    correction.add(source);
    correction.scale.setScalar(0.01);
    correction.rotation.set(Math.PI / 2, 0, Math.PI / 2, "ZYX");
    source.traverse((node) => {
      if (!node.isMesh) return;
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (let i = 0; i < materials.length; i++) {
        const old = materials[i];
        const material = new THREE.MeshStandardMaterial({
          color: old.color, roughness: old.name.includes("windows") ? 0.25 : 0.72,
          metalness: old.name.includes("metal") ? 0.45 : 0.08,
        });
        material.name = old.name;
        if (old.name.includes("headlight")) {
          material.emissive.set(0xffc882); material.emissiveIntensity = 0.7;
        }
        materials[i] = material;
        old.dispose();
      }
      node.material = Array.isArray(node.material) ? materials : materials[0];
    });
    this.model = mergeModel(correction);
    this.objects.forEach((object) => this.attachModel(object));
    this.dirtyPose = true;
    this.needsRender = true;
    this.callbacks.onModelReady();
  }

  attachModel(object) {
    if (object.track.kind === "drone" || object.track.kind === "object") return;
    if (!this.model) return;
    object.model = new THREE.Group();
    object.model.add(this.model.clone(true));
    object.group.add(object.model);
  }

  setDataset(dataset) {
    this.revision = (this.revision || 0) + 1;
    if (this.content) { this.scene.remove(this.content); disposeTree(this.content); }
    // Model resources are shared by clones. Reloading datasets must not dispose them.
    this.model = null;
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.objects.clear();
    this.cameraObjects = [];
    this.observationObjects = [];
    this.dataset = dataset;
    this.time = dataset.start;
    this.selected = dataset.tracks[0].id;
    const { min, max } = dataset.bounds;
    const floor = Math.min(0, min[2] - 4);
    const extent = Math.max(100, max[0] - min[0] + 150, max[1] - min[1] + 150);
    const cx = (min[0] + max[0]) / 2, cy = (min[1] + max[1]) / 2;
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(extent * 3, extent * 3),
      new THREE.MeshStandardMaterial({ color: "#171e24", roughness: 1 }),
    );
    ground.position.set(cx, cy, floor - 0.06);
    this.content.add(ground);
    const grid = new THREE.GridHelper(Math.ceil(extent / 10) * 10, Math.ceil(extent / 5), 0x3d4c57, 0x28353e);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(cx, cy, floor);
    grid.material.transparent = true;
    grid.material.opacity = 0.6;
    this.content.add(grid);
    for (const track of dataset.tracks) this.createObject(track);
    this.liveTrack = null;
    for (const reconstruction of dataset.reconstructions || []) {
      const lines = lineSegments(new Float32Array(reconstruction.cameraIds.length * 6), 0x69b9ed, 0.5);
      lines.name = `observation-rays:${reconstruction.id}`;
      this.content.add(lines);
      const marker = !reconstruction.hasTrack && reconstruction.validCount
        ? reconstruction.kind === "drone" ? createDroneModel("#69b9ed")
          : new THREE.Mesh(new THREE.OctahedronGeometry(0.6), new THREE.MeshStandardMaterial({ color: "#69b9ed" }))
        : null;
      if (marker) { marker.visible = false; this.content.add(marker); }
      this.observationObjects.push({ reconstruction, lines, marker });
    }
    for (const camera of dataset.cameras) {
      const group = new THREE.Group();
      const d = 14, halfWidth = d * 23.76 / (2 * camera.focal), halfHeight = halfWidth * 9 / 16;
      const corners = [[d, -halfWidth, -halfHeight], [d, halfWidth, -halfHeight], [d, halfWidth, halfHeight], [d, -halfWidth, halfHeight]];
      const lines = [];
      corners.forEach((corner, i) => lines.push(0, 0, 0, ...corner, ...corner, ...corners[(i + 1) % 4]));
      group.add(lineSegments(lines, 0x83a1b5, 0.45));
      const marker = new THREE.Mesh(new THREE.OctahedronGeometry(1.3), new THREE.MeshBasicMaterial({ color: 0x91b8cb }));
      group.add(marker);
      this.content.add(group);
      this.cameraObjects.push({ group, track: camera, sample: {} });
    }
    this.seek(dataset.start);
    this.focus();
    const revision = this.revision;
    this.loadModel().catch((error) => {
      if (!this.disposed && revision === this.revision) this.callbacks.onError(`Модель CargoTruck: ${error.message}`);
    });
  }

  createObject(track) {
    const group = new THREE.Group();
    const ringRadius = track.kind === "drone" ? 1 : 2.5;
    const ring = new THREE.Mesh(new THREE.RingGeometry(ringRadius - 0.04, ringRadius, 40), new THREE.MeshBasicMaterial({ color: track.color, side: THREE.DoubleSide, transparent: true }));
    ring.position.z = track.kind === "drone" ? -0.35 : -1.25;
    group.add(ring);
    const trail = buildTrail(track);
    this.content.add(group, trail.route, trail.travelled, trail.tip);
    const object = { track, group, ring, ...trail, sample: {}, heading: null, enabled: true };
    if (track.kind === "drone") {
      object.model = createDroneModel(track.color);
      group.add(object.model);
    } else if (track.kind === "object") {
      group.add(new THREE.Mesh(new THREE.OctahedronGeometry(0.6), new THREE.MeshStandardMaterial({ color: track.color })));
    }
    this.objects.set(track.id, object);
    return object;
  }

  // Онлайн-трек (offline/online.py): точки приходят по WebSocket, след пересобирается с каждой новой точкой.
  // track.live = true; track.liveVel — скорость в последней точке (м/с) для движения модели между точками.
  setLiveTrack(track) {
    if (!this.dataset) return;
    const old = track && this.objects.get(track.id);
    if (old) {
      for (const line of [old.route, old.travelled, old.tip]) { this.content.remove(line); line.geometry.dispose(); line.material.dispose(); }
      Object.assign(old, buildTrail(track), { track });
      this.content.add(old.route, old.travelled, old.tip);
    } else if (track) {
      const object = this.createObject(track);
      this.attachModel(object);
    }
    if (!track && this.liveTrack) {
      const object = this.objects.get(this.liveTrack);
      if (object) {
        for (const item of [object.group, object.route, object.travelled, object.tip]) { this.content.remove(item); disposeTree(item); }
        this.objects.delete(this.liveTrack);
      }
    }
    this.liveTrack = track ? track.id : null;
    this.dirtyPose = true;
    this.needsRender = true;
  }

  resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height || this.disposed) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.needsRender = true;
  }

  updateLoop() {
    this.lastTick = 0;
    this.frameTimes = [];
    this.renderer.setAnimationLoop(this.active && !document.hidden ? (now) => this.tick(now) : null);
  }

  setActive(active) {
    this.active = active;
    this.needsRender = true;
    this.updateLoop();
  }

  setPlaying(playing) {
    if (!this.dataset) return;
    if (playing && this.time >= this.dataset.end) this.seek(this.dataset.start);
    this.playing = playing;
    this.frameTimes = [];
    this.lastTick = 0;
    this.needsRender = true;
    if (!playing) { this.updateObjects(0); this.emitFrame(); }
    this.callbacks.onPlayback(playing);
  }

  seek(time) {
    if (!this.dataset) return;
    this.time = Math.max(this.dataset.start, Math.min(this.dataset.end, time));
    this.dirtyPose = true;
    this.needsRender = true;
    this.updateObjects(0);
    this.emitFrame();
  }

  select(id) {
    this.selected = id;
    this.dirtyPose = true;
    this.needsRender = true;
    if (this.follow) this.focus();
    this.emitFrame();
  }

  setObjectVisible(id, visible) {
    const object = this.objects.get(id);
    if (!object) return;
    object.enabled = visible;
    this.dirtyPose = true;
    this.needsRender = true;
  }

  focus(overview = false) {
    if (!this.dataset) return;
    this.needsRender = true;
    if (overview) {
      const { min, max } = this.dataset.bounds;
      this.controls.target.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
      const vertical = this.camera.fov * Math.PI / 180;
      const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * this.camera.aspect);
      const radius = Math.max(20, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2);
      const distance = radius / Math.sin(Math.min(vertical, horizontal) / 2) * 1.12;
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3(-0.45, -0.65, 0.65).normalize().multiplyScalar(distance));
    } else {
      const object = this.objects.get(this.selected);
      if (!object) {
        const rays = observationRays(this.dataset.imageTracks, this.dataset.cameras, this.selected, this.time);
        if (!rays.length) return;
        const { origin, direction } = rays[0];
        const reconstruction = this.dataset.reconstructions.find((item) => item.id === this.selected);
        const frame = reconstruction?.frames[Math.round(this.time * reconstruction.fps)];
        if (frame?.position) {
          this.controls.target.fromArray(frame.position);
          this.camera.position.copy(this.controls.target).add(new THREE.Vector3(-5, -7, 5));
        } else {
          this.controls.target.fromArray(origin).addScaledVector(this.temp.fromArray(direction), 60);
          this.camera.position.copy(this.controls.target).add(new THREE.Vector3(-65, -85, 65));
        }
        this.previousTarget.copy(this.controls.target);
        this.controls.update();
        return;
      }
      const p = sampleTrack(object.track, this.time, object.sample);
      this.controls.target.set(p.x, p.y, p.z);
      const distance = object.track.kind === "drone" ? 0.18 : 1;
      this.camera.position.set(p.x - 24 * distance, p.y - 34 * distance, p.z + 24 * distance);
    }
    this.previousTarget.copy(this.controls.target);
    this.controls.update();
  }

  updateObjects(dt) {
    for (const object of this.objects.values()) {
      const p = sampleTrack(object.track, this.time, object.sample);
      if (object.track.live) {
        const last = object.track.times.at(-1), ahead = this.time - last;
        if (ahead > 0 && ahead < 3) {
          const v = object.track.liveVel || [0, 0, 0], i = (object.track.times.length - 1) * 3;
          p.x = object.track.positions[i] + v[0] * ahead;
          p.y = object.track.positions[i + 1] + v[1] * ahead;
          p.z = object.track.positions[i + 2] + v[2] * ahead;
          p.speed = Math.hypot(v[0], v[1], v[2]);
          p.visible = true;
        }
      }
      object.group.visible = p.visible && object.enabled;
      object.route.visible = object.travelled.visible = object.tip.visible = object.enabled;
      object.group.position.set(p.x, p.y, p.z);
      if (this.dirtyPose || object.heading == null) object.heading = p.heading;
      else if (p.speed > 0.01) {
        const angle = Math.atan2(Math.sin(p.heading - object.heading), Math.cos(p.heading - object.heading));
        object.heading += angle * (1 - Math.exp(-dt * 10));
      }
      if (object.model) object.model.rotation.z = object.heading;
      object.ring.material.opacity = object.track.id === this.selected ? 1 : 0.45;
      object.travelled.geometry.setDrawRange(0, object.endIndices[p.index] || 0);
      const offset = p.index * 3;
      const a = object.tip.geometry.attributes.position;
      a.setXYZ(0, object.track.positions[offset], object.track.positions[offset + 1], object.track.positions[offset + 2]);
      a.setXYZ(1, p.x, p.y, p.z);
      a.needsUpdate = true;
      object.tip.visible = object.enabled && p.visible;
      if (this.follow && object.track.id === this.selected && p.visible) {
        this.temp.set(p.x, p.y, p.z).sub(this.previousTarget);
        this.camera.position.add(this.temp);
        this.controls.target.add(this.temp);
        this.previousTarget.set(p.x, p.y, p.z);
      }
    }
    for (const object of this.cameraObjects) {
      const p = sampleTrack(object.track, this.time, object.sample);
      object.group.position.set(p.x, p.y, p.z);
      const index = p.index * 3;
      const next = Math.min(p.index + 1, object.track.times.length - 1) * 3;
      const r = object.track.rotations;
      const angle = (axis) => {
        const from = r[index + axis] * Math.PI / 180, to = r[next + axis] * Math.PI / 180;
        return from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * p.alpha;
      };
      object.group.rotation.set(-angle(0), -angle(1), angle(2), "ZYX");
    }
    for (const { reconstruction, lines, marker } of this.observationObjects || []) {
      const rays = observationRays(this.dataset.imageTracks, this.dataset.cameras, reconstruction.id, this.time);
      const frame = reconstruction.frames[Math.round(this.time * reconstruction.fps)];
      const position = lines.geometry.attributes.position;
      rays.forEach((ray, i) => {
        position.setXYZ(i * 2, ...ray.origin);
        const end = frame?.position || ray.origin.map((value, axis) => value + ray.direction[axis] * 300);
        position.setXYZ(i * 2 + 1, ...end);
      });
      position.needsUpdate = true;
      lines.geometry.setDrawRange(0, rays.length * 2);
      lines.visible = reconstruction.id === this.selected && this.objects.get(reconstruction.id)?.enabled !== false;
      if (marker) {
        marker.visible = Boolean(frame?.position);
        if (frame?.position) marker.position.fromArray(frame.position);
      }
    }
    this.dirtyPose = false;
  }

  emitFrame() {
    const object = this.objects.get(this.selected);
    const reconstruction = this.dataset?.reconstructions?.find((item) => item.id === this.selected);
    const frame = reconstruction?.frames[Math.round(this.time * reconstruction.fps)];
    const sample = object ? { ...object.sample } : frame?.position ? {
      x: frame.position[0], y: frame.position[1], z: frame.position[2],
      visible: true, speed: 0, distance: 0, motionAvailable: false,
    } : null;
    this.callbacks.onFrame({ time: this.time, sample });
  }

  tick(now) {
    if (this.disposed) return;
    const delta = this.lastTick ? (now - this.lastTick) / 1000 : 0;
    const frameTime = this.lastTick ? now - this.lastTick : 0;
    this.lastTick = now;
    if (this.dataset && this.liveClock) {
      const live = Math.max(this.dataset.start, Math.min(this.dataset.end, this.liveClock()));
      if (live < this.time) this.dirtyPose = true;
      this.time = live;
    } else if (this.dataset && this.playing) {
      const next = advancePlayback(this.time, delta * this.rate, this.dataset.start, this.dataset.end, this.loop);
      if (next.time < this.time) this.dirtyPose = true;
      this.time = next.time;
      if (next.ended) this.setPlaying(false);
    }
    const updatePose = this.playing || this.dirtyPose || Boolean(this.liveClock);
    if (this.dataset && updatePose) this.updateObjects(delta);
    const controlsChanged = this.controls.update();
    if (this.needsRender || updatePose || controlsChanged) {
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
      if (frameTime > 0) this.frameTimes.push(frameTime);
    }
    if ((this.playing || this.liveClock) && now - this.lastUi >= 100 && this.dataset) { this.lastUi = now; this.emitFrame(); }
    if (now - this.lastStats >= 1000 && this.frameTimes.length) {
      const frames = this.frameTimes.sort((a, b) => a - b);
      const avg = frames.reduce((sum, value) => sum + value, 0) / frames.length;
      this.callbacks.onStats({
        fps: Math.round(1000 / avg), frameMs: frames[Math.min(frames.length - 1, Math.floor(frames.length * 0.95))].toFixed(1),
        calls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles,
      });
      this.frameTimes = [];
      this.lastStats = now;
    }
  }

  dispose() {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    document.removeEventListener("visibilitychange", this.visibility);
    this.renderer.domElement.removeEventListener("webglcontextlost", this.contextLost);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    disposeTree(this.scene);
    if (this.model) disposeTree(this.model);
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.objects.clear();
  }
}
