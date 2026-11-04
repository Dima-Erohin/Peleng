import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { createDataset, createTrack } from "./data.js";
import { TrajectoryScene } from "./scene.js";

function sceneFor(source = "estimate") {
  const track = { ...createTrack({ id: "truck", units: "m", samples: Array.from({ length: 60 }, (_, i) => ({
    t: i / 30, position: [i / 2, i === 30 ? 5 : 0, 3],
  })) }), source, kind: "truck", color: "#69b9ed" };
  const scene = Object.create(TrajectoryScene.prototype);
  Object.assign(scene, {
    objects: new Map(), content: new THREE.Group(), temp: new THREE.Vector3(), previousTarget: new THREE.Vector3(),
    cameraObjects: [], observationObjects: [], dataset: createDataset([track]), time: 1.01, selected: track.id,
    follow: false, showRoute: false, dirtyPose: true, camera: new THREE.PerspectiveCamera(48, 1.5, 0.1, 5000),
    controls: { target: new THREE.Vector3(), enableDamping: true, update() {} },
  });
  scene.createObject(track);
  return scene;
}

test("route toggle changes visibility even while paused; trail tip uses display rather than raw coordinates", () => {
  const scene = sceneFor();
  scene.updateObjects(0);
  const object = scene.objects.get("truck");
  assert.equal(object.route.visible, false);
  scene.setShowRoute(true);
  scene.updateObjects(0);
  assert.equal(object.route.visible, true);
  const tip = object.tip.geometry.attributes.position;
  assert.ok(Math.abs(tip.getY(0)) < 0.2);
  assert.equal(object.track.positions[91], 5);
  scene.setShowRoute(false);
  scene.updateObjects(0);
  assert.equal(object.route.visible, false);
});

test("reference positions are never smoothed and camera framing can return from overview to the object", () => {
  const scene = sceneFor("reference");
  const object = scene.objects.get("truck");
  assert.equal(object.motionTrack, object.track);
  scene.updateObjects(0);
  scene.focus(true);
  const overview = scene.camera.position.clone();
  scene.focus();
  assert.ok(scene.camera.position.distanceTo(overview) > 10);
  assert.ok(scene.controls.target.distanceTo(object.group.position) < 1e-8);
  const closeup = scene.camera.position.clone();
  scene.focus();
  assert.ok(scene.camera.position.distanceTo(closeup) < 1e-8);
  assert.equal(scene.controls.enableDamping, true);
});

test("live correction approaches a new position instead of teleporting, but seek remains exact", () => {
  const scene = sceneFor("online");
  const object = scene.objects.get("truck");
  object.track.live = true;
  scene.updateObjects(0);
  const before = object.group.position.clone();
  scene.time += 0.1;
  scene.updateObjects(1 / 60);
  const moved = object.group.position.distanceTo(before);
  assert.ok(moved > 0 && moved < 1);
  scene.dirtyPose = true;
  scene.updateObjects(0);
  assert.ok(Math.abs(object.group.position.x - scene.time * 15) < 1e-8);
});
