import * as THREE from "three";

export function createDroneModel(color) {
  const root = new THREE.Group();
  root.name = "drone-model";
  const shell = new THREE.MeshStandardMaterial({ color: "#dce4e8", roughness: 0.55, metalness: 0.2 });
  const frame = new THREE.MeshStandardMaterial({ color: "#34424b", roughness: 0.7 });
  const rotor = new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8), shell);
  body.scale.set(1.3, 0.8, 0.5);
  root.add(body);
  for (const x of [-0.65, 0.65]) {
    for (const y of [-0.65, 0.65]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(x, y), 0.07, 0.07), frame);
      arm.position.set(x / 2, y / 2, 0);
      arm.rotation.z = Math.atan2(y, x);
      const disc = new THREE.Mesh(new THREE.CircleGeometry(0.37, 16), rotor);
      disc.position.set(x, y, 0.1);
      root.add(arm, disc);
    }
  }
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6), frame);
  lens.position.set(0.35, 0, -0.1);
  root.add(lens);
  return root;
}
