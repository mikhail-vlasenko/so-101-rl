import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

/** Authored environment and optics illustrations; never change observation data. */
function solidBox(size, position, color, radius = 0.1) {
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(...size, 3, radius), new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function stroke(points, color, opacity = 1) {
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p))), new THREE.LineBasicMaterial({ color, transparent: true, opacity, toneMapped: false }));
}

function inscription(text, position, width, color, rotation) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 128;
  const context = canvas.getContext('2d');
  context.fillStyle = color;
  context.font = '500 60px Arial';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, 512, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, width / 8), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide }));
  mesh.position.set(...position);
  mesh.rotation.set(...rotation);
  return mesh;
}

function addExhibit(scene, mounts) {
  // A physical cutaway display table: the top remains at the common Y=0 plane.
  scene.add(solidBox([49, 3.6, 65], [6, -1.82, 19], '#bbb09e', 0.6));
  scene.add(solidBox([47.8, 0.45, 63.8], [6, -0.24, 19], '#e1d8c6', 0.18));
  scene.add(solidBox([41, 1.2, 57], [6, -4.1, 19], '#6a7166', 0.25));
  scene.add(solidBox([24, 0.08, 13], [0, -0.015, 0], '#b1beb0', 0.025));

  // Surface ticks and a travel track make distance tangible without a HUD.
  scene.add(stroke([[-10, 0.065, 5], [10, 0.065, 5]], '#728473'));
  for (let x = -10; x <= 10; x++) {
    scene.add(stroke([[x, 0.07, 5], [x, 0.07, x % 5 === 0 ? 5.65 : 5.3]], '#728473'));
  }
  scene.add(inscription('SURFACE / POSITION', [0, 0.08, -5.1], 13, '#4e6555', [-Math.PI / 2, 0, 0]));
  scene.add(inscription('STEREO STUDY     /     01', [6, -1.5, 51.56], 27, '#4e514a', [0, 0, 0]));

  const start = mounts[0].clone();
  const end = mounts[1].clone();
  const rail = solidBox([start.distanceTo(end) + 5, 0.8, 3.6], [(start.x + end.x) / 2, 0.45, (start.z + end.z) / 2], '#9b998c', 0.12);
  scene.add(rail);
  for (const mount of mounts) {
    const cable = new THREE.CatmullRomCurve3([
      new THREE.Vector3(mount.x + 0.8, mount.y - 1.5, mount.z),
      new THREE.Vector3(mount.x + 1.5, mount.y * 0.55, mount.z + 0.5),
      new THREE.Vector3(mount.x + 1.3, 0.9, mount.z + 1.5),
      new THREE.Vector3(mount.x - 1, 0.6, mount.z + 5),
    ]);
    scene.add(new THREE.Mesh(new THREE.TubeGeometry(cable, 36, 0.085, 6, false), new THREE.MeshStandardMaterial({ color: '#555e59', roughness: 0.95 })));
    const socket = new THREE.Mesh(new THREE.TorusGeometry(2.35, 0.12, 8, 48), new THREE.MeshStandardMaterial({ color: '#7f8979', roughness: 0.7 }));
    socket.rotation.x = Math.PI / 2;
    socket.position.set(mount.x, 0.92, mount.z);
    scene.add(socket);
  }
  // Low exhibit walls frame the workspace without enclosing the camera rig.
  scene.add(solidBox([49, 5, 0.8], [6, 2.5, -13], '#d6cbb5', 0.2));
  scene.add(solidBox([0.8, 5, 65], [-18.1, 2.5, 19], '#d6cbb5', 0.2));
}

function convexHull(points) {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const half = (values) => {
    const result = [];
    for (const p of values) {
      while (result.length >= 2 && cross(result[result.length - 2], result[result.length - 1], p) <= 0) result.pop();
      result.push(p);
    }
    return result.slice(0, -1);
  };
  return [...half(sorted), ...half(sorted.reverse())];
}

function addOptics(scene, mounts, target, size) {
  const colors = ['#549cdf', '#ba8cd9'];
  mounts.forEach((mount, index) => {
    const color = colors[index];
    const direction = target.clone().sub(mount).normalize();
    const eye = mount.clone().addScaledVector(direction, 2.1);
    const distance = eye.distanceTo(target) * 0.5;
    const fieldOfView = 24;
    const virtualCamera = new THREE.PerspectiveCamera(fieldOfView, 16 / 9, 0.1, 100);
    virtualCamera.position.copy(eye);
    virtualCamera.lookAt(target);
    virtualCamera.updateMatrixWorld(true);
    const height = 2 * distance * Math.tan(THREE.MathUtils.degToRad(fieldOfView / 2));
    const width = height * 16 / 9;
    const plane = new THREE.Group();
    plane.position.copy(eye).addScaledVector(direction, distance);
    plane.quaternion.copy(virtualCamera.quaternion);
    const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ color: '#10203b', transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false }));
    plane.add(backdrop);
    plane.add(stroke([[-width / 2, -height / 2, 0], [width / 2, -height / 2, 0], [width / 2, height / 2, 0], [-width / 2, height / 2, 0], [-width / 2, -height / 2, 0]], color, 0.65));

    // Project the actual box corners, rather than inventing a generic mask shape.
    const projected = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      const point = target.clone().add(new THREE.Vector3(x * size[0] / 2, y * size[1] / 2, z * size[2] / 2)).project(virtualCamera);
      projected.push(new THREE.Vector2(point.x * width / 2, point.y * height / 2));
    }
    const mask = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(convexHull(projected))), new THREE.MeshBasicMaterial({ color: '#55e989', side: THREE.DoubleSide, transparent: true, opacity: 0.8, depthWrite: false, toneMapped: false }));
    mask.position.z = 0.015;
    plane.add(mask);
    plane.add(inscription(`VIEW ${index === 0 ? 'A' : 'B'}`, [0, height / 2 + 0.8, 0], 7, color, [0, 0, 0]));
    scene.add(plane);
    plane.updateMatrixWorld(true);
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => plane.localToWorld(new THREE.Vector3(x * width / 2, y * height / 2, 0)));
    for (const corner of corners) scene.add(stroke([eye.toArray(), corner.toArray()], color, 0.3));
    for (let i = 0; i < corners.length; i++) {
      const points = [eye, corners[i], corners[(i + 1) % corners.length]];
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flatMap((p) => p.toArray()), 3));
      scene.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.025, depthWrite: false, toneMapped: false })));
    }
    scene.add(stroke([plane.position.toArray(), target.toArray()], color, 0.45));
  });
  // Spatial reticle marks the workspace, rather than filling the world with grid.
  for (const radius of [8, 11]) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(radius, radius + 0.035, 96), new THREE.MeshBasicMaterial({ color: '#35577c', side: THREE.DoubleSide, transparent: true, opacity: 0.8 }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(target.x - 5, 0.025, 0);
    scene.add(ring);
  }
}

export function addSceneEnvironment(scene, style, mounts, target, size) {
  if (style === 'porcelain') addExhibit(scene, mounts);
  if (style === 'blueprint') addOptics(scene, mounts, target, size);
}
