import * as THREE from 'three';
import { LIVE_X, SNAPSHOT_X } from './motionDraft.js';
import { CAMERA_AIM_X, CAMERA_IMAGE_FOV_DEG, CAMERA_A_COLOR, CAMERA_B_COLOR } from './sceneConfig.js';

/** Static editorial keyframes. Camera panels are renders of our 3D model,
 * not recorded frames. The shutter sequence is slowed for illustration.
 */
const SHUTTER_STEP_CM = 1;
const POST_SHUTTER_TRAVEL_CM = 1.4;
export const SECOND_SHUTTER_X = LIVE_X - POST_SHUTTER_TRAVEL_CM;
export const FIRST_SHUTTER_X = SECOND_SHUTTER_X - SHUTTER_STEP_CM;

export const STORY_FRAMES = {
  views: { title: '01 · Two views', note: 'The same stationary sponge, seen from two cameras.', eye: [10, 10, 0], target: [10, 14, 26] },
  masks: { title: '02 · Find the sponge', note: 'Object masks and their image centers.', eye: [10, 10, 0], target: [10, 14, 26] },
  position: { title: '03 · Position', note: 'An oblique top view reveals the triangle between two fixed cameras and the position estimate.', eye: [30, 63, 20], target: [5, 5, 20], fov: 36 },
  surface: { title: '04 · Surface', note: 'Matching details form a visible-surface cloud; the viewer sweeps left to reveal its empty back.', eye: [-21, 8, -12], orbitFrom: [2.7, 11, 18.6], orbitTargetFrom: [-5, 4.5, 0], orbitDurationMs: 3600, target: [-5, 2, 0] },
  timing: { title: '05a · Left shutter', note: 'After the sponge moves, the left camera freezes one visible corner.', eye: [12, 17, 33], target: [0, 6, 2] },
  second: { title: '05b · Right shutter', note: 'The sponge shifts slightly before the right camera freezes the same corner.', eye: [12, 17, 33], target: [0, 6, 2] },
  depth: { title: '05c · False depth', note: 'Two views of a moving corner can give an incorrect depth.', eye: [20, 12, 14], target: [6, 2, -2], fov: 35 },
  reject: { title: '05d · Cloud held', note: 'The orange example disappears; the unchanged green cloud stays behind the moving position estimate.', eye: [12, 11, 24], target: [0, 1.6, 0] },
  refresh: { title: '06 · Refresh', note: 'The sponge has settled; a new surface measurement replaces the old one.', eye: [12, 7, 16], target: [5, 2, 0] },
};

export function textLabel(text, location, width, color = '#c5d8e8') {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.font = '400 52px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(text, 512, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, toneMapped: false }));
  sprite.scale.set(width, width / 8, 1);
  sprite.position.set(...location);
  sprite.renderOrder = 20;
  return sprite;
}

function stroke(points, color, opacity = 1) {
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity, toneMapped: false }));
}

export function beam(start, end, color, radius = 0.06) {
  const path = new THREE.CatmullRomCurve3([start, end]);
  return new THREE.Mesh(new THREE.TubeGeometry(path, 1, radius, 8, false), new THREE.MeshBasicMaterial({ color, toneMapped: false }));
}

export function dot(point, color, radius = 0.14) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), new THREE.MeshBasicMaterial({ color, toneMapped: false }));
  mesh.position.copy(point);
  return mesh;
}

export function capture(renderer, objects, mount, objectX, masked, resources, corner = null, cornerColor = CAMERA_A_COLOR, detailTexture = null, cropToObject = false) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#152131');
  scene.add(new THREE.HemisphereLight('#d6eaff', '#24343e', 2));
  const light = new THREE.DirectionalLight('#fff3d8', 3);
  light.position.set(-4, 12, 8);
  scene.add(light);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: '#283a4c', roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
  const object = new THREE.Mesh(objects.sponge.geometry, new THREE.MeshStandardMaterial({ color: detailTexture ? '#ffffff' : '#c3a341', roughness: 0.85, map: detailTexture }));
  object.position.set(objectX, objects.sponge.position.y, 0);
  scene.add(object);
  const camera = new THREE.PerspectiveCamera(CAMERA_IMAGE_FOV_DEG, 16 / 9, 0.1, 200);
  camera.position.copy(mount);
  camera.lookAt(CAMERA_AIM_X, object.position.y, 0);
  camera.updateMatrixWorld(true);
  if (cropToObject) {
    // Magnify a crop of the fixed optical view; do not aim the physical camera
    // at the object or change which world features correspond between images.
    const center = object.position.clone().project(camera);
    const left = THREE.MathUtils.clamp((center.x * 0.5 + 0.5) * 1024 - 256, 0, 512);
    const top = THREE.MathUtils.clamp((-center.y * 0.5 + 0.5) * 576 - 144, 0, 288);
    camera.setViewOffset(1024, 576, left, top, 512, 288);
  }
  if (masked) {
    const overlay = new THREE.Mesh(objects.sponge.geometry, new THREE.MeshBasicMaterial({ color: '#42d568', transparent: true, opacity: 0.57, depthWrite: false, toneMapped: false }));
    overlay.position.copy(object.position);
    overlay.scale.setScalar(1.012);
    scene.add(overlay);
    const marker = dot(object.position, '#94f7ff', 0.16);
    marker.material.depthTest = false;
    marker.renderOrder = 10;
    scene.add(marker);
    resources.push(overlay.material, marker.geometry, marker.material);
  }
  if (corner) {
    const marker = dot(corner, cornerColor, 0.22);
    marker.material.depthTest = false;
    marker.renderOrder = 10;
    scene.add(marker);
    resources.push(marker.geometry, marker.material);
  }
  const target = new THREE.WebGLRenderTarget(1024, 576);
  target.texture.colorSpace = THREE.SRGBColorSpace;
  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  resources.push(target, ground.geometry, ground.material, object.material);
  return { texture: target.texture, camera };
}

export function panel(scene, image, position, width, caption, color = '#a4bdcc', facingBack = false) {
  const height = width * 9 / 16;
  const group = new THREE.Group();
  group.position.set(...position);
  if (facingBack) group.rotation.y = Math.PI;
  const frame = new THREE.Mesh(new THREE.BoxGeometry(width + 0.16, height + 0.16, 0.16), new THREE.MeshStandardMaterial({ color: '#354858', roughness: 0.6, metalness: 0.3 }));
  group.add(frame);
  const accent = new THREE.Mesh(new THREE.BoxGeometry(width, 0.055, 0.05), new THREE.MeshBasicMaterial({ color, toneMapped: false }));
  accent.position.set(0, height / 2 + 0.08, 0.13);
  group.add(accent);
  const picture = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: image.texture, toneMapped: false }));
  picture.position.z = 0.09;
  group.add(picture);
  scene.add(group);
  const captionLabel = textLabel(caption, [position[0], position[1] + height / 2 + 0.8, position[2]], width, color);
  scene.add(captionLabel);
  group.updateMatrixWorld(true);
  return {
    group,
    frame,
    captionLabel,
    project(point) {
      const uv = point.clone().project(image.camera);
      return group.localToWorld(new THREE.Vector3(uv.x * width / 2, uv.y * height / 2, 0.13));
    },
  };
}

export function ghost(objects, x, color) {
  const mesh = new THREE.Mesh(objects.sponge.geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12, depthWrite: false }));
  mesh.position.set(x, objects.sponge.position.y, 0);
  mesh.scale.setScalar(1.018);
  objects.sponge.geometry.computeBoundingBox();
  const size = objects.sponge.geometry.boundingBox.getSize(new THREE.Vector3());
  const bounds = new THREE.BoxGeometry(size.x, size.y, size.z);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(bounds), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.65 }));
  bounds.dispose();
  mesh.add(edges);
  return mesh;
}

export function spongeCorner(objects, centerX) {
  objects.sponge.geometry.computeBoundingBox();
  const corner = objects.sponge.geometry.boundingBox.max;
  return new THREE.Vector3(centerX + corner.x, objects.sponge.position.y + corner.y, corner.z);
}

function shutterPanel(scene, image, x, caption, color, active, dimmed) {
  const view = panel(scene, image, [x, 10.5, 7], 10, caption, color);
  if (active) {
    view.frame.material.color.set(color);
    view.frame.material.emissive.set(color);
    view.frame.material.emissiveIntensity = 0.25;
  }
  if (dimmed) {
    const shade = new THREE.Mesh(new THREE.PlaneGeometry(10, 10 * 9 / 16), new THREE.MeshBasicMaterial({ color: '#101a28', toneMapped: false }));
    shade.position.z = 0.15;
    view.group.add(shade);
    scene.add(textLabel('Waiting', [x, 10.5, 7.2], 6, '#8091a4'));
  }
  return view;
}

export function closestRayPoint(a, u, b, v) {
  const w = a.clone().sub(b);
  const cosine = u.dot(v);
  const denominator = 1 - cosine * cosine;
  if (denominator < 1e-8) throw new Error('Illustration rays are parallel');
  const s = (cosine * v.dot(w) - u.dot(w)) / denominator;
  const t = (v.dot(w) - cosine * u.dot(w)) / denominator;
  return a.clone().addScaledVector(u, s).add(b.clone().addScaledVector(v, t)).multiplyScalar(0.5);
}

export function buildStoryFrame(objects, id, renderer) {
  const resources = [];
  const { scene, sponge, cloud, position, captureCameras } = objects;
  const mounts = captureCameras.map((rig) => rig.lens.clone());
  for (const object of [cloud, position, objects.surfaceLabel, objects.positionLabel, objects.surfaceLeader, objects.positionLeader, objects.arrow]) object.visible = false;
  for (const rig of captureCameras) rig.guide.visible = false;
  sponge.position.x = SNAPSHOT_X;
  const stationary = sponge.position.clone();

  if (['views', 'masks', 'surface'].includes(id)) {
    // Rig stays physically intact, but offstage for these image-focused shots.
    for (const rig of captureCameras) rig.caption.visible = false;
    const masked = id === 'masks';
    const first = capture(renderer, objects, mounts[0], SNAPSHOT_X, masked, resources);
    const second = capture(renderer, objects, mounts[1], SNAPSHOT_X, masked, resources);
    const opening = id === 'views' || id === 'masks';
    const width = 10;
    const y = opening ? 11 : 7.5;
    const z = opening ? 22 : -4;
    const center = opening ? 10 : id === 'surface' ? -5 : 0;
    const left = panel(scene, first, [center - width / 2 - 0.7, y, z], width, 'Camera A', CAMERA_A_COLOR, opening);
    const right = panel(scene, second, [center + width / 2 + 0.7, y, z], width, 'Camera B', CAMERA_B_COLOR, opening);
    sponge.visible = false;
    if (id === 'masks') {
      scene.add(textLabel('“sponge”', [10, 17.7, 22], 10, '#9af3aa'));
      objects.imageSubtitle = textLabel('Find the object’s pixels', [10, 7.1, 22], 13);
      scene.add(objects.imageSubtitle);
    }
    if (id === 'views') {
      objects.imageSubtitle = textLabel('One sponge. Two viewpoints.', [10, 7.1, 22], 13);
      scene.add(objects.imageSubtitle);
    }
    if (id === 'surface') {
      cloud.visible = true;
      const context = new THREE.Group();
      context.add(left.group, left.captionLabel, right.group, right.captionLabel);
      scene.add(context);
      objects.surfaceContext = context;
      scene.add(textLabel('Visible surface', [-5, 4.2, 0], 10, '#b7f0b1'));
      const features = [new THREE.Vector3(-3.3, 2.535, 1.2), new THREE.Vector3(-6.6, 2.535, 0.2)];
      for (const point of features) {
        const a = left.project(point);
        const b = right.project(point);
        context.add(dot(a, '#f3ca7a', 0.08), dot(b, '#f3ca7a', 0.08));
        context.add(stroke([a, b], '#f3ca7a', 0.7));
        context.add(stroke([a, point, b], '#71ac98', 0.3));
      }
    }
  }

    if (id === 'timing' || id === 'second') {
    const rightFired = id === 'second';
    const firstCorner = spongeCorner(objects, FIRST_SHUTTER_X);
    const secondCorner = spongeCorner(objects, SECOND_SHUTTER_X);
    for (const rig of captureCameras) rig.caption.visible = false;
    const firstImage = capture(renderer, objects, mounts[0], FIRST_SHUTTER_X, false, resources, firstCorner);
    const secondImage = capture(renderer, objects, mounts[1], rightFired ? SECOND_SHUTTER_X : FIRST_SHUTTER_X, false, resources, rightFired ? secondCorner : null, CAMERA_B_COLOR);
    const left = shutterPanel(scene, firstImage, -5.7, rightFired ? 'Camera A · saved' : 'Camera A · shutter', CAMERA_A_COLOR, !rightFired, false);
    const right = shutterPanel(scene, secondImage, 5.7, rightFired ? 'Camera B · shutter' : 'Camera B · waiting', CAMERA_B_COLOR, rightFired, !rightFired);
    objects.shutterPanels = [left, right];
    sponge.position.x = rightFired ? SECOND_SHUTTER_X : FIRST_SHUTTER_X;
    position.position.x = sponge.position.x;
    cloud.visible = true;
    position.visible = true;
    objects.arrow.visible = true;
    const firstEvidence = new THREE.Group();
    firstEvidence.add(ghost(objects, FIRST_SHUTTER_X, CAMERA_A_COLOR), dot(firstCorner, CAMERA_A_COLOR, 0.22), stroke([left.project(firstCorner), firstCorner], CAMERA_A_COLOR, 0.6));
    scene.add(firstEvidence);
    objects.shutterEvidence = [firstEvidence];
    if (rightFired) {
      const secondEvidence = new THREE.Group();
      secondEvidence.add(ghost(objects, SECOND_SHUTTER_X, CAMERA_B_COLOR), dot(secondCorner, CAMERA_B_COLOR, 0.22), stroke([right.project(secondCorner), secondCorner], CAMERA_B_COLOR, 0.6));
      scene.add(secondEvidence);
      objects.shutterEvidence.push(secondEvidence);
    }
    objects.shutterTitles = new THREE.Group();
    objects.shutterTitles.add(textLabel(rightFired ? 'Right shutter · same corner, later' : 'Left shutter · corner frozen', [0, 18, 7], 21));
    objects.shutterTitles.add(textLabel('Shown in slow motion', [0, 16.4, 7], 14, '#d9be91'));
    scene.add(objects.shutterTitles);
  }

  if (id === 'position' || id === 'depth') {
    const errorDemo = id === 'depth';
    const points = errorDemo
      ? [spongeCorner(objects, FIRST_SHUTTER_X), spongeCorner(objects, SECOND_SHUTTER_X)]
      : [stationary.clone().add(new THREE.Vector3(0, 0, 0.18)), stationary.clone().add(new THREE.Vector3(0.12, 0, -0.18))];
    const directions = points.map((p, i) => p.clone().sub(mounts[i]).normalize());
    const estimate = closestRayPoint(mounts[0], directions[0], mounts[1], directions[1]);
    objects.triangulationRays = [];
    objects.triangulationFeatures = [];
    for (let i = 0; i < 2; i++) {
      const color = i === 0 ? CAMERA_A_COLOR : CAMERA_B_COLOR;
      const end = mounts[i].clone().addScaledVector(directions[i], mounts[i].distanceTo(estimate) + 3);
      const ray = beam(mounts[i], end, color);
      scene.add(ray);
      objects.triangulationRays.push({ mesh: ray, origin: mounts[i] });
      const feature = dot(points[i], color, errorDemo ? 0.24 : 0.14);
      if (errorDemo) { feature.material.depthTest = false; feature.renderOrder = 12; }
      scene.add(feature);
      objects.triangulationFeatures.push(feature);
    }
    if (errorDemo) {
      sponge.visible = false;
      scene.add(ghost(objects, FIRST_SHUTTER_X, CAMERA_A_COLOR), ghost(objects, SECOND_SHUTTER_X, CAMERA_B_COLOR));
      const falseDepth = new THREE.Group();
      falseDepth.add(dot(estimate, '#ff9462', 0.4));
      falseDepth.add(beam(points[0].clone().add(points[1]).multiplyScalar(0.5), estimate, '#ff9462', 0.035));
      objects.falseDepthLabel = textLabel('Movement mistaken for depth', [estimate.x, estimate.y + 3.7, estimate.z], 23, '#ffb38a');
      falseDepth.add(objects.falseDepthLabel);
      scene.add(falseDepth);
      objects.falseDepth = falseDepth;
      objects.depthTitle = textLabel('Same corner · different moments', [6, 6.5, 0], 20);
      scene.add(objects.depthTitle);
      scene.add(stroke([points[0], points[1]], '#d5e4f0', 0.6));
    } else {
      sponge.material.transparent = true;
      sponge.material.opacity = 0.22;
      sponge.material.depthWrite = false;
      position.visible = true;
      position.position.copy(estimate);
      position.scale.setScalar(2.2);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 8, 64), new THREE.MeshBasicMaterial({ color: '#91e9f2', transparent: true, opacity: 0.7, toneMapped: false }));
      ring.rotation.x = Math.PI / 2;
      ring.position.set(estimate.x, 0.08, estimate.z);
      scene.add(ring);
      objects.triangulationRing = ring;
      objects.triangulationTitle = textLabel('Approximate position', [estimate.x, estimate.y + 5, estimate.z], 25, '#91e9f2');
      scene.add(objects.triangulationTitle);
    }
  }

  if (id === 'reject') {
    sponge.position.x = LIVE_X;
    position.visible = true;
    cloud.visible = true;
    objects.arrow.visible = true;
    objects.surfaceLabel.visible = true;
    objects.surfaceLeader.visible = true;
    objects.positionLabel.visible = true;
    objects.positionLeader.visible = true;
    for (const rig of captureCameras) rig.caption.visible = false;
    objects.heldTitle = textLabel('No new surface update', [0, 6.8, 0], 19, '#b7f0b1');
    scene.add(objects.heldTitle);
  }

  if (id === 'refresh') {
    sponge.position.x = LIVE_X;
    sponge.material.transparent = true;
    sponge.material.opacity = 0.14;
    sponge.material.depthWrite = false;
    cloud.visible = true;
    // A new accepted measurement at the settled position, not transported memory.
    cloud.position.x = LIVE_X - SNAPSHOT_X;
    position.visible = true;
    scene.add(textLabel('Still again · surface refreshed', [5, 5.4, 0], 16, '#b7f0b1'));
    scene.add(textLabel('Position + surface', [5, 0.2, 4], 10, '#a2edf5'));
    for (const rig of captureCameras) rig.caption.visible = false;
  }
  return resources;
}
