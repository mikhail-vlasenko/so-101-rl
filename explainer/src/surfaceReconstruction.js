import * as THREE from 'three';
import { capture, panel, beam, dot, textLabel } from './storyFrames.js';
import { CAMERA_A_COLOR, CAMERA_B_COLOR } from './sceneConfig.js';
import { SNAPSHOT_X } from './motionDraft.js';
import { SURFACE_MATCHES } from './filmTimeline.js';

/** The three explained points are actual cloud samples, not extra oversized
 * markers. Their surface UVs put recognizable pores at the same physical
 * locations in both captured views; all remaining samples fill in afterward.
 */
export function buildSurfaceReconstruction(objects, renderer) {
  const resources = [];
  const { scene, sponge, cloud, captureCameras } = objects;
  for (const object of [cloud, objects.position, objects.surfaceLabel, objects.positionLabel, objects.surfaceLeader, objects.positionLeader, objects.arrow]) object.visible = false;
  for (const rig of captureCameras) { rig.guide.visible = false; rig.caption.visible = false; }
  sponge.position.x = SNAPSHOT_X;
  sponge.updateMatrixWorld(true);

  const selected = SURFACE_MATCHES.map(({ sampleIndex }) => sampleIndex);
  const matrices = Array.from({ length: cloud.count }, (_, index) => {
    const matrix = new THREE.Matrix4();
    cloud.getMatrixAt(index, matrix);
    return matrix;
  });
  const points = selected.map((index) => new THREE.Vector3().setFromMatrixPosition(matrices[index]));
  const ordered = [...selected, ...matrices.map((_, index) => index).filter((index) => !selected.includes(index))];
  ordered.forEach((source, index) => cloud.setMatrixAt(index, matrices[source]));
  cloud.instanceMatrix.needsUpdate = true;

  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 512;
  const ctx = canvas.getContext('2d');
  // Bake the gold base into the texture so its gray markings are not tinted
  // brown by multiplication with the sponge's gold material color.
  ctx.fillStyle = sponge.material.color.clone().multiply(new THREE.Color('#f0ead1')).getStyle();
  ctx.fillRect(0, 0, 512, 512);
  const normals = [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0)];
  const raycaster = new THREE.Raycaster();
  points.forEach((point, index) => {
    const normal = normals[selected[index] % 3];
    raycaster.set(point.clone().addScaledVector(normal, 0.5), normal.clone().negate());
    const hits = raycaster.intersectObject(sponge);
    if (!hits.length) throw new Error(`Surface feature ${index} misses the sponge`);
    const uv = hits[0].uv;
    const x = uv.x * 512;
    const y = (1 - uv.y) * 512;
    ctx.fillStyle = '#686868';
    ctx.beginPath();
    ctx.ellipse(x, y, 12 + index * 2, 9, index * 0.65, 0, Math.PI * 2);
    ctx.fill();
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  resources.push(texture);
  const images = captureCameras.map((rig, index) => capture(renderer, objects, rig.lens, SNAPSHOT_X, false, resources, null, index === 0 ? CAMERA_A_COLOR : CAMERA_B_COLOR, texture, true));
  const views = [
    panel(scene, images[0], [-10.7, 7.5, -4], 10, 'Camera A', CAMERA_A_COLOR),
    panel(scene, images[1], [0.7, 7.5, -4], 10, 'Camera B', CAMERA_B_COLOR),
  ];
  sponge.visible = false;
  cloud.visible = true;
  const context = new THREE.Group();
  for (const view of views) context.add(view.group, view.captionLabel);
  scene.add(context);
  objects.surfaceContext = context;
  objects.surfaceMatches = points.map((point) => {
    const a = views[0].project(point);
    const b = views[1].project(point);
    const features = [dot(a, '#f3ca7a', 0.065), dot(b, '#f3ca7a', 0.065)];
    for (const feature of features) {
      feature.material.transparent = true;
      feature.material.depthTest = false;
      feature.material.depthWrite = false;
      feature.renderOrder = 13;
    }
    context.add(...features);
    const lines = [
      { mesh: beam(a, b, '#f3ca7a', 0.015), origin: a, phase: 'connection' },
      { mesh: beam(a, point, '#5cec55', 0.018), origin: a, phase: 'rays' },
      { mesh: beam(b, point, '#5cec55', 0.018), origin: b, phase: 'rays' },
    ];
    for (const { mesh } of lines) {
      mesh.material.transparent = true;
      mesh.material.depthWrite = false;
      mesh.material.depthTest = false;
      mesh.renderOrder = 12;
      context.add(mesh);
    }
    return { features, lines };
  });
  scene.add(textLabel('Visible surface', [-5, 4.2, 0], 10, '#b7f0b1'));
  return resources;
}
