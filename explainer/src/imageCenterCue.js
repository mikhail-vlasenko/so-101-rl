import * as THREE from 'three';

function turn(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/** Project the authored convex sponge silhouette into the image plane. This
 * cue illustrates mask-centroid reduction, not a physical surface measurement.
 */
export function imageSilhouette(sponge, camera) {
  sponge.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const vertices = sponge.geometry.attributes.position;
  const projected = Array.from({ length: vertices.count }, (_, index) => new THREE.Vector3()
    .fromBufferAttribute(vertices, index).applyMatrix4(sponge.matrixWorld).project(camera));
  projected.sort((a, b) => a.x - b.x || a.y - b.y);
  const unique = projected.filter((point, index) => index === 0 || point.x !== projected[index - 1].x || point.y !== projected[index - 1].y);
  const lower = [];
  const upper = [];
  for (const point of unique) {
    while (lower.length >= 2 && turn(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
    lower.push(point);
  }
  for (const point of unique.toReversed()) {
    while (upper.length >= 2 && turn(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
    upper.push(point);
  }
  const outline = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  if (outline.length < 3) throw new Error('The image-center cue needs a visible silhouette');
  let area = 0;
  const center = new THREE.Vector3();
  outline.forEach((a, index) => {
    const b = outline[(index + 1) % outline.length];
    const cross = a.x * b.y - b.x * a.y;
    area += cross;
    center.x += (a.x + b.x) * cross;
    center.y += (a.y + b.y) * cross;
  });
  if (area === 0) throw new Error('The image-center cue has zero area');
  center.divideScalar(3 * area);
  const depth = sponge.position.clone().project(camera).z;
  center.z = depth;
  center.unproject(camera);
  const offsets = outline.map((point) => new THREE.Vector3(point.x, point.y, depth).unproject(camera).sub(center));
  return { center, offsets };
}

export function buildImageCenterCue(sponge, camera) {
  const { center, offsets } = imageSilhouette(sponge, camera);
  const group = new THREE.Group();
  group.position.copy(center);
  const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(offsets),
    new THREE.LineBasicMaterial({ color: '#94f7ff', transparent: true, opacity: 0, depthTest: false, depthWrite: false, toneMapped: false, fog: false }));
  outline.renderOrder = 28;
  group.add(outline);
  const lengths = offsets.map((point, index) => point.distanceTo(offsets[(index + 1) % offsets.length]));
  const perimeter = lengths.reduce((sum, length) => sum + length, 0);
  const particles = Array.from({ length: 12 }, (_, index) => {
    let distance = perimeter * index / 12;
    let edge = 0;
    while (edge < lengths.length - 1 && distance > lengths[edge]) distance -= lengths[edge++];
    const origin = offsets[edge].clone().lerp(offsets[(edge + 1) % offsets.length], distance / lengths[edge]);
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 8),
      new THREE.MeshBasicMaterial({ color: '#94f7ff', transparent: true, opacity: 0, depthTest: false, depthWrite: false, toneMapped: false, fog: false }));
    mesh.renderOrder = 29;
    group.add(mesh);
    return { mesh, origin };
  });
  return { group, outline, particles, center };
}
