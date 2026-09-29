import * as THREE from 'three';
import { buildStoryFrame, capture, panel, ghost, spongeCorner, beam, dot, closestRayPoint, textLabel } from './storyFrames.js';
import { CAMERA_A_COLOR, CAMERA_B_COLOR } from './sceneConfig.js';
import { SHUTTERS, spongeX, introView } from './filmTimeline.js';
import { LIVE_X, SNAPSHOT_X } from './motionDraft.js';
import { buildSurfaceReconstruction } from './surfaceReconstruction.js';
import { buildImageCenterCue } from './imageCenterCue.js';

/** Continuous film scenes reuse the gallery geometry and stereo calculations.
 * Captured silhouettes and image textures are immutable; the live sponge is
 * a separate object that continues moving after both exposures.
 */
export function buildFilmShot(objects, id, renderer) {
  if (id === 'intro') {
    const resources = buildStoryFrame(objects, 'position', renderer);
    const mask = new THREE.Mesh(objects.sponge.geometry, new THREE.MeshBasicMaterial({ color: '#42d568', transparent: true, opacity: 0, depthWrite: false, toneMapped: false }));
    mask.position.copy(objects.sponge.position);
    mask.scale.setScalar(1.012);
    objects.scene.add(mask);
    objects.modelMask = mask;
    const view = introView(8, objects.captureCameras[0].lens.toArray());
    const camera = new THREE.PerspectiveCamera(view.fov, 16 / 9, 0.1, 2000);
    camera.position.set(...view.eye);
    camera.lookAt(...view.target);
    objects.imageCenterCue = buildImageCenterCue(objects.sponge, camera);
    objects.positionEstimate = objects.position.position.clone();
    objects.scene.add(objects.imageCenterCue.group);
    objects.sponge.material.opacity = 1;
    objects.sponge.material.transparent = false;
    objects.sponge.material.depthWrite = true;
    return resources;
  }
  if (id === 'surface') return buildSurfaceReconstruction(objects, renderer);
  if (id !== 'movement') throw new Error(`Unknown film shot: ${id}`);

  const resources = buildStoryFrame(objects, 'reject', renderer);
  const fadingShadow = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, alphaHash: true });
  objects.spongeShadowOpacity = { value: 1 };
  fadingShadow.onBeforeCompile = (shader) => {
    shader.uniforms.fadingShadowOpacity = objects.spongeShadowOpacity;
    shader.fragmentShader = 'uniform float fadingShadowOpacity;\n' + shader.fragmentShader.replace('vec4 diffuseColor = vec4( 1.0 );', 'vec4 diffuseColor = vec4(1.0, 1.0, 1.0, fadingShadowOpacity);');
  };
  objects.sponge.customDepthMaterial = fadingShadow;
  resources.push(fadingShadow);
  const { scene } = objects;
  objects.heldTitle.visible = false;
  objects.surfaceLabel.scale.set(7.4, 0.925, 1);
  objects.positionLabel.scale.set(7.4, 0.925, 1);
  objects.positionLabel.position.y = 5.85;
  const mounts = objects.captureCameras.map((rig) => rig.lens.clone());
  const positions = [spongeX(SHUTTERS.first), spongeX(SHUTTERS.second)];
  const colors = [CAMERA_A_COLOR, CAMERA_B_COLOR];
  const corners = positions.map((x) => spongeCorner(objects, x));
  const directions = corners.map((corner, index) => corner.clone().sub(mounts[index]).normalize());
  const estimate = closestRayPoint(mounts[0], directions[0], mounts[1], directions[1]);
  objects.shutterPanels = [];
  objects.shutterContexts = [];
  objects.shutterEvidence = [];
  objects.triangulationRays = [];
  for (let index = 0; index < 2; index++) {
    const image = capture(renderer, objects, mounts[index], positions[index], false, resources, corners[index], colors[index]);
    const view = panel(scene, image, [index === 0 ? -4.7 : 4.7, 6.6, -4], 8.5, index === 0 ? 'Camera A · captured' : 'Camera B · captured', colors[index]);
    const context = new THREE.Group();
    context.add(view.group, view.captionLabel);
    scene.add(context);
    objects.shutterPanels.push(view);
    objects.shutterContexts.push(context);
    const evidence = new THREE.Group();
    const feature = dot(corners[index], colors[index], 0.22);
    feature.material.transparent = true;
    feature.material.depthTest = false;
    feature.material.depthWrite = false;
    feature.renderOrder = 40;
    evidence.add(ghost(objects, positions[index], colors[index]), feature);
    scene.add(evidence);
    objects.shutterEvidence.push(evidence);
    const end = mounts[index].clone().addScaledVector(directions[index], mounts[index].distanceTo(estimate) + 3);
    const ray = beam(mounts[index], end, colors[index]);
    scene.add(ray);
    objects.triangulationRays.push({ mesh: ray, origin: mounts[index] });
  }
  const falseDepth = new THREE.Group();
  const errorPoint = dot(estimate, '#ff9462', 0.4);
  const errorLeader = beam(corners[0].clone().add(corners[1]).multiplyScalar(0.5), estimate, '#ff9462', 0.035);
  // This is explanatory evidence, not a physical object occluded by the live sponge.
  for (const overlay of [errorPoint, errorLeader]) {
    overlay.material.transparent = true;
    overlay.material.depthTest = false;
    overlay.material.depthWrite = false;
    overlay.renderOrder = 45;
  }
  falseDepth.add(errorPoint, errorLeader);
  falseDepth.add(textLabel('Movement mistaken for depth', [estimate.x, estimate.y + 3.7, estimate.z], 13.8, '#ffb38a'));
  scene.add(falseDepth);
  objects.falseDepth = falseDepth;
  // Separate accepted geometry: the old snapshot is never moved with the sponge.
  objects.refreshedCloud = objects.cloud.clone();
  objects.refreshedCloud.material = objects.cloud.material.clone();
  objects.refreshedCloud.position.x = LIVE_X - SNAPSHOT_X;
  objects.refreshedCloud.visible = false;
  scene.add(objects.refreshedCloud);
  return resources;
}
