import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import armModelXml from '../../so101/so101.xml?raw';
import { SCENE_STYLES, applySceneStyle } from './sceneStyles.js';
import { addSceneEnvironment } from './sceneEnvironments.js';
import { DRAFT_DURATION, LIVE_X, SNAPSHOT_X, sampleDraft } from './motionDraft.js';
import { STORY_FRAMES, buildStoryFrame } from './storyFrames.js';
import { CAMERA_AIM_X, CAMERA_A_COLOR, CAMERA_B_COLOR, CAMERA_PAIR_SHIFT_X, SPONGE_SIZE } from './sceneConfig.js';

/** Authored 3D scene; units are centimeters, Y is up.
 * The shared draft timeline moves the live estimate independently of the held
 * surface. Offline rendering and interactive playback sample identical state.
 * Surface samples are seeded illustrations, not measured perception data.
 */
function cameraMountPosition(name) {
  // MuJoCo accepts comments containing CLI flags (--), unlike browser XML parsers.
  const xml = new DOMParser().parseFromString(armModelXml.replace(/<!--[\s\S]*?-->/g, ''), 'application/xml');
  const parseError = xml.querySelector('parsererror');
  if (parseError) throw new Error(`Invalid arm model XML: ${parseError.textContent}`);
  const body = xml.querySelector(`body[name="${name}"]`);
  if (!body || !body.hasAttribute('pos')) throw new Error(`Missing camera mount position: ${name}`);
  const coordinates = body.getAttribute('pos').trim().split(/\s+/).map(Number);
  if (coordinates.length !== 3 || coordinates.some((value) => !Number.isFinite(value))) {
    throw new Error(`Invalid camera mount position: ${name}`);
  }
  const [x, y, z] = coordinates;
  // Workspace origin is an authored point at base (0.20, 0, 0) meters.
  // Rotate MuJoCo Z-up into Three.js Y-up, preserving distances and handedness.
  return [(x - 0.20) * 100 + CAMERA_PAIR_SHIFT_X, z * 100, -y * 100];
}

function random(seed) {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function label(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.font = '400 62px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(text, 512, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true, toneMapped: false, fog: false }));
  sprite.renderOrder = 10;
  sprite.scale.set(7.4, 0.925, 1);
  return sprite;
}

function line(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
}

function addCaptureCamera(scene, name, location, accent, textColor) {
  const [x, y, z] = location;
  const housing = new THREE.MeshStandardMaterial({ color: '#414d5c', roughness: 0.42, metalness: 0.55 });
  const dark = new THREE.MeshStandardMaterial({ color: '#101720', roughness: 0.4, metalness: 0.45 });
  const rig = new THREE.Group();
  rig.scale.setScalar(3);
  rig.position.set(x, y, z);
  rig.lookAt(CAMERA_AIM_X, SPONGE_SIZE[1] / 2, 0);
  const body = new THREE.Mesh(new RoundedBoxGeometry(2.1, 1.05, 0.85, 3, 0.12), housing);
  body.castShadow = true;
  rig.add(body);
  const bezel = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.43, 0.28, 32), dark);
  bezel.rotation.x = Math.PI / 2;
  bezel.position.z = 0.52;
  rig.add(bezel);
  const glass = new THREE.Mesh(new THREE.CircleGeometry(0.32, 32), new THREE.MeshStandardMaterial({ color: '#173953', roughness: 0.13, metalness: 0.8 }));
  glass.position.z = 0.668;
  rig.add(glass);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.027, 8, 40), new THREE.MeshBasicMaterial({ color: accent, toneMapped: false }));
  ring.position.z = 0.675;
  rig.add(ring);
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.055, 10, 8), new THREE.MeshBasicMaterial({ color: accent, toneMapped: false }));
  led.position.set(0.79, 0.12, 0.44);
  rig.add(led);
  scene.add(rig);

  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, y - 1.5, 16), housing);
  stand.position.set(x, (y - 1.5) / 2, z);
  stand.castShadow = true;
  scene.add(stand);
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(2, 2.2, 0.4, 32), dark);
  foot.position.set(x, 0.2, z);
  foot.castShadow = true;
  scene.add(foot);
  const caption = label(name, textColor);
  caption.scale.set(11, 1.375, 1);
  caption.position.set(x, y + 3.2, z);
  scene.add(caption);

  // Illustrative optical-axis guide, not a measured calibration or depth ray.
  rig.updateMatrixWorld(true);
  const lens = rig.localToWorld(new THREE.Vector3(0, 0, 0.7));
  const guide = line([lens.toArray(), [CAMERA_AIM_X, SPONGE_SIZE[1] / 2, 0]], accent, 0.2);
  scene.add(guide);
  return { caption, guide, lens, ring, rig };
}

export function buildScene(style) {
  const theme = SCENE_STYLES[style];
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(theme.background);
  scene.fog = new THREE.FogExp2(theme.background, 0.003);
  scene.add(new THREE.HemisphereLight('#b8d4ef', '#172019', 1.1));

  const key = new THREE.DirectionalLight('#e7f1ff', 3.5);
  key.position.set(-5, 14, 8);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -55, right: 55, top: 55, bottom: -55, near: 0.5, far: 100 });
  key.shadow.normalBias = 0.035;
  key.shadow.bias = -0.0001;
  key.shadow.radius = 4;
  scene.add(key);
  const rim = new THREE.DirectionalLight('#65b5df', 2.2);
  rim.position.set(4, 8, -10);
  scene.add(rim);

  const floorMaterial = new THREE.MeshStandardMaterial({ color: theme.floor, roughness: 0.88, metalness: 0.12, dithering: true });
  if (style === 'studio') {
    // Draw the grid in the floor shader: separate near-coplanar line geometry
    // was depth-fighting with the large table and disappearing at shallow angles.
    floorMaterial.onBeforeCompile = (shader) => {
      shader.uniforms.tableGridColor = { value: new THREE.Color('#60778b') };
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 tableXZ;');
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\ntableXZ = (modelMatrix * vec4(transformed, 1.0)).xz;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 tableXZ;\nuniform vec3 tableGridColor;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `#include <color_fragment>
        vec2 cell = tableXZ / 2.0;
        vec2 footprint = max(fwidth(cell), vec2(0.0001));
        vec2 distanceToLine = abs(fract(cell - 0.5) - 0.5) / footprint;
        float lineCoverage = 1.0 - min(min(distanceToLine.x, distanceToLine.y), 1.0);
        float fade = 1.0 - smoothstep(55.0, 95.0, length(tableXZ));
        float antialiasFade = 1.0 - smoothstep(0.2, 0.8, max(footprint.x, footprint.y));
        diffuseColor.rgb = mix(diffuseColor.rgb, tableGridColor, lineCoverage * fade * antialiasFade * 0.35);
      `);
    };
  }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), floorMaterial);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = style === 'porcelain' ? -5 : 0;
  floor.receiveShadow = true;
  scene.add(floor);
  if (style !== 'studio') {
    const grid = new THREE.GridHelper(120, 60, theme.grid, theme.grid);
    grid.position.y = 0.006;
    grid.material.transparent = true;
    grid.material.opacity = 0.27;
    scene.add(grid);
  }

  const captureCameras = [
    addCaptureCamera(scene, 'Camera A', cameraMountPosition('tag_cam_mount'), CAMERA_A_COLOR, CAMERA_A_COLOR),
    addCaptureCamera(scene, 'Camera B', cameraMountPosition('tag_cam_aux_mount'), CAMERA_B_COLOR, CAMERA_B_COLOR),
  ];

  const sponge = new THREE.Mesh(new RoundedBoxGeometry(...SPONGE_SIZE, 3, 0.075), new THREE.MeshStandardMaterial({ color: '#c3a341', roughness: 0.76, metalness: 0.02 }));
  sponge.name = 'sponge';
  sponge.position.set(LIVE_X, SPONGE_SIZE[1] / 2 + 0.035, 0);
  sponge.castShadow = true;
  sponge.receiveShadow = true;
  scene.add(sponge);

  // Equal-sized instances share one material. Only their positions vary.
  const count = 510;
  const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.047, 10, 8), new THREE.MeshBasicMaterial({ color: '#5cec55', toneMapped: false, fog: false }), count);
  const rng = random(127);
  const transform = new THREE.Matrix4();
  for (let i = 0; i < count; i++) {
    let x = (rng() - 0.5) * SPONGE_SIZE[0];
    let y = rng() * SPONGE_SIZE[1];
    let z = (rng() - 0.5) * SPONGE_SIZE[2];
    const face = i % 3;
    if (face === 0) y = SPONGE_SIZE[1];
    if (face === 1) z = SPONGE_SIZE[2] / 2;
    if (face === 2) x = SPONGE_SIZE[0] / 2;
    transform.makeTranslation(x + SNAPSHOT_X, y + 0.035, z);
    dots.setMatrixAt(i, transform);
  }
  scene.add(dots);

  // Marker sits at the estimated center, rendered visibly through the object.
  const position = new THREE.Mesh(new THREE.SphereGeometry(0.18, 24, 16), new THREE.MeshBasicMaterial({ color: '#4cdeee', depthTest: false, toneMapped: false, fog: false }));
  position.position.copy(sponge.position);
  position.renderOrder = 10;
  scene.add(position);

  const surfaceLabel = label('Surface snapshot', theme.surfaceText);
  surfaceLabel.scale.multiplyScalar(1.6);
  surfaceLabel.position.set(SNAPSHOT_X, 5.1, 0);
  scene.add(surfaceLabel);
  const surfaceLeader = line([[SNAPSHOT_X, 4.65, 0], [SNAPSHOT_X, 3.3, 0], [SNAPSHOT_X + 0.5, 2.7, 0]], '#75b47c', 0.7);
  scene.add(surfaceLeader);
  const positionLabel = label('Position estimate', theme.positionText);
  positionLabel.scale.multiplyScalar(1.6);
  positionLabel.position.set(LIVE_X, 5.1, 0);
  scene.add(positionLabel);
  const positionLeader = line([[LIVE_X, 4.65, 0], [LIVE_X, 3.3, 0], [LIVE_X, 1.3, 0]], '#62b9c7', 0.8);
  scene.add(positionLeader);

  // A world-space floor arrow establishes lateral displacement without a HUD.
  const arrow = new THREE.Group();
  arrow.add(line([[-1.5, 0.025, 3.2], [3.4, 0.025, 3.2]], '#8096a6', 0.75));
  arrow.add(line([[2.8, 0.025, 2.85], [3.4, 0.025, 3.2], [2.8, 0.025, 3.55]], '#8096a6', 0.75));
  scene.add(arrow);
  applySceneStyle(scene, style);
  addSceneEnvironment(scene, style, [new THREE.Vector3(...cameraMountPosition('tag_cam_mount')), new THREE.Vector3(...cameraMountPosition('tag_cam_aux_mount'))], sponge.position, SPONGE_SIZE);
  return { scene, sponge, cloud: dots, position, surfaceLabel, surfaceLeader, positionLabel, positionLeader, arrow, captureCameras };
}

export function PerceptionScene({ style, motion, keyframe, onTime, onPlaying }) {
  const mount = useRef(null);
  useEffect(() => {
    const container = mount.current;
    const objects = buildScene(style);
    const { scene } = objects;
    const camera = new THREE.PerspectiveCamera(43, 16 / 9, 0.1, 2000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    const frameResources = keyframe ? buildStoryFrame(objects, keyframe, renderer) : [];
    renderer.domElement.setAttribute('aria-label', 'Two cameras on stands observe a sponge with a cyan position estimate. A green surface point cloud stays at its earlier location. Drag to orbit.');
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.06;
    controls.minDistance = 12;
    controls.maxDistance = 150;
    let time = 0;
    let playing = false;
    let lastTick = null;
    let authoredCamera = motion;
    let surfaceOrbitActive = false;
    let surfaceOrbitStart = null;
    const params = new URLSearchParams(window.location.search);
    const exporting = params.has('export');
    const orbitSurface = keyframe === 'surface' && !params.has('frame');
    const applyTime = (seconds, moveCamera) => {
      const frame = sampleDraft(seconds);
      time = frame.time;
      objects.sponge.position.x = frame.liveX;
      objects.position.position.x = frame.liveX;
      objects.positionLabel.position.x = frame.liveX;
      objects.positionLabel.scale.set(7.4, 0.925, 1);
      objects.surfaceLabel.scale.set(7.4, 0.925, 1);
      objects.positionLeader.position.x = frame.liveX - LIVE_X;
      for (const annotation of [objects.positionLabel, objects.surfaceLabel, objects.positionLeader, objects.surfaceLeader]) {
        annotation.material.opacity = frame.labels;
      }
      objects.arrow.visible = frame.labels > 0;
      for (const captureCamera of objects.captureCameras) {
        captureCamera.caption.material.opacity = frame.rigLabels;
      }
      if (moveCamera) {
        camera.position.set(...frame.eye);
        controls.target.set(...frame.target);
        camera.lookAt(controls.target);
      }
      if (!exporting) onTime(time);
      return frame;
    };
    const pause = () => { playing = false; lastTick = null; onPlaying(false); };
    const seek = (seconds) => {
      pause();
      authoredCamera = true;
      applyTime(seconds, true);
      renderer.render(scene, camera);
    };
    const play = () => {
      if (time >= DRAFT_DURATION) time = 0;
      authoredCamera = true;
      playing = true;
      lastTick = null;
      onPlaying(true);
    };
    const reset = () => {
      surfaceOrbitActive = false;
      if (motion) { seek(0); return; }
      if (keyframe === 'surface') objects.surfaceContext.visible = false;
      const view = keyframe ? STORY_FRAMES[keyframe] : SCENE_STYLES[style];
      authoredCamera = false;
      camera.fov = view.fov || 43;
      camera.updateProjectionMatrix();
      camera.position.set(...view.eye);
      controls.target.set(...view.target);
      controls.update();
    };
    const startSurfaceOrbit = () => {
      if (!orbitSurface) return;
      const frame = STORY_FRAMES.surface;
      surfaceOrbitActive = true;
      surfaceOrbitStart = null;
      authoredCamera = true;
      objects.surfaceContext.visible = true;
      objects.surfaceContext.traverse((object) => {
        if (!object.material) return;
        if (object.userData.surfaceOpacity === undefined) object.userData.surfaceOpacity = object.material.opacity;
        object.material.transparent = true;
        object.material.opacity = object.userData.surfaceOpacity;
      });
      camera.position.set(...frame.orbitFrom);
      controls.target.set(...frame.orbitTargetFrom);
      camera.lookAt(controls.target);
    };
    reset();
    startSurfaceOrbit();
    const explore = () => { pause(); surfaceOrbitActive = false; authoredCamera = false; };
    controls.addEventListener('start', explore);
    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    const exportImage = () => {
      renderer.render(scene, camera);
      const link = document.createElement('a');
      link.download = keyframe ? `perception-${keyframe}.png` : `perception-scene-05-${style}.png`;
      link.href = renderer.domElement.toDataURL('image/png');
      link.click();
    };
    window.addEventListener('perception-reset-camera', reset);
    window.addEventListener('perception-replay-orbit', startSurfaceOrbit);
    window.addEventListener('perception-export-frame', exportImage);
    const seekEvent = (event) => seek(event.detail);
    if (motion) {
      window.addEventListener('perception-play', play);
      window.addEventListener('perception-pause', pause);
      window.addEventListener('perception-seek', seekEvent);
    }
    if (exporting) {
      // Offline exporter drives the same state sampler, independent of wall time.
      window.perceptionDraft = {
        renderFrame(seconds) {
          const state = applyTime(seconds, true);
          renderer.render(scene, camera);
          return { png: renderer.domElement.toDataURL('image/png'), state };
        },
      };
    }
    renderer.setAnimationLoop((now) => {
      if (surfaceOrbitActive) {
        if (surfaceOrbitStart === null) surfaceOrbitStart = now;
        const frame = STORY_FRAMES.surface;
        const progress = Math.min(1, (now - surfaceOrbitStart) / frame.orbitDurationMs);
        const eased = progress * progress * (3 - 2 * progress);
        // Clear the image panels before the viewer crosses behind them.
        const fade = THREE.MathUtils.smoothstep(progress, 0.1, 0.5);
        objects.surfaceContext.traverse((object) => {
          if (object.material) object.material.opacity = object.userData.surfaceOpacity * (1 - fade);
        });
        objects.surfaceContext.visible = fade < 1;
        const initialTarget = new THREE.Vector3(...frame.orbitTargetFrom);
        const finalTarget = new THREE.Vector3(...frame.target);
        const target = initialTarget.clone().lerp(finalTarget, eased);
        const from = new THREE.Vector3(...frame.orbitFrom).sub(initialTarget);
        const to = new THREE.Vector3(...frame.eye).sub(finalTarget);
        const angle = Math.atan2(from.x, from.z) + (Math.atan2(to.x, to.z) - Math.atan2(from.x, from.z)) * eased;
        const radius = Math.hypot(from.x, from.z) + (Math.hypot(to.x, to.z) - Math.hypot(from.x, from.z)) * eased;
        camera.position.set(target.x + Math.sin(angle) * radius, target.y + from.y + (to.y - from.y) * eased, target.z + Math.cos(angle) * radius);
        controls.target.copy(target);
        camera.lookAt(target);
        if (progress === 1) { surfaceOrbitActive = false; authoredCamera = false; }
      }
      if (playing) {
        if (lastTick !== null) time = Math.min(DRAFT_DURATION, time + (now - lastTick) / 1000);
        lastTick = now;
        applyTime(time, true);
        if (time >= DRAFT_DURATION) pause();
      }
      if (!authoredCamera) controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      if (exporting) delete window.perceptionDraft;
      window.removeEventListener('perception-play', play);
      window.removeEventListener('perception-pause', pause);
      window.removeEventListener('perception-seek', seekEvent);
      controls.removeEventListener('start', explore);
      renderer.setAnimationLoop(null);
      observer.disconnect();
      window.removeEventListener('perception-reset-camera', reset);
      window.removeEventListener('perception-replay-orbit', startSurfaceOrbit);
      window.removeEventListener('perception-export-frame', exportImage);
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line || object instanceof THREE.Sprite) {
          object.geometry?.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) {
            material.map?.dispose();
            material.dispose();
          }
        }
        if (object instanceof THREE.Light && object.shadow) object.shadow.dispose();
      });
      renderer.dispose();
      for (const resource of frameResources) resource.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [style, motion, keyframe, onTime, onPlaying]);
  return <div ref={mount} className="scene" />;
}
