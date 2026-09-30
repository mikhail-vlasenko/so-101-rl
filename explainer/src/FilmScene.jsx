import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildScene } from './PerceptionScene.jsx';
import { STORY_FRAMES } from './storyFrames.js';
import { buildFilmShot } from './filmShots.js';
import { buildCaptionLayer } from './captionLayer.js';
import { LIVE_X } from './motionDraft.js';
import { CAMERA_A_COLOR, CAMERA_B_COLOR } from './sceneConfig.js';
import { FILM_DURATION, FILM_SHOTS, REFRESH_AT, REFRESH_END, SHUTTERS, SHUTTER_EFFECTS, INTRO_TIMING, LABEL_TIMING, DEPTH_CLEANUP, SURFACE_CONTEXT_FADE, DEPTH_RECONSTRUCTION, ease, sampleFilm, surfaceView, surfaceReconstruction, introView, movementView } from './filmTimeline.js';

function groupMaterials(materials, group) {
  return materials.filter(({ object }) => {
    let parent = object.parent;
    while (parent) {
      if (parent === group) return true;
      parent = parent.parent;
    }
    return false;
  });
}

function setOpacity(materials, amount) {
  for (const { object, opacity, depthWrite } of materials) {
    object.material.transparent = true;
    object.material.depthWrite = amount >= 0.99 && depthWrite;
    object.material.opacity = opacity * amount;
  }
}

/** Continuous lens entry/pullback and movement shots keep the physical rig
 * fixed. Render targets dissolve chapter transitions; shutter events happen
 * within one scene. One canvas supplies exportable captions.
 */
export function FilmScene({ onTime, onPlaying }) {
  const mount = useRef(null);
  useEffect(() => {
    const container = mount.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.domElement.setAttribute('aria-label', `The complete ${FILM_DURATION}-second perception explainer, with captions.`);
    container.appendChild(renderer.domElement);

    const shots = {};
    for (const id of new Set(FILM_SHOTS.map((shot) => shot.id))) {
      const objects = buildScene('studio');
      const resources = buildFilmShot(objects, id, renderer);
      objects.position.material.transparent = true;
      objects.position.material.depthWrite = false;
      objects.position.renderOrder = 30;
      const materials = [];
      objects.scene.traverse((object) => {
        if (object.material) materials.push({ object, opacity: object.material.opacity, depthWrite: object.material.depthWrite });
        if (object instanceof THREE.DirectionalLight && object.shadow) object.shadow.mapSize.set(1024, 1024);
      });
      const frame = STORY_FRAMES[id === 'intro' ? 'position' : id === 'movement' ? 'reject' : id];
      const camera = new THREE.PerspectiveCamera(frame.fov || 43, 16 / 9, 0.1, 2000);
      camera.position.set(...frame.eye);
      camera.lookAt(...frame.target);
      const cloudCount = objects.cloud.count;
      const context = id === 'surface' ? groupMaterials(materials, objects.surfaceContext) : [];
      for (const { object } of context) { object.material.transparent = true; object.material.depthWrite = false; }
      const exposureMaterials = id === 'movement' ? objects.shutterContexts.map((group) => groupMaterials(materials, group)) : [];
      const evidenceMaterials = id === 'movement' ? objects.shutterEvidence.map((group) => groupMaterials(materials, group)) : [];
      const falseDepthMaterials = id === 'movement' ? groupMaterials(materials, objects.falseDepth) : [];
      shots[id] = { objects, resources, frame, context, exposureMaterials, evidenceMaterials, falseDepthMaterials, camera, cloudCount, spongeColor: objects.sponge.material.color.clone(), shadowX: null, shadowOpacity: null };
    }

    // Preserve dark gradients until the final output conversion, then dither
    // the 8-bit display. Intermediate 8-bit linear buffers produced hard bands.
    const targets = [new THREE.WebGLRenderTarget(1280, 720, { type: THREE.HalfFloatType, samples: 4 }), new THREE.WebGLRenderTarget(1280, 720, { type: THREE.HalfFloatType, samples: 4 })];
    const composite = new THREE.Scene();
    const screenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    screenCamera.position.z = 5;
    const background = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: targets[0].texture, depthTest: false, depthWrite: false, toneMapped: false, dithering: true }));
    const foreground = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: targets[1].texture, transparent: true, depthTest: false, depthWrite: false, toneMapped: false, dithering: true }));
    foreground.renderOrder = 1;
    const hudCanvas = document.createElement('canvas');
    hudCanvas.width = 1280;
    hudCanvas.height = 720;
    const ctx = hudCanvas.getContext('2d');
    const hudTexture = new THREE.CanvasTexture(hudCanvas);
    hudTexture.colorSpace = THREE.SRGBColorSpace;
    hudTexture.generateMipmaps = false;
    hudTexture.minFilter = THREE.LinearFilter;
    const hud = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ map: hudTexture, transparent: true, depthTest: false, depthWrite: false, toneMapped: false }));
    hud.renderOrder = 2;
    const black = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: '#080d15', transparent: true, depthTest: false, depthWrite: false, toneMapped: false }));
    black.renderOrder = 6;
    composite.add(background, foreground, hud, black);
    const captions = buildCaptionLayer(composite);

    const controls = new OrbitControls(shots.intro.camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 10;
    controls.maxDistance = 150;
    controls.maxPolarAngle = Math.PI / 2 - 0.04;
    let time = 0;
    let playing = false;
    let exploring = false;
    let lastTick = null;
    let hudKey = '';
    const exporting = new URLSearchParams(window.location.search).has('export');

    const drawHud = (state) => {
      const prompt = state.time >= INTRO_TIMING.lensEntryEnd && state.time < INTRO_TIMING.pullbackStart;
      const key = `${state.chapter.number}|${prompt}`;
      if (key === hudKey) return;
      hudKey = key;
      ctx.clearRect(0, 0, 1280, 720);
      const top = ctx.createLinearGradient(0, 0, 0, 110);
      top.addColorStop(0, '#080d15e8');
      top.addColorStop(1, '#080d1500');
      ctx.fillStyle = top;
      ctx.fillRect(0, 0, 1280, 110);
      ctx.textAlign = 'left';
      ctx.font = '500 16px Arial';
      ctx.fillStyle = '#83a7b9';
      ctx.fillText(state.chapter.number, 40, 43);
      ctx.fillStyle = '#dbe8f3';
      ctx.font = '400 20px Arial';
      ctx.fillText(state.chapter.title, 76, 43);
      if (prompt) {
        ctx.textAlign = 'right';
        ctx.fillStyle = '#9af3aa';
        ctx.font = '400 18px Arial';
        ctx.fillText('PROMPT  “sponge”', 1240, 43);
      }
      // Keep the subtitle backdrop through cue gaps; toggling it with the text
      // made the whole lower scene flash between captions and chapters.
      const bottom = ctx.createLinearGradient(0, 520, 0, 720);
      bottom.addColorStop(0, '#080d1500');
      bottom.addColorStop(1, '#080d15ed');
      ctx.fillStyle = bottom;
      ctx.fillRect(0, 520, 1280, 200);
      hudTexture.needsUpdate = true;
    };

    const animateShot = (id, state) => {
      const shot = shots[id];
      const { objects, camera, frame } = shot;
      let eye = frame.eye;
      let target = frame.target;
      let fov = frame.fov || 43;
      if (id === 'intro') {
        ({ eye, target, fov } = introView(state.time, objects.captureCameras[0].lens.toArray()));
        const pullback = ease(state.time, INTRO_TIMING.pullbackStart, INTRO_TIMING.pullbackEnd);
        objects.modelMask.material.opacity = 0.57 * ease(state.time, INTRO_TIMING.maskStart, INTRO_TIMING.maskEnd) * (1 - ease(state.time, INTRO_TIMING.maskFadeStart, INTRO_TIMING.maskFadeEnd));
        objects.modelMask.visible = objects.modelMask.material.opacity > 0;
        objects.sponge.material.transparent = false;
        objects.sponge.material.opacity = 1;
        objects.sponge.material.depthWrite = true;
        objects.sponge.material.color.copy(shot.spongeColor).multiplyScalar(1 - 0.65 * state.positionEmphasis);
        const cue = objects.imageCenterCue;
        cue.group.visible = state.centerCueOpacity > 0;
        cue.outline.material.opacity = 0.28 * state.centerCueOpacity;
        cue.outline.scale.setScalar(1 - state.centerConvergence);
        for (const { mesh, origin } of cue.particles) {
          mesh.position.copy(origin).multiplyScalar(1 - state.centerConvergence);
          mesh.material.opacity = 0.85 * state.centerCueOpacity;
        }
        objects.position.position.copy(cue.center).lerp(objects.positionEstimate, ease(state.time, INTRO_TIMING.pullbackStart, INTRO_TIMING.positionBlendEnd));
        objects.position.scale.setScalar(state.introPointScale * (1 + 1.2 * pullback));
        const rays = ease(state.time, INTRO_TIMING.raysStart, INTRO_TIMING.raysEnd);
        for (const { mesh, origin } of objects.triangulationRays) {
          mesh.scale.setScalar(rays);
          mesh.position.copy(origin).multiplyScalar(1 - rays);
          mesh.material.transparent = true;
          mesh.material.opacity = 1 - 0.78 * state.positionEmphasis;
        }
        for (const feature of objects.triangulationFeatures) {
          feature.visible = state.time >= INTRO_TIMING.featuresStart;
          feature.material.transparent = true;
          feature.material.opacity = 1 - 0.78 * state.positionEmphasis;
        }
        objects.triangulationRing.visible = state.time >= INTRO_TIMING.raysEnd;
        objects.triangulationRing.material.opacity = 0.7 * (1 - state.positionEmphasis);
        objects.triangulationTitle.visible = state.time >= INTRO_TIMING.raysEnd;
        for (const rig of objects.captureCameras) rig.caption.visible = state.time < INTRO_TIMING.rigLabelsHide || state.time >= INTRO_TIMING.rigLabelsReturn;
        // The viewer crosses the camera's own housing on lens exit.
        objects.captureCameras[0].rig.visible = objects.captureCameras[0].lens.distanceTo(new THREE.Vector3(...eye)) > 4;
      }
      if (id === 'surface') {
        ({ eye, target } = surfaceView(state.time));
        const reconstruction = surfaceReconstruction(state.time, shot.cloudCount);
        objects.cloud.count = reconstruction.pointCount;
        objects.surfaceMatches.forEach((match, index) => {
          const progress = reconstruction.matches[index];
          for (const feature of match.features) feature.visible = progress.visible;
          for (const { mesh, origin, phase } of match.lines) {
            const growth = phase === 'connection' ? progress.connection : progress.rays;
            mesh.visible = growth > 0;
            mesh.scale.setScalar(growth);
            mesh.position.copy(origin).multiplyScalar(1 - growth);
          }
        });
        const contextOpacity = 1 - ease(state.time, SURFACE_CONTEXT_FADE.start, SURFACE_CONTEXT_FADE.end);
        objects.surfaceContext.visible = contextOpacity > 0;
        for (const { object, opacity } of shot.context) object.material.opacity = opacity * contextOpacity;
      }
      if (id === 'movement') {
        ({ eye, target, fov } = movementView(state.time));
        objects.sponge.position.x = state.liveX;
        objects.position.position.x = state.liveX;
        objects.positionLabel.position.x = state.liveX;
        objects.positionLeader.position.x = state.liveX - LIVE_X;
        const refresh = ease(state.time, REFRESH_AT, REFRESH_END);
        const labels = (1 - ease(state.time, LABEL_TIMING.fadeStart, LABEL_TIMING.fadeEnd) * (1 - ease(state.time, LABEL_TIMING.returnStart, LABEL_TIMING.returnEnd))) * (1 - refresh);
        for (const annotation of [objects.surfaceLabel, objects.surfaceLeader]) annotation.material.opacity = labels * state.heldCloudOpacity;
        for (const annotation of [objects.positionLabel, objects.positionLeader]) annotation.material.opacity = labels * state.spongeOpacity;
        objects.position.visible = state.spongeOpacity > 0;
        objects.position.material.opacity = state.spongeOpacity;
        objects.arrow.visible = state.time < REFRESH_AT;
        objects.cloud.material.transparent = true;
        objects.cloud.material.opacity = state.heldCloudOpacity * (1 - refresh);
        objects.cloud.material.depthWrite = objects.cloud.material.opacity === 1;
        objects.cloud.visible = refresh < 1;
        objects.refreshedCloud.visible = state.time >= REFRESH_AT;
        objects.refreshedCloud.count = Math.round(shot.cloudCount * refresh);
        objects.sponge.visible = state.spongeOpacity > 0;
        objects.sponge.material.transparent = true;
        objects.sponge.material.opacity = state.spongeOpacity;
        objects.sponge.material.depthWrite = state.spongeOpacity === 1;
        objects.spongeShadowOpacity.value = state.spongeOpacity;
        for (let index = 0; index < 2; index++) {
          const shutterTime = index === 0 ? SHUTTERS.first : SHUTTERS.second;
          const captured = index === 0 ? state.firstCaptured : state.secondCaptured;
          const panelOpacity = ease(state.time, shutterTime, shutterTime + SHUTTER_EFFECTS.panelRevealSeconds) * (1 - ease(state.time, DEPTH_RECONSTRUCTION.start, DEPTH_RECONSTRUCTION.start + SHUTTER_EFFECTS.panelFadeSeconds));
          const evidenceOpacity = captured ? 1 - ease(state.time, DEPTH_CLEANUP.evidenceFadeStart, DEPTH_CLEANUP.evidenceFadeEnd) : 0;
          objects.shutterContexts[index].visible = panelOpacity > 0;
          setOpacity(shot.exposureMaterials[index], panelOpacity);
          objects.shutterEvidence[index].visible = evidenceOpacity > 0;
          setOpacity(shot.evidenceMaterials[index], evidenceOpacity);
          const flash = state.time >= shutterTime ? Math.max(0, 1 - (state.time - shutterTime) / SHUTTER_EFFECTS.flashSeconds) : 0;
          objects.shutterPanels[index].frame.material.emissive.set(index === 0 ? CAMERA_A_COLOR : CAMERA_B_COLOR);
          objects.shutterPanels[index].frame.material.emissiveIntensity = 0.15 + flash * 2;
          objects.captureCameras[index].ring.material.color.set(index === 0 ? CAMERA_A_COLOR : CAMERA_B_COLOR).lerp(new THREE.Color('#ffffff'), flash);
        }
        const rays = state.depthRays;
        for (const { mesh, origin } of objects.triangulationRays) {
          mesh.scale.setScalar(rays);
          mesh.position.copy(origin).multiplyScalar(1 - rays);
        }
        objects.falseDepth.visible = state.falseDepth > 0;
        setOpacity(shot.falseDepthMaterials, state.falseDepth);
      }
      if (!exploring) {
        camera.fov = fov;
        camera.updateProjectionMatrix();
        camera.position.set(...eye);
        camera.lookAt(...target);
        if (state.shot === id) {
          controls.object = camera;
          controls.target.set(...target);
        }
      }
      return shot;
    };

    const renderShot = (id, state, renderTarget) => {
      const shot = animateShot(id, state);
      renderer.shadowMap.needsUpdate = shot.shadowX !== shot.objects.sponge.position.x || shot.shadowOpacity !== shot.objects.sponge.material.opacity;
      renderer.setRenderTarget(renderTarget);
      renderer.render(shot.objects.scene, shot.camera);
      shot.shadowX = shot.objects.sponge.position.x;
      shot.shadowOpacity = shot.objects.sponge.material.opacity;
    };
    const render = (seconds) => {
      const state = sampleFilm(seconds);
      time = state.time;
      if (state.dissolve < 1) renderShot(state.previous, state, targets[0]);
      renderShot(state.shot, state, targets[1]);
      foreground.material.opacity = state.dissolve;
      black.material.opacity = state.ending;
      drawHud(state);
      captions.update(state);
      renderer.setRenderTarget(null);
      renderer.render(composite, screenCamera);
      return state;
    };
    const pause = () => { playing = false; lastTick = null; onPlaying(false); };
    const seek = (seconds) => { pause(); exploring = false; render(seconds); onTime(time); };
    const play = () => {
      if (time >= FILM_DURATION) time = 0;
      exploring = false;
      playing = true;
      lastTick = null;
      onPlaying(true);
    };
    const explore = () => { pause(); exploring = true; };
    const reset = () => seek(0);
    const seekEvent = (event) => seek(event.detail);
    const exportImage = () => {
      render(time);
      const link = document.createElement('a');
      link.download = `perception-film-${time.toFixed(1)}s.png`;
      link.href = renderer.domElement.toDataURL('image/png');
      link.click();
    };
    controls.addEventListener('start', explore);
    window.addEventListener('perception-play', play);
    window.addEventListener('perception-pause', pause);
    window.addEventListener('perception-seek', seekEvent);
    window.addEventListener('perception-reset-camera', reset);
    window.addEventListener('perception-export-frame', exportImage);
    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      renderer.setSize(width, height);
      const drawingSize = renderer.getDrawingBufferSize(new THREE.Vector2());
      hudCanvas.width = drawingSize.x;
      hudCanvas.height = drawingSize.y;
      ctx.setTransform(drawingSize.x / 1280, 0, 0, drawingSize.y / 720, 0, 0);
      captions.resize(drawingSize.x, drawingSize.y);
      hudKey = '';
      for (const rt of targets) rt.setSize(drawingSize.x, drawingSize.y);
      for (const shot of Object.values(shots)) {
        shot.camera.aspect = width / height;
        shot.camera.updateProjectionMatrix();
      }
      render(time);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    if (exporting) {
      window.perceptionFilm = {
        renderFrame(seconds) {
          exploring = false;
          const state = render(seconds);
          return { png: renderer.domElement.toDataURL('image/png'), state, hudPixels: { width: hudCanvas.width, height: hudCanvas.height }, captionPixels: captions.stats() };
        },
      };
    } else {
      renderer.setAnimationLoop((now) => {
        if (playing) {
          if (lastTick !== null) time = Math.min(FILM_DURATION, time + (now - lastTick) / 1000);
          lastTick = now;
          render(time);
          onTime(time);
          if (time >= FILM_DURATION) pause();
        } else if (exploring) { controls.update(); render(time); }
      });
    }
    return () => {
      if (exporting) delete window.perceptionFilm;
      renderer.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      window.removeEventListener('perception-play', play);
      window.removeEventListener('perception-pause', pause);
      window.removeEventListener('perception-seek', seekEvent);
      window.removeEventListener('perception-reset-camera', reset);
      window.removeEventListener('perception-export-frame', exportImage);
      for (const shot of Object.values(shots)) {
        shot.objects.scene.traverse((object) => {
          if (object.geometry) object.geometry.dispose();
          if (object.material) { object.material.map?.dispose(); object.material.dispose(); }
          if (object instanceof THREE.Light && object.shadow) object.shadow.dispose();
        });
        for (const resource of shot.resources) resource.dispose();
      }
      for (const mesh of [background, foreground, hud, black]) { mesh.geometry.dispose(); mesh.material.dispose(); }
      hudTexture.dispose();
      captions.dispose();
      for (const rt of targets) rt.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [onTime, onPlaying]);
  return <div ref={mount} className="scene" />;
}
