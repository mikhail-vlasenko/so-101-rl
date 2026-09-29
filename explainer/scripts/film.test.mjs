import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FILM_DURATION, FILM_ENDING, FILM_CAPTIONS, FILM_FPS, SHUTTERS, REFRESH_AT, MOTION_START, MOTION_END, MOTION_SPEED, MOTION_EASE_SECONDS, MOTION_SEGMENTS, SPONGE_FADE, SURFACE_MATCHES, SURFACE_FILL_END, INTRO_VIEW_FOV_DEG, INTRO_LENS_OFFSET_CM, CENTER_GATHER, DEPTH_VIEW, surfaceReconstruction, sampleFilm, surfaceView, introView, movementView } from '../src/filmTimeline.js';
import { imageSilhouette, buildImageCenterCue } from '../src/imageCenterCue.js';
import { LIVE_X, SNAPSHOT_X } from '../src/motionDraft.js';
import { STORY_FRAMES } from '../src/storyFrames.js';
import { CAMERA_AIM_X, SPONGE_SIZE } from '../src/sceneConfig.js';

test('the opening introduces a general target, approximate position, then surface shape', () => {
  assert.equal(sampleFilm(1).caption, 'How do we perceive an object\nwe want to interact with?');
  assert.ok(sampleFilm(5).caption.includes('target object’s pixels'));
  assert.ok(sampleFilm(12).caption.includes('approximate 3D position'));
  assert.ok(sampleFilm(17).caption.includes('not its shape'));
  assert.ok(sampleFilm(20).caption.includes('surface point cloud'));
  assert.ok(sampleFilm(24).caption.includes('shape, size,'));
  assert.ok(FILM_CAPTIONS.every(({ text }) => !text.includes('gives us depth')));
  assert.equal(sampleFilm(17).shot, 'intro');
  assert.equal(sampleFilm(17).positionEmphasis, 1);
  assert.equal(surfaceReconstruction(17, 510).pointCount, 0);
});

test('the mask outline converges before lens exit and the position-only hold', () => {
  assert.equal(sampleFilm(CENTER_GATHER.start).centerConvergence, 0);
  assert.equal(sampleFilm(CENTER_GATHER.start).introPointScale, 0);
  const middle = sampleFilm((CENTER_GATHER.start + CENTER_GATHER.end) / 2);
  assert.ok(middle.centerConvergence > 0 && middle.centerConvergence < 1);
  assert.ok(middle.centerCueOpacity > 0);
  assert.equal(sampleFilm(CENTER_GATHER.end).centerConvergence, 1);
  assert.equal(sampleFilm(CENTER_GATHER.fadeEnd).centerCueOpacity, 0);
  assert.equal(sampleFilm(CENTER_GATHER.fadeEnd).introPointScale, 1);
  assert.ok(CENTER_GATHER.fadeEnd < 10.5);
  assert.equal(sampleFilm(15.3).positionEmphasis, 0);
});

test('image-center markers start on the projected silhouette, not arbitrary interior points', () => {
  const sponge = new THREE.Mesh(new THREE.BoxGeometry(6, 2.5, 4), new THREE.MeshBasicMaterial());
  const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.1, 200);
  camera.position.set(0, 0, 40);
  camera.lookAt(0, 0, 0);
  const silhouette = imageSilhouette(sponge, camera);
  assert.equal(silhouette.offsets.length, 4);
  assert.ok(silhouette.center.length() < 1e-10);
  const cue = buildImageCenterCue(sponge, camera);
  assert.equal(cue.particles.length, 12);
  for (const { origin } of cue.particles) {
    const xExtent = Math.abs(silhouette.offsets[0].x);
    const yExtent = Math.abs(silhouette.offsets[0].y);
    assert.ok(Math.abs(Math.abs(origin.x) - xExtent) < 1e-10 || Math.abs(Math.abs(origin.y) - yExtent) < 1e-10);
    assert.ok(Math.abs(origin.z) < 1e-10);
  }
  const offset = new THREE.Vector3(1.5, 0.75, 0);
  sponge.position.copy(offset);
  const shifted = imageSilhouette(sponge, camera);
  assert.ok(shifted.center.x > 0 && shifted.center.y > 0);
  assert.ok(Math.abs(shifted.center.z) < 1e-10);
  cue.group.traverse((object) => {
    if (object.geometry) object.geometry.dispose();
    if (object.material) object.material.dispose();
  });
  sponge.geometry.dispose();
  sponge.material.dispose();
});

test('the same corner is captured at two positions before false depth appears', () => {
  const first = sampleFilm(SHUTTERS.first);
  const second = sampleFilm(SHUTTERS.second);
  assert.equal(second.liveX - first.liveX, 1);
  assert.equal(first.firstCaptured, true);
  assert.equal(first.secondCaptured, false);
  assert.equal(second.secondCaptured, true);
  assert.equal(first.falseDepth, 0);
  assert.equal(second.falseDepth, 0);
  assert.ok(sampleFilm(43).falseDepth > 0);
  assert.equal(sampleFilm(46.6).falseDepth, 0);
});

test('the sponge keeps positive, continuous velocity across both shutters', () => {
  for (const time of [SHUTTERS.first, SHUTTERS.second]) {
    const before = sampleFilm(time - 0.001).liveX;
    const capture = sampleFilm(time).liveX;
    const after = sampleFilm(time + 0.001).liveX;
    assert.ok(before < capture && capture < after);
    assert.ok(Math.abs((capture - before) - (after - capture)) < 0.00001);
  }
  assert.ok(sampleFilm(39.5).liveX > sampleFilm(SHUTTERS.second).liveX);
  assert.equal(sampleFilm(49).liveX, LIVE_X);
  for (const time of [32, SHUTTERS.first, SHUTTERS.second, 39.5]) {
    assert.equal(sampleFilm(time).shot, 'movement');
    assert.equal(sampleFilm(time).dissolve, 1);
    assert.deepEqual(movementView(time), movementView(32));
  }
});

test('visible motion is twice as fast and the shutter gap shrinks proportionally', () => {
  assert.equal(MOTION_SPEED, 1);
  assert.equal(SHUTTERS.second - SHUTTERS.first, 1);
  for (const [start, end] of [[MOTION_START + MOTION_EASE_SECONDS, SPONGE_FADE.outEnd], [SPONGE_FADE.inStart, MOTION_END - MOTION_EASE_SECONDS]]) {
    for (let time = start + 0.1; time < end; time += 0.1) {
      assert.ok(Math.abs((sampleFilm(time).liveX - sampleFilm(time - 0.1).liveX) / 0.1 - MOTION_SPEED) < 1e-10);
    }
  }
  for (const time of [SPONGE_FADE.outEnd, SPONGE_FADE.inStart]) {
    assert.ok(Math.abs(sampleFilm(time - 0.0001).liveX - sampleFilm(time + 0.0001).liveX) < 0.001);
  }
});

test('motion eases from rest and settles with continuous velocity at every join', () => {
  const delta = 0.001;
  const velocity = (time) => (sampleFilm(time + delta).liveX - sampleFilm(time).liveX) / delta;
  assert.ok(velocity(MOTION_START) < 0.00001);
  assert.ok(velocity(MOTION_END - delta) < 0.00001);
  for (const time of MOTION_SEGMENTS.map(({ start }) => start).concat(MOTION_END)) {
    assert.ok(Math.abs(velocity(time - delta) - velocity(time)) < 0.00001, `Velocity discontinuity at ${time}`);
  }
  assert.ok(velocity(48.6) < velocity(48.2));
  assert.ok(velocity(48.8) > 0);
});

test('the live sponge holds for 500 ms after capture, then fades and returns while moving', () => {
  assert.equal(sampleFilm(SHUTTERS.first).spongeOpacity, 1);
  assert.equal(sampleFilm(SHUTTERS.second).spongeOpacity, 1);
  assert.equal(SPONGE_FADE.outStart, SHUTTERS.second + 0.5);
  assert.ok(Math.abs(SPONGE_FADE.outEnd - SPONGE_FADE.outStart - 0.8) < 1e-10);
  assert.equal(sampleFilm(SHUTTERS.second + 0.499).spongeOpacity, 1);
  assert.equal(sampleFilm(SPONGE_FADE.outStart).spongeOpacity, 1);
  assert.ok(sampleFilm(SPONGE_FADE.outStart).liveX > sampleFilm(SHUTTERS.second).liveX);
  assert.ok(sampleFilm(SPONGE_FADE.outStart + 0.4).spongeOpacity < 1);
  assert.equal(sampleFilm(SPONGE_FADE.outEnd).spongeOpacity, 0);
  assert.equal(sampleFilm(43).spongeOpacity, 0);
  assert.equal(sampleFilm(SPONGE_FADE.inStart).spongeOpacity, 0);
  const returning = sampleFilm(48);
  assert.ok(returning.spongeOpacity > 0 && returning.spongeOpacity < 1);
  assert.ok(sampleFilm(48.1).liveX > returning.liveX);
  assert.equal(sampleFilm(SPONGE_FADE.inEnd).spongeOpacity, 1);
  assert.ok(sampleFilm(SPONGE_FADE.inEnd + 0.1).liveX > sampleFilm(SPONGE_FADE.inEnd).liveX);
  assert.equal(sampleFilm(49.2).spongeOpacity, 1);
  assert.equal(sampleFilm(49.2).liveX, LIVE_X);
});

test('the final refresh stays in the same scene and view', () => {
  for (const time of [48.2, 49, REFRESH_AT - 0.01, REFRESH_AT, REFRESH_AT + 0.01, 55]) {
    assert.equal(sampleFilm(time).shot, 'movement');
    assert.equal(sampleFilm(time).dissolve, 1);
    assert.deepEqual(movementView(time), movementView(48.2));
  }
  assert.equal(sampleFilm(SHUTTERS.first).caption, '…the cameras can capture at different moments.');
  assert.equal(sampleFilm(SHUTTERS.second).caption, sampleFilm(SHUTTERS.first).caption);
});

test('captions give the stereo reliability explanation more time and omit redundant summaries', () => {
  const shutters = FILM_CAPTIONS.find(({ text }) => text.startsWith('That makes detailed stereo'));
  const captures = FILM_CAPTIONS.find(({ text }) => text.includes('cameras can capture'));
  assert.ok(shutters.start < 46.1);
  assert.ok(shutters.end - shutters.start >= 5);
  assert.ok(captures.end - captures.start < 3);
  assert.equal(sampleFilm(28).caption, '');
  assert.equal(sampleFilm(53).caption, '');
  assert.equal(FILM_DURATION, 55);
  assert.ok(FILM_CAPTIONS.every(({ text, end }) => !text.includes('Surface detail') && end <= FILM_DURATION));
  assert.equal(sampleFilm(FILM_DURATION).ending, 1);
  for (let index = 1; index < FILM_CAPTIONS.length; index++) assert.ok(FILM_CAPTIONS[index].start >= FILM_CAPTIONS[index - 1].end);
});

test('the final caption fades before a one-second scene hold and the final scene fade', () => {
  assert.equal(sampleFilm(FILM_ENDING.captionFadeStart).captionOpacity, 1);
  assert.equal(sampleFilm(52.75).captionOpacity, 0.5);
  assert.equal(sampleFilm(FILM_ENDING.captionEnd).captionOpacity, 0);
  for (const time of [53, 53.5, 54]) {
    const state = sampleFilm(time);
    assert.equal(state.caption, '');
    assert.equal(state.ending, 0);
    assert.equal(state.spongeOpacity, 1);
    assert.equal(state.liveX, LIVE_X);
    assert.equal(state.shot, 'movement');
    assert.deepEqual(movementView(time), movementView(53));
  }
  assert.equal(sampleFilm(54.5).ending, 0.5);
  assert.equal(sampleFilm(FILM_DURATION).ending, 1);
});

test('movement introduces the timing problem before its consequence and refresh', () => {
  assert.equal(sampleFilm(30).caption, 'When the sponge starts moving…');
  assert.ok(sampleFilm(35).caption.includes('different moments'));
  assert.ok(sampleFilm(41).caption.includes('mistaken for depth'));
  assert.ok(sampleFilm(46).caption.includes('unreliable during movement'));
  assert.equal(sampleFilm(50).caption, 'So we refresh the cloud\nwhen the sponge settles.');
  assert.equal(sampleFilm(MOTION_START).heldCloudOpacity, 1);
  assert.ok(sampleFilm(30).heldCloudOpacity < 1);
  assert.equal(sampleFilm(32).heldCloudOpacity, 0.5);
  assert.equal(sampleFilm(43).heldCloudOpacity, sampleFilm(32).heldCloudOpacity);
  assert.equal(sampleFilm(32).snapshotX, SNAPSHOT_X);
});

test('stage two stays just in front of the physical lens with wider framing', () => {
  const lens = [7, 21, 38];
  const view = introView(6, lens);
  assert.ok(Math.abs(Math.hypot(...view.eye.map((value, index) => value - lens[index])) - INTRO_LENS_OFFSET_CM) < 1e-10);
  assert.deepEqual(view.target, [CAMERA_AIM_X, SPONGE_SIZE[1] / 2, 0]);
  assert.equal(view.fov, INTRO_VIEW_FOV_DEG);
  assert.deepEqual(introView(4, lens), introView(10.5, lens));
  assert.deepEqual(introView(16, lens).eye, STORY_FRAMES.position.eye);
  assert.deepEqual(introView(16, lens).target, STORY_FRAMES.position.target);
  for (const time of [1.7, 4, 10.5, 16]) {
    const before = introView(time - 0.0001, lens);
    const after = introView(time + 0.0001, lens);
    assert.ok(Math.abs(before.fov - after.fov) < 0.00001);
    before.eye.forEach((value, index) => assert.ok(Math.abs(value - after.eye[index]) < 0.00001));
  }
  for (const time of [0, 4, 11, 18.9]) assert.equal(sampleFilm(time).shot, 'intro');
});

test('depth explanation rises to a downward-looking view and returns continuously', () => {
  assert.deepEqual(movementView(43), DEPTH_VIEW);
  assert.ok(DEPTH_VIEW.eye[1] - DEPTH_VIEW.target[1] > 25);
  for (const time of [40, 42, 46, 48.2]) {
    const before = movementView(time - 0.0001);
    const after = movementView(time + 0.0001);
    before.eye.forEach((value, index) => assert.ok(Math.abs(value - after.eye[index]) < 0.00001));
  }
});

test('the held cloud never follows movement; only a new refresh changes its location', () => {
  let previousX = SNAPSHOT_X;
  for (let frame = 0; frame < FILM_DURATION * FILM_FPS; frame++) {
    const time = frame / FILM_FPS;
    const state = sampleFilm(time);
    assert.ok(state.liveX >= previousX - 1e-10, `Sponge moved backwards at ${time}`);
    previousX = state.liveX;
    assert.equal(state.snapshotX, time < REFRESH_AT ? SNAPSHOT_X : LIVE_X);
    assert.ok(state.falseDepth === 0 || state.secondCaptured);
  }
  assert.ok(FILM_DURATION <= 60);
});

test('leftward surface orbit uses the approved start and rear view', () => {
  const frame = STORY_FRAMES.surface;
  const start = surfaceView(23);
  const end = surfaceView(27);
  start.eye.forEach((value, i) => assert.ok(Math.abs(value - frame.orbitFrom[i]) < 1e-10));
  end.eye.forEach((value, i) => assert.ok(Math.abs(value - frame.eye[i]) < 1e-10));
  assert.deepEqual(start.target, frame.orbitTargetFrom);
  assert.deepEqual(end.target, frame.target);
  assert.ok(end.eye[0] < end.target[0] && end.eye[2] < end.target[2]);
});

test('surface reconstruction matches pixels, then draws rays, then adds each point', () => {
  assert.equal(SURFACE_MATCHES.length, 3);
  let previousDuration = Infinity;
  SURFACE_MATCHES.forEach(({ start, matchEnd, rayEnd }, index) => {
    const duration = rayEnd - start;
    assert.ok(duration < previousDuration);
    previousDuration = duration;
    assert.equal(surfaceReconstruction(start, 510).pointCount, index);
    const matched = surfaceReconstruction(matchEnd, 510);
    assert.equal(matched.matches[index].connection, 1);
    assert.equal(matched.matches[index].rays, 0);
    assert.equal(matched.pointCount, index);
    assert.equal(surfaceReconstruction(rayEnd - 0.001, 510).pointCount, index);
    const reconstructed = surfaceReconstruction(rayEnd, 510);
    assert.equal(reconstructed.matches[index].rays, 1);
    assert.equal(reconstructed.pointCount, index + 1);
  });
  assert.equal(surfaceReconstruction(19, 510).pointCount, 0);
  assert.equal(surfaceReconstruction(SURFACE_FILL_END, 510).pointCount, 510);
  assert.equal(surfaceReconstruction(26, 510).pointCount, 510);
});

test('seeking is independent of playback history and finite time is required', () => {
  const saved = sampleFilm(35.7);
  sampleFilm(55);
  sampleFilm(0);
  assert.deepEqual(sampleFilm(35.7), saved);
  assert.deepEqual(sampleFilm(-2), sampleFilm(0));
  assert.deepEqual(sampleFilm(100), sampleFilm(FILM_DURATION));
  assert.throws(() => sampleFilm(NaN), /finite/);
});
