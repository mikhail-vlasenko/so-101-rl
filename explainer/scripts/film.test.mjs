import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FILM_DURATION, FILM_MAX_DURATION, FILM_ENDING, FILM_CAPTIONS, FILM_CUES, FILM_SCRIPT, FILM_SHOTS, FILM_FPS, CAPTION_ROLL_SECONDS, SHUTTERS, REFRESH_AT, MOTION_START, MOTION_END, MOTION_SPEED, MOTION_EASE_SECONDS, MOTION_SEGMENTS, SPONGE_FADE, SURFACE_MATCHES, SURFACE_FILL_END, SURFACE_ORBIT, SURFACE_CONTEXT_FADE, INTRO_TIMING, INTRO_VIEW_FOV_DEG, INTRO_LENS_OFFSET_CM, CENTER_GATHER, DEPTH_VIEW, DEPTH_RECONSTRUCTION, DEPTH_CLEANUP, captionStack, surfaceReconstruction, sampleFilm, surfaceView, introView, movementView } from '../src/filmTimeline.js';
import { imageSilhouette, buildImageCenterCue } from '../src/imageCenterCue.js';
import { LIVE_X, SNAPSHOT_X } from '../src/motionDraft.js';
import { STORY_FRAMES } from '../src/storyFrames.js';
import { CAMERA_AIM_X, SPONGE_SIZE } from '../src/sceneConfig.js';
import { FILM_REVIEW_TIMES } from './film-review.mjs';

const middle = ({ start, end }) => (start + end) / 2;

test('review samples follow the live cues and stay within the current cut', () => {
  assert.equal(FILM_REVIEW_TIMES[0], 0);
  assert.equal(FILM_REVIEW_TIMES.at(-1), FILM_DURATION);
  assert.equal(new Set(FILM_REVIEW_TIMES).size, FILM_REVIEW_TIMES.length);
  for (const [index, time] of FILM_REVIEW_TIMES.entries()) {
    assert.ok(time >= 0 && time <= FILM_DURATION);
    if (index > 0) assert.ok(time > FILM_REVIEW_TIMES[index - 1]);
  }
  for (const cue of FILM_CAPTIONS) assert.ok(FILM_REVIEW_TIMES.some((time) => time > cue.start && time < cue.end));
});

test('captions have unique stable IDs, contiguous bounds and a derived plain-text script', () => {
  assert.equal(new Set(FILM_CAPTIONS.map(({ id }) => id)).size, FILM_CAPTIONS.length);
  for (const [index, cue] of FILM_CAPTIONS.entries()) {
    assert.equal(FILM_CUES[cue.id], cue);
    assert.ok(cue.start < cue.end);
    assert.ok(cue.text.trim().length > 0);
    assert.ok(cue.text.split('\n').length <= 2);
    assert.equal(sampleFilm(middle(cue)).captionId, cue.id);
    assert.equal(sampleFilm(middle(cue)).caption, cue.text);
    if (index > 0) assert.equal(cue.start, FILM_CAPTIONS[index - 1].end);
  }
  assert.equal(FILM_CAPTIONS.at(-1).end, FILM_ENDING.captionEnd);
  assert.equal(FILM_SCRIPT, FILM_CAPTIONS.map(({ text }) => text.replaceAll('\n', ' ')).join(' '));
  assert.ok(!FILM_SCRIPT.includes('\n'));
});

test('the position-only measurement precedes surface reconstruction', () => {
  const time = middle(FILM_CUES.positionOnly);
  assert.equal(sampleFilm(time).shot, 'intro');
  assert.equal(sampleFilm(time).positionEmphasis, 1);
  assert.equal(surfaceReconstruction(time, 510).pointCount, 0);
  assert.ok(FILM_CUES.position.end <= FILM_CUES.matchDetails.start);
});

test('caption reading stack previews the next phrase and smoothly promotes it', () => {
  assert.ok(CAPTION_ROLL_SECONDS <= 0.2);
  const settled = captionStack(middle(FILM_CUES.findTarget));
  assert.deepEqual(settled, [
    { text: FILM_CAPTIONS[1].text, slot: 0, emphasis: 1, opacity: 1 },
    { text: FILM_CAPTIONS[2].text, slot: 1, emphasis: 0, opacity: 1 },
  ]);
  const start = FILM_CAPTIONS[2].start;
  assert.deepEqual(captionStack(start), settled, 'A preview must not jump when it becomes current');
  const moving = captionStack(start + CAPTION_ROLL_SECONDS / 2);
  assert.equal(moving.length, 3);
  const [outgoing, current, next] = moving;
  assert.ok(Math.abs(outgoing.slot + 0.5) < 1e-10);
  assert.ok(Math.abs(current.slot - 0.5) < 1e-10);
  assert.ok(Math.abs(current.emphasis - 0.5) < 1e-10);
  assert.ok(Math.abs(next.slot - 1.5) < 1e-10);
  assert.equal(current.text, FILM_CAPTIONS[2].text);
  assert.equal(next.text, FILM_CAPTIONS[3].text);
  const arrived = captionStack(start + CAPTION_ROLL_SECONDS);
  assert.equal(arrived.length, 2);
  assert.equal(arrived[0].slot, 0);
  assert.equal(arrived[0].emphasis, 1);
  assert.equal(arrived[1].slot, 1);
  for (const cue of FILM_CAPTIONS) {
    const readable = captionStack(cue.start + 0.2);
    assert.equal(readable[0].slot, 0, 'Every caption must settle within 200 ms');
    assert.equal(readable[0].emphasis, 1);
  }
});

test('caption stack is seekable, previews across scenes, and leaves the ending empty', () => {
  const time = FILM_CUES.matchDetails.start + CAPTION_ROLL_SECONDS / 2;
  const first = captionStack(time);
  captionStack(FILM_ENDING.captionEnd - 0.1);
  captionStack(middle(FILM_CUES.opening));
  assert.deepEqual(captionStack(time), first);
  const beforeSurface = captionStack(middle(FILM_CUES.positionOnly));
  assert.equal(beforeSurface[1].text, FILM_CUES.matchDetails.text);
  assert.deepEqual(captionStack(0), []);
  assert.deepEqual(captionStack(FILM_CAPTIONS[0].start), []);
  const ending = captionStack(FILM_ENDING.captionEnd - 0.1);
  assert.equal(ending.length, 1);
  assert.equal(ending[0].text, FILM_CAPTIONS.at(-1).text);
  assert.deepEqual(captionStack(FILM_ENDING.captionEnd), []);
  assert.deepEqual(captionStack(FILM_DURATION), []);
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
  assert.ok(CENTER_GATHER.fadeEnd < INTRO_TIMING.pullbackStart);
  assert.equal(sampleFilm(INTRO_TIMING.emphasisStart).positionEmphasis, 0);
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
  assert.ok(sampleFilm(DEPTH_RECONSTRUCTION.pointEnd).falseDepth > 0);
  assert.equal(sampleFilm(DEPTH_CLEANUP.fadeEnd).falseDepth, 0);
});

test('the sponge keeps positive, continuous velocity across both shutters', () => {
  for (const time of [SHUTTERS.first, SHUTTERS.second]) {
    const before = sampleFilm(time - 0.001).liveX;
    const capture = sampleFilm(time).liveX;
    const after = sampleFilm(time + 0.001).liveX;
    assert.ok(before < capture && capture < after);
    assert.ok(Math.abs((capture - before) - (after - capture)) < 0.00001);
  }
  assert.ok(sampleFilm(SPONGE_FADE.outEnd).liveX > sampleFilm(SHUTTERS.second).liveX);
  assert.equal(sampleFilm(MOTION_END).liveX, LIVE_X);
  for (const time of [MOTION_START, SHUTTERS.first, SHUTTERS.second, SPONGE_FADE.outStart]) {
    assert.equal(sampleFilm(time).shot, 'movement');
    assert.equal(sampleFilm(time).dissolve, 1);
    assert.deepEqual(movementView(time), movementView(MOTION_START));
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
  assert.ok(velocity(MOTION_END - 0.4) < velocity(MOTION_END - 0.8));
  assert.ok(velocity(SPONGE_FADE.inEnd) > 0);
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
  assert.equal(sampleFilm(middle(FILM_CUES.depthWarning)).spongeOpacity, 0);
  assert.equal(sampleFilm(SPONGE_FADE.inStart).spongeOpacity, 0);
  const returning = sampleFilm(SPONGE_FADE.inStart + 1);
  assert.ok(returning.spongeOpacity > 0 && returning.spongeOpacity < 1);
  assert.ok(sampleFilm(SPONGE_FADE.inStart + 1.1).liveX > returning.liveX);
  assert.equal(sampleFilm(SPONGE_FADE.inEnd).spongeOpacity, 1);
  assert.ok(sampleFilm(SPONGE_FADE.inEnd + 0.1).liveX > sampleFilm(SPONGE_FADE.inEnd).liveX);
  assert.equal(sampleFilm(MOTION_END + 0.2).spongeOpacity, 1);
  assert.equal(sampleFilm(MOTION_END + 0.2).liveX, LIVE_X);
});

test('the final refresh stays in the same scene and view', () => {
  for (const time of [DEPTH_CLEANUP.viewReturnEnd, MOTION_END, REFRESH_AT - 0.01, REFRESH_AT, REFRESH_AT + 0.01, FILM_DURATION]) {
    assert.equal(sampleFilm(time).shot, 'movement');
    assert.equal(sampleFilm(time).dissolve, 1);
    assert.deepEqual(movementView(time), movementView(DEPTH_CLEANUP.viewReturnEnd));
  }
  assert.equal(sampleFilm(SHUTTERS.first).captionId, FILM_CUES.captures.id);
  assert.equal(sampleFilm(SHUTTERS.second).caption, sampleFilm(SHUTTERS.first).caption);
});

test('captions cover both shutters and explain the mismatch before the held-snapshot response', () => {
  const mismatch = FILM_CUES.depthWarning;
  const held = FILM_CUES.holdAndTrack;
  const captures = FILM_CUES.captures;
  const visibleSurface = FILM_CUES.surfaceSnapshot;
  assert.ok(visibleSurface.end - visibleSurface.start >= 3);
  assert.equal(visibleSurface.end, MOTION_START);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.rayEnd).captionId, mismatch.id);
  assert.equal(sampleFilm(SPONGE_FADE.inStart).captionId, held.id, 'The held-snapshot block must also cover the live position returning');
  assert.ok(mismatch.end - mismatch.start <= 5, 'The combined warning should not linger after its explanation');
  assert.ok(DEPTH_RECONSTRUCTION.pointEnd < mismatch.end, 'The false-depth point must finish appearing before the response');
  assert.equal(held.start, mismatch.end);
  assert.ok(captures.start <= SHUTTERS.first && captures.end > SHUTTERS.second);
  assert.ok(captures.end - captures.start < 3);
  assert.equal(sampleFilm(FILM_ENDING.captionEnd).caption, '');
  assert.equal(sampleFilm(FILM_DURATION).ending, 1);
  for (let index = 1; index < FILM_CAPTIONS.length; index++) assert.ok(FILM_CAPTIONS[index].start >= FILM_CAPTIONS[index - 1].end);
});

test('the final caption fades before a one-second scene hold and the final scene fade', () => {
  assert.equal(sampleFilm(FILM_ENDING.captionFadeStart).captionOpacity, 1);
  assert.equal(sampleFilm((FILM_ENDING.captionFadeStart + FILM_ENDING.captionEnd) / 2).captionOpacity, 0.5);
  assert.equal(sampleFilm(FILM_ENDING.captionEnd).captionOpacity, 0);
  const holdStart = FILM_ENDING.captionEnd;
  const holdEnd = holdStart + FILM_ENDING.holdSeconds;
  for (const time of [holdStart, (holdStart + holdEnd) / 2, holdEnd]) {
    const state = sampleFilm(time);
    assert.equal(state.caption, '');
    assert.equal(state.ending, 0);
    assert.equal(state.spongeOpacity, 1);
    assert.equal(state.liveX, LIVE_X);
    assert.equal(state.shot, 'movement');
    assert.deepEqual(movementView(time), movementView(holdStart));
  }
  assert.equal(FILM_ENDING.holdSeconds, 1);
  assert.equal(sampleFilm((holdEnd + FILM_DURATION) / 2).ending, 0.5);
  assert.equal(sampleFilm(FILM_DURATION).ending, 1);
});

test('the held cloud fades halfway with movement, before the fresh measurement', () => {
  assert.equal(sampleFilm(MOTION_START - 0.001).liveX, SNAPSHOT_X);
  assert.equal(sampleFilm(MOTION_START).captionId, FILM_CUES.movement.id);
  assert.equal(sampleFilm(MOTION_START).heldCloudOpacity, 1);
  assert.ok(sampleFilm(MOTION_START + 1).heldCloudOpacity < 1);
  assert.equal(sampleFilm(MOTION_START + 3).heldCloudOpacity, 0.5);
  assert.equal(sampleFilm(middle(FILM_CUES.depthWarning)).heldCloudOpacity, 0.5);
  assert.equal(sampleFilm(MOTION_START + 1).snapshotX, SNAPSHOT_X);
});

test('stage two stays just in front of the physical lens with wider framing', () => {
  const lens = [7, 21, 38];
  const view = introView(middle(FILM_CUES.findTarget), lens);
  assert.ok(Math.abs(Math.hypot(...view.eye.map((value, index) => value - lens[index])) - INTRO_LENS_OFFSET_CM) < 1e-10);
  assert.deepEqual(view.target, [CAMERA_AIM_X, SPONGE_SIZE[1] / 2, 0]);
  assert.equal(view.fov, INTRO_VIEW_FOV_DEG);
  assert.deepEqual(introView(INTRO_TIMING.lensEntryEnd, lens), introView(INTRO_TIMING.pullbackStart, lens));
  assert.deepEqual(introView(INTRO_TIMING.pullbackEnd, lens).eye, STORY_FRAMES.position.eye);
  assert.deepEqual(introView(INTRO_TIMING.pullbackEnd, lens).target, STORY_FRAMES.position.target);
  for (const time of [INTRO_TIMING.orbitEnd, INTRO_TIMING.lensEntryEnd, INTRO_TIMING.pullbackStart, INTRO_TIMING.pullbackEnd]) {
    const before = introView(time - 0.0001, lens);
    const after = introView(time + 0.0001, lens);
    assert.ok(Math.abs(before.fov - after.fov) < 0.00001);
    before.eye.forEach((value, index) => assert.ok(Math.abs(value - after.eye[index]) < 0.00001));
  }
  for (const cue of [FILM_CUES.opening, FILM_CUES.findTarget, FILM_CUES.combineCenters, FILM_CUES.positionOnly]) assert.equal(sampleFilm(middle(cue)).shot, 'intro');
});

test('depth explanation rises to a downward-looking view and returns continuously', () => {
  assert.deepEqual(movementView(DEPTH_RECONSTRUCTION.pointEnd), DEPTH_VIEW);
  assert.ok(DEPTH_VIEW.eye[1] - DEPTH_VIEW.target[1] > 25);
  for (const time of [SPONGE_FADE.outStart, DEPTH_RECONSTRUCTION.rayEnd, DEPTH_CLEANUP.viewReturnStart, DEPTH_CLEANUP.viewReturnEnd]) {
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
  assert.ok(FILM_DURATION <= FILM_MAX_DURATION);
});

test('leftward surface orbit uses the approved start and rear view', () => {
  const frame = STORY_FRAMES.surface;
  const start = surfaceView(SURFACE_ORBIT.start);
  const end = surfaceView(SURFACE_ORBIT.end);
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
  assert.equal(surfaceReconstruction(SURFACE_MATCHES[0].start - 0.01, 510).pointCount, 0);
  assert.equal(surfaceReconstruction(SURFACE_FILL_END, 510).pointCount, 510);
  assert.equal(surfaceReconstruction(SURFACE_FILL_END + 0.1, 510).pointCount, 510);
});

test('a single surface point holds through its explanation before the cloud builds', () => {
  const pointCue = FILM_CUES.surfacePoint;
  const cloudCue = FILM_CUES.surfaceCloud;
  assert.ok(SURFACE_MATCHES[0].rayEnd < pointCue.start);
  assert.equal(SURFACE_MATCHES[1].start, cloudCue.start);
  for (let time = pointCue.start; time < pointCue.end; time += 0.1) {
    const reconstruction = surfaceReconstruction(time, 510);
    assert.equal(reconstruction.pointCount, 1);
    assert.equal(reconstruction.matches[1].visible, false);
    assert.deepEqual(surfaceView(time), surfaceView(SURFACE_ORBIT.start));
  }
  assert.equal(SURFACE_CONTEXT_FADE.start, SURFACE_ORBIT.start);
  assert.ok(SURFACE_CONTEXT_FADE.start > SURFACE_MATCHES.at(-1).rayEnd);
  assert.ok(surfaceReconstruction(cloudCue.start + 1.5, 510).pointCount > 3);
});

test('the complete cloud holds before the orbit, which overlaps the snapshot explanation', () => {
  assert.ok(SURFACE_ORBIT.start > SURFACE_FILL_END);
  const holdTime = (SURFACE_FILL_END + SURFACE_ORBIT.start) / 2;
  assert.equal(surfaceReconstruction(holdTime, 510).pointCount, 510);
  assert.deepEqual(surfaceView(holdTime), surfaceView(SURFACE_ORBIT.start));
  const snapshotCue = FILM_CUES.surfaceSnapshot;
  assert.ok(SURFACE_ORBIT.start < snapshotCue.start && SURFACE_ORBIT.end > snapshotCue.start);
  assert.equal(sampleFilm(SURFACE_ORBIT.end).shot, 'surface');
});

test('the stage-five scene establishes before movement begins', () => {
  const transition = FILM_SHOTS.find(({ id }) => id === 'movement');
  const transitionEnd = transition.at + transition.dissolve;
  assert.ok(transitionEnd < MOTION_START);
  const holdTime = (transitionEnd + MOTION_START) / 2;
  const held = sampleFilm(holdTime);
  assert.equal(held.shot, 'movement');
  assert.equal(held.dissolve, 1);
  assert.equal(held.liveX, SNAPSHOT_X);
  assert.equal(held.spongeOpacity, 1);
  assert.equal(held.heldCloudOpacity, 1);
  assert.equal(held.firstCaptured, false);
  assert.deepEqual(movementView(holdTime), movementView(MOTION_START));
  assert.equal(sampleFilm(MOTION_START).shot, 'movement');
  assert.equal(sampleFilm(MOTION_START).dissolve, 1);
});

test('depth rays start immediately after the live sponge fades, then reveal the wrong point', () => {
  assert.equal(DEPTH_RECONSTRUCTION.start, SPONGE_FADE.outEnd);
  assert.ok(DEPTH_RECONSTRUCTION.start > SHUTTERS.second);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.start).spongeOpacity, 0);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.start).depthRays, 0);
  assert.ok(sampleFilm(DEPTH_RECONSTRUCTION.start + 0.01).depthRays > 0);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.rayEnd).depthRays, 1);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.rayEnd).falseDepth, 0);
  assert.ok(sampleFilm(DEPTH_RECONSTRUCTION.rayEnd + 0.01).falseDepth > 0);
  assert.equal(sampleFilm(DEPTH_RECONSTRUCTION.pointEnd).falseDepth, 1);
});

test('seeking is independent of playback history and finite time is required', () => {
  const saved = sampleFilm(35.7);
  sampleFilm(FILM_DURATION);
  sampleFilm(0);
  assert.deepEqual(sampleFilm(35.7), saved);
  assert.deepEqual(sampleFilm(-2), sampleFilm(0));
  assert.deepEqual(sampleFilm(100), sampleFilm(FILM_DURATION));
  assert.throws(() => sampleFilm(NaN), /finite/);
});
