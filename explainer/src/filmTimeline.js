import { LIVE_X, SNAPSHOT_X } from './motionDraft.js';
import { STORY_FRAMES } from './storyFrames.js';
import { CAMERA_AIM_X, SPONGE_SIZE } from './sceneConfig.js';

/** Edit narration and editorial timing here, not in renderers, tests or docs.
 * Cue IDs are stable animation anchors; cue ends and the transcript are derived.
 * Seconds are illustrative, not hardware latencies. Sampling stays seekable.
 */
const captionEnd = 53;
export const FILM_ENDING = { captionFadeStart: captionEnd - 0.5, captionEnd, holdSeconds: 1, fadeSeconds: 1 };
export const FILM_DURATION = FILM_ENDING.captionEnd + FILM_ENDING.holdSeconds + FILM_ENDING.fadeSeconds;
export const FILM_MAX_DURATION = 59;
export const FILM_FPS = 30;
export const CAPTION_ROLL_SECONDS = 0.18;

const captionCues = [
  { id: 'opening', start: 0.3, text: 'How do we perceive an object\nwe want to interact with?' },
  { id: 'findTarget', start: 4.3, text: 'First, a vision model finds\nthe target object’s pixels' },
  { id: 'trackTarget', start: 7.5, text: 'and tracks them in both camera views.' },
  { id: 'flatImage', start: 9.8, text: 'Each image is flat.' },
  { id: 'combineCenters', start: 11.2, text: 'But combining the centers of those pixels\nfrom two viewpoints' },
  { id: 'position', start: 13.8, text: 'gives us an approximate position\nin three dimensions.' },
  { id: 'positionOnly', start: 16.2, text: 'That tells us where the object is,\nbut not its shape.' },
  { id: 'matchDetails', start: 19.2, text: 'For that, we match details across the images.' },
  { id: 'surfacePoint', start: 21.5, text: 'Each matched detail gives us another point\non the visible surface.' },
  { id: 'surfaceCloud', start: 24.9, text: 'These become a cloud of surface points,\nshowing its shape and size.' },
  { id: 'surfaceSnapshot', start: 27.6, text: 'This is a snapshot of the surfaces\nthe cameras can see.' },
  { id: 'movement', start: 32.8, text: 'However, when the object moves,\nour independent cameras' },
  { id: 'captures', start: 36.2, text: 'can capture it at different moments.' },
  { id: 'depthWarning', start: 39.1, text: 'That movement can be mistaken for depth,\nmaking the surface measurement unreliable.' },
  { id: 'holdAndTrack', start: 43.5, text: 'So, during movement, we keep the last surface snapshot\nwhile the simpler position estimate follows the object.' },
  { id: 'refresh', start: 50, text: 'Once the object settles, we refresh the cloud.' },
];
export const FILM_CAPTIONS = captionCues.map((cue, index) => ({
  ...cue, end: index + 1 < captionCues.length ? captionCues[index + 1].start : FILM_ENDING.captionEnd,
}));
export const FILM_CUES = Object.fromEntries(FILM_CAPTIONS.map((cue) => [cue.id, cue]));
export const FILM_SCRIPT = FILM_CAPTIONS.map(({ text }) => text.replaceAll('\n', ' ')).join(' ');

// Millisecond precision keeps anchored events on their intended frame boundaries.
function after(time, seconds) { return Math.round((time + seconds) * 1000) / 1000; }

export const REFRESH_AT = after(FILM_CUES.refresh.start, 1);
export const REFRESH_END = after(FILM_ENDING.captionEnd, -0.4);
export const MOTION_START = FILM_CUES.movement.start;
export const MOTION_END = after(FILM_CUES.refresh.start, -1);
export const MOTION_SPEED = 1; // cm/s
const CAPTURE_DISPLACEMENT = 1;
const firstShutter = after(FILM_CUES.captures.start, 0.5);
export const SHUTTERS = { first: firstShutter, second: after(firstShutter, CAPTURE_DISPLACEMENT / MOTION_SPEED) };
export const SHUTTER_EFFECTS = { panelRevealSeconds: 0.25, panelFadeSeconds: 1.3, flashSeconds: 0.4 };
const spongeFadeOutStart = SHUTTERS.second + 0.5;
export const SPONGE_FADE = { outStart: spongeFadeOutStart, outEnd: spongeFadeOutStart + 0.8, inStart: after(MOTION_END, -2), inEnd: after(MOTION_END, -0.2) };
export const DEPTH_RECONSTRUCTION = {
  start: SPONGE_FADE.outEnd,
  rayEnd: SPONGE_FADE.outEnd + 1.2,
  pointEnd: SPONGE_FADE.outEnd + 1.8,
};
export const FILM_VIDEO = '/renders/perception-explainer-v4.mp4';
export const FILM_CAPTION_FILE = '/renders/perception-explainer-v4.vtt';
export const INTRO_VIEW_FOV_DEG = 30;
export const INTRO_LENS_OFFSET_CM = 0.6;
const lensEntryEnd = after(FILM_CUES.findTarget.start, -0.3);
const pullbackStart = after(FILM_CUES.flatImage.start, 0.7);
const pullbackEnd = after(FILM_CUES.positionOnly.start, -0.2);
const positionBlendEnd = after(FILM_CUES.position.start, 0.7);
export const INTRO_TIMING = {
  orbitEnd: 1.7, lensEntryEnd, pullbackStart, pullbackEnd,
  maskStart: after(lensEntryEnd, 0.4), maskEnd: after(lensEntryEnd, 2.8),
  maskFadeStart: after(pullbackStart, 2), maskFadeEnd: after(pullbackEnd, -0.5),
  positionBlendEnd, raysStart: after(FILM_CUES.combineCenters.start, -0.2), raysEnd: positionBlendEnd,
  featuresStart: after(FILM_CUES.combineCenters.start, 0.8), rigLabelsHide: 2,
  rigLabelsReturn: after(FILM_CUES.position.start, -0.8),
  emphasisStart: after(pullbackEnd, -0.7), emphasisEnd: after(FILM_CUES.positionOnly.start, 0.3),
};
export const CENTER_GATHER = { start: after(FILM_CUES.trackTarget.start, -0.3), end: after(FILM_CUES.trackTarget.start, 1.2), fadeEnd: after(FILM_CUES.trackTarget.start, 1.7) };
export const DEPTH_VIEW = { eye: [18, 30, 20], target: [3, 2, 1], fov: 38 };
export const DEPTH_CLEANUP = {
  fadeStart: after(SPONGE_FADE.inStart, -1.2), fadeEnd: after(SPONGE_FADE.inStart, -0.4),
  viewReturnStart: after(SPONGE_FADE.inStart, -1), viewReturnEnd: after(SPONGE_FADE.inStart, 1.2),
  evidenceFadeStart: after(SPONGE_FADE.inStart, -1), evidenceFadeEnd: after(SPONGE_FADE.inStart, 0.2),
};
export const LABEL_TIMING = { fadeStart: after(SHUTTERS.first, -0.8), fadeEnd: SHUTTERS.first, returnStart: after(SPONGE_FADE.inStart, 0.1), returnEnd: DEPTH_CLEANUP.viewReturnEnd };
export const SURFACE_MATCHES = [
  { sampleIndex: 0, start: after(FILM_CUES.matchDetails.start, 0.45), matchEnd: after(FILM_CUES.matchDetails.start, 1.05), rayEnd: after(FILM_CUES.matchDetails.start, 1.75) },
  { sampleIndex: 1, start: FILM_CUES.surfaceCloud.start, matchEnd: after(FILM_CUES.surfaceCloud.start, 0.28), rayEnd: after(FILM_CUES.surfaceCloud.start, 0.63) },
  { sampleIndex: 2, start: after(FILM_CUES.surfaceCloud.start, 0.69), matchEnd: after(FILM_CUES.surfaceCloud.start, 0.84), rayEnd: after(FILM_CUES.surfaceCloud.start, 1.06) },
];
export const SURFACE_FILL_END = after(FILM_CUES.surfaceCloud.start, 2);
const dissolveSeconds = 0.65;
const movementEstablishSeconds = 0.8;
const movementTransitionStart = after(MOTION_START, -movementEstablishSeconds - dissolveSeconds);
export const SURFACE_ORBIT = { start: after(SURFACE_FILL_END, 0.6), end: after(movementTransitionStart, -0.15) };
export const SURFACE_CONTEXT_FADE = { start: SURFACE_ORBIT.start, end: after(SURFACE_ORBIT.start, 1.2) };
export const MOTION_EASE_SECONDS = 1;
const visibleDistance = MOTION_SPEED * (SPONGE_FADE.outEnd - MOTION_START + MOTION_END - SPONGE_FADE.inStart - MOTION_EASE_SECONDS);
const hiddenSpeed = (LIVE_X - SNAPSHOT_X - visibleDistance - MOTION_SPEED * MOTION_EASE_SECONDS)
  / (SPONGE_FADE.inStart - SPONGE_FADE.outEnd - MOTION_EASE_SECONDS);
// Smooth velocity ramps preserve the fast shutter motion and continuous hidden
// travel, then settle at the original endpoint before the refresh caption.
export const MOTION_SEGMENTS = [
  { start: MOTION_START, end: MOTION_START + MOTION_EASE_SECONDS, from: 0, to: MOTION_SPEED },
  { start: MOTION_START + MOTION_EASE_SECONDS, end: SPONGE_FADE.outEnd, from: MOTION_SPEED, to: MOTION_SPEED },
  { start: SPONGE_FADE.outEnd, end: SPONGE_FADE.outEnd + MOTION_EASE_SECONDS, from: MOTION_SPEED, to: hiddenSpeed },
  { start: SPONGE_FADE.outEnd + MOTION_EASE_SECONDS, end: SPONGE_FADE.inStart - MOTION_EASE_SECONDS, from: hiddenSpeed, to: hiddenSpeed },
  { start: SPONGE_FADE.inStart - MOTION_EASE_SECONDS, end: SPONGE_FADE.inStart, from: hiddenSpeed, to: MOTION_SPEED },
  { start: SPONGE_FADE.inStart, end: MOTION_END - MOTION_EASE_SECONDS, from: MOTION_SPEED, to: MOTION_SPEED },
  { start: MOTION_END - MOTION_EASE_SECONDS, end: MOTION_END, from: MOTION_SPEED, to: 0 },
];

const surfaceStart = after(FILM_CUES.matchDetails.start, -0.2);
export const FILM_SHOTS = [
  { at: 0, id: 'intro', dissolve: 0 },
  { at: surfaceStart, id: 'surface', dissolve: dissolveSeconds },
  { at: movementTransitionStart, id: 'movement', dissolve: dissolveSeconds },
];

export const FILM_CHAPTERS = [
  { at: 0, title: 'Perceiving a target object', number: '01' },
  { at: INTRO_TIMING.lensEntryEnd, title: 'Find the target object', number: '02' },
  { at: INTRO_TIMING.raysStart, title: 'Position in space', number: '03' },
  { at: surfaceStart, title: 'A surface snapshot', number: '04' },
  { at: MOTION_START, title: 'What happens during movement?', number: '05' },
  { at: MOTION_END, title: 'Two complementary views', number: '06' },
];

export function ease(time, start, end) {
  const progress = Math.max(0, Math.min(1, (time - start) / (end - start)));
  return progress * progress * (3 - 2 * progress);
}

/** Current and upcoming phrases share fixed caption rows. At each contiguous
 * cue boundary the gray preview becomes the main text without jumping; the
 * outgoing phrase fades above it and the next preview rises in from below.
 */
export function captionStack(time) {
  const index = FILM_CAPTIONS.findIndex((cue) => time >= cue.start && time < cue.end);
  if (index < 0) return [];
  const current = FILM_CAPTIONS[index];
  const previous = index > 0 ? FILM_CAPTIONS[index - 1] : null;
  const next = index + 1 < FILM_CAPTIONS.length ? FILM_CAPTIONS[index + 1] : null;
  const rolling = previous !== null && previous.end === current.start;
  const progress = ease(time, current.start, current.start + CAPTION_ROLL_SECONDS);
  const items = [];
  if (rolling && progress < 1) items.push({ text: previous.text, slot: 0 - progress, emphasis: 1, opacity: 1 - progress });
  items.push({ text: current.text, slot: rolling ? 1 - progress : 0, emphasis: rolling ? progress : 1, opacity: rolling ? 1 : progress });
  if (next) items.push({ text: next.text, slot: rolling ? 2 - progress : 1, emphasis: 0, opacity: progress });
  return items.filter(({ opacity }) => opacity > 0);
}

export function surfaceReconstruction(time, totalPoints) {
  const matches = SURFACE_MATCHES.map(({ start, matchEnd, rayEnd }) => ({
    visible: time >= start,
    connection: ease(time, start, matchEnd),
    rays: ease(time, matchEnd, rayEnd),
  }));
  const completed = SURFACE_MATCHES.filter(({ rayEnd }) => time >= rayEnd).length;
  const fill = ease(time, SURFACE_MATCHES.at(-1).rayEnd, SURFACE_FILL_END);
  return { matches, pointCount: completed + Math.floor((totalPoints - SURFACE_MATCHES.length) * fill) };
}

function blend(a, b, progress) {
  return a.map((value, index) => value + (b[index] - value) * progress);
}

function bezier(a, b, c, d, progress) {
  const remaining = 1 - progress;
  return a.map((value, index) => remaining ** 3 * value + 3 * remaining ** 2 * progress * b[index] + 3 * remaining * progress ** 2 * c[index] + progress ** 3 * d[index]);
}

export function introView(time, lens) {
  const sceneTarget = [5, 5, 18];
  const opticalTarget = [CAMERA_AIM_X, SPONGE_SIZE[1] / 2, 0];
  const lensDistance = Math.hypot(...opticalTarget.map((value, index) => value - lens[index]));
  const lensFront = blend(lens, opticalTarget, INTRO_LENS_OFFSET_CM / lensDistance);
  const orbit = ease(time, 0, INTRO_TIMING.orbitEnd);
  const angle = -0.9 + (-1.75 + 0.9) * orbit;
  const orbitEye = [5 + Math.sin(angle) * 56, 33 - 4 * orbit, 18 + Math.cos(angle) * 56];
  const orbitEnd = [5 + Math.sin(-1.75) * 56, 29, 18 + Math.cos(-1.75) * 56];
  const pullback = STORY_FRAMES.position;
  if (time <= INTRO_TIMING.orbitEnd) return { eye: orbitEye, target: sceneTarget, fov: 43 };
  if (time < INTRO_TIMING.lensEntryEnd) {
    const progress = ease(time, INTRO_TIMING.orbitEnd, INTRO_TIMING.lensEntryEnd);
    const front = blend(lensFront, opticalTarget, 0.2);
    return { eye: bezier(orbitEnd, [-30, 30, 16], front, lensFront, progress), target: blend(sceneTarget, opticalTarget, progress), fov: 43 + (INTRO_VIEW_FOV_DEG - 43) * progress };
  }
  if (time <= INTRO_TIMING.pullbackStart) return { eye: lensFront, target: opticalTarget, fov: INTRO_VIEW_FOV_DEG };
  const progress = ease(time, INTRO_TIMING.pullbackStart, INTRO_TIMING.pullbackEnd);
  return { eye: blend(lensFront, pullback.eye, progress), target: blend(opticalTarget, pullback.target, progress), fov: INTRO_VIEW_FOV_DEG + (pullback.fov - INTRO_VIEW_FOV_DEG) * progress };
}

export function movementView(time) {
  const wide = STORY_FRAMES.reject;
  const depth = DEPTH_VIEW;
  const intoDepth = ease(time, SPONGE_FADE.outStart, DEPTH_RECONSTRUCTION.rayEnd);
  const outOfDepth = ease(time, DEPTH_CLEANUP.viewReturnStart, DEPTH_CLEANUP.viewReturnEnd);
  const close = intoDepth * (1 - outOfDepth);
  return { eye: blend(wide.eye, depth.eye, close), target: blend(wide.target, depth.target, close), fov: 43 + (depth.fov - 43) * close };
}

export function spongeX(time) {
  if (time <= MOTION_START) return SNAPSHOT_X;
  if (time >= MOTION_END) return LIVE_X;
  let position = SNAPSHOT_X;
  for (const { start, end, from, to } of MOTION_SEGMENTS) {
    const duration = end - start;
    const progress = Math.max(0, Math.min(1, (time - start) / duration));
    // Integral of smoothstep velocity gives zero acceleration at every join.
    position += duration * (from * progress + (to - from) * (progress ** 3 - 0.5 * progress ** 4));
  }
  return position;
}

export function surfaceView(time) {
  const frame = STORY_FRAMES.surface;
  const progress = ease(time, SURFACE_ORBIT.start, SURFACE_ORBIT.end);
  const target = blend(frame.orbitTargetFrom, frame.target, progress);
  const from = frame.orbitFrom.map((value, i) => value - frame.orbitTargetFrom[i]);
  const to = frame.eye.map((value, i) => value - frame.target[i]);
  const startAngle = Math.atan2(from[0], from[2]);
  const angle = startAngle + (Math.atan2(to[0], to[2]) - startAngle) * progress;
  const radius = Math.hypot(from[0], from[2]) + (Math.hypot(to[0], to[2]) - Math.hypot(from[0], from[2])) * progress;
  return {
    eye: [target[0] + Math.sin(angle) * radius, target[1] + from[1] + (to[1] - from[1]) * progress, target[2] + Math.cos(angle) * radius],
    target,
  };
}

export function sampleFilm(seconds) {
  if (!Number.isFinite(seconds)) throw new Error('Film time must be finite');
  const time = Math.max(0, Math.min(FILM_DURATION, seconds));
  const index = FILM_SHOTS.findLastIndex((shot) => time >= shot.at);
  const shot = FILM_SHOTS[index];
  const chapter = FILM_CHAPTERS.findLast((entry) => time >= entry.at);
  const caption = FILM_CAPTIONS.find((cue) => time >= cue.start && time < cue.end);
  const liveX = spongeX(time);
  return {
    time,
    shot: shot.id,
    previous: index === 0 ? shot.id : FILM_SHOTS[index - 1].id,
    dissolve: shot.dissolve === 0 ? 1 : ease(time, shot.at, shot.at + shot.dissolve),
    chapter,
    caption: caption ? caption.text : '',
    captionId: caption ? caption.id : null,
    captionStack: captionStack(time),
    captionOpacity: 1 - ease(time, FILM_ENDING.captionFadeStart, FILM_ENDING.captionEnd),
    centerConvergence: ease(time, CENTER_GATHER.start, CENTER_GATHER.end),
    centerCueOpacity: ease(time, CENTER_GATHER.start, CENTER_GATHER.start + 0.2) * (1 - ease(time, CENTER_GATHER.end - 0.2, CENTER_GATHER.fadeEnd)),
    introPointScale: ease(time, CENTER_GATHER.end - 0.35, CENTER_GATHER.fadeEnd),
    positionEmphasis: ease(time, INTRO_TIMING.emphasisStart, INTRO_TIMING.emphasisEnd),
    liveX,
    heldCloudOpacity: 1 - 0.5 * ease(time, MOTION_START, MOTION_START + 3),
    spongeOpacity: 1 - ease(time, SPONGE_FADE.outStart, SPONGE_FADE.outEnd) + ease(time, SPONGE_FADE.inStart, SPONGE_FADE.inEnd),
    snapshotX: time < REFRESH_AT ? SNAPSHOT_X : LIVE_X,
    firstCaptured: time >= SHUTTERS.first,
    secondCaptured: time >= SHUTTERS.second,
    depthRays: ease(time, DEPTH_RECONSTRUCTION.start, DEPTH_RECONSTRUCTION.rayEnd) * (1 - ease(time, DEPTH_CLEANUP.fadeStart, DEPTH_CLEANUP.fadeEnd)),
    falseDepth: ease(time, DEPTH_RECONSTRUCTION.rayEnd, DEPTH_RECONSTRUCTION.pointEnd) * (1 - ease(time, DEPTH_CLEANUP.fadeStart, DEPTH_CLEANUP.fadeEnd)),
    ending: ease(time, FILM_DURATION - FILM_ENDING.fadeSeconds, FILM_DURATION),
  };
}
