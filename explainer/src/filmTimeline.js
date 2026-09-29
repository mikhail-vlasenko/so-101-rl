import { LIVE_X, SNAPSHOT_X } from './motionDraft.js';
import { STORY_FRAMES } from './storyFrames.js';
import { CAMERA_AIM_X, SPONGE_SIZE } from './sceneConfig.js';

/** Editorial seconds, not hardware latencies. Every frame is sampled directly
 * from time, so seeks and export have the same shutters and held geometry.
 */
export const FILM_ENDING = { captionFadeStart: 52.5, captionEnd: 53, holdSeconds: 1, fadeSeconds: 1 };
export const FILM_DURATION = FILM_ENDING.captionEnd + FILM_ENDING.holdSeconds + FILM_ENDING.fadeSeconds;
export const FILM_FPS = 30;
export const REFRESH_AT = 51;
export const MOTION_START = 29;
export const MOTION_END = 49;
export const MOTION_SPEED = 1; // cm/s: twice the previous visible movement speed.
const CAPTURE_DISPLACEMENT = 1;
export const SHUTTERS = { first: 34.6, second: 34.6 + CAPTURE_DISPLACEMENT / MOTION_SPEED };
const spongeFadeOutStart = SHUTTERS.second + 0.5;
export const SPONGE_FADE = { outStart: spongeFadeOutStart, outEnd: spongeFadeOutStart + 0.8, inStart: 47, inEnd: 48.8 };
export const FILM_VIDEO = '/renders/perception-explainer-v4.mp4';
export const FILM_CAPTION_FILE = '/renders/perception-explainer-v4.vtt';
export const INTRO_VIEW_FOV_DEG = 30;
export const INTRO_LENS_OFFSET_CM = 0.6;
export const CENTER_GATHER = { start: 7.2, end: 8.7, fadeEnd: 9.2 };
export const DEPTH_VIEW = { eye: [18, 30, 20], target: [3, 2, 1], fov: 38 };
export const SURFACE_MATCHES = [
  { sampleIndex: 0, start: 19.65, matchEnd: 20.25, rayEnd: 20.95 },
  { sampleIndex: 1, start: 21.03, matchEnd: 21.31, rayEnd: 21.66 },
  { sampleIndex: 2, start: 21.72, matchEnd: 21.87, rayEnd: 22.09 },
];
export const SURFACE_FILL_END = 23;
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

export const FILM_SHOTS = [
  { at: 0, id: 'intro', dissolve: 0 },
  { at: 19, id: 'surface', dissolve: 0.65 },
  { at: 27.1, id: 'movement', dissolve: 0.65 },
];

export const FILM_CHAPTERS = [
  { at: 0, title: 'Perceiving a target object', number: '01' },
  { at: 4, title: 'Find the target object', number: '02' },
  { at: 11, title: 'Position in space', number: '03' },
  { at: 19, title: 'A surface snapshot', number: '04' },
  { at: 29, title: 'What happens during movement?', number: '05' },
  { at: 49, title: 'Two complementary views', number: '06' },
];

export const FILM_CAPTIONS = [
  { start: 0.3, end: 4.3, text: 'How do we perceive an object\nwe want to interact with?' },
  { start: 4.3, end: 7.7, text: 'First, a vision model finds\nthe target object’s pixels…' },
  { start: 7.7, end: 10.8, text: '…and tracks them in both camera views.' },
  { start: 11.3, end: 15.3, text: 'The two image centers give us\nan approximate 3D position.' },
  { start: 15.3, end: 18.7, text: 'That tells us where it is—not its shape.' },
  { start: 19.3, end: 23, text: 'Matching details across the images\nbuilds a surface point cloud.' },
  { start: 23, end: 27.1, text: 'This shows its shape, size,\nand how it’s turned.' },
  { start: 29, end: 33.8, text: 'When the sponge starts moving…' },
  { start: 34.3, end: 37, text: '…the cameras can capture at different moments.' },
  { start: 37, end: 43.3, text: '…and that motion can be mistaken for depth.' },
  { start: 43.3, end: 49, text: 'That makes detailed stereo depth\nunreliable during movement.' },
  { start: 49.2, end: FILM_ENDING.captionEnd, text: 'So we refresh the cloud\nwhen the sponge settles.' },
];

export function ease(time, start, end) {
  const progress = Math.max(0, Math.min(1, (time - start) / (end - start)));
  return progress * progress * (3 - 2 * progress);
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
  const orbit = ease(time, 0, 1.7);
  const angle = -0.9 + (-1.75 + 0.9) * orbit;
  const orbitEye = [5 + Math.sin(angle) * 56, 33 - 4 * orbit, 18 + Math.cos(angle) * 56];
  const orbitEnd = [5 + Math.sin(-1.75) * 56, 29, 18 + Math.cos(-1.75) * 56];
  const pullback = STORY_FRAMES.position;
  if (time <= 1.7) return { eye: orbitEye, target: sceneTarget, fov: 43 };
  if (time < 4) {
    const progress = ease(time, 1.7, 4);
    const front = blend(lensFront, opticalTarget, 0.2);
    return { eye: bezier(orbitEnd, [-30, 30, 16], front, lensFront, progress), target: blend(sceneTarget, opticalTarget, progress), fov: 43 + (INTRO_VIEW_FOV_DEG - 43) * progress };
  }
  if (time <= 10.5) return { eye: lensFront, target: opticalTarget, fov: INTRO_VIEW_FOV_DEG };
  const progress = ease(time, 10.5, 16);
  return { eye: blend(lensFront, pullback.eye, progress), target: blend(opticalTarget, pullback.target, progress), fov: INTRO_VIEW_FOV_DEG + (pullback.fov - INTRO_VIEW_FOV_DEG) * progress };
}

export function movementView(time) {
  const wide = STORY_FRAMES.reject;
  const depth = DEPTH_VIEW;
  const intoDepth = ease(time, 40, 42);
  const outOfDepth = ease(time, 46, 48.2);
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
  const progress = ease(time, 23, 27);
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
    captionOpacity: 1 - ease(time, FILM_ENDING.captionFadeStart, FILM_ENDING.captionEnd),
    centerConvergence: ease(time, CENTER_GATHER.start, CENTER_GATHER.end),
    centerCueOpacity: ease(time, CENTER_GATHER.start, CENTER_GATHER.start + 0.2) * (1 - ease(time, CENTER_GATHER.end - 0.2, CENTER_GATHER.fadeEnd)),
    introPointScale: ease(time, CENTER_GATHER.end - 0.35, CENTER_GATHER.fadeEnd),
    positionEmphasis: ease(time, 15.3, 16.5),
    liveX,
    heldCloudOpacity: 1 - 0.5 * ease(time, MOTION_START, MOTION_START + 3),
    spongeOpacity: 1 - ease(time, SPONGE_FADE.outStart, SPONGE_FADE.outEnd) + ease(time, SPONGE_FADE.inStart, SPONGE_FADE.inEnd),
    snapshotX: time < REFRESH_AT ? SNAPSHOT_X : LIVE_X,
    firstCaptured: time >= SHUTTERS.first,
    secondCaptured: time >= SHUTTERS.second,
    falseDepth: ease(time, 41.5, 42.5) * (1 - ease(time, 45.8, 46.6)),
    ending: ease(time, FILM_DURATION - FILM_ENDING.fadeSeconds, FILM_DURATION),
  };
}
