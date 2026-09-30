import { FILM_CAPTIONS, FILM_DURATION, FILM_ENDING, FILM_SHOTS, INTRO_TIMING, CENTER_GATHER, SURFACE_MATCHES, SURFACE_ORBIT, SURFACE_FILL_END, SHUTTERS, SPONGE_FADE, DEPTH_RECONSTRUCTION, DEPTH_CLEANUP, REFRESH_AT, REFRESH_END } from '../src/filmTimeline.js';

/** Review the actual cues and event boundaries; retiming needs no parallel list. */
export const FILM_REVIEW_TIMES = [...new Set([
  0,
  ...FILM_CAPTIONS.map(({ start, end }) => (start + end) / 2),
  ...FILM_SHOTS.map(({ at, dissolve }) => at + dissolve / 2),
  INTRO_TIMING.orbitEnd, INTRO_TIMING.lensEntryEnd, INTRO_TIMING.pullbackStart, INTRO_TIMING.pullbackEnd,
  CENTER_GATHER.start, CENTER_GATHER.end, CENTER_GATHER.fadeEnd,
  ...SURFACE_MATCHES.flatMap(({ start, matchEnd, rayEnd }) => [start, matchEnd, rayEnd]),
  SURFACE_FILL_END, SURFACE_ORBIT.start, SURFACE_ORBIT.end,
  SHUTTERS.first, SHUTTERS.second,
  SPONGE_FADE.outStart, SPONGE_FADE.outEnd, SPONGE_FADE.inStart, SPONGE_FADE.inEnd,
  DEPTH_RECONSTRUCTION.start + 0.3, DEPTH_RECONSTRUCTION.rayEnd, DEPTH_RECONSTRUCTION.pointEnd,
  DEPTH_CLEANUP.viewReturnEnd, REFRESH_AT, REFRESH_END,
  FILM_ENDING.captionFadeStart, FILM_ENDING.captionEnd,
  FILM_ENDING.captionEnd + FILM_ENDING.holdSeconds / 2,
  FILM_DURATION - FILM_ENDING.fadeSeconds, FILM_DURATION - FILM_ENDING.fadeSeconds / 2, FILM_DURATION,
].map((time) => Math.round(time * 1000) / 1000))].sort((a, b) => a - b);
