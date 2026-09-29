/** Shared time-based motion for browser playback and offline frame export. */
export const DRAFT_DURATION = 14;
export const DRAFT_FPS = 30;
export const SNAPSHOT_X = -5;
export const LIVE_X = 5;

function ease(t, start, end) {
  const x = Math.max(0, Math.min(1, (t - start) / (end - start)));
  return x * x * (3 - 2 * x);
}

function blend(a, b, amount) {
  return a.map((value, i) => value + (b[i] - value) * amount);
}

export function sampleDraft(seconds) {
  if (!Number.isFinite(seconds)) throw new Error('Timeline time must be finite');
  const time = Math.max(0, Math.min(DRAFT_DURATION, seconds));
  const dolly = ease(time, 1.5, 6.5);
  const movement = ease(time, 7, 11);
  return {
    time,
    eye: blend([43, 39, 64], [12, 11, 24], dolly),
    target: blend([5, 7, 20], [0, 1.6, 0], dolly),
    liveX: SNAPSHOT_X + (LIVE_X - SNAPSHOT_X) * movement,
    snapshotX: SNAPSHOT_X,
    labels: ease(time, 8, 9.5),
    rigLabels: 1 - ease(time, 2, 4),
    phase: time < 1.5 ? 'Establish the two cameras' : time < 6.5 ? 'Move into the workspace' : time < 7 ? 'Hold before movement' : time < 11 ? 'Position follows · surface stays' : 'Hold the separation',
  };
}
