import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DRAFT_DURATION, SNAPSHOT_X, LIVE_X, sampleDraft } from '../src/motionDraft.js';

test('camera finishes its move before object motion; snapshot never follows', () => {
  const start = sampleDraft(0);
  const close = sampleDraft(6.5);
  const moving = sampleDraft(9);
  const end = sampleDraft(DRAFT_DURATION);
  assert.equal(start.liveX, start.snapshotX);
  assert.equal(close.liveX, SNAPSHOT_X);
  assert.notDeepEqual(start.eye, close.eye);
  assert.deepEqual(close.eye, moving.eye);
  assert.deepEqual(moving.eye, end.eye);
  assert.ok(moving.liveX > SNAPSHOT_X && moving.liveX < LIVE_X);
  for (let time = 0; time <= DRAFT_DURATION; time += 0.1) assert.equal(sampleDraft(time).snapshotX, SNAPSHOT_X);
  assert.equal(end.liveX, LIVE_X);
  assert.deepEqual(sampleDraft(11).eye, end.eye);
});

test('scrubbing backwards produces the same scene state as linear playback', () => {
  const first = sampleDraft(8.75);
  sampleDraft(14);
  sampleDraft(2);
  assert.deepEqual(sampleDraft(8.75), first);
  assert.deepEqual(sampleDraft(-1), sampleDraft(0));
  assert.deepEqual(sampleDraft(100), sampleDraft(DRAFT_DURATION));
  assert.throws(() => sampleDraft(NaN), /finite/);
});
