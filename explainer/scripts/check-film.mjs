import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { FILM_DURATION, FILM_ENDING, FILM_CAPTIONS, FILM_CUES, FILM_SCRIPT, CAPTION_ROLL_SECONDS, MOTION_START } from '../src/filmTimeline.js';
import { FILM_REVIEW_TIMES } from './film-review.mjs';

await mkdir(new URL('../keyframes/', import.meta.url), { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  // Chromium requests an implicit favicon even for the clean render page.
  await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204 }));
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('http://127.0.0.1:4173/?frame&export&film');
  await page.waitForFunction(() => Boolean(window.perceptionFilm), null, { timeout: 60000 });
  await page.evaluate(() => document.fonts.ready);
  const render = (seconds) => page.evaluate((time) => window.perceptionFilm.renderFrame(time), seconds);
  const captionCache = (await render(0)).captionPixels;
  assert.equal(captionCache.width, 1280);
  assert.equal(captionCache.height, 96);
  assert.equal(captionCache.textures, FILM_CAPTIONS.length);
  const captionTransition = FILM_CUES.trackTarget.start;
  const transitionSamples = [captionTransition, captionTransition + CAPTION_ROLL_SECONDS / 2, captionTransition + CAPTION_ROLL_SECONDS];
  for (const time of [...FILM_REVIEW_TIMES, ...transitionSamples]) {
    const result = await render(time);
    assert.deepEqual(result.captionPixels, captionCache, 'Caption motion and cue changes must not rerasterize text');
    assert.equal(result.state.time, time);
    await writeFile(new URL(`../keyframes/film-${time.toFixed(3)}s.png`, import.meta.url), Buffer.from(result.png.split(',')[1], 'base64'));
    console.log(`Reviewed ${time}s: ${result.state.shot}`);
  }
  const seekSamples = [...FILM_CAPTIONS.map(({ start, end }) => (start + end) / 2), ...transitionSamples];
  for (const time of seekSamples) {
    const first = await render(time);
    await render(FILM_DURATION - 0.5);
    await render(1);
    const repeated = await render(time);
    assert.equal(repeated.png, first.png, `Seeking must reproduce the same pixels at ${time}s`);
  }
  const captionRegion = (seconds) => page.evaluate((time) => {
    window.perceptionFilm.renderFrame(time);
    const crop = document.createElement('canvas');
    crop.width = 1280;
    crop.height = 170;
    crop.getContext('2d').drawImage(document.querySelector('canvas'), 0, 550, 1280, 170, 0, 0, 1280, 170);
    return crop.toDataURL('image/png');
  }, seconds);
  assert.equal(await captionRegion(captionTransition - 0.000001), await captionRegion(captionTransition), 'Caption promotion must start from the exact preview pixels, not jump');
  // This camera and scene are static across the final caption fade. A text-free strip
  // must stay pixel-identical instead of flashing when the subtitle vanishes.
  const subtitleBackdrop = (seconds) => page.evaluate((time) => {
    window.perceptionFilm.renderFrame(time);
    const crop = document.createElement('canvas');
    crop.width = 80;
    crop.height = 100;
    crop.getContext('2d').drawImage(document.querySelector('canvas'), 0, 620, 80, 100, 0, 0, 80, 100);
    return crop.toDataURL('image/png');
  }, seconds);
  assert.equal(await subtitleBackdrop((FILM_ENDING.captionFadeStart + FILM_ENDING.captionEnd) / 2), await subtitleBackdrop(FILM_ENDING.captionEnd + 0.2), 'Subtitle backdrop must persist after the caption fades');
  const holdStart = FILM_ENDING.captionEnd;
  const heldEnding = await render(holdStart);
  assert.equal((await render(holdStart + FILM_ENDING.holdSeconds / 2)).png, heldEnding.png, 'The final scene must hold after the caption fades');
  assert.equal((await render(holdStart + FILM_ENDING.holdSeconds)).png, heldEnding.png, 'The scene fade must wait until the one-second hold finishes');
  assert.equal((await render(FILM_DURATION)).state.ending, 1);
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.waitForFunction(() => document.querySelector('canvas').width === 1600);
  const enlarged = await render(6);
  assert.deepEqual(enlarged.hudPixels, { width: 1600, height: 900 });
  assert.equal(enlarged.captionPixels.width, 1600);
  assert.equal(enlarged.captionPixels.height, 120);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.waitForFunction(() => document.querySelector('canvas').width === 1280);
  assert.deepEqual((await render(6)).hudPixels, { width: 1280, height: 720 });
  assert.deepEqual(errors, []);
  const retina = await browser.newPage({ viewport: { width: 800, height: 450 }, deviceScaleFactor: 2 });
  await retina.route('**/favicon.ico', (route) => route.fulfill({ status: 204 }));
  retina.on('pageerror', (error) => errors.push(error.message));
  retina.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await retina.goto('http://127.0.0.1:4173/?frame&export&film');
  await retina.waitForFunction(() => Boolean(window.perceptionFilm));
  const highDpi = await retina.evaluate(() => window.perceptionFilm.renderFrame(6));
  assert.deepEqual(highDpi.hudPixels, { width: 1600, height: 900 });
  assert.equal(highDpi.captionPixels.width, 1600);
  assert.equal(highDpi.captionPixels.height, 120);
  await writeFile(new URL('../keyframes/film-hidpi.png', import.meta.url), Buffer.from(highDpi.png.split(',')[1], 'base64'));
  await retina.close();
  assert.deepEqual(errors, []);
  await page.goto('http://127.0.0.1:4173/?film');
  await page.locator('canvas').waitFor();
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).getAttribute('max'), String(FILM_DURATION));
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForTimeout(500);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  const paused = await page.getByRole('slider', { name: 'Timeline' }).inputValue();
  await page.waitForTimeout(200);
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).inputValue(), paused);
  await page.getByRole('slider', { name: 'Timeline' }).fill(String(MOTION_START + 1));
  await page.getByRole('button', { name: 'Reset view' }).click();
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).inputValue(), '0');
  await page.getByRole('button', { name: 'Storyboard frames' }).click();
  assert.equal(await page.getByRole('button', { name: '01 · Two views' }).count(), 1);
  await page.getByRole('button', { name: 'Full video', exact: true }).click();
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).getAttribute('max'), String(FILM_DURATION));
  await page.getByText('Narration script', { exact: true }).click();
  assert.equal(await page.locator('details').filter({ has: page.getByText('Narration script', { exact: true }) }).locator('p').textContent(), FILM_SCRIPT);
  assert.deepEqual(errors, []);
  console.log('Checked exact repeated renders, film playback, pause, scrubbing, reset, and gallery switching.');
} finally { await browser.close(); }
