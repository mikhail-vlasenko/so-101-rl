import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { FILM_DURATION } from '../src/filmTimeline.js';

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
  for (const time of [0, 1.6, 3, 4, 6, 7.3, 7.8, 8.3, 8.7, 9.2, 10, 12, 15, 17, 18.6, 19.7, 20.1, 20.6, 20.95, 21.4, 22, 22.4, 23, 26.8, 28, 32, 34.8, 35.6, 36, 36.4, 37.8, 39.5, 43, 43.5, 47, 47.6, 48, 48.8, 49, 50.9, 51, 51.8, 52.5, 52.8, 53, 53.5, 54, 54.5, 55]) {
    const result = await render(time);
    assert.equal(result.state.time, time);
    await writeFile(new URL(`../keyframes/film-${time.toFixed(1)}s.png`, import.meta.url), Buffer.from(result.png.split(',')[1], 'base64'));
    console.log(`Reviewed ${time}s: ${result.state.shot}`);
  }
  for (const time of [6, 7.8, 8.7, 9.2, 12, 17, 21.5, 32, 35.7, 43, 47.6, 48, 51.8, 53]) {
    const first = await render(time);
    await render(FILM_DURATION - 0.5);
    await render(1);
    const repeated = await render(time);
    assert.equal(repeated.png, first.png, `Seeking must reproduce the same pixels at ${time}s`);
  }
  // This camera and scene are static across the cue gap. A text-free strip
  // must stay pixel-identical instead of flashing when the subtitle vanishes.
  const subtitleBackdrop = (seconds) => page.evaluate((time) => {
    window.perceptionFilm.renderFrame(time);
    const crop = document.createElement('canvas');
    crop.width = 80;
    crop.height = 100;
    crop.getContext('2d').drawImage(document.querySelector('canvas'), 0, 620, 80, 100, 0, 0, 80, 100);
    return crop.toDataURL('image/png');
  }, seconds);
  assert.equal(await subtitleBackdrop(18.6), await subtitleBackdrop(18.8), 'Subtitle backdrop must persist between cues');
  const heldEnding = await render(53);
  assert.equal((await render(53.5)).png, heldEnding.png, 'The final scene must hold after the caption fades');
  assert.equal((await render(54)).png, heldEnding.png, 'The scene fade must wait until the one-second hold finishes');
  assert.equal((await render(FILM_DURATION)).state.ending, 1);
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.waitForFunction(() => document.querySelector('canvas').width === 1600);
  const enlarged = await render(6);
  assert.deepEqual(enlarged.hudPixels, { width: 1600, height: 900 });
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
  await page.getByRole('slider', { name: 'Timeline' }).fill('32');
  await page.getByRole('button', { name: 'Reset view' }).click();
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).inputValue(), '0');
  await page.getByRole('button', { name: 'Storyboard frames' }).click();
  assert.equal(await page.getByRole('button', { name: '01 · Two views' }).count(), 1);
  await page.getByRole('button', { name: 'Full video', exact: true }).click();
  assert.equal(await page.getByRole('slider', { name: 'Timeline' }).getAttribute('max'), String(FILM_DURATION));
  assert.deepEqual(errors, []);
  console.log('Checked exact repeated renders, film playback, pause, scrubbing, reset, and gallery switching.');
} finally { await browser.close(); }
