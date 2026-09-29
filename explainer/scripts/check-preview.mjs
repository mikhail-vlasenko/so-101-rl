import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';

await mkdir(new URL('../keyframes/', import.meta.url), { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/?draft');
  await page.locator('canvas').waitFor();
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.waitForFunction(() => Number(document.querySelector('input[aria-label="Timeline"]').value) > 0.3);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  const pausedTime = await page.getByLabel('Timeline').inputValue();
  await page.waitForTimeout(250);
  assert.equal(await page.getByLabel('Timeline').inputValue(), pausedTime);
  await page.getByLabel('Timeline').fill('9.5');
  assert.equal(await page.locator('output').textContent(), '9.5 / 14s');
  await page.getByRole('button', { name: 'Reset view', exact: true }).click();
  assert.equal(await page.getByLabel('Timeline').inputValue(), '0');

  await page.goto('http://127.0.0.1:5173/?frame&export&style=studio');
  await page.waitForFunction(() => Boolean(window.perceptionDraft));
  for (const time of [0, 6.5, 9.5, 14]) {
    const first = await page.evaluate((t) => window.perceptionDraft.renderFrame(t), time);
    const second = await page.evaluate((t) => window.perceptionDraft.renderFrame(t), time);
    assert.equal(first.png, second.png, `Repeated render at ${time}s must be identical`);
    await writeFile(new URL(`../keyframes/motion-check-${time}.png`, import.meta.url), Buffer.from(first.png.split(',')[1], 'base64'));
  }
  assert.deepEqual(errors, []);
  console.log('Playback, pause, scrub, reset, repeated frame rendering, and browser errors checked.');
} finally {
  await browser.close();
}
