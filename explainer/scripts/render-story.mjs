import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { STORY_FRAMES } from '../src/storyFrames.js';

const output = new URL('../public/keyframes/', import.meta.url);
await mkdir(output, { recursive: true });
await mkdir(new URL('../keyframes/', import.meta.url), { recursive: true });
const browser = await chromium.launch({
  executablePath: '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const ids = process.argv.length > 2 ? process.argv.slice(2) : Object.keys(STORY_FRAMES);
  for (const id of ids) {
    if (!Object.hasOwn(STORY_FRAMES, id)) throw new Error(`Unknown storyboard frame: ${id}`);
    await page.goto(`http://127.0.0.1:5173/?frame&keyframe=${id}`);
    await page.locator('canvas').waitFor();
    await page.waitForTimeout(600);
    if (errors.length) throw new Error(`Browser error on ${id}: ${errors.join('; ')}`);
    await page.locator('canvas').screenshot({ path: new URL(`story-${id}.png`, output).pathname });
    console.log(`Rendered ${id}`);
  }
  if (ids.length === Object.keys(STORY_FRAMES).length) {
    await page.goto('http://127.0.0.1:5173/?keyframe=views');
    await page.waitForFunction(() => [...document.querySelectorAll('.frame-picker img')].every((image) => image.complete && image.naturalWidth > 0));
    assert.equal(await page.locator('.frame-picker button').count(), ids.length);
    await page.getByRole('button', { name: '05c · False depth' }).click();
    assert.equal(new URL(page.url()).searchParams.get('keyframe'), 'depth');
    assert.equal(await page.locator('canvas').count(), 1);
    await page.getByRole('button', { name: '04 · Surface' }).click();
    await page.waitForTimeout(200);
    const orbitStart = await page.locator('canvas').evaluate((canvas) => canvas.toDataURL('image/png'));
    await page.waitForTimeout(STORY_FRAMES.surface.orbitDurationMs + 200);
    const orbitEnd = await page.locator('canvas').evaluate((canvas) => canvas.toDataURL('image/png'));
    assert.notEqual(orbitStart, orbitEnd, 'The surface viewer orbit should change the rendered view');
    await page.getByRole('button', { name: 'Replay orbit' }).click();
    await page.waitForTimeout(200);
    const orbitReplay = await page.locator('canvas').evaluate((canvas) => canvas.toDataURL('image/png'));
    assert.notEqual(orbitEnd, orbitReplay, 'Replay orbit should return to its starting view');
    await page.getByRole('button', { name: '01 · Two views' }).click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: new URL('../keyframes/story-gallery.png', import.meta.url).pathname, fullPage: true });
    if (errors.length) throw new Error(`Browser error in gallery: ${errors.join('; ')}`);
    console.log('Checked gallery thumbnails and frame selection.');
  }
} finally {
  await browser.close();
}
