/** Exact frame export from a production preview, isolated from dev hot reload. */
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DRAFT_DURATION, DRAFT_FPS } from '../src/motionDraft.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.resolve(root, process.argv[2] || 'public/renders/perception-motion-v2.mp4');
await mkdir(path.dirname(output), { recursive: true });
await mkdir(path.join(root, 'keyframes'), { recursive: true });
await access('/usr/bin/chromium');
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let encoder;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:4173/?frame&export&style=studio');
  await page.waitForFunction(() => Boolean(window.perceptionDraft));
  await page.evaluate(() => document.fonts.ready);
  encoder = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-n', '-f', 'image2pipe', '-framerate', String(DRAFT_FPS), '-vcodec', 'png', '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', output], { stdio: ['pipe', 'inherit', 'inherit'] });
  const completion = once(encoder, 'close');
  const frames = DRAFT_DURATION * DRAFT_FPS;
  for (let index = 0; index < frames; index++) {
    if (errors.length) throw new Error(errors.join('\n'));
    const result = await page.evaluate((seconds) => window.perceptionDraft.renderFrame(seconds), index / DRAFT_FPS);
    const png = Buffer.from(result.png.split(',')[1], 'base64');
    if (!encoder.stdin.write(png)) await once(encoder.stdin, 'drain');
    if ([0, 195, 285, frames - 1].includes(index)) {
      await writeFile(path.join(root, 'keyframes', `${path.parse(output).name}-${String(index).padStart(3, '0')}.png`), png);
    }
    if (index % DRAFT_FPS === 0) console.log(`Rendered ${index / DRAFT_FPS}s / ${DRAFT_DURATION}s`);
  }
  encoder.stdin.end();
  const [code] = await completion;
  if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
  console.log(`Saved ${output}`);
} finally {
  if (encoder && encoder.exitCode === null) encoder.kill('SIGTERM');
  await browser.close();
}
