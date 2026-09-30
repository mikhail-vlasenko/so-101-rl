/** Full, captioned H.264 film; exact timeline frames from the production site. */
import { chromium } from 'playwright-core';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FILM_DURATION, FILM_FPS, FILM_CAPTIONS, FILM_VIDEO } from '../src/filmTimeline.js';
import { FILM_REVIEW_TIMES } from './film-review.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.resolve(root, process.argv[2] || `public${FILM_VIDEO}`);
await mkdir(path.dirname(output), { recursive: true });
await mkdir(path.join(root, 'keyframes'), { recursive: true });
const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let encoder;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('http://127.0.0.1:4173/?frame&export&film');
  await page.waitForFunction(() => Boolean(window.perceptionFilm), null, { timeout: 60000 });
  await page.evaluate(() => document.fonts.ready);
  encoder = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-n', '-f', 'image2pipe', '-framerate', String(FILM_FPS), '-vcodec', 'png', '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', output], { stdio: ['pipe', 'inherit', 'inherit'] });
  const completion = once(encoder, 'close');
  const reviewFrames = new Set(FILM_REVIEW_TIMES.map((seconds) => Math.round(seconds * FILM_FPS)));
  for (let index = 0; index < FILM_DURATION * FILM_FPS; index++) {
    if (errors.length) throw new Error(errors.join('\n'));
    const result = await page.evaluate((seconds) => window.perceptionFilm.renderFrame(seconds), index / FILM_FPS);
    const png = Buffer.from(result.png.split(',')[1], 'base64');
    if (!encoder.stdin.write(png)) await once(encoder.stdin, 'drain');
    if (reviewFrames.has(index)) await writeFile(path.join(root, 'keyframes', `film-${(index / FILM_FPS).toFixed(1)}s.png`), png);
    if (index % FILM_FPS === 0) console.log(`Rendered ${index / FILM_FPS}s / ${FILM_DURATION}s`);
  }
  encoder.stdin.end();
  const [code] = await completion;
  if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
  const timestamp = (seconds) => {
    const millis = Math.round(seconds * 1000);
    return `00:${String(Math.floor(millis / 60000)).padStart(2, '0')}:${String(Math.floor(millis / 1000) % 60).padStart(2, '0')}.${String(millis % 1000).padStart(3, '0')}`;
  };
  const captions = 'WEBVTT\n\n' + FILM_CAPTIONS.map((cue) => `${timestamp(cue.start)} --> ${timestamp(cue.end)}\n${cue.text}\n`).join('\n');
  await writeFile(output.replace(/\.mp4$/, '.vtt'), captions);
  console.log(`Saved ${output}`);
} finally {
  if (encoder && encoder.exitCode === null) encoder.kill('SIGTERM');
  await browser.close();
}
