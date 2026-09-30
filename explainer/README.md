# Perception explainer

Code-native React + Three.js dark-studio film, with an orbitable keyframe gallery
and earlier visual studies. No image-model assets. The default page opens the
captioned film paused; narration audio is not attached yet.

## Run and review

```bash
cd explainer
npm ci
npm run dev
```

Open the URL printed by Vite. `?film` opens the film, `?keyframe=views` opens the
gallery, and `?frame` hides the preview shell. Drag to pause and orbit; playback
returns to the authored camera. Reset view resets the film; Export PNG saves
the current composition. The gallery's surface frame also has Replay orbit.

The film's “Narration script” panel is generated from its captions. To print
the same script with its word count and current duration:

```bash
npm run script
```

## Where to edit

- `src/filmTimeline.js`: the only source of narration, cue boundaries, chapters,
  animation timing and duration limit. Each caption has a stable ID. Its end is
  derived from the next cue's start. Related events reference cues or preceding
  events: cloud building follows the cloud cue, movement follows the movement
  cue, and shutter rays follow the sponge fade. Keep those links when retiming.
- `storyboard.md`: audience, visual intent and explanation boundaries, not a
  second copy of the script or a timestamp-by-timestamp shot list.
- `src/sceneConfig.js` and `src/PerceptionScene.jsx`: shared scale, rig geometry,
  materials, lighting and floor. `src/storyFrames.js` owns gallery compositions
  and reusable camera-image/ray primitives.
- `src/filmShots.js`, `src/surfaceReconstruction.js` and `src/imageCenterCue.js`:
  authored film geometry. `src/FilmScene.jsx` applies timeline timing to it.
  `src/captionLayer.js` caches native-resolution glyphs and animates caption quads.
- `src/main.jsx` and `src/style.css`: preview controls, transcript and technical
  notes. Styles and alternative sets live in `sceneStyles.js` and `sceneEnvironments.js`.

Routine script or timing changes should need only `filmTimeline.js`. Tests check
cue structure, causal ordering, geometry and smooth motion, not a copied script
or a fixed word count. Review/export sample times are derived from the timeline
in `scripts/film-review.mjs`.

## Verify

```bash
npm run test:film
npm run test:motion
npm run build
npm run preview -- --port 4173 --strictPort
```

With the production preview running, `node scripts/check-film.mjs` checks
deterministic seeking, caption continuity and caching, native text resolution,
playback controls, the persistent subtitle backdrop, and the final scene hold.
It saves review PNGs in `keyframes/`. `node scripts/check-preview.mjs` checks the
earlier motion draft.

## Gallery and visual studies

With the development server running, `npm run render:story` renders gallery
thumbnails. Pass frame names to render a subset, such as
`npm run render:story -- views surface refresh`. The gallery does not require
thumbnails to render its live scenes.

`?draft` opens the earlier motion study. Static style links are
`?style=studio&still`, `?style=porcelain`, and `?style=blueprint`.
These studies are separate from the film's editorial timeline.

## Export and version control

Do not export a video for routine iterations; render an MP4 only when requested.
With Chromium at `/usr/bin/chromium`, FFmpeg on PATH and the production preview
running, use `npm run render:film -- public/renders/<new-name>.mp4`. The exporter
uses timeline duration, frame rate and captions, writing an H.264 MP4 and WebVTT
companion. It refuses to overwrite an existing video. `npm run render:motion`
exports the earlier study.

The download links say “Last exported” because those artifacts may lag the live
film. Their paths live in `filmTimeline.js`; use a new path for each export.

Track authored source, scripts, docs, package manifest and lockfile. Dependencies,
build output, review PNGs, gallery thumbnails and video exports are ignored.
Render/check scripts create output directories. A fresh checkout needs no
exported assets to play the live film.
