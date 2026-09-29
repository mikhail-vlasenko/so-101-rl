# Perception explainer visuals

React + Three.js dark-studio film and storyboard. The default page plays the
complete 55-second captioned film, initially paused; `?film` opens it directly.
Nine
keyframes cover the two camera views, object masks, position rays, visible
surface reconstruction, two successive shutters, a false depth example, the
held cloud after that example disappears, and a fresh surface measurement after
the sponge settles. The gallery opens at
`?keyframe=views`; each frame can be orbited and exported as a PNG.

The scene uses a perspective camera, lit beveled object, shadow-receiving
floor with an anti-aliased grid drawn directly into its material, and uniform
green point markers. The grid stays on the floor, without coplanar depth
fighting. Half-float film render targets and output dithering preserve smooth
dark gradients instead of introducing intermediate color bands.
Camera positions use the saved mounts in `so101/so101.xml`, converted from
meters into centimeters around base position (0.20, 0, 0), then shifted together
16 cm along the sponge's movement direction. This authored offset keeps both
cameras on the visible side of the snapshot's right face while preserving
their measured separation and height.
The sponge is 6 × 4 × 2.5 cm; the saved rig has roughly 11 cm camera separation
and 40 cm horizontal distance to the workspace. Camera housings and stands are
stylized. Both rendered views and the visible camera models use one fixed,
authored aim and field of view; the cameras never follow the sponge. This is
not a reproduction of the calibrated camera rotations or fields of view.
Camera A stays cyan and camera B violet even when the viewing angle reverses
their left-to-right order on screen.
The surface points and the two camera images are deterministic illustrations,
not captured data. The shutter interval is expanded for explanation; the two
capture positions differ by 1 cm in the authored geometry. All geometry,
renders, and labels are built from code; image-model assets are not used.

```bash
cd explainer
npm install
npm run dev
```

## Version control

Track the authored source, timeline, scripts, documentation, package manifest,
and dependency lockfile. Generated review images (`keyframes/`), gallery
thumbnails (`public/keyframes/`), exports (`public/renders/`), dependencies,
and the built site are ignored; existing local artifacts are not deleted.
Render/check scripts create their output directories when needed.

On a fresh checkout, run `npm ci` and `npm run dev`. The film itself needs no
exported assets. To populate gallery thumbnails, run `npm run render:story`
with the development server running. Previously exported MP4/VTT downloads
are local artifacts, not included in Git; export a new video only on request.

## Full film

The opening frames the sponge as an example target, asking how we perceive an
object we want to interact with. Image centers answer where it is; the surface
cloud adds information about shape, size, and orientation, rather than
introducing depth as though the position estimate had none.
The full film follows the six chapters in `storyboard.md`: scene orbit, masks,
approximate position, a surface-only cloud with the approved leftward viewer
orbit, lateral movement and two staggered shutters, then a fresh cloud after
the sponge settles. The opening orbits the fixed rig, places the viewer 0.6 cm
in front of camera A's lens with a wider 30° illustrative field of view,
highlights the sponge pixels there, then continuously
pulls out into the two-ray diagram. From 7.2–8.7 seconds, twelve small cyan
markers and a faint outline converge from the projected mask boundary onto
its image center. The marker origins and centroid come from the projected
silhouette of the authored convex sponge; the physical object and mask do not
shrink. The transient cue disappears by 9.2 seconds. During pullback the image
center marker blends into the illustrative triangulated position. From
15.3–16.5 seconds the opaque sponge reference and viewing rays dim, leaving
the position dot emphasized until surface reconstruction starts at 19 seconds.
No measurement brackets or orientation axes are added.
There are no floating image panels in the
opening. Step 5 is also one continuous scene: the sponge moves through both
shutters and past both saved silhouettes at a cruising speed of 1 cm/s,
with a one-second smooth velocity ramp from rest and another into the final
stop. The hidden speed changes also use smooth ramps, so there are no velocity
jumps. The capture gap is proportionally reduced to 1 second, preserving the
same 1 cm displacement. There are no capture-time pauses or cuts. One caption covers both exposures:
“…the cameras can capture at different moments.” It holds for 2.7 seconds,
following “When the sponge starts moving…” and leading into “…and that motion
can be mistaken for depth.” The stereo-depth reliability conclusion starts
at 43.3 seconds and holds for 5.7 seconds. The transition summary and final
“Surface detail when still / Position while moving” recap are omitted; the
film finishes with the refresh. Its final caption fades at 52.5–53 seconds;
the finished sponge and cloud hold without subtitle text at 53–54, followed
by a one-second scene fade at 54–55 seconds.
After the second capture, the live sponge and its position marker stay fully
visible for 500 ms, then fade out over 0.8 seconds, leaving the frozen exposure
evidence unobstructed. At 47
seconds they fade back in while moving right, easing to a stop during the last
second. The sponge is fully visible by 48.8 seconds, stops at 49, and precedes the
“So we refresh the cloud when the sponge settles” cue at 49.2. Unseen travel is compressed during the depth
explanation without jumping the world position. The shadow fades with the
sponge. Only the two small captured images and silhouettes freeze. The viewer moves
up to a downward-looking view of the corner evidence and ray plane for the
depth explanation, then pulls back smoothly.
The same top-right corner is captured twice before the
orange false-depth example is drawn. The green held cloud stays at its original
location throughout motion, easing to 50% opacity during the first three seconds
and remaining half-visible through the explanation. The ending remains in that same scene and viewer
pose: new points grow on the settled sponge while the old cloud fades away.
It does not transport the held one. Outside the explicit fades, the gold
sponge is opaque, so the table grid cannot show through it. The physical
camera rig never moves.

The surface reconstruction first magnifies two crops of the fixed camera
views, with a sparse texture consisting of three recognizable patches instead
of dense pores. One yellow correspondence grows
between the same textured feature in both images; only afterward do two green
rays grow toward its 3D point, and only when they arrive is that cloud sample
revealed. The next two matches repeat progressively faster, then the rest
of the 510 equal-sized points fill in without extra correspondence animations.
The three demonstrated points are reordered samples of the same cloud, not
larger or separately colored additions. The approved leftward orbit follows.

`src/filmTimeline.js` owns the editorial timing and caption cues;
`src/FilmScene.jsx` renders the approved geometry from `src/storyFrames.js`;
`src/filmShots.js` assembles the continuous opening and exposure scene from
those shared primitives.
`src/surfaceReconstruction.js` builds the textured crop and matching geometry;
its staged timing is single-sourced in `src/filmTimeline.js`.
`src/imageCenterCue.js` builds the silhouette-to-center illustration; its cue
and position emphasis are sampled from that same timeline.
Browser playback, seeking, and offline export sample the same timeline. Captions
are drawn at the actual drawing-buffer resolution, including device pixel
ratio, and resized with the viewport rather than stretching a fixed 720p text
texture. Offline export includes the same text, with a WebVTT companion.
The subtitle gradient remains visible through caption gaps and scene changes;
only the text changes, avoiding flashes in the lower part of the frame.
The complete cut is silent and uses procedural illustrations. The
approved spoken script remains in `storyboard.md` for a subsequent voiceover.
Expandable notes on the webpage explain the actual segmentation, stereo, and
held-surface behavior and distinguish illustrative shutter timing from measured
camera delay.

With the production preview running on port 4173:

```bash
npm run test:film
node scripts/check-film.mjs
```

Do not export video for routine visual iterations; render an MP4 only when
requested. The current webpage is newer than the last exported film, so the
download link is explicitly labeled “Last exported MP4.”
The last export is `public/renders/perception-explainer-v4.mp4`, H.264, 1280 × 720,
30 fps, exactly 58 seconds; the caption companion is
`public/renders/perception-explainer-v4.vtt`. The Full video page links both.
The previous v1, v2 and v3 MP4s are retained for comparison.
Export preserves existing videos; pass a new MP4 path for a subsequent version.
The browser check verifies repeated frames after backward seeks, native text
resolution after viewport resizing and at 2× device pixel ratio, playback,
pause, scrubbing, reset, and switching between the film and gallery. Review
stills are saved as `keyframes/film-*.png`.

Open the local URL printed by Vite. Select Storyboard frames to compare the nine
shots, or open a frame directly with `?keyframe=views` (replace `views` with
`masks`, `position`, `surface`, `timing`, `second`, `depth`, `reject`, or `refresh`).
The surface frame sweeps left from a front oblique view toward the empty back
of the measured cloud. The image panels fade during the turn to keep that
surface unobstructed; Replay orbit starts it again. The physical cameras stay fixed.
The Motion draft button (or `?draft`) opens the earlier 14-second dark studio
motion draft, initially paused. Play, pause,
replay, or scrub the timeline. Dragging pauses playback for free orbit; Play
returns to the authored camera. The Scene styles button opens static alternatives.
Add `?frame` for the composition alone,
without the preview toolbar. Drag to orbit, scroll to zoom, and right-drag to
pan. Reset view restores the authored keyframe. Export PNG saves the current
camera view at the canvas resolution.

The style selector opens each treatment at its authored viewing angle.
Orbit and zoom still work in every scene. Static direct links use `?style=studio&still`, `?style=porcelain`, or
`?style=blueprint`; add `&frame` for clean capture. Dark studio is the original
lit treatment. Tabletop exhibit uses a layered display plinth, a marked workspace,
a mounting rail, ceramic hardware, and routed cables. Optical space adds two
floating image planes, view volumes, and masks projected from the actual box
geometry. These are explanatory virtual projections with an authored field of
view, not recorded camera images or calibrated frusta. All share the same point
samples, object scale, and rig positions. Style definitions and material treatment
live in `src/sceneStyles.js`; set geometry lives in `src/sceneEnvironments.js`.
`npm run build` produces the static site in `dist/`.

With the development server running on port 5173, `npm run render:story` renders
all nine 1280×720 PNGs into `public/keyframes/` and checks the gallery. The
gallery uses these PNGs as its thumbnails. To render only a selection, pass
frame names, e.g. `npm run render:story -- views surface refresh`.

## Motion draft and export

The shared `src/motionDraft.js` sampler defines a wide hold (0–1.5 s), camera
move (1.5–6.5 s), settling beat (6.5–7 s), sideways sponge motion (7–11 s),
and final hold (11–14 s). The cloud never moves or refreshes in this draft.
The position marker moves with the sponge. Narration and shutter timing are
not included yet.

With Vite running on port 5173, Chromium at `/usr/bin/chromium`, and FFmpeg on PATH:

```bash
npm run test:motion
node scripts/check-preview.mjs
npm run build
npm run preview -- --port 4173 --strictPort
```

In another terminal, run `npm run render:motion`. Export uses the production
preview on port 4173 so development hot reload cannot interrupt a render.

The exporter samples exactly 420 frames at 1280×720, 30 fps and writes an H.264
MP4 to `public/renders/perception-motion-v2.mp4`. The page's Download MP4 link
serves that artifact. Browser render time does not affect video timing. Existing
videos are not overwritten; pass a new path for another version:

```bash
npm run render:motion -- public/renders/perception-motion-v3.mp4
```

`?frame&export&style=studio` exposes a render-only browser API used by the export
and preview-check scripts. The preview check exercises playback, pause, seek,
reset, and identical repeated frame renders, then saves review stills.

Edit `src/PerceptionScene.jsx` to adjust geometry, lighting, camera, and labels;
edit `src/style.css` for the preview shell. The composition is framed at 16:9.
Motion is limited to the selected dark studio treatment. Earlier style frames in `keyframes/` are retained
as iterations and are not consumed by the scene.
