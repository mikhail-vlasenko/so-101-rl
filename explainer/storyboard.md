# Perceiving a target object — visual intent

Audience: curious people with no robotics or machine-learning background.
The sponge is an example target, not a restriction on the pipeline.

Narration, exact timing and the duration limit live only in
[`src/filmTimeline.js`](src/filmTimeline.js). Read the generated “Narration script”
panel in the preview, or run `npm run script`. Do not maintain a second script,
word count or timed shot list here. This document explains visual decisions
that are not obvious from the code.

## Scene and scope

Use the dark-studio 3D scene and a fixed tabletop grid. The viewer can move;
the physical cameras cannot. Preserve camera A's cyan and camera B's violet
identity even when the viewing angle reverses their screen order.
Uniform green surface points and a cyan position dot distinguish shape from
location. Keep the subtitle backdrop continuous between cues.

Arm tracking, controller inputs, distance encoding and observation ages are
outside this story. Technical implementation details belong in the preview's
expandable notes and the real-system modules linked below.

## Story beats

1. **Enter the camera view.** Orbit the scene, then fly to just in front of
   camera A's lens. Show the full target, not a close-up from near the sponge.
   This establishes that the following mask belongs to a camera image.
2. **Find image pixels.** Highlight the target's silhouette and gather boundary
   markers toward its center. The mask and physical object do not shrink;
   this is a reduction to an image center, not surface reconstruction.
3. **Locate the object.** Pull out into the two-ray triangle, viewed obliquely
   from above and to one side. Briefly dim the reference object and rays to
   emphasize the position-only estimate before introducing shape.
4. **Measure visible surface.** Match a recognizable texture feature between
   magnified crops, connect it in yellow, then converge green rays onto a point.
   Hold that first point for its explanation. Build the next two matches faster,
   then fill the rest with the cloud cue. Sweep left toward the empty back
   while explaining the surface snapshot. Return to the sponge view only as
   the movement explanation begins. Do not add dimensions or orientation axes.
5. **Explain independent shutters.** The sponge and position dot move smoothly;
   the old cloud stays fixed and half-visible. Capture the same physical corner
   twice without stopping the sponge or exaggerating its displacement again.
   After the post-capture hold, fade the live object, then immediately draw the
   two rays. Raise the viewer to reveal their plane and incorrect depth point.
   The saved images and silhouettes are evidence, not a new accepted cloud.
6. **Refresh after settling.** Fade the sponge back in while it is still moving,
   then ease into rest. Grow a fresh cloud in place and fade the old measurement
   without moving it. Stay in the same scene. Fade the final caption, hold the
   finished scene without text, then fade to dark; no extra recap.

## Illustration boundaries

All measurements and camera images in the film are procedural illustrations.
Only observed faces get points; hidden faces are not filled. Rig mounts use the
saved scene positions with an authored lateral shift and aim, not calibrated
image views. Shutter spacing and hidden travel are expanded/compressed for
explanation, not measured hardware timing. Receipt times are not exposure times.

The approximate position can also have timing errors. Stillness permits a
surface refresh but does not guarantee a valid measurement. If recorded data
is introduced, preserve noise, missing points and real processing delays, or
make editorial time compression explicit.

## Technical source anchors

- [Prompting and mask tracking](../real/tracking/sam_seg.py)
- [Object tracking and surface refresh](../real/rollout/object_obs.py)
- [Stereo geometry and filtering](../real/tracking/dense_stereo.py)
- [Position tracking and static gate](../src/shape_obs.py)

Production follow-ups live in [TODO.md](../TODO.md).
