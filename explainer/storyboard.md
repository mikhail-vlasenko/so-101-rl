# How we perceive a target object

Storyboard and narration, draft 13. Video target: 55 seconds, hard maximum
60 seconds including the ending. Landscape 16:9. Audience: curious people
with no robotics or machine-learning background. The quoted passages are the
complete spoken script. Timings are editorial targets, not system latencies.

The opening asks how we perceive an object we want to interact with; the sponge
is an example, not a restriction on the perception pipeline. Image centers
answer where the object is; matched surface details add shape, size, and
orientation. Its central moment is lateral sponge movement:
the position dot follows, the point cloud stays put, and a staggered
shutter sequence explains why.

## Visual language

Use the authored dark-studio 3D scene, with rendered camera images and animated
geometry growing out of them. Keep a fixed tabletop reference and consistent
camera arrangement. The sponge is gold and its surface points are green;
distinguish the cloud labeled “Surface snapshot” from the dot labeled
“Position estimate” with shapes and labels. In scene 5, show both in the same
spatial view so their separation is immediately apparent.
Keep the physical cameras fixed throughout; move only the viewer. Give camera A
the same cyan accent and camera B the same violet accent in every view, even
when their left-to-right screen order reverses with the viewing angle.

Keep on-screen text short, with space below the action for captions. Arm
tracking, controller inputs, distance encoding, and measurement ages are outside
the story. Technical explanations can expand in webpage notes.

The current full film is a procedural illustration, documented in webpage
notes, with uniform cloud samples on only the three observed faces. If recorded
data is introduced, preserve its noise and cloud gaps. Keep sponge speed
constant through the shutter demonstration; no extra timing badge or
per-camera subtitle. Explain the expanded illustrative shutter timing in the
webpage notes. No invented numerical latency or shutter-offset claims.

## Storyboard at a glance

| Time | Scene | Main visual | On-screen text |
| --- | --- | --- | --- |
| 0:00–0:04 | Two cameras | Orbit the fixed scene, then enter camera A's view | How do we perceive an object we want to interact with? |
| 0:04–0:11 | Find the target object | Highlight its pixels; gather boundary markers toward their center | Find the target object's pixels |
| 0:11–0:19 | Estimate position | Pull out as both rays appear; hold on the position dot | Where it is—not its shape |
| 0:19–0:29 | Measure surface | Matched image details become a cloud | Shape, size, and how it's turned |
| 0:29–0:49 | Movement and shutter timing | Sponge and position dot move sideways; cloud stays; staggered exposures explain the mismatch | Different moments |
| 0:49–0:55 | Surface refresh | A new cloud replaces the snapshot; caption fades, scene holds, then fades | So we refresh the cloud when the sponge settles. |

## 1. Two cameras — 0:00–0:04

**Narration**

> How do we perceive an object we want to interact with?

**Visual:** Start with a short orbit around the tabletop scene, showing the
stationary sponge and fixed camera pair. Fly into camera A's optical viewpoint
and settle on the sponge, using its authored aim and image field of view.
Continue in the same 3D scene into the next two chapters. Keep the opening
question visible until 0:04.3, just after arriving at camera A's view.

## 2. Find the target object — 0:04–0:11

**Narration**

> First, a vision model finds the target object’s pixels…
> …and tracks them in both camera views.

**Visual:** Place the viewer just in front of camera A's lens, not near the
sponge. Use wider framing so this reads as the full scene from that camera
position rather than a close-up. Show the prompt “sponge” and fill its
visible pixels with a green mask. At 0:07.2–0:08.7, gather twelve small cyan
markers inward from the projected silhouette boundary, with a faint contracting
outline. Reveal the center dot as they arrive, then fade the cue by 0:09.2.
Keep the mask and physical object unchanged: this is a reduction of image
pixels to their center, not a cloud of 3D surface measurements. Explain that
the model tracks pixels in both cameras without introducing floating images.
Keep the same framing and save object movement for scene 5.

**Webpage note:** Segmentation identifies object pixels. SAM 3 initializes
the masks from a text prompt; SAM 2 tracks them. An evaluation tag visible
on the sponge is not used to locate the object on this path.

## 3. Estimate position — 0:11–0:19

**Narration**

> The two image centers give us an approximate 3D position.
> That tells us where it is—not its shape.

**Visual:** Pull the viewer continuously out of camera A and up to one side of
the fixed camera pair. Fade the green mask off the opaque sponge as both
viewing rays extend toward the approximate position. The baseline and triangle
become clear in this oblique top view. From 0:15.3, dim the still-opaque sponge
reference and the rays, leaving the bright position dot and its label dominant.
Hold this position-only measurement beat before the cloud builds at 0:19.
Do not add dimension brackets or orientation axes. Keep the tabletop grid as a reference;
do not force the rays to intersect perfectly. There is no cut or image-panel
layout during these first three chapters.

**Webpage note:** Mask-centroid triangulation estimates position. Each view’s
center can correspond to a different physical surface point. Camera calibration
provides the shared geometry; its procedure stays outside the video.

## 4. Measure surface — 0:19–0:29

**Narration**

> Matching details across the images builds a surface point cloud.
> This shows its shape, size, and how it’s turned.

**Visual:**

- 0:19–0:23: Magnify two fixed-view image crops and give the sponge recognizable
  sparse texture with a few recognizable patches, not dense pores. Draw a yellow connection between corresponding texture details
  in the two images. Then grow two green rays from those image points toward
  their shared surface location. Only after the rays arrive, reveal its green
  cloud point. Take a little longer on the first match, then make the second
  faster and the third faster still: only three explained triangulations.
  Fill in the remaining equal-sized
  green points without more correspondence animations.
- 0:23–0:27: With the cloud complete, begin at the front oblique view and
  sweep the viewer left, continuing past the side toward the back. Show the
  empty rear faces so the points read as a measured surface rather than a
  filled object. Use this orbit for the shape/size/orientation line; let the
  visible empty back carry the unmeasured-surface limitation without another
  spoken caption. Fade the floating image crops and correspondence lines early
  in the sweep to keep the cloud unobstructed. Keep both capture cameras fixed.
- 0:27–0:29: Settle into a fixed tabletop view containing the cloud, position
  dot, and faint sponge reference. Label both observations. This is the exact
  composition that starts scene 5.

**Webpage note:** Dense stereo aligns images, matches detail, and filters
unreliable geometry. This pipeline uses StereoSGBM. The cloud describes visible
surface; it does not fill in hidden faces.

## 5. Movement and shutter timing — 0:29–0:49

**Narration**

> When the sponge starts moving, the cameras can capture at different moments.
> And that motion can be mistaken for depth.
> That makes detailed stereo depth unreliable during movement.

**Visual:**

- 0:29–0:34: Move the sponge laterally across the tabletop at twice the previous
  cruising speed, easing gently out of rest, clearly sideways
  on screen. The position dot follows; the cloud remains at its original
  measured location, fading gently to 50% opacity over the first three seconds.
  Keep the camera view fixed and all three visible. Introduce movement before
  explaining why the surface snapshot must be held. Keep
  this same scene and viewer pose through both exposures.
- 0:34–0:37: Keep the same sponge speed. Flash the left
  camera shutter while the sponge continues moving. Leave a translucent sponge
  silhouette at that captured position and reveal a small saved camera image
  above the scene, marking its visible top-right corner. Do not stop the live
  sponge at the exposure.
- 0:35–0:40: Let the sponge shift a small amount to the right, then flash the
  right camera shutter while the sponge continues moving. Leave its second
  silhouette alongside the first and highlight the same physical corner in
  the second image. Keep the spatial
  shift consistent with the following depth diagram; do not magnify it again.
  Continue the live sponge past both captured silhouettes after the second
  shutter. Halve the interval between captures to preserve the same spatial
  displacement. Fade the live sponge, its shadow and position marker out
  starting 500 ms after the second shutter, over 0.8 seconds; leave the captured
  silhouettes visible. Keep the live sponge moving during this brief hold.
  Keep cruising speed constant through both shutters. One caption
  covers both captures: “…the cameras can capture at different moments.” Keep this
  caption brief, then show the motion/depth explanation from 0:37. Start the
  independent-shutter conclusion at 0:43.3 and keep it visible until 0:49.
- 0:40–0:46: Only after both silhouettes and camera images are visible, draw
  one viewing ray from each camera through the two corner observations. Reveal
  their incorrect depth point, labeled “Movement mistaken for depth.” The
  original held cloud stays fixed and visible at 50% opacity. Raise the viewer and
  look down at the frozen corner evidence to reveal the camera-ray plane,
  fading the small images
  rather than cutting to another diagram. The live sponge is now hidden,
  leaving only the two saved corners and silhouettes as evidence.
  Leave timing caveats in the webpage notes, not an extra film badge.
- 0:46–0:49: Fade away the orange false-depth example, rays, and exposure
  silhouettes, then pull back to the lateral-motion view. The green held cloud
  stays exactly where it was. At 0:47, fade the live sponge and position dot back
  in while they are still moving right at the faster speed. Finish the fade
  just before the refresh cue, easing into the stop at 0:49, with no separate
  shutter-shot transitions. Compress only the invisible travel; do not jump the
  sponge's position.

**Webpage note:** Rectification aligns image geometry, not exposure times.
Independent unsynchronized shutters can cause wrong depths or rejected matches
during motion. This is why our pipeline gates surface refreshes on stillness.
The simpler position estimate is approximate and can also have timing errors;
it is not immune to the mismatch.

**Production notes:** Choose a lateral direction that makes the displacement
clear in the shared view and camera images. Use the same small movement between
the two shutters and in the depth calculation. Receipt timestamps are not proof
of exposure times. The erroneous depth point is a separate
explanatory inset, not a new cloud published by the running pipeline.

## 6. Surface refresh — 0:49–0:55

**Narration**

> So we refresh the cloud when the sponge settles.

**Visual:**

- 0:49–0:53: The sponge stops at its new location. After a visible settling
  beat, draw a successful fresh point cloud directly on this same sponge,
  fading the old cloud without moving it. Keep the exact scene and viewer pose
  from the end of step 5; no cut, scene dissolve, or camera reset. Keep the
  sponge opaque so the tabletop grid cannot show through it.
- 0:52.5–0:53: Fade the last caption gently while keeping the scene visible.
- 0:53–0:54: Hold the finished sponge and fresh cloud with no subtitle text.
- 0:54–0:55: Fade the whole scene to dark. No summary caption or additional outro.

**Production note:** Use a successful recorded refresh or label the sequence
as illustration. Stillness and visibility permit a refresh; they do not
guarantee a valid cloud. Preserve the real settling/processing interval in
replay or make any time compression explicit. The position dot represents
location only. Never translate or rotate the held cloud to follow it.

## Timing and shared webpage treatment

The current 55-second webpage has animated geometry and captions,
with a downloadable WebVTT companion; it is silent pending voiceover recording.
The exact caption and animation timings live in `src/filmTimeline.js`. The
webpage is the current visual iteration; export a new video only on request.
The position moves from its original location before camera A captures; it moves
one further centimeter before B captures the same corner, then continues beyond
both saved positions before settling. These editorial
times and distances illustrate the issue, not a measured camera delay.

Aim for a conversational read, leaving room for the visual reveals. Record a
scratch voiceover against these timings before animation polish. If it runs
long, shorten wording or simplify transitions; keep the export under one minute
rather than speeding up the explanation. The final caption fade, one-second
scene hold, and one-second scene fade are included in the 55-second timeline.

The webpage plays the same complete short story. Pausing, scrubbing, expandable
technical notes, and an optional orbitable cloud provide additional depth
without lengthening the video. Resuming restores the authored camera view.
Captions and a transcript support muted viewing. Sound is never the sole cue
for a measurement refresh.

## Asset specification

| Asset | Required content |
| --- | --- |
| Camera frame pairs | One still–lateral movement–still sponge sequence, visible in both views |
| Perception exports | Matching masks, live positions, accepted clouds, camera geometry, and timestamps for replay alignment |
| Camera diagram | Tabletop, sponge, and two cameras consistent with the selected recording |
| Timing illustration | Left then right shutter, two silhouettes of the same moving corner, and erroneous depth point; explicitly illustrative |
| Voiceover and captions | Script timed to the 55-second composition, including the final hold and fade |

Use one sequence as the backbone. Existing dataset frames and clouds can supply
material, but verify available timing before presenting it as live behavior.
No external pickup footage or arm action is required.

## Source anchors for production review

Module documentation remains the source of technical contracts.

- [Text prompting and mask tracking](../real/tracking/sam_seg.py)
- [Object tracking and surface refresh](../real/rollout/object_obs.py)
- [Stereo geometry and filtering](../real/tracking/dense_stereo.py)
- [Position tracking and static gate](../src/shape_obs.py)

Production follow-ups live in [TODO.md](../TODO.md).
