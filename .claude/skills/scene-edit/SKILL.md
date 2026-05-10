---
name: scene-edit
description: Edit MuJoCo scene XML files with visual verification
---

When making changes to MuJoCo scene XML files (`so101/*.xml`):

1. **Make the scene changes** in the XML file(s).

2. **Write a screenshot script** to `scripts/screenshot_scene.py` that:
   - Loads the modified scene and calls `mj_forward`
   - Uses `mujoco.Renderer` + `PIL.Image` to save screenshots to `screenshots/`
   - Configures `MjvCamera` (lookat, distance, azimuth, elevation) so the region of interest is clearly visible
   - Produces at most 2 screenshots (e.g. a close-up and an angled view)
   - See `screenshot_ring.py` in the project root for reference

3. **Run the script** with `python scripts/screenshot_scene.py` and **read the output images** to verify the result.

4. **Iterate** — if the scene doesn't look right, adjust the XML and re-run until the desired outcome is reached.
