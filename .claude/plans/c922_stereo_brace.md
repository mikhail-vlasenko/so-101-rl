# C922 stereo-camera brace

## Goal

Design a rigid horizontal brace that ties the two existing tripod-mounted Logitech
C922 cameras together, preserving their current approximately 109 mm optical-center
baseline and relative orientation. The tripods continue to carry the cameras; the
brace only prevents relative drift.

## Plan

1. Install the `earthtojake/text-to-cad` CAD skill and use build123d as the
   authoritative parametric model. Keep generated STEP and 3MF/STL artifacts
   reproducible from the Python source.
2. Download the free C922 reference STL from MakerWorld, record its source/license,
   and import it as reference geometry only. Its non-manifold edges are acceptable
   because it will not participate in booleans or become printable geometry. Check
   its scale against Logitech's 95 x 24 x 29 mm envelope and a few physical caliper
   measurements at the intended contact surfaces.
3. Model two repeatable, removable camera saddles/clamps joined by a stiff ribbed
   beam. Parameterize the optical-center spacing, camera clearance, wall thickness,
   fasteners, and print tolerance. Keep the lenses, microphones, hinges, tripod
   mounts, and USB cable exits unobstructed, and avoid forcing misaligned tripods
   together during installation.
4. Render the brace with two instances of the reference camera and validate that the
   generated brace is one or more intentional watertight solids, has the requested
   baseline, and meets minimum wall/clearance dimensions.
5. Print one saddle as a small fit test before printing the full brace. Update only
   measured fit parameters, then print and install the complete part without moving
   the cameras from their accepted view geometry.
6. After installation, re-run stereo alignment, checkerboard calibration, the
   table-anchor reference capture, and `real.diagnostics.snapshot_cam_mount` for
   both `main` and `aux` cameras as required by `TODO.md`.

## Deliverables

- Parametric build123d source and documented dimensions
- C922 reference mesh kept separate from manufactured geometry
- STEP assembly/reference output and printable 3MF/STL files
- Fit-test saddle and final brace
