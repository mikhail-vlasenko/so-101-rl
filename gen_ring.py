"""Generate XML for a hexagonal wall around the place target.

Defines the hexagon by its 6 vertices, then creates a box for each
side connecting adjacent vertices. Each wall is extended slightly past
the vertices so adjacent walls overlap at corners (no gaps).
"""

import math

# Geometry parameters
R = 0.05         # circumradius: center to vertex
t = 0.005        # wall thickness
h_half = 0.015   # wall half-height

# 6 vertices (pointy-top: first vertex at 90°)
vertices = []
for i in range(6):
    angle = math.radians(90 + i * 60)
    vertices.append((R * math.cos(angle), R * math.sin(angle)))

# Extension past each vertex to seal corners
extend = t

print(f"# Circumradius: {R}")
print(f"# Wall thickness: {t}")
print(f"# Corner extension: {extend:.5f}")
print()

for i in range(6):
    v0 = vertices[i]
    v1 = vertices[(i + 1) % 6]

    cx = (v0[0] + v1[0]) / 2
    cy = (v0[1] + v1[1]) / 2

    dx = v1[0] - v0[0]
    dy = v1[1] - v0[1]
    side_len = math.sqrt(dx**2 + dy**2)
    side_angle = math.atan2(dy, dx)

    # Outward normal = side_angle - π/2 (CCW winding)
    # Note: so101.xml sets <compiler angle="radian"/>
    normal_angle_rad = side_angle - math.pi / 2
    half_len = side_len / 2 + extend

    print(f'      <geom type="box" size="{t/2:.5f} {half_len:.5f} {h_half}"'
          f' pos="{cx:.5f} {cy:.5f} {h_half}"'
          f' euler="0 0 {normal_angle_rad:.5f}"'
          f' rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>')
