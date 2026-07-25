"""Generate one print-ready PDF of fiducial markers at exact physical sizes.

Lays out, on a single A4 sheet, every id in `marker_spec.TAG_SIZE_MM` in both
families: the 4 cm table tags (10-12) and the 2 cm arm/sponge tags (0-7, the
latter including the eval-only sponge set 3-7).

Each marker is drawn at its true millimetre size, so printed at 100%
(no fit-to-page) the outer black square matches the caption — measure an edge
with calipers to confirm the printer did not rescale. IDs are unique within
each family (ArUco id N and AprilTag id N are different markers in different
dictionaries).

Run:
    conda run -n mujoco_env python -m scripts.make_markers
"""
import os

import cv2.aruco as aruco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from real.marker_spec import FAMILIES, TAG_SIZE_MM

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0   # A4
MARGIN_MM = 12.0
GAP_MM = 8.0                          # spacing between printed cells
QUIET_MODULES = 1.0                   # white border baked around each tag, in marker modules
MARKER_PX = 400
MM_PER_IN = 25.4

OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "markers", "markers.pdf"))


def build_tags():
    """Flat list of (family, dict_id, tag_id, edge_mm), grouped 4 cm then 2 cm."""
    tags = []
    for size_mm in (40.0, 20.0):
        ids = sorted(tid for tid, s in TAG_SIZE_MM.items() if s == size_mm)
        for fam, dict_id in FAMILIES.items():
            for tid in ids:
                tags.append((fam, dict_id, tid, size_mm))
    return tags


def mm_to_fig(x_mm, y_mm):
    return x_mm / PAGE_W_MM, y_mm / PAGE_H_MM


def marker_cell(dict_id, tag_id):
    """Marker padded with a QUIET_MODULES-wide white border.

    Returns (image, cell_factor) where cell_factor is the padded edge divided by
    the black-square edge — so a tag whose black square is E mm prints in a cell
    of E * cell_factor mm, with the extra being white quiet zone.
    """
    d = aruco.getPredefinedDictionary(dict_id)
    total_modules = d.markerSize + 2          # data grid + 1 black border each side
    pad = int(round(QUIET_MODULES * MARKER_PX / total_modules))
    img = aruco.generateImageMarker(d, tag_id, MARKER_PX)
    img = np.pad(img, pad, mode="constant", constant_values=255)
    return img, (MARKER_PX + 2 * pad) / MARKER_PX


def render(tags, out_path):
    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(PAGE_W_MM / MM_PER_IN, PAGE_H_MM / MM_PER_IN))

        fig.text(*mm_to_fig(MARGIN_MM, PAGE_H_MM - MARGIN_MM),
                 "fiducial markers  —  print at 100% (no fit-to-page); verify an edge with calipers",
                 ha="left", va="top", fontsize=11, weight="bold")

        # 50 mm reference bar to catch printer rescaling.
        bar_y = PAGE_H_MM - MARGIN_MM - 8
        x0, _ = mm_to_fig(MARGIN_MM, bar_y)
        x1, yb = mm_to_fig(MARGIN_MM + 50.0, bar_y)
        fig.add_artist(Line2D([x0, x1], [yb, yb], transform=fig.transFigure, color="black", lw=1.2))
        for xt in (MARGIN_MM, MARGIN_MM + 50.0):
            tx, _ = mm_to_fig(xt, bar_y)
            _, ty0 = mm_to_fig(0, bar_y - 1.5)
            _, ty1 = mm_to_fig(0, bar_y + 1.5)
            fig.add_artist(Line2D([tx, tx], [ty0, ty1], transform=fig.transFigure, color="black", lw=1.2))
        fig.text(*mm_to_fig(MARGIN_MM + 53, bar_y), "= 50 mm", ha="left", va="center", fontsize=8)

        # Flow layout: new row whenever the (size, family) group changes.
        cursor_x = MARGIN_MM
        row_top = PAGE_H_MM - MARGIN_MM - 22
        row_h = 0.0
        prev_key = None
        for fam, dict_id, tag_id, edge_mm in tags:
            img, cell_factor = marker_cell(dict_id, tag_id)
            cell_mm = edge_mm * cell_factor    # printed cell = black square + white quiet zone
            key = (edge_mm, fam)
            wrap = key != prev_key or cursor_x + cell_mm > PAGE_W_MM - MARGIN_MM
            if wrap and prev_key is not None:
                row_top -= row_h + GAP_MM
                cursor_x = MARGIN_MM
                row_h = 0.0
            prev_key = key

            mleft, mbottom = mm_to_fig(cursor_x, row_top - cell_mm)
            cell_w, cell_h = cell_mm / PAGE_W_MM, cell_mm / PAGE_H_MM
            ax = fig.add_axes([mleft, mbottom, cell_w, cell_h])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Cut guide at the quiet-zone edge — cut on the grey line to keep the white border.
            fig.add_artist(Rectangle((mleft, mbottom), cell_w, cell_h, transform=fig.transFigure,
                                     fill=False, edgecolor="0.7", lw=0.4))
            short_fam = "april" if fam == "apriltag" else fam
            fig.text(*mm_to_fig(cursor_x, row_top - cell_mm - 1.5),
                     f"{short_fam} {tag_id}  {edge_mm:.0f}mm",
                     ha="left", va="top", fontsize=6.5)
            cursor_x += cell_mm + GAP_MM
            row_h = max(row_h, cell_mm + GAP_MM)

        pdf.savefig(fig)
        plt.close(fig)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    render(build_tags(), OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
