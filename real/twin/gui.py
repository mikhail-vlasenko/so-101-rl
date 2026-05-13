"""Tkinter panel for the digital-twin tool.

Tkinter (not DearPyGui) is used because DPG creates its own GLFW/EGL context
which segfaults when coexisting with MuJoCo's passive viewer on Wayland. Tk
talks to X11 (or XWayland) and has no overlap with MuJoCo's GL pipeline.

`run()` must be called from the main thread — Tk requires it.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Callable

import numpy as np

from .mapping import JointMaps, raw_to_norm

MIRROR = "MIRROR"
CONTROL = "CONTROL"

# Tk on HiDPI / Wayland-via-XWayland doesn't auto-scale. Use negative font
# sizes — in Tk, negative means "pixels", which is DPI-independent (positive
# is points, which depends on `tk scaling` and is finicky).
# Requires an Xft-enabled tk (e.g. `tk=*=xft_*` from conda-forge); the noxft
# build only sees the bitmap `fixed` font and won't render at this size.
FONT_PX = -18
FONT_PX_HEADER = -19


class TwinState:
    def __init__(self, n: int):
        self.lock = threading.Lock()
        self.bus_lock = threading.Lock()
        self.mode: str = MIRROR
        self.direction: np.ndarray = np.ones(n, dtype=np.int8)
        self.latest_raw_read: np.ndarray = np.full(n, 2048, dtype=np.int64)
        self.targets_rad: np.ndarray = np.zeros(n, dtype=np.float64)
        self.prev_raw_target: np.ndarray = np.full(n, 2048, dtype=np.int64)
        self.read_hz: float = 0.0
        self.last_error: str = ""
        self.stop: bool = False


def _set_global_fonts(root: tk.Tk) -> tkfont.Font:
    for fname in ("TkDefaultFont", "TkTextFont", "TkHeadingFont",
                  "TkMenuFont", "TkFixedFont", "TkTooltipFont", "TkIconFont"):
        tkfont.nametofont(fname).configure(size=FONT_PX)
    mono = tkfont.nametofont("TkFixedFont")
    style = ttk.Style()
    base = ("TkDefaultFont", FONT_PX)
    # Some ttk themes ignore the root `.` style for specific widget classes,
    # so configure each class we use.
    for cls in (".", "TLabel", "TButton", "TCheckbutton", "TRadiobutton",
                "TFrame", "TLabelframe", "TSeparator", "TScale"):
        style.configure(cls, font=base)
    style.configure("Estop.TButton",
                    font=("TkDefaultFont", FONT_PX_HEADER, "bold"),
                    foreground="white", background="#c0392b")
    return mono


def run(
    state: TwinState,
    jm: JointMaps,
    enter_control: Callable[[], None],
    enter_mirror: Callable[[], None],
    estop: Callable[[], None],
) -> None:
    n = len(jm.items)
    root = tk.Tk()
    root.title("SO-101 Digital Twin")
    mono = _set_global_fonts(root)

    # --- top bar: mode + e-stop ---
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Mode:").pack(side="left")
    mode_var = tk.StringVar(value=MIRROR)
    slider_widgets: list[ttk.Scale] = []
    slider_vars: list[tk.DoubleVar] = []

    def apply_mode_state() -> None:
        for sld in slider_widgets:
            sld.configure(state=("normal" if mode_var.get() == CONTROL else "disabled"))

    def on_mode_change() -> None:
        if mode_var.get() == CONTROL:
            enter_control()
            with state.lock:
                tgt = state.targets_rad.copy()
            for i, v in enumerate(slider_vars):
                v.set(float(tgt[i]))
        else:
            enter_mirror()
        apply_mode_state()

    ttk.Radiobutton(top, text="MIRROR", variable=mode_var, value=MIRROR,
                    command=on_mode_change).pack(side="left", padx=6)
    ttk.Radiobutton(top, text="CONTROL", variable=mode_var, value=CONTROL,
                    command=on_mode_change).pack(side="left", padx=6)

    def do_estop() -> None:
        estop()
        mode_var.set(MIRROR)
        apply_mode_state()
    ttk.Button(top, text="E-STOP", style="Estop.TButton",
               command=do_estop).pack(side="right")

    ttk.Separator(root, orient="horizontal").pack(fill="x")

    # --- joints: per-joint frame with info row + slider below ---
    raw_labels: list[ttk.Label] = []
    nrm_labels: list[ttk.Label] = []
    rad_labels: list[ttk.Label] = []

    def make_dir_cb(idx: int, var: tk.BooleanVar):
        def cb() -> None:
            with state.lock:
                state.direction[idx] = -1 if var.get() else 1
                state.prev_raw_target = state.latest_raw_read.copy()
        return cb

    def make_sld_cb(idx: int, var: tk.DoubleVar):
        def cb(_value: str) -> None:
            with state.lock:
                if state.mode == CONTROL:
                    state.targets_rad[idx] = float(var.get())
        return cb

    for i, j in enumerate(jm.items):
        jframe = ttk.Frame(root, padding=(10, 6))
        jframe.pack(fill="x")

        info = ttk.Frame(jframe)
        info.pack(fill="x")
        ttk.Label(info, text=j.name, width=14, anchor="w",
                  font=("TkDefaultFont", FONT_PX_HEADER, "bold")).pack(side="left")
        dv = tk.BooleanVar(value=False)
        ttk.Checkbutton(info, text="inv", variable=dv,
                        command=make_dir_cb(i, dv)).pack(side="left", padx=8)
        lbl_raw = ttk.Label(info, text="raw=----", width=12, font=mono)
        lbl_raw.pack(side="left", padx=4); raw_labels.append(lbl_raw)
        lbl_nrm = ttk.Label(info, text="norm=----", width=14, font=mono)
        lbl_nrm.pack(side="left", padx=4); nrm_labels.append(lbl_nrm)
        lbl_rad = ttk.Label(info, text="rad=----", width=14, font=mono)
        lbl_rad.pack(side="left", padx=4); rad_labels.append(lbl_rad)
        mid_color = "black" if abs(j.cal_mid - 2048) < 20 else "#c47a00"
        ttk.Label(info, text=f"mid={j.cal_mid:.0f}", font=mono,
                  foreground=mid_color).pack(side="left", padx=4)

        sv = tk.DoubleVar(value=0.0)
        sld = ttk.Scale(jframe, from_=j.xml_low, to=j.xml_high, variable=sv,
                        orient="horizontal", state="disabled",
                        command=make_sld_cb(i, sv))
        sld.pack(fill="x", pady=(4, 0))
        slider_widgets.append(sld)
        slider_vars.append(sv)

        if i < n - 1:
            ttk.Separator(root, orient="horizontal").pack(fill="x")

    ttk.Separator(root, orient="horizontal").pack(fill="x")

    # --- status ---
    status_lbl = ttk.Label(root, text="", padding=(10, 4))
    status_lbl.pack(fill="x")
    qpos_lbl = ttk.Label(root, text="", padding=(10, 0), font=mono)
    qpos_lbl.pack(fill="x", pady=(0, 8))

    def refresh() -> None:
        if state.stop:
            root.quit()
            return
        with state.lock:
            raw = state.latest_raw_read.copy()
            direction = state.direction.copy()
            read_hz = state.read_hz
            mode = state.mode
            last_error = state.last_error
        # External mode changes (e.g. from gamepad) need to be reflected in
        # the radiobuttons and slider enable state.
        if mode != mode_var.get():
            mode_var.set(mode)
            apply_mode_state()
        norm = raw_to_norm(raw, jm.cal_min(), jm.cal_max(), direction)
        rad = jm.xml_low() + norm * (jm.xml_high() - jm.xml_low())
        for i in range(n):
            raw_labels[i]["text"] = f"raw={int(raw[i]):5d}"
            nrm_labels[i]["text"] = f"norm={norm[i]:+0.3f}"
            rad_labels[i]["text"] = f"rad={rad[i]:+0.3f}"
        if mode == MIRROR:
            # Slider callback no-ops when mode != CONTROL, so this is safe.
            # Skipped in CONTROL: programmatic .set() fires the slider callback
            # and would race the gamepad integrator's writes to targets_rad.
            for i in range(n):
                slider_vars[i].set(float(rad[i]))
        err = f"  |  err: {last_error}" if last_error else ""
        status_lbl["text"] = f"mode={mode}   read={read_hz:5.1f} Hz{err}"
        qpos_lbl["text"] = "qpos: [" + ", ".join(f"{x:+0.3f}" for x in rad) + "]"
        root.after(33, refresh)

    def on_close() -> None:
        state.stop = True
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(33, refresh)
    root.mainloop()
    state.stop = True
    root.destroy()
