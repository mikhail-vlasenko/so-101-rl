"""Process lifecycle for panel-launched scripts.

One `Runner` owns every child process the panel starts: it enforces hardware
exclusivity (one serial-bus user, one camera user — including the panel's own
camera stream, which registers itself as an external holder), allocates stream
ports from a fixed pool, captures each child's merged stdout/stderr into a
ring buffer plus a log file under `logs/panel/`, and implements the two-stage
stop: SIGINT to the child's process group first (so the scripts' own handlers
do torque-off / CSV+PNG saving), SIGKILL only after a grace period.

Children run with `start_new_session=True`, so panel-process signals never
reach them, and with PYTHONUNBUFFERED so log lines arrive as they happen.
Their stdin is `/dev/null`; terminal-interactive tools must be launched from a
terminal rather than through the panel.
Streamed launches additionally get MUJOCO_GL=egl: offscreen rendering must
not depend on the panel host's display.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from panel.registry import REPO_ROOT, Resource, ScriptSpec, build_command

PANEL_LOG_DIR = REPO_ROOT / "logs" / "panel"
STREAM_PORTS = range(8801, 8820)
STOP_GRACE_S = 15.0
RING_BUFFER_LINES = 2000


class ResourceBusyError(Exception):
    """A required hardware resource is already held; message names the holder."""


@dataclass
class Run:
    run_id: str
    spec: ScriptSpec
    argv: list[str]            # full argv including the interpreter
    values: dict[str, str]     # original form values, kept for restart
    proc: subprocess.Popen
    log_path: Path
    started_at: float
    stream_port: int | None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=RING_BUFFER_LINES))
    total_lines: int = 0
    cond: threading.Condition = field(default_factory=threading.Condition)
    returncode: int | None = None
    stop_requested: bool = False

    @property
    def running(self) -> bool:
        return self.returncode is None

    def append_line(self, line: str) -> None:
        with self.cond:
            self.lines.append(line)
            self.total_lines += 1
            self.cond.notify_all()

    def lines_since(self, idx: int) -> tuple[list[str], int]:
        """Lines with index >= idx (clamped to what the ring still holds)."""
        with self.cond:
            first = self.total_lines - len(self.lines)
            skip = max(0, idx - first)
            new = list(self.lines)[skip:]
            return new, self.total_lines

    def wait_lines(self, idx: int, timeout: float) -> tuple[list[str], int, bool]:
        """Block until lines beyond idx exist or the run finishes.

        Returns (new_lines, next_idx, finished)."""
        with self.cond:
            self.cond.wait_for(
                lambda: self.total_lines > idx or self.returncode is not None,
                timeout=timeout)
        new, next_idx = self.lines_since(idx)
        return new, next_idx, self.returncode is not None


class Runner:
    def __init__(self, python: str = sys.executable,
                 log_dir: Path = PANEL_LOG_DIR,
                 stream_ports: range = STREAM_PORTS,
                 stop_grace_s: float = STOP_GRACE_S) -> None:
        self._python = python
        self._log_dir = log_dir
        self._stream_ports = stream_ports
        self._stop_grace_s = stop_grace_s
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        self._counter = 0
        # Non-subprocess holders (the in-server camera stream). name -> resources.
        self._external_holders: dict[str, frozenset[Resource]] = {}

    # ---- resource accounting (single source of truth for exclusivity) ----

    def held_resources(self) -> dict[Resource, str]:
        """Resource -> human-readable holder name, for every live claim."""
        with self._lock:
            held: dict[Resource, str] = {}
            for run in self._runs.values():
                if run.running:
                    for r in run.spec.resources:
                        held[r] = f"run {run.run_id} ({run.spec.title})"
            for name, resources in self._external_holders.items():
                for r in resources:
                    held[r] = name
            return held

    def claim_external(self, name: str, resources: set[Resource]) -> None:
        """Claim resources for a non-subprocess holder; raises ResourceBusyError."""
        held = self.held_resources()
        for r in resources:
            if r in held:
                raise ResourceBusyError(f"{r.value} is in use by {held[r]}")
        with self._lock:
            self._external_holders[name] = frozenset(resources)

    def release_external(self, name: str) -> None:
        with self._lock:
            del self._external_holders[name]

    # ---- lifecycle ----

    def start(self, spec: ScriptSpec, values: dict[str, str], stream: bool) -> Run:
        held = self.held_resources()
        for r in spec.resources:
            if r in held:
                raise ResourceBusyError(f"{r.value} is in use by {held[r]}")

        stream_port = self._allocate_stream_port() if stream else None
        argv = [self._python, *build_command(spec, values, stream_port=stream_port)]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if stream_port is not None:
            env["MUJOCO_GL"] = "egl"

        with self._lock:
            self._counter += 1
            run_id = f"{spec.id}-{self._counter}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{run_id}.log"

        proc = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        run = Run(run_id=run_id, spec=spec, argv=argv, values=values,
                  proc=proc, log_path=log_path, started_at=time.time(),
                  stream_port=stream_port)
        with self._lock:
            self._runs[run_id] = run
        threading.Thread(target=self._read_output, args=(run,),
                         name=f"panel-reader-{run_id}", daemon=True).start()
        return run

    def _allocate_stream_port(self) -> int:
        with self._lock:
            used = {r.stream_port for r in self._runs.values()
                    if r.running and r.stream_port is not None}
        for port in self._stream_ports:
            if port not in used:
                return port
        raise ResourceBusyError(
            f"no free stream ports in {self._stream_ports.start}-{self._stream_ports.stop - 1}")

    def _read_output(self, run: Run) -> None:
        with run.log_path.open("w") as log_file:
            header = f"$ {' '.join(run.argv)}"
            log_file.write(header + "\n")
            run.append_line(header)
            for line in run.proc.stdout:
                line = line.rstrip("\n")
                log_file.write(line + "\n")
                log_file.flush()
                run.append_line(line)
            code = run.proc.wait()
            footer = f"panel: process exited with code {code}"
            log_file.write(footer + "\n")
        with run.cond:
            run.returncode = code
            run.cond.notify_all()
        run.append_line(footer)

    def stop(self, run_id: str) -> Run:
        """SIGINT the run's process group; SIGKILL after the grace period."""
        run = self.get(run_id)
        if not run.running:
            return run
        run.stop_requested = True
        run.append_line("panel: stop requested, sending SIGINT")
        try:
            os.killpg(run.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return run  # exited between the running check and the signal
        threading.Thread(target=self._kill_after_grace, args=(run,),
                         name=f"panel-grace-{run_id}", daemon=True).start()
        return run

    def _kill_after_grace(self, run: Run) -> None:
        deadline = time.monotonic() + self._stop_grace_s
        while time.monotonic() < deadline:
            if not run.running:
                return
            time.sleep(0.1)
        if run.running:
            run.append_line(
                f"panel: still alive {self._stop_grace_s:.0f}s after SIGINT, sending SIGKILL")
            try:
                os.killpg(run.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # exited in the same instant

    def get(self, run_id: str) -> Run:
        with self._lock:
            return self._runs[run_id]

    def runs(self) -> list[Run]:
        """All runs, newest first."""
        with self._lock:
            return sorted(self._runs.values(),
                          key=lambda r: r.started_at, reverse=True)

    def shutdown(self) -> None:
        """Stop every live run (used at panel exit)."""
        for run in self.runs():
            if run.running:
                self.stop(run.run_id)
