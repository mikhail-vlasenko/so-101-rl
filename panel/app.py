"""FastAPI control panel: pages + JSON API over the Runner.

No auth, no TLS: anything reachable on the panel port can launch scripts,
including ones that move the real arm. Localhost-only by default.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from panel.camera_service import CameraService, default_settings
from panel.registry import (
    PAGES,
    REPO_ROOT,
    SCRIPTS,
    get_spec,
    validate_registry,
)
from panel.runner import ResourceBusyError, Run, Runner
from panel.settings import SettingsStore
from panel.streamer import STREAM_BOUNDARY

PANEL_DIR = Path(__file__).resolve().parent

PAGE_TITLES = {
    "train": "Train",
    "sim": "Sim",
    "real": "Real arm",
    "camera": "Camera",
    "sysid": "Sysid",
}

# Browsable artifact roots for /files/{root}/... — nothing outside these.
FILE_ROOTS = {
    "rollouts": REPO_ROOT / "rollouts",
    "sysid_logs": REPO_ROOT / "sysid_logs",
    "logs": REPO_ROOT / "logs",
}

SSE_KEEPALIVE_S = 15.0


def run_summary(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "spec_id": run.spec.id,
        "title": run.spec.title,
        "running": run.running,
        "returncode": run.returncode,
        "stop_requested": run.stop_requested,
        "started_at": run.started_at,
        "uptime_s": (time.time() - run.started_at) if run.running else None,
        "stream_port": run.stream_port,
        "argv": run.argv,
    }


def create_app(runner: Runner | None = None) -> FastAPI:
    validate_registry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # On shutdown — including a --reload restart — SIGINT every live run
        # (so scripts torque-off / save) and release the camera device.
        yield
        app.state.runner.shutdown()
        app.state.camera.stop()  # no-op when the stream was never started

    app = FastAPI(title="SO-101 panel", lifespan=lifespan)
    app.state.runner = runner if runner is not None else Runner()
    app.state.camera = CameraService(app.state.runner)
    app.state.settings = SettingsStore()

    templates = Jinja2Templates(directory=str(PANEL_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(PANEL_DIR / "static")), name="static")

    @app.exception_handler(ResourceBusyError)
    async def busy_handler(request: Request, exc: ResourceBusyError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # ---- pages ----

    def page_ctx(active: str) -> dict:
        return {"active": active, "page_titles": PAGE_TITLES}

    @app.get("/")
    def dashboard(request: Request):
        return templates.TemplateResponse(
            request, "dashboard.html", page_ctx("dashboard"))

    @app.get("/page/{page}")
    def scripts_page(request: Request, page: str):
        if page not in PAGES or page == "camera":
            raise HTTPException(404, f"unknown page {page!r}")
        specs = [s for s in SCRIPTS if s.page == page]
        return templates.TemplateResponse(
            request, "scripts_page.html",
            {**page_ctx(page), "specs": specs, "page_title": PAGE_TITLES[page]})

    @app.get("/camera")
    def camera_page(request: Request):
        specs = [s for s in SCRIPTS if s.page == "camera"]
        return templates.TemplateResponse(
            request, "camera.html",
            {**page_ctx("camera"), "specs": specs,
             "camera_streaming": app.state.camera.is_running(),
             "camera_defaults": default_settings()})

    @app.get("/artifacts")
    def artifacts_page(request: Request):
        rollout_dir = FILE_ROOTS["rollouts"]
        rollouts = []
        if rollout_dir.is_dir():
            for png in sorted(rollout_dir.glob("*.png"),
                              key=lambda p: p.stat().st_mtime, reverse=True):
                csv = png.with_suffix(".csv")
                rollouts.append({
                    "png": png.name,
                    "csv": csv.name if csv.exists() else None,
                    "mtime": time.strftime("%Y-%m-%d %H:%M",
                                           time.localtime(png.stat().st_mtime)),
                })
        sysid_dir = FILE_ROOTS["sysid_logs"]
        sysid_plots = []
        if (sysid_dir / "plots").is_dir():
            sysid_plots = sorted(p.name for p in (sysid_dir / "plots").glob("*.png"))
        report = sysid_dir / "report.tsv"
        report_text = report.read_text() if report.exists() else None

        checkpoints = []
        logs_dir = FILE_ROOTS["logs"]
        for d in sorted(logs_dir.glob("*_*")):
            ckpt_dir = d / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
            checkpoints.append({
                "log_dir": d.name,
                "count": len(zips),
                "latest": zips[-1].name if zips else None,
                "has_best": (d / "best_model.zip").exists(),
            })
        return templates.TemplateResponse(
            request, "artifacts.html",
            {**page_ctx("artifacts"), "rollouts": rollouts,
             "sysid_plots": sysid_plots, "report_text": report_text,
             "checkpoints": checkpoints})

    @app.get("/run/{run_id}")
    def run_page(request: Request, run_id: str):
        try:
            run = app.state.runner.get(run_id)
        except KeyError:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return templates.TemplateResponse(
            request, "run.html", {**page_ctx(run.spec.page), "run": run,
                                  "summary": run_summary(run)})

    # ---- API ----

    @app.post("/api/run/{spec_id}")
    async def api_run(spec_id: str, request: Request):
        try:
            spec = get_spec(spec_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        body = await request.json()
        values: dict[str, str] = body["values"]
        stream = bool(body.get("stream", False))
        try:
            run = app.state.runner.start(spec, values, stream=stream)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"run_id": run.run_id, "stream_port": run.stream_port}

    @app.post("/api/stop/{run_id}")
    def api_stop(run_id: str):
        try:
            run = app.state.runner.stop(run_id)
        except KeyError:
            raise HTTPException(404, f"unknown run {run_id!r}")
        return run_summary(run)

    @app.post("/api/restart/{run_id}")
    def api_restart(run_id: str):
        """Relaunch a finished run with its original values (fresh stream port)."""
        try:
            old = app.state.runner.get(run_id)
        except KeyError:
            raise HTTPException(404, f"unknown run {run_id!r}")
        if old.running:
            raise HTTPException(409, f"run {run_id} is still running")
        run = app.state.runner.start(old.spec, old.values,
                                     stream=old.stream_port is not None)
        return {"run_id": run.run_id, "stream_port": run.stream_port}

    @app.get("/api/status")
    def api_status():
        return {
            "runs": [run_summary(r) for r in app.state.runner.runs()],
            "camera_streaming": app.state.camera.is_running(),
        }

    @app.get("/api/logs/{run_id}")
    def api_logs(run_id: str, request: Request, start: int = 0):
        try:
            run = app.state.runner.get(run_id)
        except KeyError:
            raise HTTPException(404, f"unknown run {run_id!r}")

        def event_stream():
            idx = start
            while True:
                lines, idx, finished = run.wait_lines(idx, timeout=SSE_KEEPALIVE_S)
                for line in lines:
                    yield f"data: {json.dumps(line)}\n\n"
                if finished:
                    yield f"event: done\ndata: {json.dumps(run.returncode)}\n\n"
                    return
                if not lines:
                    yield ": keepalive\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/checkpoints/{log_dir}")
    def api_checkpoints(log_dir: str):
        if "/" in log_dir or log_dir.startswith("."):
            raise HTTPException(400, "bad log dir")
        base = FILE_ROOTS["logs"] / log_dir
        options = ["latest", "best"]
        # Promoted/named checkpoints live directly in the log dir (e.g.
        # stage4_light_run222.zip); auto-saved ones live under checkpoints/.
        # List both so either is selectable, newest first within each group.
        for src in (base, base / "checkpoints"):
            if src.is_dir():
                zips = sorted((p for p in src.glob("*.zip") if p.name != "best_model.zip"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
                options += [str(p.relative_to(REPO_ROOT)) for p in zips]
        return {"log_dir": log_dir, "options": options,
                "exists": base.is_dir(), "has_best": (base / "best_model.zip").exists()}

    # ---- ui settings (form-field persistence) ----

    @app.get("/api/settings")
    def api_settings():
        return app.state.settings.all()

    @app.post("/api/settings")
    async def api_settings_set(request: Request):
        body = await request.json()
        app.state.settings.set(body["scope"], body["field"], body["value"])
        return {"ok": True}

    @app.post("/api/settings/reset")
    def api_settings_reset():
        app.state.settings.reset()
        return {"ok": True}

    # ---- camera ----

    @app.post("/api/camera/start")
    async def api_camera_start(request: Request):
        body = await request.json()
        defaults = default_settings()
        family = body.get("family") or "aruco"
        exposure = int(body["exposure"]) if str(body.get("exposure", "")).strip() else defaults["exposure"]
        gain = int(body["gain"]) if str(body.get("gain", "")).strip() else defaults["gain"]
        try:
            app.state.camera.start(family, exposure, gain)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"streaming": True, "family": family, "exposure": exposure, "gain": gain}

    @app.post("/api/camera/stop")
    def api_camera_stop():
        app.state.camera.stop()
        return {"streaming": False}

    @app.post("/api/camera/exposure")
    async def api_camera_exposure(request: Request):
        body = await request.json()
        if not app.state.camera.is_running():
            raise HTTPException(409, "camera stream is not running")
        app.state.camera.set_exposure(int(body["value"]))
        return {"exposure": app.state.camera.exposure}

    @app.post("/api/camera/gain")
    async def api_camera_gain(request: Request):
        body = await request.json()
        if not app.state.camera.is_running():
            raise HTTPException(409, "camera stream is not running")
        app.state.camera.set_gain(int(body["value"]))
        return {"gain": app.state.camera.gain}

    @app.get("/camera/stream")
    def camera_stream():
        camera = app.state.camera
        if not camera.is_running():
            raise HTTPException(503, "camera stream is not running")

        def mjpeg():
            seq = 0
            while camera.is_running():
                frame = camera.box.wait_newer(seq, timeout=1.0)
                if frame is None:
                    continue
                jpeg, seq = frame
                yield (f"--{STREAM_BOUNDARY}\r\n"
                       f"Content-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpeg)}\r\n\r\n").encode() + jpeg + b"\r\n"

        return StreamingResponse(
            mjpeg(),
            media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")

    @app.get("/files/{root}/{path:path}")
    def files(root: str, path: str):
        if root not in FILE_ROOTS:
            raise HTTPException(404, f"unknown file root {root!r}")
        base = FILE_ROOTS[root]
        target = (base / path).resolve()
        if not target.is_relative_to(base.resolve()):
            raise HTTPException(403, "path escapes the artifact root")
        if not target.is_file():
            raise HTTPException(404, f"{root}/{path} not found")
        return FileResponse(target)

    return app
