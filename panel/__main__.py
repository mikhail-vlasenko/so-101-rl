"""Run the control panel: `python -m panel [--host 127.0.0.1] [--port 8800]`.

Defaults to localhost. There is no auth, so only pass `--host 0.0.0.0` (LAN
exposure) on a trusted network. Run inside the `mujoco_env` conda env.
"""

import argparse

import uvicorn

from panel.app import create_app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    args = p.parse_args()

    app = create_app()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        app.state.runner.shutdown()
        app.state.camera.stop()  # no-op when the stream was never started
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
