"""Run the control panel: `python -m panel [--host 127.0.0.1] [--port 8800]`.

Defaults to localhost. There is no auth, so only pass `--host 0.0.0.0` (LAN
exposure) on a trusted network. Run inside the `mujoco_env` conda env.

`--reload` is a dev convenience: uvicorn watches panel/*.py and restarts the
server on edits. Each restart SIGINTs every live run (panel cleanup), so don't
use it while a real-arm script is mid-execution. Templates and static assets
already reload per request, so you only need this for Python changes.
"""

import argparse
from pathlib import Path

import uvicorn

from panel.app import create_app

PANEL_DIR = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--reload", action="store_true",
                   help="restart the server when panel/*.py changes (dev only)")
    args = p.parse_args()

    if args.reload:
        uvicorn.run("panel.app:create_app", factory=True,
                    host=args.host, port=args.port, log_level="warning",
                    reload=True, reload_dirs=[str(PANEL_DIR)])
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port,
                    log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
