"""Persistent UI settings for the panel's web forms.

Every form field and toggle the panel renders (camera capture controls, each
script card's argument inputs) is persisted here as a saved *override* keyed by
`scope/field`, so the value the user last entered is pre-filled on the next
page load. Only overrides live in this file — the original defaults stay in
code (registry arg defaults, `camera_service.default_settings`), so resetting
is just wiping the overrides and letting those code defaults render again.

The store is a single JSON file guarded by a lock; FastAPI may call into it
from concurrent request handlers.
"""

from __future__ import annotations

import json
import threading

from panel.registry import REPO_ROOT

SETTINGS_PATH = REPO_ROOT / "logs" / "panel" / "ui_settings.json"


class SettingsStore:
    """Thread-safe `{scope: {field: value}}` override map backed by a file."""

    def __init__(self, path=SETTINGS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, object]] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def all(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return json.loads(json.dumps(self._data))  # deep copy for the caller

    def set(self, scope: str, field: str, value: object) -> None:
        with self._lock:
            self._data.setdefault(scope, {})[field] = value
            self._flush()

    def reset(self) -> None:
        with self._lock:
            self._data = {}
            self._flush()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
