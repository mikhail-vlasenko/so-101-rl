"""UI settings persistence: store round-trips and the /api/settings endpoints."""

import asyncio

import httpx

from panel.app import create_app
from panel.settings import SettingsStore


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "sub" / "ui_settings.json"  # parent dir created on flush
    store = SettingsStore(path=path)
    assert store.all() == {}

    store.set("camera", "exposure", "500")
    store.set("camera", "gain", 80)
    store.set("script:rollout_real", "__stream__", True)

    expected = {"camera": {"exposure": "500", "gain": 80},
                "script:rollout_real": {"__stream__": True}}
    assert store.all() == expected
    assert SettingsStore(path=path).all() == expected  # reload from disk


def test_all_returns_a_copy(tmp_path):
    store = SettingsStore(path=tmp_path / "s.json")
    store.set("camera", "gain", 80)
    snapshot = store.all()
    snapshot["camera"]["gain"] = 999
    assert store.all()["camera"]["gain"] == 80


def test_reset_clears_disk(tmp_path):
    path = tmp_path / "s.json"
    store = SettingsStore(path=path)
    store.set("camera", "gain", 80)
    store.reset()
    assert store.all() == {}
    assert SettingsStore(path=path).all() == {}


def test_settings_api_roundtrip(tmp_path):
    app = create_app()
    app.state.settings = SettingsStore(path=tmp_path / "s.json")  # isolate from repo

    async def exercise_api():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as client:
            assert (await client.get("/api/settings")).json() == {}

            await client.post(
                "/api/settings",
                json={"scope": "camera", "field": "exposure", "value": "500"})
            await client.post(
                "/api/settings",
                json={"scope": "camera", "field": "gain", "value": 80})
            assert (await client.get("/api/settings")).json() == {
                "camera": {"exposure": "500", "gain": 80}}

            await client.post("/api/settings/reset")
            assert (await client.get("/api/settings")).json() == {}

    asyncio.run(exercise_api())
