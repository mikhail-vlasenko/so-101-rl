"""Runner lifecycle tests against tests/dummy_panel_script.py — no hardware.

The dummy claims Resource.SERIAL in its spec so exclusivity can be exercised
without a real bus; resource accounting never touches devices.
"""

import asyncio
import time

import httpx
import pytest

from panel.app import create_app
from panel.registry import ArgSpec, Resource, ScriptSpec, validate_registry
from panel.runner import ResourceBusyError, Runner

DUMMY_SPEC = ScriptSpec(
    id="dummy", title="Dummy", page="sim",
    module="tests.panel.dummy_panel_script", arg_style="argparse",
    description="test stand-in",
    args=(ArgSpec("--stubborn", "flag", "Ignore SIGINT"),),
    resources=(Resource.SERIAL,),
    supports_stream=True,
)


def wait_for(predicate, timeout=10.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def runner(tmp_path):
    r = Runner(log_dir=tmp_path / "panel_logs", stream_ports=range(28801, 28803),
               stop_grace_s=1.0)
    yield r
    r.shutdown()
    for run in r.runs():
        wait_for(lambda: not run.running, what=f"{run.run_id} to die")


def test_dummy_spec_is_valid():
    validate_registry((DUMMY_SPEC,))


def test_sigint_path_clean_exit(runner):
    run = runner.start(DUMMY_SPEC, {}, stream=False)
    wait_for(lambda: run.total_lines >= 3, what="output lines")
    lines, _ = run.lines_since(0)
    assert lines[0].startswith("$ ")          # command echo
    assert "READY" in lines

    runner.stop(run.run_id)
    wait_for(lambda: not run.running, what="clean exit")
    assert run.returncode == 0
    lines, _ = run.lines_since(0)
    assert "CLEAN EXIT" in lines
    assert run.log_path.read_text().count("CLEAN EXIT") == 1


def test_sigkill_after_grace(runner):
    run = runner.start(DUMMY_SPEC, {"stubborn": "on"}, stream=False)
    wait_for(lambda: run.total_lines >= 2, what="output lines")

    t0 = time.monotonic()
    runner.stop(run.run_id)
    wait_for(lambda: not run.running, what="SIGKILL")
    assert time.monotonic() - t0 >= 1.0       # grace was honored
    assert run.returncode == -9
    lines, _ = run.lines_since(0)
    assert any("SIGKILL" in l for l in lines)


def test_resource_conflict_names_holder(runner):
    first = runner.start(DUMMY_SPEC, {}, stream=False)
    wait_for(lambda: first.total_lines >= 2, what="first run up")

    with pytest.raises(ResourceBusyError, match="serial is in use by run"):
        runner.start(DUMMY_SPEC, {}, stream=False)

    runner.stop(first.run_id)
    wait_for(lambda: not first.running, what="first run exit")
    second = runner.start(DUMMY_SPEC, {}, stream=False)
    assert second.running
    runner.stop(second.run_id)


def test_external_holder_blocks_and_releases(runner):
    runner.claim_external("camera page stream", {Resource.SERIAL})
    with pytest.raises(ResourceBusyError, match="camera page stream"):
        runner.start(DUMMY_SPEC, {}, stream=False)
    with pytest.raises(ResourceBusyError):
        runner.claim_external("second claimant", {Resource.SERIAL})
    runner.release_external("camera page stream")
    run = runner.start(DUMMY_SPEC, {}, stream=False)
    assert run.running
    runner.stop(run.run_id)


def test_stream_port_allocation_and_exhaustion(tmp_path):
    spec_free = ScriptSpec(
        id="dummy_free", title="Dummy (no hardware)", page="sim",
        module="tests.panel.dummy_panel_script", arg_style="argparse",
        description="test stand-in", supports_stream=True)
    runner = Runner(log_dir=tmp_path / "panel_logs",
                    stream_ports=range(28801, 28803), stop_grace_s=1.0)
    try:
        a = runner.start(spec_free, {}, stream=True)
        b = runner.start(spec_free, {}, stream=True)
        assert (a.stream_port, b.stream_port) == (28801, 28802)
        assert "--stream-port" in a.argv and "28801" in a.argv
        with pytest.raises(ResourceBusyError, match="no free stream ports"):
            runner.start(spec_free, {}, stream=True)
        runner.stop(a.run_id)
        wait_for(lambda: not a.running, what="port release")
        c = runner.start(spec_free, {}, stream=True)
        assert c.stream_port == 28801
    finally:
        runner.shutdown()
        for run in runner.runs():
            wait_for(lambda: not run.running, what="shutdown")


def test_restart_endpoint_relaunches_with_original_values(runner):
    async def exercise_api():
        transport = httpx.ASGITransport(app=create_app(runner))
        async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as client:
            assert (await client.post("/api/restart/nope")).status_code == 404

            run = runner.start(DUMMY_SPEC, {"stubborn": ""}, stream=False)
            wait_for(lambda: run.total_lines >= 2, what="run up")
            response = await client.post(f"/api/restart/{run.run_id}")
            assert response.status_code == 409

            runner.stop(run.run_id)
            wait_for(lambda: not run.running, what="exit")
            response = await client.post(f"/api/restart/{run.run_id}")
            assert response.status_code == 200
            rerun = runner.get(response.json()["run_id"])
            assert rerun.run_id != run.run_id
            assert rerun.argv == run.argv
            runner.stop(rerun.run_id)

    asyncio.run(exercise_api())


def test_wait_lines_reports_finish(runner):
    run = runner.start(DUMMY_SPEC, {}, stream=False)
    lines, idx, finished = run.wait_lines(0, timeout=10.0)
    assert lines and not finished
    runner.stop(run.run_id)
    wait_for(lambda: not run.running, what="exit")
    _, _, finished = run.wait_lines(idx, timeout=10.0)
    assert finished
