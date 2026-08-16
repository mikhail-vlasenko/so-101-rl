"""Warm lift-session lifecycle contracts without camera or arm hardware."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np

from real.rollout.rollout_lift import EpisodeResult, LiftRolloutSession
from real.tracking.sam_seg import SAMPromptNoMatchError
from real.twin.constants import FOLDED_REST_QPOS


class FakeLoop:
    def __init__(self, execute=False):
        self.boots = 0
        self.ends = 0
        self.rests = []
        self.execute = execute

    def boot(self):
        self.boots += 1

    def end_episode(self):
        self.ends += 1

    def return_to_rest(self, qpos, duration_s, action_scale, settle_s,
                       should_stop):
        self.rests.append(
            (qpos, duration_s, action_scale, settle_s, should_stop()))
        return True

    def set_execute(self, execute):
        self.execute = execute


def test_interrupted_episode_disables_torque_boundary():
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.args = Namespace(interactive=True, max_steps=10)
    session.loop = FakeLoop()
    session._reset_episode_scene = lambda: None

    result = session._run_episode({"flag": True})

    assert result.interrupted
    assert result.rows == []
    assert session.loop.boots == 1
    assert session.loop.ends == 1
    assert session.loop.rests == []


def test_completed_episode_returns_to_folded_rest_before_torque_off():
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.args = Namespace(interactive=False, max_steps=0)
    session.loop = FakeLoop(execute=True)
    session.config = SimpleNamespace(
        rest_qpos=np.array(FOLDED_REST_QPOS),
        rest_duration_s=5.0,
        rest_action_scale=0.025,
        rest_settle_s=1.0,
    )
    session._reset_episode_scene = lambda: None

    result = session._run_episode({"flag": False})

    assert not result.interrupted
    assert session.loop.boots == 1
    assert session.loop.ends == 1
    assert len(session.loop.rests) == 1
    np.testing.assert_allclose(session.loop.rests[0][0], session.config.rest_qpos)


def test_execute_mode_toggles_at_interactive_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.loop = FakeLoop(execute=False)
    commands = iter(("e", ""))
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    monkeypatch.setattr(
        "real.rollout.rollout_lift.signal.signal",
        lambda signal_number, handler: None)

    assert session._prompt_for_episode(0)
    assert session.loop.execute


def test_ctrl_c_exits_interactive_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.loop = FakeLoop(execute=False)
    monkeypatch.setattr(
        "builtins.input", lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(
        "real.rollout.rollout_lift.signal.signal",
        lambda signal_number, handler: None)

    assert not session._prompt_for_episode(0)


def test_rest_command_parks_and_disables_torque_from_execute_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.loop = FakeLoop(execute=True)
    session.config = SimpleNamespace(
        rest_qpos=np.array(FOLDED_REST_QPOS),
        rest_duration_s=5.0,
        rest_action_scale=0.025,
        rest_settle_s=1.0,
    )
    commands = iter(("r", KeyboardInterrupt))

    def prompt(_message):
        command = next(commands)
        if command is KeyboardInterrupt:
            raise KeyboardInterrupt
        return command

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(
        "real.rollout.rollout_lift.install_sigint_flag",
        lambda: {"flag": False})
    monkeypatch.setattr(
        "real.rollout.rollout_lift.signal.signal",
        lambda signal_number, handler: None)

    assert not session._prompt_for_episode(0)
    assert session.loop.boots == 1
    assert session.loop.ends == 1
    assert len(session.loop.rests) == 1


def test_rest_command_does_not_move_from_dry_run_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.loop = FakeLoop(execute=False)
    commands = iter(("r", KeyboardInterrupt))

    def prompt(_message):
        command = next(commands)
        if command is KeyboardInterrupt:
            raise KeyboardInterrupt
        return command

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(
        "real.rollout.rollout_lift.signal.signal",
        lambda signal_number, handler: None)

    assert not session._prompt_for_episode(0)
    assert session.loop.boots == 0
    assert session.loop.ends == 0
    assert session.loop.rests == []


def test_missing_object_can_retry_without_restarting_source():
    class FakeObjectSource:
        def __init__(self):
            self.starts = 0

        def start(self):
            self.starts += 1
            if self.starts == 1:
                raise SAMPromptNoMatchError("SAM3 found no sponge")

    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.args = Namespace(interactive=True)
    session.camera_object = FakeObjectSource()
    prompts = []
    session._prompt_for_object_retry = lambda error: prompts.append(str(error)) or True

    assert session._start_object_source()
    assert session.camera_object.starts == 2
    assert prompts == ["SAM3 found no sponge"]


def test_interrupted_episode_returns_to_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.args = Namespace(interactive=True)
    prompts = iter((True, True, False))
    results = iter((
        EpisodeResult([], interrupted=True, viewer_closed=False),
        EpisodeResult([], interrupted=False, viewer_closed=False),
    ))
    prepared = []
    paused = []
    saved = []
    session._prompt_for_episode = lambda episode: next(prompts)
    session._prepare_camera_episode = lambda: prepared.append(True)
    session._pause_camera_pipeline = lambda: paused.append(True)
    session._run_episode = lambda stopped: next(results)
    session._save_episode = lambda rows, episode: saved.append(episode)
    monkeypatch.setattr(
        "real.rollout.rollout_lift.install_sigint_flag",
        lambda: {"flag": False})
    monkeypatch.setattr(
        "real.rollout.rollout_lift.signal.signal",
        lambda signal_number, handler: None)

    session.run()

    assert len(prepared) == 2
    assert len(paused) == 2
    assert saved == [0, 1]


def test_camera_pipeline_resumes_for_validation_and_pauses_at_prompt():
    calls = []

    class Markers:
        def resume(self):
            calls.append("markers resume")

        def warmup(self):
            calls.append("markers warmup")
            return 0.0

        def pause(self):
            calls.append("markers pause")

    class Object:
        def resume(self):
            calls.append("object resume")

        def prepare_episode(self):
            calls.append("object prepare")

        def pause(self):
            calls.append("object pause")

    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.camera_markers = Markers()
    session.camera_object = Object()

    session._prepare_camera_episode()
    session._pause_camera_pipeline()

    assert calls == [
        "markers resume",
        "object resume",
        "markers warmup",
        "object prepare",
        "object pause",
        "markers pause",
    ]
