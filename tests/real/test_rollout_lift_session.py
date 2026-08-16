"""Warm lift-session lifecycle contracts without camera or arm hardware."""

from argparse import Namespace

from real.rollout.rollout_lift import EpisodeResult, LiftRolloutSession


class FakeLoop:
    def __init__(self):
        self.boots = 0
        self.ends = 0

    def boot(self):
        self.boots += 1

    def end_episode(self):
        self.ends += 1


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


def test_interrupted_episode_returns_to_prompt(monkeypatch):
    session = LiftRolloutSession.__new__(LiftRolloutSession)
    session.args = Namespace(interactive=True)
    prompts = iter((True, True, False))
    results = iter((
        EpisodeResult([], interrupted=True, viewer_closed=False),
        EpisodeResult([], interrupted=False, viewer_closed=False),
    ))
    prepared = []
    saved = []
    session._prompt_for_episode = lambda episode: next(prompts)
    session._prepare_camera_episode = lambda: prepared.append(True)
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
    assert saved == [0, 1]
