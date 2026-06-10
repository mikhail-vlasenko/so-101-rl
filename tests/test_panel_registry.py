"""Registry consistency + build_command golden tests (no hardware, no launches)."""

import pytest

from panel.registry import SCRIPTS, build_command, get_spec, validate_registry


def test_registry_validates():
    validate_registry()


def test_get_spec_unknown_id_raises():
    with pytest.raises(KeyError):
        get_spec("nope")


def test_every_page_reachable():
    pages = {s.page for s in SCRIPTS}
    assert pages == {"train", "sim", "real", "camera", "sysid"}


def test_hydra_command_golden():
    spec = get_spec("eval")
    argv = build_command(spec, {
        "env": "lift",
        "model": "best",
        "episodes": "5",
        "render": "false",
    })
    assert argv == ["-m", "src.eval", "env=lift", "+model=best",
                    "+episodes=5", "render=false"]


def test_argparse_command_golden_with_flag_and_stream():
    spec = get_spec("rollout_lift")
    argv = build_command(spec, {
        "model": "latest",
        "execute": "on",
        "slow": "3",
        "seed": "0",
    }, stream_port=8801)
    assert argv == ["-m", "real.rollout_lift",
                    "--model", "latest", "--execute", "--slow", "3",
                    "--seed", "0",
                    "--no-view", "--stream-port", "8801"]


def test_empty_values_are_omitted():
    spec = get_spec("train")
    assert build_command(spec, {"env": "", "seed": "", "resume": ""}) == \
        ["-m", "src.train"]


def test_flag_false_is_omitted():
    spec = get_spec("rollout_lift")
    argv = build_command(spec, {"execute": "off", "no-view": ""})
    assert argv == ["-m", "real.rollout_lift"]


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown form fields"):
        build_command(get_spec("eval"), {"rm_rf": "yes"})


def test_unparseable_int_rejected():
    with pytest.raises(ValueError):
        build_command(get_spec("eval"), {"episodes": "ten"})


def test_unparseable_float_rejected():
    with pytest.raises(ValueError):
        build_command(get_spec("rollout_lift"), {"slow": "fast"})


def test_bad_choice_rejected():
    with pytest.raises(ValueError, match="not in"):
        build_command(get_spec("train"), {"env": "moonbase"})


def test_bad_flag_value_rejected():
    with pytest.raises(ValueError, match="bad flag value"):
        build_command(get_spec("rollout_lift"), {"execute": "maybe"})


def test_stream_port_on_nonstreaming_spec_rejected():
    with pytest.raises(ValueError, match="does not support streaming"):
        build_command(get_spec("read_kp"), {}, stream_port=8801)


def test_hydra_stream_port_form():
    argv = build_command(get_spec("eval"), {}, stream_port=8802)
    assert argv == ["-m", "src.eval", "render=false", "stream_port=8802"]
