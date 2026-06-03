# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for resources/parse_env.py.

Acceptance criterion (issue #2): env parser unit-tested on at least 2 envs.
Covers one manager-based env (Isaac-Velocity-Flat-Ayg-v0) and one direct env
(Isaac-Velocity-Flat-Ayg-Direct-v0).

These are *integration tests*. `parse_env.py` launches the Isaac Sim AppLauncher
(needed because `isaaclab` imports `pxr` transitively), so we invoke it via
subprocess and assert on the emitted env_schema.json. This avoids importing
`parse_env` directly into pytest, which would conflict with AppLauncher's
singleton sim app.

Run:
    cd cf_lab
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      .venv/bin/python -m pytest \\
      .claude/skills/auto_train/resources/tests/test_parse_env.py -v -s

The `-s` flag streams Isaac Sim boot logs (helps spot failures). Each test takes
~30-60 s of Isaac Sim startup time.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_PARSER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "parse_env.py"))
# Prefer cf_lab venv python; fall back to current sys.executable
_REPO_VENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".venv", "bin", "python"))
if os.path.isfile(_REPO_VENV):
    _VENV_PYTHON = _REPO_VENV
else:
    _VENV_PYTHON = shutil.which("python") or sys.executable


def _run_parser(task_id: str, tmpdir: str) -> dict:
    out_json = os.path.join(tmpdir, "schema.json")
    out_md = os.path.join(tmpdir, "schema.md")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    cmd = [_VENV_PYTHON, _PARSER, "--task", task_id, "--output-json", out_json, "--output-md", out_md]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        pytest.fail(
            f"parse_env.py exited {result.returncode} for {task_id}\n"
            f"STDOUT tail: {result.stdout[-1500:]}\n"
            f"STDERR tail: {result.stderr[-1500:]}"
        )
    assert os.path.isfile(out_json), f"missing {out_json}"
    assert os.path.isfile(out_md), f"missing {out_md}"
    with open(out_json) as f:
        schema = json.load(f)
    assert "error" not in schema, f"parser reported error: {schema.get('error')}"
    schema["_md_path"] = out_md
    return schema


@pytest.fixture(scope="session")
def flat_schema():
    with tempfile.TemporaryDirectory() as tmp:
        yield _run_parser("Isaac-Velocity-Flat-Ayg-v0", tmp)


@pytest.fixture(scope="session")
def direct_schema():
    with tempfile.TemporaryDirectory() as tmp:
        yield _run_parser("Isaac-Velocity-Flat-Ayg-Direct-v0", tmp)


# ─── Manager-based env ───


def test_flat_basic_fields(flat_schema):
    assert flat_schema["task"] == "Isaac-Velocity-Flat-Ayg-v0"
    assert flat_schema["is_manager_based"] is True
    # Isaac Lab default: sim.dt=0.005, decimation=4 → 50 Hz control
    assert flat_schema["control_freq_hz"] == 50.0
    assert flat_schema["decimation"] == 4


def test_flat_has_velocity_tracking_rewards(flat_schema):
    reward_names = {r["name"] for r in flat_schema.get("rewards", [])}
    assert "track_lin_vel_xy_exp" in reward_names, reward_names
    assert "track_ang_vel_z_exp" in reward_names, reward_names
    # AYG-specific terms from AygRewardsCfg
    assert "feet_regulation" in reward_names
    assert "foot_clearance" in reward_names


def test_flat_reward_weights_match_source(flat_schema):
    """track_lin_vel_xy_exp.weight = 2.0 per rough_env_cfg.py:99 (flat inherits)."""
    rewards = {r["name"]: r for r in flat_schema.get("rewards", [])}
    assert rewards["track_lin_vel_xy_exp"]["weight"] == 2.0
    assert rewards["track_ang_vel_z_exp"]["weight"] == 1.0
    # AygFlatEnvCfg.__post_init__ overrides feet_air_time to 0.25
    assert rewards["feet_air_time"]["weight"] == 0.25


def test_flat_action_scale(flat_schema):
    """actions.joint_pos.scale = 0.25 per rough_env_cfg.py:79."""
    actions = {a["name"]: a for a in flat_schema.get("actions", [])}
    assert "joint_pos" in actions
    # scale is a top-level field on the JointPositionActionCfg, not inside params
    a = actions["joint_pos"]
    params = a.get("params") or {}
    # Some Isaac Lab versions place scale in params; others as a sibling attr
    found = False
    for candidate in (params.get("scale"), a.get("scale")):
        if candidate == 0.25:
            found = True
            break
    assert found, f"expected joint_pos scale=0.25, got params={params} action={a}"


def test_flat_observations_policy_group(flat_schema):
    obs = flat_schema.get("observations", {})
    assert "policy" in obs
    policy_term_names = {t["name"] for t in obs["policy"]["terms"] if "name" in t}
    for required in ("base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands"):
        assert required in policy_term_names, f"missing {required} in {policy_term_names}"


def test_flat_terrain_is_plane(flat_schema):
    scene = flat_schema.get("scene", {})
    assert scene.get("terrain_type") == "plane"
    # Flat env disables height scanner
    assert scene.get("height_scanner_present") is False


def test_flat_markdown_renders(flat_schema):
    with open(flat_schema["_md_path"]) as f:
        md = f.read()
    assert "# Env Schema: Isaac-Velocity-Flat-Ayg-v0" in md
    assert "## Rewards" in md
    assert "track_lin_vel_xy_exp" in md


# ─── Direct env ───


def test_direct_basic_fields(direct_schema):
    assert direct_schema["task"] == "Isaac-Velocity-Flat-Ayg-Direct-v0"
    assert direct_schema["is_manager_based"] is False
    # Direct env: sim.dt = 1/200, decimation = 4 → 50 Hz
    assert direct_schema["control_freq_hz"] == 50.0
    assert direct_schema["decimation"] == 4


def test_direct_action_obs_spaces(direct_schema):
    # Per ayg_env_cfg.py:69-70
    assert direct_schema.get("action_space") == 12
    assert direct_schema.get("observation_space") == 49


def test_direct_reward_scales_extracted(direct_schema):
    scales = direct_schema.get("reward_scales") or {}
    assert "lin_vel_reward_scale" in scales
    assert "yaw_rate_reward_scale" in scales
    assert "feet_air_time_reward_scale" in scales
    # Values per ayg_env_cfg.py:120-132
    assert scales["lin_vel_reward_scale"] == 2.0
    assert scales["yaw_rate_reward_scale"] == 1.0


def test_direct_reward_params(direct_schema):
    params = direct_schema.get("reward_params") or {}
    # Per ayg_env_cfg.py:135-137
    assert params.get("feet_air_time_threshold") == 0.4
    assert params.get("base_height_target") == 0.35
    assert params.get("foot_clearance_target") == 0.10


if __name__ == "__main__":
    # Allow `.venv/bin/python test_parse_env.py` for quick interactive runs
    sys.exit(pytest.main([__file__, "-v", "-s"]))
