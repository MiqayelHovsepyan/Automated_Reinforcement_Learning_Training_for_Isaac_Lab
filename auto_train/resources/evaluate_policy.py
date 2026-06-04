# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical + visual policy evaluation for auto_train verification loop.

Runs a trained checkpoint headlessly with a task-appropriate camera, collects
step-by-step metrics (tracking error, foot contact pattern, energy, survival,
posture, action smoothness) and a distribution-shift check against training-time
observation normalizer stats. Writes:

  - eval_report.json — structured numerical results
  - <log_dir>/videos/play/<...>.mp4 — rollout video

Consumed by run_phase.py and merged into phase_report.json under `evaluation`.

Usage:
    python evaluate_policy.py \
        --task=Isaac-Velocity-Flat-Ayg-Play-v0 \
        --checkpoint=logs/rsl_rl/.../model_500.pt \
        --eval-steps=1000 --num-envs=4 --video --headless \
        --report-path=logs/rsl_rl/.../eval_report.json

Acceptance criteria (issue #2 §2): video + numerical reports per candidate,
plus comparison against normalized training-time data.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher


# Script lives at .claude/skills/auto_train/resources/ — go up 4 levels to cf_lab root
_cf_lab_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_cf_lab_dir, "scripts", "rsl_rl"))
import cli_args  # isort: skip


# ─── CLI ───

parser = argparse.ArgumentParser(description="Evaluate a trained policy with numerical metrics + video.")
parser.add_argument("--task", type=str, required=True, help="Play-variant task id, e.g. Isaac-Velocity-Flat-Ayg-Play-v0")
# NOTE: --checkpoint is provided by cli_args.add_rsl_rl_args() (dest=checkpoint); do not redefine here
# (cli_args drift added --checkpoint, which previously caused an argparse conflict). It is required in practice;
# we validate presence after parsing.
parser.add_argument("--report-path", type=str, required=True, help="Where to write eval_report.json")

parser.add_argument("--eval-steps", type=int, default=1000, help="Steps to roll out per env (default 1000 ≈ 20 s @ 50 Hz)")
parser.add_argument("--num-envs", type=int, default=4, help="Eval envs (small for clear video; metrics aggregate across envs)")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video-length", type=int, default=None, help="Video length in steps. Defaults to --eval-steps.")
parser.add_argument("--terrain-aware-camera", action="store_true", default=True,
                    help="Pick camera based on task name (rough → elevated side-back; flat → side).")
parser.add_argument("--camera-distance", type=float, default=None,
                    help="Override camera distance (m). If unset, terrain-aware default.")
parser.add_argument("--camera-height", type=float, default=None,
                    help="Override camera height (m). If unset, terrain-aware default.")
parser.add_argument("--camera-azimuth", type=float, default=None,
                    help="Override camera azimuth (deg). If unset, terrain-aware default.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent cfg entry point")
parser.add_argument("--seed", type=int, default=None)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required (path to model_*.pt)")

# Force-enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv so hydra doesn't choke
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports — must come after AppLauncher."""

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    ViewerCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import cf_lab.tasks  # noqa: F401


# ─── Helpers ───


def _pick_camera(task_id: str) -> tuple[float, float, float]:
    """Return (distance, height, azimuth_deg) appropriate for the task's terrain."""
    name = task_id.lower()
    if "rough" in name or "parkour" in name:
        # Elevated side-back to see terrain traversal
        return 3.0, 1.2, 120.0
    if "wtw" in name:
        # Side, slightly elevated
        return 2.2, 0.6, 90.0
    # Flat / default
    return 2.0, 0.4, 90.0


def _safe_tensor_to_numpy(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    # rsl-rl / Isaac Lab may hand back (obs, extras) tuples or dict/TensorDict obs.
    if isinstance(t, (tuple, list)):
        return _safe_tensor_to_numpy(t[0]) if len(t) else None
    if hasattr(t, "items"):  # dict / TensorDict — prefer the policy group
        if "policy" in t:
            return _safe_tensor_to_numpy(t["policy"])
        for v in t.values():
            return _safe_tensor_to_numpy(v)
        return None
    try:
        return np.asarray(t)
    except Exception:
        return None


def _extract_obs_normalizer(runner) -> dict | None:
    """Try to pull running mean/var from an RSL-RL obs normalizer.

    Returns dict with `mean`, `std`, `count` arrays, or None if normalization
    isn't enabled / can't be located. Several RSL-RL versions store it
    differently (`obs_normalizer`, `actor_obs_normalizer`, `policy.obs_norm`).
    """
    candidates = []
    for attr in ("obs_normalizer", "actor_obs_normalizer", "critic_obs_normalizer"):
        if hasattr(runner, attr):
            candidates.append(getattr(runner, attr))
    pol = getattr(runner, "policy", None) or getattr(runner, "alg", None)
    if pol is not None:
        for attr in ("obs_normalizer", "obs_norm", "actor_obs_normalizer"):
            if hasattr(pol, attr):
                candidates.append(getattr(pol, attr))
    for norm in candidates:
        if norm is None:
            continue
        mean = getattr(norm, "running_mean", None)
        var = getattr(norm, "running_var", None)
        if mean is None:
            mean = getattr(norm, "mean", None)
        if var is None:
            std = getattr(norm, "std", None)
            if std is not None:
                var_arr = _safe_tensor_to_numpy(std) ** 2
            else:
                var_arr = None
        else:
            var_arr = _safe_tensor_to_numpy(var)
        if mean is not None and var_arr is not None:
            return {
                "mean": _safe_tensor_to_numpy(mean).flatten().tolist(),
                "std": np.sqrt(np.maximum(var_arr, 1e-12)).flatten().tolist(),
                "source": type(norm).__name__,
            }
    return None


def _get_command_target(env) -> np.ndarray | None:
    """Best-effort: read commanded base velocity (xy + yaw) from the env command manager."""
    try:
        cmd_mgr = env.unwrapped.command_manager
        cmd = cmd_mgr.get_command("base_velocity")  # shape (num_envs, 3): vx, vy, wz
        return _safe_tensor_to_numpy(cmd)
    except Exception:
        return None


def _get_base_velocity(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(lin_vel_b, ang_vel_b) in body frame from the robot articulation."""
    try:
        robot = env.unwrapped.scene["robot"]
        lin = _safe_tensor_to_numpy(robot.data.root_lin_vel_b)
        ang = _safe_tensor_to_numpy(robot.data.root_ang_vel_b)
        return lin, ang
    except Exception:
        return None, None


def _get_foot_contacts(env) -> np.ndarray | None:
    """Per-env per-foot contact bool array (num_envs, 4) in LF/RF/LH/RH order."""
    try:
        cs = env.unwrapped.scene["contact_forces"]
        # Find foot body indices
        foot_re = ".*_Foot"
        body_ids, _ = cs.find_bodies(foot_re)
        forces = cs.data.net_forces_w[:, body_ids, :]  # (num_envs, n_feet, 3)
        mag = torch.linalg.norm(forces, dim=-1)  # (num_envs, n_feet)
        return _safe_tensor_to_numpy(mag > 1.0)
    except Exception:
        return None


def _get_joint_state(env) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """(joint_pos, joint_vel, applied_torque) from the robot."""
    try:
        robot = env.unwrapped.scene["robot"]
        jp = _safe_tensor_to_numpy(robot.data.joint_pos)
        jv = _safe_tensor_to_numpy(robot.data.joint_vel)
        # applied_torque is the effective torque; available as data.applied_torque
        jt = _safe_tensor_to_numpy(getattr(robot.data, "applied_torque", None))
        return jp, jv, jt
    except Exception:
        return None, None, None


def _get_base_pose(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(base_height, projected_gravity_z) from the robot."""
    try:
        robot = env.unwrapped.scene["robot"]
        h = _safe_tensor_to_numpy(robot.data.root_pos_w[..., 2])
        pg = _safe_tensor_to_numpy(robot.data.projected_gravity_b[..., 2])
        return h, pg
    except Exception:
        return None, None


# ─── Metric aggregation ───


class _RolloutMetrics:
    """Per-step accumulators that become the eval_report when finalized."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.steps = 0
        self.alive_per_step: list[float] = []  # fraction alive
        self.alive_mask = np.ones(num_envs, dtype=bool)
        # Tracking error
        self.cmd_lin_xy_err_sq: list[float] = []
        self.cmd_lin_z_err_sq: list[float] = []
        self.cmd_yaw_err_sq: list[float] = []
        # Contacts (foot contact sequence; aggregated post-hoc)
        self.contact_seq: list[np.ndarray] = []  # each (num_envs, 4)
        # Energy / torques
        self.energy_sum = 0.0
        self.energy_count = 0
        # Posture
        self.base_heights: list[np.ndarray] = []
        self.proj_grav_z: list[np.ndarray] = []
        # Action smoothness
        self.action_diff_sq_sum = 0.0
        self.action_diff_count = 0
        # Observations (for distribution shift)
        self.obs_samples: list[np.ndarray] = []

    def update(self, env, obs, action, prev_action, dones):
        if dones is not None:
            done_np = _safe_tensor_to_numpy(dones).astype(bool)
            # Mark anyone done as no-longer-alive (latching; first death counts)
            self.alive_mask &= ~done_np
        self.alive_per_step.append(float(self.alive_mask.mean()))

        # Tracking error
        cmd = _get_command_target(env)
        lin_b, ang_b = _get_base_velocity(env)
        if cmd is not None and lin_b is not None and ang_b is not None:
            err_xy = np.linalg.norm(cmd[:, :2] - lin_b[:, :2], axis=-1)
            err_yaw = np.abs(cmd[:, 2] - ang_b[:, 2])
            self.cmd_lin_xy_err_sq.append(float(np.mean(err_xy ** 2)))
            self.cmd_lin_z_err_sq.append(float(np.mean(lin_b[:, 2] ** 2)))
            self.cmd_yaw_err_sq.append(float(np.mean(err_yaw ** 2)))

        # Contacts
        contacts = _get_foot_contacts(env)
        if contacts is not None:
            self.contact_seq.append(contacts.astype(np.uint8))

        # Energy
        _, jv, jt = _get_joint_state(env)
        if jv is not None and jt is not None:
            self.energy_sum += float(np.mean(np.sum(np.abs(jt) * np.abs(jv), axis=-1)))
            self.energy_count += 1

        # Posture
        h, pg = _get_base_pose(env)
        if h is not None:
            self.base_heights.append(h)
        if pg is not None:
            self.proj_grav_z.append(pg)

        # Action smoothness
        if action is not None and prev_action is not None:
            diff = _safe_tensor_to_numpy(action - prev_action)
            self.action_diff_sq_sum += float(np.mean(np.sum(diff ** 2, axis=-1)))
            self.action_diff_count += 1

        # Obs samples — collect first env's obs for distribution shift
        if obs is not None:
            obs_np = _safe_tensor_to_numpy(obs)
            if obs_np is not None and obs_np.ndim >= 2:
                # Subsample every 5 steps to bound memory
                if self.steps % 5 == 0:
                    self.obs_samples.append(obs_np[:1].flatten())

        self.steps += 1

    def finalize(self, train_normalizer: dict | None, foot_names: list[str]) -> dict:
        # Survival horizons
        def horizon(threshold: float) -> int | None:
            for i, a in enumerate(self.alive_per_step):
                if a < threshold:
                    return i
            return None

        alive_500 = self.alive_per_step[min(500, len(self.alive_per_step) - 1)] if self.alive_per_step else None
        alive_1000 = self.alive_per_step[min(1000, len(self.alive_per_step) - 1)] if self.alive_per_step else None

        # Tracking — RMSE
        def rmse(seq: list[float]) -> float | None:
            return float(np.sqrt(np.mean(seq))) if seq else None

        tracking = {
            "lin_vel_xy_rmse": rmse(self.cmd_lin_xy_err_sq),
            "lin_vel_z_rmse": rmse(self.cmd_lin_z_err_sq),
            "yaw_rate_rmse": rmse(self.cmd_yaw_err_sq),
        }

        # Gait: duty cycle + regularity per foot
        gait: dict = {"duty_cycles": {}, "regularity_std": None, "arrhythmic": None}
        if self.contact_seq:
            seq = np.stack(self.contact_seq, axis=0)  # (T, num_envs, n_feet)
            duty = seq.mean(axis=(0, 1))  # (n_feet,)
            for i, name in enumerate(foot_names[: duty.shape[0]]):
                gait["duty_cycles"][name] = float(duty[i])
            # Touchdown intervals per env per foot — std as regularity proxy
            interval_stds = []
            for env_i in range(seq.shape[1]):
                for foot_i in range(seq.shape[2]):
                    contact = seq[:, env_i, foot_i].astype(int)
                    # rising edges
                    diffs = np.diff(contact)
                    touchdown_steps = np.where(diffs == 1)[0]
                    if len(touchdown_steps) > 2:
                        intervals = np.diff(touchdown_steps)
                        interval_stds.append(float(np.std(intervals)))
            gait["regularity_std"] = float(np.mean(interval_stds)) if interval_stds else None
            # Arrhythmic if any duty is <0.15 or >0.85, or regularity_std > 10 steps
            arrhythmic = False
            for v in duty:
                if v < 0.15 or v > 0.85:
                    arrhythmic = True
                    break
            if gait["regularity_std"] is not None and gait["regularity_std"] > 10.0:
                arrhythmic = True
            gait["arrhythmic"] = arrhythmic

        # Posture
        posture: dict = {}
        if self.base_heights:
            heights = np.concatenate([h.flatten() for h in self.base_heights])
            posture["base_height_mean"] = float(np.mean(heights))
            posture["base_height_std"] = float(np.std(heights))
        if self.proj_grav_z:
            pg_z = np.concatenate([p.flatten() for p in self.proj_grav_z])
            # projected_gravity_b[z] near -1 when upright (gravity points down in body frame)
            posture["upright_frac"] = float(np.mean(pg_z < -0.85))

        # Action smoothness
        action_smoothness = (self.action_diff_sq_sum / self.action_diff_count) if self.action_diff_count else None

        # Distribution shift
        distribution_shift: dict = {}
        if train_normalizer is not None and self.obs_samples:
            eval_obs = np.stack(self.obs_samples, axis=0)  # (T_sub, obs_dim)
            eval_mean = eval_obs.mean(axis=0)
            train_mean = np.asarray(train_normalizer["mean"])
            train_std = np.asarray(train_normalizer["std"])
            # Align dims (in case of mismatch, take min)
            n = min(len(train_mean), eval_mean.shape[0])
            shifts = np.abs(eval_mean[:n] - train_mean[:n]) / np.maximum(train_std[:n], 1e-6)
            distribution_shift["obs_shift_magnitude"] = float(np.mean(shifts))
            # Top-K shifted dims
            topk = np.argsort(shifts)[::-1][:10]
            distribution_shift["top_shifted_dims"] = [
                {"idx": int(i), "shift": float(shifts[i])} for i in topk
            ]
            distribution_shift["obs_dim"] = int(n)
            distribution_shift["normalizer_source"] = train_normalizer.get("source")
        elif self.obs_samples:
            distribution_shift["obs_shift_magnitude"] = None
            distribution_shift["note"] = "training-time normalizer not found in checkpoint; eval-only stats reported"
            eval_obs = np.stack(self.obs_samples, axis=0)
            distribution_shift["eval_obs_mean_l2"] = float(np.linalg.norm(eval_obs.mean(axis=0)))

        warnings: list[str] = []
        if tracking.get("lin_vel_xy_rmse") is not None and tracking["lin_vel_xy_rmse"] > 0.25:
            warnings.append("lin_vel_xy_rmse_above_threshold")
        if tracking.get("yaw_rate_rmse") is not None and tracking["yaw_rate_rmse"] > 0.3:
            warnings.append("yaw_rate_rmse_above_threshold")
        if alive_1000 is not None and alive_1000 < 0.9:
            warnings.append("survival_at_1000_below_threshold")
        if gait.get("arrhythmic"):
            warnings.append("arrhythmic_gait")
        if posture.get("upright_frac") is not None and posture["upright_frac"] < 0.95:
            warnings.append("upright_frac_below_threshold")
        if distribution_shift.get("obs_shift_magnitude") and distribution_shift["obs_shift_magnitude"] > 1.0:
            warnings.append("obs_shift_magnitude_above_threshold")

        return {
            "tracking": tracking,
            "gait": gait,
            "energy_proxy": (self.energy_sum / self.energy_count) if self.energy_count else None,
            "survival": {
                "alive_at_500_steps": alive_500,
                "alive_at_1000_steps": alive_1000,
                "horizon_50pct_step": horizon(0.5),
                "horizon_95pct_step": horizon(0.95),
            },
            "posture": posture,
            "action_smoothness": action_smoothness,
            "distribution_shift": distribution_shift,
            "warnings": warnings,
            "total_steps": self.steps,
            "num_envs": self.num_envs,
        }


# ─── Main rollout ───


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    # ─── Camera ───
    dist, height, az = _pick_camera(args_cli.task)
    if args_cli.camera_distance is not None:
        dist = args_cli.camera_distance
    if args_cli.camera_height is not None:
        height = args_cli.camera_height
    if args_cli.camera_azimuth is not None:
        az = args_cli.camera_azimuth
    az_rad = math.radians(az)
    cam_x = dist * math.cos(az_rad)
    cam_y = -dist * math.sin(az_rad)
    cam_z = height
    env_cfg.viewer = ViewerCfg(
        eye=(cam_x, cam_y, cam_z),
        lookat=(0.0, 0.0, 0.25),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    print(f"[EVAL] Camera: dist={dist} height={height} azimuth={az}° (eye=({cam_x:.2f},{cam_y:.2f},{cam_z:.2f}))")
    print(f"[EVAL] Eval steps={args_cli.eval_steps}, num_envs={args_cli.num_envs}")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_length = args_cli.video_length or args_cli.eval_steps
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[EVAL] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    train_normalizer = _extract_obs_normalizer(runner)
    if train_normalizer is None:
        print("[EVAL] Note: no training-time obs normalizer found in checkpoint.")
    else:
        print(f"[EVAL] Obs normalizer loaded from: {train_normalizer.get('source')}")

    # Foot names in canonical order
    foot_names = ["LF", "RF", "LH", "RH"]

    metrics = _RolloutMetrics(num_envs=args_cli.num_envs)
    obs = env.get_observations()
    prev_action = None

    with torch.inference_mode():
        for step in range(args_cli.eval_steps):
            action = policy(obs)
            obs_next, _, dones, _ = env.step(action)
            metrics.update(env, obs, action, prev_action, dones)
            policy_nn.reset(dones)
            prev_action = action
            obs = obs_next
            if step % 100 == 0:
                print(f"[EVAL] step {step}/{args_cli.eval_steps}, alive={metrics.alive_per_step[-1]:.2f}")

    report = metrics.finalize(train_normalizer, foot_names)
    report.update({
        "task": args_cli.task,
        "checkpoint": resume_path,
        "log_dir": log_dir,
        "video_path": None,  # filled in by run_phase.py after the run
    })

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.report_path)) or ".", exist_ok=True)
    with open(args_cli.report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[EVAL] Report written: {args_cli.report_path}")

    # Print a quick health summary so the calling subprocess captures it in stdout
    track = report.get("tracking", {})
    surv = report.get("survival", {})
    print(
        "[EVAL_SUMMARY] "
        f"lin_xy_rmse={track.get('lin_vel_xy_rmse')} yaw_rmse={track.get('yaw_rate_rmse')} "
        f"alive_1000={surv.get('alive_at_1000_steps')} "
        f"warnings={report.get('warnings')}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
