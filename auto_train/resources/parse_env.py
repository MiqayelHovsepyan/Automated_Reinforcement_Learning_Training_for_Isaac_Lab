# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Programmatic env-config parser for auto_train.

Takes a registered task ID (e.g., Isaac-Velocity-Flat-Ayg-v0), instantiates the
@configclass env config (and rsl_rl agent config) and emits:

  - env_schema.json — machine-readable structured dump
  - env_schema.md   — markdown rendering for Claude's iteration-1 context

Used by run_phase.py before training so the tuner sees the full reward / obs /
action / event / curriculum surface alongside QUADRUPED_PRIOR_ART.md.

Note: instantiating an Isaac Lab @configclass transitively pulls in `isaaclab`,
which imports `pxr` (USD bindings) — so this script launches the Isaac Sim
AppLauncher in headless mode just like `play_for_inspection.py`. Boot cost is
~30 s; `run_phase.py` only invokes parse_env once per phase (or skips if a
schema already exists in the scratch dir).

Usage:
    python parse_env.py --task Isaac-Velocity-Flat-Ayg-v0 \
        --output-json env_schema.json --output-md env_schema.md
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


# ─── CLI + AppLauncher (must come before isaaclab core imports) ───

parser = argparse.ArgumentParser(description="Parse a registered Isaac Lab env config to JSON + Markdown.")
parser.add_argument("--task", required=True, help="Registered gym task id, e.g. Isaac-Velocity-Flat-Ayg-v0")
parser.add_argument("--output-json", default=None, help="Path for env_schema.json (default: ./env_schema.json)")
parser.add_argument("--output-md", default=None, help="Path for env_schema.md (default: ./env_schema.md)")
AppLauncher.add_app_launcher_args(parser)
_args_cli, _hydra_args = parser.parse_known_args()
# Always force headless — we don't need rendering for static config introspection
_args_cli.headless = True
sys.argv = [sys.argv[0]] + _hydra_args

app_launcher = AppLauncher(_args_cli)
simulation_app = app_launcher.app

"""Rest of imports — must come after AppLauncher."""

import dataclasses
import importlib
import json
from typing import Any


def _safe_qualname(obj: Any) -> str:
    if obj is None:
        return "None"
    mod = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or repr(obj)
    return f"{mod}.{name}" if mod else str(name)


def _to_jsonable(value: Any, _depth: int = 0) -> Any:
    if _depth > 6:
        return f"<truncated:{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in value.items()}
    if callable(value):
        return _safe_qualname(value)
    if hasattr(value, "name") and hasattr(value, "body_names"):
        return {
            "_type": "SceneEntityCfg",
            "name": getattr(value, "name", None),
            "body_names": getattr(value, "body_names", None),
            "joint_names": getattr(value, "joint_names", None),
        }
    if dataclasses.is_dataclass(value):
        return {f.name: _to_jsonable(getattr(value, f.name, None), _depth + 1) for f in dataclasses.fields(value)}
    if hasattr(value, "__dict__"):
        try:
            return {k: _to_jsonable(v, _depth + 1) for k, v in vars(value).items() if not k.startswith("_")}
        except Exception:
            return repr(value)
    return repr(value)


def _term_to_dict(term: Any, term_name: str) -> dict:
    info: dict[str, Any] = {"name": term_name}
    # `func` for RewTerm/ObsTerm/EventTerm/DoneTerm/CurrTerm,
    # `class_type` for ActionTermCfg-derived classes.
    for attr in ("func", "class_type", "weight", "mode", "interval_range_s", "time_out",
                 "clip", "noise", "params", "scale", "use_default_offset", "asset_name",
                 "joint_names", "body_names", "command_name"):
        if hasattr(term, attr):
            val = getattr(term, attr)
            info[attr] = _to_jsonable(val)
    keep = {"name", "func", "class_type", "weight"}
    return {k: v for k, v in info.items() if v is not None or k in keep}


def _is_term_like(term: Any) -> bool:
    """Heuristic: is this configclass a manager Term (vs a nested group cfg)?"""
    return any(hasattr(term, a) for a in ("func", "class_type"))


def _walk_manager_group(group: Any) -> list[dict]:
    if group is None:
        return []
    terms: list[dict] = []
    if dataclasses.is_dataclass(group):
        for f in dataclasses.fields(group):
            term = getattr(group, f.name, None)
            if term is None:
                continue
            if _is_term_like(term):
                terms.append(_term_to_dict(term, f.name))
            elif dataclasses.is_dataclass(term):
                nested = _walk_manager_group(term)
                if nested:
                    terms.append({"name": f.name, "_subgroup": nested})
    return terms


def _control_freq_hz(cfg: Any) -> float | None:
    sim = getattr(cfg, "sim", None)
    dt = getattr(sim, "dt", None) if sim is not None else None
    decimation = getattr(cfg, "decimation", None)
    if dt and decimation:
        try:
            return round(1.0 / (float(dt) * int(decimation)), 3)
        except Exception:
            return None
    return None


def _direct_env_summary(cfg: Any) -> dict:
    out: dict[str, Any] = {}
    for attr in ("episode_length_s", "decimation", "action_scale", "action_space", "observation_space", "state_space"):
        if hasattr(cfg, attr):
            out[attr] = _to_jsonable(getattr(cfg, attr))
    out["control_freq_hz"] = _control_freq_hz(cfg)
    reward_scales: dict[str, float] = {}
    reward_params: dict[str, Any] = {}
    for attr in dir(cfg):
        if attr.startswith("_") or attr in out:
            continue
        try:
            val = getattr(cfg, attr)
        except Exception:
            continue
        if attr.endswith("_reward_scale") and isinstance(val, (int, float)):
            reward_scales[attr] = float(val)
        elif attr.endswith(("_threshold", "_target")) and isinstance(val, (int, float)):
            reward_params[attr] = float(val)
    out["reward_scales"] = reward_scales
    out["reward_params"] = reward_params
    out["events"] = _walk_manager_group(getattr(cfg, "events", None))
    return out


def parse_env_cfg(task_id: str) -> dict:
    """Instantiate the env cfg for `task_id` and return a structured schema."""
    importlib.import_module("cf_lab.tasks")
    import gymnasium as gym

    spec = gym.spec(task_id)
    entry_point = spec.kwargs.get("env_cfg_entry_point")
    if not entry_point:
        raise RuntimeError(f"Task {task_id!r} has no env_cfg_entry_point")
    rsl_rl_ep = spec.kwargs.get("rsl_rl_cfg_entry_point")

    if ":" in entry_point:
        mod_name, cls_name = entry_point.split(":", 1)
    else:
        mod_name, _, cls_name = entry_point.rpartition(".")
    cfg_cls = getattr(importlib.import_module(mod_name), cls_name)
    cfg = cfg_cls()

    schema: dict[str, Any] = {
        "task": task_id,
        "env_cfg_entry_point": entry_point,
        "rsl_rl_cfg_entry_point": rsl_rl_ep,
        "env_cfg_class": _safe_qualname(cfg_cls),
        "is_manager_based": dataclasses.is_dataclass(getattr(cfg, "rewards", None)),
        "control_freq_hz": _control_freq_hz(cfg),
        "decimation": getattr(cfg, "decimation", None),
        "sim_dt": getattr(getattr(cfg, "sim", None), "dt", None),
        "episode_length_s": getattr(cfg, "episode_length_s", None),
    }

    if schema["is_manager_based"]:
        schema["rewards"] = _walk_manager_group(getattr(cfg, "rewards", None))
        schema["actions"] = _walk_manager_group(getattr(cfg, "actions", None))
        schema["terminations"] = _walk_manager_group(getattr(cfg, "terminations", None))
        schema["events"] = _walk_manager_group(getattr(cfg, "events", None))
        schema["curriculum"] = _walk_manager_group(getattr(cfg, "curriculum", None))
        schema["commands"] = _walk_manager_group(getattr(cfg, "commands", None))
        obs_cfg = getattr(cfg, "observations", None)
        obs: dict[str, Any] = {}
        if obs_cfg is not None and dataclasses.is_dataclass(obs_cfg):
            for f in dataclasses.fields(obs_cfg):
                group = getattr(obs_cfg, f.name, None)
                if group is None:
                    continue
                obs[f.name] = {
                    "terms": _walk_manager_group(group),
                    "enable_corruption": getattr(group, "enable_corruption", None),
                }
        schema["observations"] = obs
        scene = getattr(cfg, "scene", None)
        if scene is not None:
            schema["scene"] = {
                "num_envs": getattr(scene, "num_envs", None),
                "env_spacing": getattr(scene, "env_spacing", None),
                "terrain_type": getattr(getattr(scene, "terrain", None), "terrain_type", None),
                "height_scanner_present": getattr(scene, "height_scanner", None) is not None,
            }
    else:
        schema.update(_direct_env_summary(cfg))

    if rsl_rl_ep:
        try:
            if ":" in rsl_rl_ep:
                mod_name, cls_name = rsl_rl_ep.split(":", 1)
            else:
                mod_name, _, cls_name = rsl_rl_ep.rpartition(".")
            agent_cls = getattr(importlib.import_module(mod_name), cls_name)
            agent_cfg = agent_cls()
            schema["agent"] = _to_jsonable(agent_cfg)
        except Exception as e:
            schema["agent_error"] = f"{type(e).__name__}: {e}"

    return schema


def schema_to_markdown(schema: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Env Schema: {schema['task']}")
    lines.append("")
    lines.append(f"- **Config class:** `{schema['env_cfg_class']}`")
    lines.append(f"- **Manager-based:** {schema['is_manager_based']}")
    lines.append(f"- **Control frequency:** {schema.get('control_freq_hz')} Hz "
                 f"(sim_dt={schema.get('sim_dt')}, decimation={schema.get('decimation')})")
    lines.append(f"- **Episode length:** {schema.get('episode_length_s')} s")
    lines.append("")

    if schema["is_manager_based"]:
        rewards = schema.get("rewards") or []
        if rewards:
            lines.append("## Rewards")
            lines.append("")
            lines.append("| Name | Weight | Func | Key params |")
            lines.append("|------|-------:|------|------------|")
            for r in rewards:
                params = r.get("params") or {}
                param_summary = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3]) if isinstance(params, dict) else ""
                lines.append(f"| `{r['name']}` | {r.get('weight')} | `{r.get('func')}` | {param_summary} |")
            lines.append("")

        obs = schema.get("observations") or {}
        for group_name, group in obs.items():
            lines.append(f"## Observations / {group_name}")
            lines.append("")
            lines.append(f"- **enable_corruption:** {group.get('enable_corruption')}")
            lines.append("")
            lines.append("| Name | Func | Noise | Clip |")
            lines.append("|------|------|-------|------|")
            for t in group.get("terms", []):
                if "_subgroup" in t:
                    continue
                lines.append(f"| `{t['name']}` | `{t.get('func')}` | {t.get('noise')} | {t.get('clip')} |")
            lines.append("")

        actions = schema.get("actions") or []
        if actions:
            lines.append("## Actions")
            lines.append("")
            lines.append("| Name | Class / Func | Scale | Joint names |")
            lines.append("|------|--------------|------:|-------------|")
            for a in actions:
                callable_ref = a.get("class_type") or a.get("func") or "?"
                scale = a.get("scale")
                joints = a.get("joint_names")
                lines.append(f"| `{a['name']}` | `{callable_ref}` | {scale} | {joints} |")
            lines.append("")

        terms = schema.get("terminations") or []
        if terms:
            lines.append("## Terminations")
            lines.append("")
            lines.append("| Name | Func | time_out | Params |")
            lines.append("|------|------|----------|--------|")
            for t in terms:
                params = t.get("params") or {}
                p_summary = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3]) if isinstance(params, dict) else ""
                lines.append(f"| `{t['name']}` | `{t.get('func')}` | {t.get('time_out')} | {p_summary} |")
            lines.append("")

        events = schema.get("events") or []
        if events:
            lines.append("## Events (Domain Randomization)")
            lines.append("")
            lines.append("| Name | Mode | Func | Params |")
            lines.append("|------|------|------|--------|")
            for e in events:
                params = e.get("params") or {}
                p_summary = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3]) if isinstance(params, dict) else ""
                lines.append(f"| `{e['name']}` | {e.get('mode')} | `{e.get('func')}` | {p_summary} |")
            lines.append("")

        curr = schema.get("curriculum") or []
        if curr:
            lines.append("## Curriculum")
            lines.append("")
            for c in curr:
                lines.append(f"- `{c['name']}` — `{c.get('func')}` params={c.get('params')}")
            lines.append("")

        scene = schema.get("scene") or {}
        if scene:
            lines.append("## Scene")
            lines.append("")
            lines.append(f"- **num_envs:** {scene.get('num_envs')}")
            lines.append(f"- **env_spacing:** {scene.get('env_spacing')}")
            lines.append(f"- **terrain_type:** `{scene.get('terrain_type')}`")
            lines.append(f"- **height_scanner_present:** {scene.get('height_scanner_present')}")
            lines.append("")
    else:
        lines.append("## Direct env reward scales")
        lines.append("")
        for k, v in (schema.get("reward_scales") or {}).items():
            lines.append(f"- `{k}` = {v}")
        lines.append("")
        if schema.get("reward_params"):
            lines.append("## Direct env reward parameters")
            lines.append("")
            for k, v in schema["reward_params"].items():
                lines.append(f"- `{k}` = {v}")
            lines.append("")
        lines.append(f"- **action_space:** {schema.get('action_space')}")
        lines.append(f"- **observation_space:** {schema.get('observation_space')}")
        lines.append(f"- **state_space:** {schema.get('state_space')}")
        lines.append("")

    agent = schema.get("agent")
    if isinstance(agent, dict):
        lines.append("## Agent (RSL-RL PPO)")
        lines.append("")
        scalar_fields = {k: v for k, v in agent.items() if isinstance(v, (int, float, str, bool, type(None)))}
        for k, v in scalar_fields.items():
            lines.append(f"- `{k}` = {v}")
        for sub_name in ("algorithm", "policy"):
            sub = agent.get(sub_name)
            if isinstance(sub, dict):
                lines.append("")
                lines.append(f"### {sub_name}")
                for k, v in sub.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        lines.append(f"- `{sub_name}.{k}` = {v}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**Tuner usage:** cross-reference each reward weight above against the canonical "
                 "ranges in `QUADRUPED_PRIOR_ART.md` § 1. Flag any weight outside the surveyed range "
                 "as a hypothesis candidate in the journal.")
    return "\n".join(lines)


def main() -> None:
    try:
        schema = parse_env_cfg(_args_cli.task)
    except Exception as e:
        err = {"task": _args_cli.task, "error": f"{type(e).__name__}: {e}"}
        out_json = _args_cli.output_json or "env_schema.json"
        with open(out_json, "w") as f:
            json.dump(err, f, indent=2)
        print(f"[ERROR] parse_env failed for {_args_cli.task}: {e}", file=sys.stderr)
        sys.exit(1)

    out_json = _args_cli.output_json or "env_schema.json"
    out_md = _args_cli.output_md or "env_schema.md"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(schema, f, indent=2, default=str)
    md = schema_to_markdown(schema)
    with open(out_md, "w") as f:
        f.write(md)
    print(f"[parse_env] wrote {out_json}")
    print(f"[parse_env] wrote {out_md}")


if __name__ == "__main__":
    main()
    simulation_app.close()
