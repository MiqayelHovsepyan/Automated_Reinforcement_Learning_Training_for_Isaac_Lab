# Automated Reinforcement Learning Training for Isaac Lab with Claude Code

An automated RL training loop powered by [Claude Code](https://claude.ai/claude-code) that trains, evaluates visually **and numerically**, tunes hyperparameters, and iterates — designed to run unattended (e.g., overnight).

Built for quadruped robots trained in [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), but adaptable to any Isaac Lab RL task.

> **v3** (current) — adds grounded prior-art context, a rollout-based verification loop with distribution-shift detection, and an OOD closed-loop validation harness. See the v3 changelog in [AUTO_TRAIN_SETUP.md](AUTO_TRAIN_SETUP.md#whats-new-v3).

## How It Works

Claude Code acts as an expert RL engineer in a loop:

1. **Loads grounded context (v3)** — auto-injects `resources/prior_art/QUADRUPED_PRIOR_ART.md` (cheatsheet from ~100 surveyed Isaac quadruped repos: reward weights, obs conventions, DR ranges, PPO hparams, failure modes) plus a programmatic dump of this task's exact schema (`env_schema.md`) via `parse_env.py`.
2. **Audits** body coverage (every non-foot body must be penalized/terminated) and analyzes reward terms mathematically — now cross-referenced against the prior-art cheatsheet.
3. **Writes** parameter overrides as JSON (no source file edits in Level 1).
4. **Launches** short tuning runs (300–500 iters) via `run_phase.py` as a detached process.
5. **Waits** for completion using `wait_for_phase.py` (blocks until done, shows progress).
6. **Analyzes** three signals: TensorBoard metrics (convergence + suspicious patterns) + side-view video frames (visual gait) + **`eval_report.json` (v3)** with rollout tracking RMSE, contact patterns, energy, survival, posture, action smoothness, and **distribution shift vs the training-time obs normalizer**.
7. **Logs** everything to a journal with detailed metrics tables, CAN/CANNOT verification framework, cross-iteration comparison, and **cross-signal warnings** flagging when training metrics look fine but eval fails (or vice versa).
8. **Repeats** with tuned parameters (one variable per iteration, minimum 5 tuning iterations) until the **8-gate Production Readiness Checklist** passes, then runs a final production training.

```
┌───────────────────────────────────────────────────────────────────┐
│                       Claude Code (AI Agent)                       │
│                                                                    │
│   Load prior_art + env_schema  →  Audit  →  Override  →  Train     │
│   →  TB analyze  →  Visual inspect  →  Numerical eval  →  Decide   │
│       ↑                                                  │         │
│       └──────────── Repeat (5-15x) ──────────────────────┘         │
│                                                                    │
│   8-gate Production Readiness Checklist  →  Production Run  →  Bake│
└───────────────────────────────────────────────────────────────────┘
        │                              ↑
        ▼                              │
┌──────────────┐  phase_report.json  ┌────────────────────────┐
│ run_phase.py │ ───────────────────→│ + evaluation (v3)      │
│  (detached)  │                     │ + cross_signal_warnings│
└──────────────┘                     └────────────────────────┘
    │
    ├── parse_env.py            (v3) introspect task → env_schema.{json,md}
    ├── train_with_overrides.py       Isaac Lab training with JSON overrides
    ├── analyze_metrics.py            TensorBoard → JSON + convergence + suspicious patterns
    ├── play_for_inspection.py        side-view camera, 2-4 robots, gait inspection
    ├── extract_frames.py             MP4 → PNG for visual inspection
    └── evaluate_policy.py      (v3) rollout → tracking RMSE, gait, survival, posture,
                                      action smoothness, distribution-shift vs training obs_normalizer
```

## What's in This Repo

| File | Description |
|------|-------------|
| `auto_train/SKILL.md` | Claude Code skill definition that drives the entire auto-train loop |
| `auto_train/AUTO_TRAIN_SETUP.md` | Setup guide + v3 changelog + troubleshooting |
| `auto_train/docs/closed_loop_ood_test_report.md` | **(v3)** Template for the OOD closed-loop validation report |
| `auto_train/resources/run_phase.py` | Orchestrator: parse_env → train → metrics → play → frames → evaluate_policy → report. Produces `evaluation` and `cross_signal_warnings` keys in phase_report. |
| `auto_train/resources/train_with_overrides.py` | Modified Isaac Lab `train.py` with `--overrides-file` JSON support |
| `auto_train/resources/analyze_metrics.py` | Extracts TensorBoard events into structured JSON with convergence detection, curve shape analysis, and suspicious pattern detection |
| `auto_train/resources/play_for_inspection.py` | Custom play wrapper with side-view camera at robot height, following a single robot (2–4 envs) for visual gait inspection |
| `auto_train/resources/extract_frames.py` | Extracts evenly-spaced PNG frames from rollout videos (uses OpenCV) |
| `auto_train/resources/wait_for_phase.py` | Blocks until training completes, prints final report (replaces sleep-poll loop) |
| `auto_train/resources/parse_env.py` | **(v3)** Programmatic env-config introspection via `gymnasium.registry` → `env_schema.{json,md}` |
| `auto_train/resources/evaluate_policy.py` | **(v3)** Headless rollout eval: tracking RMSE, gait pattern, survival, posture, distribution shift |
| `auto_train/resources/prior_art/QUADRUPED_PRIOR_ART.md` | **(v3)** Distilled cheatsheet (rewards, obs, DR, PPO hparams, failure modes) — auto-loaded at iter 1 |
| `auto_train/resources/prior_art/repos.md` | **(v3)** Full catalog of ~100 Isaac quadruped repos — grep-on-demand reference |
| `auto_train/resources/tests/test_parse_env.py` | **(v3)** ≥2-env unit tests for the env parser (manager-based + direct) |
| `auto_train/resources/__init__.py` | Python package marker |

## Setup

### Prerequisites

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) installed and working
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib >= 3.0.1`)
- [Claude Code](https://claude.ai/claude-code) CLI installed
- An Isaac Lab project with registered RL tasks (e.g., `Isaac-Velocity-Flat-Ayg-v0`)

### Installation

**Single-folder copy** — everything goes into `.claude/skills/`:

```bash
# From your Isaac Lab project root
cp -r /path/to/Automated_Reinforcement_Learning_Training_for_Isaac_Lab/auto_train/ .claude/skills/auto_train/
```

Your project structure should look like:

```
your_isaac_lab_project/
├── scripts/
│   └── rsl_rl/
│       ├── train.py            # (your existing train script)
│       ├── play.py             # (your existing play script)
│       └── cli_args.py         # (your existing CLI args)
├── .claude/
│   └── skills/
│       └── auto_train/
│           ├── SKILL.md
│           ├── docs/
│           │   └── closed_loop_ood_test_report.md   # (v3)
│           ├── resources/
│           │   ├── __init__.py
│           │   ├── run_phase.py
│           │   ├── train_with_overrides.py
│           │   ├── analyze_metrics.py
│           │   ├── play_for_inspection.py
│           │   ├── extract_frames.py
│           │   ├── wait_for_phase.py
│           │   ├── parse_env.py             # (v3)
│           │   ├── evaluate_policy.py       # (v3)
│           │   ├── prior_art/               # (v3)
│           │   │   ├── QUADRUPED_PRIOR_ART.md
│           │   │   └── repos.md
│           │   └── tests/
│           │       └── test_parse_env.py    # (v3)
│           └── experiments/         # (created automatically)
│               └── .scratch/
└── logs/                            # (created automatically)
```

Then install Python dependencies (inside your Isaac Lab venv):

```bash
source .venv/bin/activate
uv pip install tbparse opencv-python-headless
```

(Optional) Run the parser unit tests to verify the install:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest .claude/skills/auto_train/resources/tests/test_parse_env.py -v
```

See [AUTO_TRAIN_SETUP.md](AUTO_TRAIN_SETUP.md) for detailed step-by-step instructions and the v3 changelog.

## Usage

### Start an Auto-Train Session

In Claude Code, invoke the skill:

```
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB
```

**Arguments format:** `<task_name> level <1|2> on <device_info> [--server=user@host:port --remote-path=...] [optional notes]`

- **Task name**: Your registered Isaac Lab task ID
- **Level 1** (Reward Tuning): Only JSON overrides — reward weights and PPO hyperparams. No source files edited. Safe and reversible.
- **Level 2** (Full Tuning): Full autonomy — can edit source configs, add/remove rewards, write new reward functions.
- **Device info**: GPU model and VRAM, used to scale `num_envs` appropriately
- **`--server` / `--remote-path`** (optional): Hybrid local/server workflow — training runs on a GPU server via SSH, rendering-dependent steps (`parse_env`, `play`, `evaluate_policy`) run locally
- **Optional notes**: Domain knowledge, known issues, focus areas (e.g., "robot tends to spider-walk", "focus on foot clearance")

### Examples

```
# Basic flat terrain training (local mode)
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 3060 12GB

# Full tuning with domain hints
/auto_train Isaac-Velocity-Rough-Ayg-v0 level 2 on A100 80GB robot tends to stumble on stairs, previous best was 15.0

# Remote mode: training on GPU server, rendering local
/auto_train Isaac-Velocity-Flat-Ayg-v0 level 2 on RTX 4090 24GB --server=root@1.2.3.4:22 --remote-path=/workspace/cf_lab

# Overnight unattended run
claude --dangerously-skip-permissions "/auto_train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB"
```

### Controls During Training

While auto-train is running, you can type:

| Command | Effect |
|---------|--------|
| `stop` | Stop after the current phase completes |
| `level 1` / `level 2` | Switch privilege level mid-session |
| `focus on X` | Redirect Claude's tuning priority (e.g., "focus on foot clearance") |

### Resuming After Disconnect

If a conversation disconnects, start a new one and say:

```
Continue auto-train, read journal at .claude/skills/auto_train/experiments/<name>/journal.md
```

Claude will read the journal, pick up from the last iteration, and continue.

## Device Scaling

The skill automatically scales `num_envs` and iteration counts based on your device:

| Device | VRAM | Recommended `num_envs` |
|--------|------|----------------------|
| Laptop GPU (RTX 3060) | 12 GB | ~2048 |
| Desktop GPU (RTX 4090) | 24 GB | ~4096 |
| Cloud GPU (A100) | 80 GB | 4096–8192 |

## Tuning Strategy

Auto-train uses **short tuning runs** (300–500 iterations) to test hypotheses quickly, with a minimum of **5 tuning iterations** before considering a production run. Each iteration produces three sources of signal:

1. **TensorBoard metrics** — convergence detection + suspicious patterns
2. **Visual inspection** — side-view frames with CAN/CANNOT framework
3. **Numerical eval (v3)** — rollout tracking RMSE, gait pattern, survival, posture, distribution-shift vs training obs_normalizer

A formal **8-gate Production Readiness Checklist** must pass before the final production run:

1. Body coverage audit (all non-foot bodies penalized/terminated)
2. Velocity tracking above threshold (the primary task objective)
3. Visual gait quality confirmed via side-view camera
4. No suspicious reward hacking patterns detected
5. Sufficient tuning iterations completed (≥ 5)
6. Key metrics showing convergence
7. **Numerical eval passes (v3)** — `lin_vel_xy_rmse < 0.25`, `yaw_rate_rmse < 0.3`, `alive_at_1000 > 0.9`, `upright_frac > 0.95`, `arrhythmic == false`
8. **Distribution shift within bounds (v3)** — `obs_shift_magnitude < 1.0` vs training-time obs_normalizer

## How the Override System Works

Level 1 tuning uses JSON override files to modify parameters without touching source code:

```json
{
  "rewards.track_lin_vel_xy_exp.weight": 2.0,
  "rewards.lin_vel_z_l2.weight": -1.5,
  "agent.learning_rate": 0.0003,
  "agent.entropy_coef": 0.008
}
```

- **Flat dot-path keys** map to `@configclass` attribute paths (e.g., `rewards.gait.weight`)
- `agent.*` prefixed keys apply to the RSL-RL agent config
- All other keys apply to the environment config
- Type casting is automatic (matches existing config types)
- Original source files remain untouched
- **Pre-flight validation** catches format errors (nested dicts, invalid JSON) before Isaac Sim boots

## Output Structure

Each auto-train session produces:

```
.claude/skills/auto_train/experiments/<experiment_name>/
└── journal.md                    # Full log of every iteration with detailed metrics + cross-signal warnings

.claude/skills/auto_train/experiments/.scratch/
├── env_schema.json               # (v3) parsed task schema (machine-readable)
├── env_schema.md                 # (v3) parsed task schema (Claude reads this)
└── current_phase_report.json     # external progress poll target

logs/rsl_rl/<task>/<timestamp>/
├── model_*.pt                    # Checkpoints
├── params/
│   ├── env.yaml                  # Full env config (with overrides applied)
│   ├── agent.yaml                # Full agent config
│   └── overrides.json            # Raw override file used
├── metrics.json                  # Extracted TensorBoard metrics with convergence analysis
├── phase_report.json             # Full phase report — now with evaluation + cross_signal_warnings
├── eval_report.json              # (v3) Rollout-based numerical eval
├── frames/
│   ├── frame_001.png–012.png     # Visual inspection frames (side-view)
│   └── frames_info.json          # Frame manifest
└── videos/
    ├── play/*.mp4                # Side-view policy rollout video
    └── eval/*.mp4                # (v3) Terrain-aware eval rollout video
```

## Key Design Decisions

### v3 additions (this version)

- **Grounded prior-art context**: `QUADRUPED_PRIOR_ART.md` distills surveyed-community ranges (reward weights, obs conventions, DR, PPO hparams, failure modes) from ~100 Isaac quadruped repos. Auto-injected at iteration 1 so Claude reasons from prior art, not from scratch each time.
- **Programmatic env schema**: `parse_env.py` instantiates the task's `@configclass` tree and dumps a structured `env_schema.{json,md}`. Cross-referenced against the cheatsheet — any reward weight outside surveyed range is flagged as a hypothesis candidate.
- **Numerical verification loop**: `evaluate_policy.py` runs the policy headlessly for 1000 steps and reports tracking RMSE, foot-contact pattern (duty cycles + regularity), energy proxy, survival horizons, posture (base height std + upright fraction), action smoothness. Terrain-aware camera (side-view for flat, elevated side-back for rough).
- **Distribution-shift check**: Loads the RSL-RL `obs_normalizer` running mean/std from the checkpoint, z-scores eval-time observations against it. `obs_shift_magnitude > 1.0` means the policy is operating outside its training distribution — caught before deployment.
- **Cross-signal warnings**: `phase_report.cross_signal_warnings` flags `clean_training_metrics_but_failed_eval` (eval catches what training reward missed) and `training_anomaly_with_distribution_shift` (suspicious training pattern + obs shift = high-confidence reward gaming).
- **OOD closed-loop validation**: Sabotaged baseline env (`Isaac-Velocity-Flat-Ayg-OOD-v0` in the host project) — running the tuner against it cold should pull weights back to within prior-art ranges autonomously. Report template at `docs/closed_loop_ood_test_report.md`.
- **8-gate Production Readiness Checklist**: Adds two v3 gates (numerical eval passes, distribution shift within bounds) on top of the v2 six.

### v2 + base design (still in effect)

- **Pre-training body coverage audit**: Before training, every robot body is verified to be covered by either termination or penalty — prevents reward exploitation through uncovered body parts (e.g., hip-walking).
- **Pre-training reward analysis**: Reward terms are mathematically analyzed for feasibility — prevents wasting iterations on misconfigured thresholds.
- **Short tuning, long production**: Tuning runs use 300–500 iterations for quick hypothesis testing. Production runs use convergence-appropriate iteration counts.
- **Convergence detection**: `analyze_metrics.py` detects where each metric plateaued, whether it converged too early, and classifies curve shapes.
- **Suspicious pattern detection**: Automatically flags reward hacking indicators (gait gaming velocity tracking, high total reward with low tracking contribution, early flatlined tracking terms).
- **Side-view visual inspection**: `play_for_inspection.py` provides a camera at robot height following a single robot (2–4 envs), replacing the useless overhead view of 50+ robots.
- **Honest visual assessment**: The skill requires explicit CAN/CANNOT verification framework — Claude must state what the camera angle can and cannot show, never claiming "good gait" without evidence.
- **Velocity tracking gate**: Velocity tracking is treated as the primary objective — poor tracking is a blocking issue that must be fixed before other tuning.
- **Hardened unattended operation**: The skill NEVER blocks on user input — no AskUserQuestion, no file creation confirmations, no option selection. All decisions are autonomous with reasoning logged in the journal.
- **Progress-aware waiting**: `run_phase.py` writes progress (iteration, reward, ETA) to the report file during training. `wait_for_phase.py` blocks until done instead of polling.
- **Pre-flight validation**: Override JSON is validated for format errors before Isaac Sim boots.
- **Detached processes**: Training runs via `nohup setsid` to survive Claude Code timeouts and conversation disconnects.
- **Journal-based resumability**: Everything is logged to `journal.md`, so training can resume from any point after a disconnect.
- **Local + Remote modes**: Local does everything on one machine. Remote sends training to a GPU server via SSH while keeping rendering-dependent steps (`parse_env`, `play`, `evaluate_policy`) on the local machine.
- **Self-contained folder**: The entire auto-train system lives in `.claude/skills/auto_train/` — single copy to set up on any project.

## License

BSD-3-Clause (following Isaac Lab's license)
