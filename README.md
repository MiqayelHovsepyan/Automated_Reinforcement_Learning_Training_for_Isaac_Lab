[README.md](https://github.com/user-attachments/files/26148987/README.md)
# Automated Reinforcement Training for Isaac Lab

An automated RL training loop powered by [Claude Code](https://claude.ai/claude-code) that trains, evaluates visually, tunes hyperparameters, and iterates — designed to run unattended (e.g., overnight).

Built for quadruped robots trained in [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), but adaptable to any Isaac Lab RL task.

## How It Works

Claude Code acts as an expert RL engineer in a loop:

1. **Reads** your task's env config (rewards, observations, terminations, PPO hyperparams)
2. **Writes** parameter overrides as JSON (no source file edits in Level 1)
3. **Launches** training via `run_phase.py` as a detached process
4. **Polls** for completion every ~2 minutes
5. **Analyzes** TensorBoard metrics + extracted video frames (visual gait inspection)
6. **Logs** everything to a journal for reproducibility and resumability
7. **Repeats** with tuned parameters until satisfied, then runs a final production training

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code (AI Agent)              │
│                                                      │
│   Reason → Write Overrides → Launch → Poll → Analyze │
│       ↑                                    │         │
│       └────────────── Repeat ──────────────┘         │
└─────────────────────────────────────────────────────┘
        │                              ↑
        ▼                              │
┌──────────────┐  metrics + frames  ┌──────────────┐
│ run_phase.py │ ─────────────────→ │ phase_report │
│  (detached)  │                    │    .json     │
└──────────────┘                    └──────────────┘
    │
    ├── train_with_overrides.py   (Isaac Lab training with JSON overrides)
    ├── analyze_metrics.py        (TensorBoard → JSON)
    ├── play.py                   (policy rollout + video)
    └── extract_frames.py         (MP4 → PNG for visual inspection)
```

## What's in This Repo

| File | Description |
|------|-------------|
| `auto_train (scripts)/run_phase.py` | Orchestrator: train → metrics → play → frames → report |
| `auto_train (scripts)/train_with_overrides.py` | Modified Isaac Lab `train.py` with `--overrides-file` JSON support |
| `auto_train (scripts)/analyze_metrics.py` | Extracts TensorBoard events into structured JSON (uses `tbparse`) |
| `auto_train (scripts)/extract_frames.py` | Extracts evenly-spaced PNG frames from rollout videos (uses OpenCV) |
| `auto_train (scripts)/__init__.py` | Python package marker |
| `SKILL.md` | Claude Code skill definition that drives the entire auto-train loop |

## Setup

### Prerequisites

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) installed and working
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib >= 3.0.1`)
- [Claude Code](https://claude.ai/claude-code) CLI installed
- An Isaac Lab project with registered RL tasks (e.g., `Isaac-Velocity-Flat-Ayg-v0`)

### Installation

1. **Copy the scripts** into your Isaac Lab project:

```bash
# From your Isaac Lab project root (e.g., cf_lab/)
cp -r /path/to/Automated_Reinforcement_Training_for_Isaac/auto_train\ \(scripts\)/ scripts/auto_train/
```

2. **Copy the skill file** into your Claude Code skills directory:

```bash
mkdir -p .claude/skills/auto-train/
cp /path/to/Automated_Reinforcement_Training_for_Isaac/SKILL.md .claude/skills/auto-train/SKILL.md
```

3. **Install Python dependencies** (inside your Isaac Lab venv):

```bash
source .venv/bin/activate
uv pip install tbparse opencv-python-headless
```

Your project structure should look like:

```
your_isaac_lab_project/
├── scripts/
│   ├── auto_train/
│   │   ├── __init__.py
│   │   ├── run_phase.py
│   │   ├── train_with_overrides.py
│   │   ├── analyze_metrics.py
│   │   └── extract_frames.py
│   └── rsl_rl/
│       ├── train.py          # (your existing train script)
│       └── play.py           # (your existing play script)
├── .claude/
│   └── skills/
│       └── auto-train/
│           └── SKILL.md
├── experiments/              # (created automatically)
└── logs/                     # (created automatically)
```

## Usage

### Start an Auto-Train Session

In Claude Code, invoke the skill:

```
/auto-train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB
```

**Arguments format:** `<task_name> level <1|2> on <device_info> [optional notes]`

- **Task name**: Your registered Isaac Lab task ID
- **Level 1** (Reward Tuning): Only JSON overrides — reward weights and PPO hyperparams. No source files edited. Safe and reversible.
- **Level 2** (Full Tuning): Full autonomy — can edit source configs, add/remove rewards, write new reward functions.
- **Device info**: GPU model and VRAM, used to scale `num_envs` appropriately
- **Optional notes**: Domain knowledge, known issues, focus areas (e.g., "robot tends to spider-walk", "focus on foot clearance")

### Examples

```
# Basic flat terrain training
/auto-train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 3060 12GB

# Full tuning with domain hints
/auto-train Isaac-Velocity-Rough-Ayg-v0 level 2 on A100 80GB robot tends to stumble on stairs, previous best was 15.0

# Overnight unattended run
claude --dangerously-skip-permissions "/auto-train Isaac-Velocity-Flat-Ayg-v0 level 1 on RTX 4090 24GB"
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
Continue auto-train, read journal at experiments/<name>/journal.md
```

Claude will read the journal, pick up from the last iteration, and continue.

## Device Scaling

The skill automatically scales `num_envs` and iteration counts based on your device:

| Device | VRAM | Recommended `num_envs` |
|--------|------|----------------------|
| Laptop GPU (RTX 3060) | 12 GB | ~2048 |
| Desktop GPU (RTX 4090) | 24 GB | ~4096 |
| Cloud GPU (A100) | 80 GB | 4096–8192 |

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

- Dot-path keys map to `@configclass` attribute paths (e.g., `rewards.gait.weight`)
- `agent.*` prefixed keys apply to the RSL-RL agent config
- All other keys apply to the environment config
- Type casting is automatic (matches existing config types)
- Original source files remain untouched

## Output Structure

Each auto-train session produces:

```
experiments/<experiment_name>/
└── journal.md                    # Full log of every iteration

logs/rsl_rl/<task>/
└── <timestamp>/
    ├── model_*.pt                # Checkpoints
    ├── params/
    │   ├── env.yaml              # Full env config (with overrides applied)
    │   ├── agent.yaml            # Full agent config
    │   └── overrides.json        # Raw override file used
    ├── metrics.json              # Extracted TensorBoard metrics
    ├── frames/
    │   ├── frame_001.png–008.png # Visual inspection frames
    │   └── frames_info.json      # Frame manifest
    └── videos/
        └── play/*.mp4            # Policy rollout video
```

## Script Details

### `run_phase.py` — Orchestrator

Coordinates the full pipeline as a single detached process:

1. Launches `train_with_overrides.py` as a subprocess
2. Optionally monitors training with abort criteria (plateau detection, reward collapse, minimum thresholds)
3. Runs `analyze_metrics.py` to extract TensorBoard data
4. Runs `play.py` to generate a rollout video
5. Runs `extract_frames.py` to extract PNG frames for visual inspection
6. Writes a `phase_report.json` (polled by Claude for completion)

**Key flags:**
- `--monitor-interval <seconds>`: Enable real-time abort monitoring
- `--abort-plateau-patience <iters>`: Stop if reward plateaus
- `--abort-min-reward-at <iter:value>`: Stop if reward too low after N iterations
- `--abort-episode-length-drop <ratio>`: Stop if reward collapses from peak
- `--report-path <path>`: External status file for polling

### `train_with_overrides.py` — Training with JSON Overrides

A modified version of the standard Isaac Lab RSL-RL `train.py` that adds `--overrides-file` support. Override files are JSON dicts of dot-path → value pairs applied at runtime.

### `analyze_metrics.py` — TensorBoard to JSON

Reads TensorBoard event files and produces structured JSON with:
- Per-scalar statistics (final, mean, std, max, min, trend)
- Separate reward term breakdown
- Trend analysis (improving / stable / degrading)

### `extract_frames.py` — Video Frame Extraction

Extracts evenly-spaced PNG frames from MP4 videos, skipping the initial 10% of frames (Isaac Sim often renders black initially). Produces a manifest JSON for programmatic access.

## Key Design Decisions

- **Visual inspection is mandatory**: Metrics alone cannot catch reward hacking. Claude reads the extracted frames every iteration to assess gait quality.
- **Detached processes**: Training runs via `nohup setsid` to survive Claude Code timeouts and conversation disconnects.
- **Journal-based resumability**: Everything is logged to `journal.md`, so training can resume from any point after a disconnect.
- **Type-safe overrides**: The override system automatically casts JSON values to match existing config types (bool, int, float, list, etc.).
- **Crash safety**: `run_phase.py` writes crash status to the report file so Claude never gets stuck polling a dead process.

## License

BSD-3-Clause (following Isaac Lab's license)
