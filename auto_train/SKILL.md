---
name: auto_train
description: Start an automated RL training loop for the AYG quadruped robot. Supports local and hybrid local/server (SSH) workflows. Trains, evaluates visually, tunes, and repeats overnight.
argument-hint: <task_name> level <1|2> on <device_info> [--server=user@host:port --remote-path=/path/to/cf_lab] [optional notes]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent
model: opus
effort: high
---

# Auto-Train Mode

You are an expert RL engineer running an automated training loop for the AYG quadruped robot.

**Session:** Auto-train `$ARGUMENTS`

## Setup

1. Parse the arguments: extract task name, level (1 or 2), device info, and any additional notes. Check for `--server=user@host:port` and `--remote-path=/path/to/cf_lab` flags. If `--server` is present, enable **remote mode** (training on server via SSH, play/analysis locally). Parse the server argument into `$SSH_USER_HOST` (e.g., `root@79.117.54.182`) and `$SSH_PORT` (e.g., `19779`). If no port is specified, default to `22`. Set `$REMOTE_CF_LAB` from `--remote-path`. If `--server` is absent, use **local mode** (all local, unchanged behavior). The user may include domain knowledge, known issues, focus areas, or hints (e.g., "robot tends to spider-walk", "previous best was 15.0", "focus on foot clearance"). Use these notes to guide your initial analysis and tuning strategy.
2. `cd cf_lab` — all commands run from the cf_lab directory. Set `$LOCAL_CF_LAB` to this absolute path.
3. Ensure dependencies: `source .venv/bin/activate && uv pip install tbparse opencv-python-headless`
4. Ensure directories: `mkdir -p .claude/skills/auto_train/experiments/.scratch`
5. **Read prior-art cheatsheet** (`.claude/skills/auto_train/resources/prior_art/QUADRUPED_PRIOR_ART.md`) and the **parsed env schema** (`.claude/skills/auto_train/experiments/.scratch/env_schema.md`). Together these are your grounded context: surveyed-community reward/obs/DR ranges (cheatsheet) plus this specific task's exact structure (schema). Cross-reference each reward weight in the schema against the cheatsheet's typical range — any weight outside the surveyed range is a hypothesis candidate. Use the deep reference at `resources/prior_art/repos.md` if you need to drill into a specific repo's choices.

   **In local mode**, `run_phase.py` auto-generates `env_schema.{md,json}` in Step 0 on the first iteration. **In remote mode**, generate it locally once during Setup before any training (the server doesn't have the needed Isaac Sim AppLauncher rendering path, and even if it did, the resulting file wouldn't get rsync'd back). Run:

   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     .venv/bin/python .claude/skills/auto_train/resources/parse_env.py \
       --task=<TASK> \
       --output-json=.claude/skills/auto_train/experiments/.scratch/env_schema.json \
       --output-md=.claude/skills/auto_train/experiments/.scratch/env_schema.md \
       --headless
   ```

   **At Level 2, after editing source configs:** delete `.claude/skills/auto_train/experiments/.scratch/env_schema.{md,json}` so the next iteration regenerates the schema against your edits (local mode does this in `run_phase.py`; remote mode means re-running the local parse_env command above). Isaac Sim boot is ~30–60 s — only pay it when the schema actually changed.
6. **(Remote mode only)** Perform one-time remote setup — see Remote Mode section below.

## Levels

**Level 1 — Reward Tuning:** Change reward weights and PPO hyperparams via JSON overrides. No source files edited. Safe and reversible.

**Level 2 — Overall Training Tuning:** Full autonomy — add/remove rewards, terminations, observations, write new reward functions, staged curriculum. Source config files are edited directly on the current branch.

## Remote Mode (Hybrid Local/Server Workflow)

When `--server` and `--remote-path` are provided, training runs on the remote server (GPU, no display), while play, frame extraction, metrics reading, and visual inspection run locally (Isaac Sim + rendering). **If `--server` is absent, skip this entire section — everything runs locally as before.**

**Requirements:**
- SSH key-based authentication configured (no password prompts — required for unattended mode)
- The remote server has cf_lab set up with a working `.venv/` and GPU
- The local machine has Isaac Sim + cf_lab fully set up with rendering capability
- Both machines have the same task registered (same `cf_lab` package)

**Variables (set once in setup, used throughout):**
- `$SSH_USER_HOST` — e.g., `root@79.117.54.182`
- `$SSH_PORT` — e.g., `19779`
- `$REMOTE_CF_LAB` — absolute path to cf_lab on the server, e.g., `/workspace/cf_lab`
- `$LOCAL_CF_LAB` — absolute path to cf_lab locally (the cwd after step 2)

### Two rsync Commands (Push and Pull)

All file transfer uses these two patterns:

**Push local → server** (before each training run — syncs everything except logs/venv/.git):
```bash
rsync -avz --progress -e "ssh -p $SSH_PORT" \
    --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='logs' \
    $LOCAL_CF_LAB/ \
    $SSH_USER_HOST:$REMOTE_CF_LAB/
```
This syncs everything in one command: override JSONs in `.claude/skills/auto_train/experiments/.scratch/`, auto_train resource scripts, source config edits (Level 2), task definitions.

**Pull logs server → local** (after training completes):
```bash
rsync -avz --progress -e "ssh -p $SSH_PORT" \
    $SSH_USER_HOST:$REMOTE_CF_LAB/logs/ \
    $LOCAL_CF_LAB/logs/
```
This brings all new training logs: checkpoints, TensorBoard events, metrics.json, phase_report.json — everything needed for local play and analysis.

### One-Time Remote Setup (per session)

Perform these steps once at the start of a remote-mode session (setup step 6):

1. **Test SSH connectivity:**
   ```bash
   ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=10 $SSH_USER_HOST "echo 'SSH OK'"
   ```
   If this fails, stop immediately. SSH key auth must be working.

2. **Verify server venv:**
   ```bash
   ssh -p $SSH_PORT $SSH_USER_HOST "ls $REMOTE_CF_LAB/.venv/bin/python"
   ```

3. **Ensure tbparse on server** (needed by `analyze_metrics.py` which runs as part of `run_phase.py`):
   ```bash
   ssh -p $SSH_PORT $SSH_USER_HOST "cd $REMOTE_CF_LAB && .venv/bin/python -c 'import tbparse; print(\"tbparse OK\")'" 2>&1
   ```
   If missing: `ssh -p $SSH_PORT $SSH_USER_HOST "cd $REMOTE_CF_LAB && source .venv/bin/activate && uv pip install tbparse"`

4. **Initial push** to sync auto_train scripts and configs to server:
   ```bash
   rsync -avz --progress -e "ssh -p $SSH_PORT" \
       --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='logs' \
       $LOCAL_CF_LAB/ \
       $SSH_USER_HOST:$REMOTE_CF_LAB/
   ```

### Remote Mode Error Handling

- **SSH failure** → stop, report error. Cannot proceed without SSH.
- **rsync push failure** → retry once. Do NOT train on stale server state.
- **rsync pull failure** → retry once. Verify `phase_report.json` exists locally before proceeding to play.
- **SSH drop during wait** → training survives (detached via `nohup setsid`). Reconnect and check report status. Re-run wait if still running.
- **Server disk check** before each training run: `ssh -p $SSH_PORT $SSH_USER_HOST "df -h $REMOTE_CF_LAB | tail -1"`. If <10 GB free, stop and log warning.

## Override JSON Format

Override files use **flat dot-paths** as keys. The values are applied directly to the `@configclass` env config or agent config at runtime.

**Correct format:**
```json
{
  "rewards.track_lin_vel_xy_exp.weight": 2.0,
  "feet_air_time_reward_scale": 4.0,
  "agent.learning_rate": 0.0003,
  "agent.entropy_coef": 0.008,
  "agent.max_iterations": 500
}
```

**Rules:**
- Keys without `agent.` prefix → applied to env config
- Keys with `agent.` prefix → prefix is stripped, then applied to agent config
- **Do NOT nest:** `{"env_cfg": {"key": val}}` is WRONG — use `{"key": val}` directly
- **Do NOT use full path prefixes:** `{"rewards.gait.weight": 30.0}` is correct, `{"env_cfg.rewards.gait.weight": 30.0}` is wrong
- Values are type-cast automatically to match the existing config type (bool, int, float, list, etc.)
- `run_phase.py` validates the JSON format before launching training — nested dicts or invalid JSON are caught immediately

**Resume format:**
- `--resume-from` expects just the **run folder name** (e.g., `2026-03-21_01-52-22`), NOT a full path like `logs/rsl_rl/experiment/2026-03-21_01-52-22`

## How it works

**Local mode:**
1. Claude reads the prior-art cheatsheet (`resources/prior_art/QUADRUPED_PRIOR_ART.md`) AND the parsed env schema (`env_schema.md`, auto-generated by `run_phase.py`). The schema is the ground truth for what reward terms exist; the cheatsheet is the surveyed-community context for what weights/ranges work.
2. Performs the Pre-Training Body Coverage Audit and Pre-Training Reward Analysis, citing the prior-art cheatsheet for each reward weight assessment ("`action_rate_l2 = -0.01` is within surveyed range [-0.02, -0.005] — leave alone").
3. Writes override JSON to `.claude/skills/auto_train/experiments/.scratch/`
4. Launches `.claude/skills/auto_train/resources/run_phase.py` as a detached process via `nohup setsid` (survives any timeout). `run_phase.py` now ALSO invokes `parse_env.py` at start (Step 0) and `evaluate_policy.py` after play (Step 4b), so every iteration produces `env_schema.{json,md}`, `eval_report.json`, and an eval video alongside the existing artifacts.
5. Waits for completion using `wait_for_phase.py` (blocks until done, prints final report)
6. Reads full `phase_report.json` (metrics + `evaluation` key with numerical eval) + PNG frames (visual) + `eval_video.mp4` when done
7. Follows the TensorBoard Analysis Protocol for convergence analysis and per-term breakdown, follows the Visual Inspection Protocol for honest gait assessment, follows the **Numerical Inspection Protocol** for tracking RMSE / contact pattern / energy / survival / distribution shift, and checks the Velocity Tracking Gate
8. Decides what to change (Claude chooses iteration count, abort criteria, param adjustments). Hypotheses MUST cite either the prior-art cheatsheet or a specific signal from the eval_report.
9. Logs everything in `.claude/skills/auto_train/experiments/<name>/journal.md` using the Enhanced Journal Format
10. Repeats until the Production Readiness Checklist passes (minimum 5 successful tuning iterations + 8 gate criteria), then runs a final production training with high iterations
11. Bakes winning weights into source config

**Remote mode** — same steps, but step 4–6 change:
4. Pushes local changes to server (rsync push), then launches `run_phase.py --skip-play --skip-eval --skip-parse-env` on the server via SSH (detached via `nohup setsid`). The server only does training + tensorboard analysis; rendering-dependent steps (parse_env, play, evaluate_policy) all run locally.
5. Waits for completion via SSH using `wait_for_phase.py` on the server
6. Pulls training logs from server (rsync pull), then runs `play_for_inspection.py`, `extract_frames.py`, **and `evaluate_policy.py`** locally. The schema dump (`env_schema.md`) was produced locally once during Setup step 5 since it doesn't depend on the training run. Reads `phase_report.json` (training metrics) + PNG frames (visual) + locally-generated `eval_report.json` (numerical + distribution shift).

All other steps (1–3, 7–11) run locally in both modes. See the Auto-Train Loop section for exact commands.

## Pre-Training Body Coverage Audit (MANDATORY before iteration 1)

Before analyzing reward math, you MUST verify that every robot body is accounted for in either termination conditions, penalty rewards, or is a foot (ground contact surface). **A missing body means the robot can exploit that part for locomotion without penalty.**

### Procedure

1. **Enumerate all robot bodies** from the URDF or asset config. Read the robot description to get the full list of links/bodies.

2. **Classify each body** into one of these categories:
   - **Foot** — intended ground contact surface (appears in foot contact rewards like air_time, gait, foot_slip)
   - **Termination body** — contact triggers episode reset (appears in `terminations.body_contact` or equivalent with `body_names`)
   - **Penalty body** — contact is penalized but does not end the episode (appears in `undesired_contacts` reward or similar penalty term with `body_names`)
   - **UNCOVERED** — not in any of the above lists

3. **Build a coverage table** and log it in the journal:

   ```
   | Body          | Foot? | Termination? | Penalty? | Status   |
   |---------------|-------|-------------|----------|----------|
   | Base          | No    | Yes         | No       | COVERED  |
   | LF_Hip        | No    | No          | Yes      | COVERED  |
   | LF_Thigh      | No    | Yes         | Yes      | COVERED  |
   | LF_Foot       | Yes   | No          | No       | FOOT     |
   | ...           |       |             |          |          |
   ```

4. **Flag any UNCOVERED body.** If a non-foot body is missing from BOTH termination AND penalty lists, this is a **BLOCKING issue**. The robot WILL learn to walk on that body part.
   - At **Level 1**: Log the gap as a critical warning. Note that you cannot fix it (source edits required). Proceed with extreme caution and monitor visual inspection extra carefully for use of the uncovered body.
   - At **Level 2**: Fix the gap before iteration 1 — add the body to either the termination or penalty list.

5. **Verify regex patterns actually match.** Check that patterns like `".*_Hip"`, `".*_Thigh"` resolve to the expected bodies. A typo in the regex could leave bodies uncovered silently.

6. **Check for foot-contact metrics.** Look for TensorBoard metrics that track what percentage of ground contact comes from feet vs other body parts. At Level 2, if no such metric exists, consider adding one — it is the most reliable programmatic check against reward hacking.

## Pre-Training Reward Analysis (MANDATORY before iteration 1)

Before starting any training, you MUST analyze each reward term mathematically:

1. **Read every reward term's formula** from the env config (weight, function, parameters)
2. **Compute theoretical max/min** for each term given robot physical params:
   - Robot dimensions: ~0.3m leg length, ~10kg mass, designed for ~1 m/s walking speed
   - Consider: at target velocity, what is the realistic swing phase duration? foot travel distance? joint velocities?
3. **Check threshold feasibility** — Example: if `feet_air_time` threshold is 0.5s and the robot walks at 1 m/s, each foot would need to travel ~0.5m in the air. For a robot with 0.3m legs, this is physically unrealistic.
4. **Identify terms likely to be negative/zero** and explain why
5. **Log this analysis** in the journal before iteration 1

This analysis prevents wasting iterations on reward terms that are fundamentally misconfigured.

## Tuning Strategy

### Iteration Count Guidance

**Tuning runs** and **production runs** have fundamentally different goals:

- **Tuning runs** test a hypothesis quickly. Use **300–500 training iterations** (not 2000). The goal is to see the *direction* of a change — is the metric improving, flat, or degrading? You do not need convergence; you need signal.
- **Production runs** train to convergence. Use the iteration count where tuning runs showed plateau onset, typically 1500–3000+ iterations depending on the task.

### Reasoning about iteration count

Before each run, explicitly state in the journal:
- **Why this iteration count?** (e.g., "400 iters — enough to see if velocity tracking responds to doubled weight")
- **What signal am I looking for?** (e.g., "error_vel_xy should drop below 1.0 by iter 200 if this weight is sufficient")
- **When would I abort early?** (e.g., "If reward is still negative at iter 150, this configuration is broken")

### Minimum tuning iterations

Plan for **at least 5 tuning iterations** before considering production. Typical flow:

1. **Iteration 1** (300–500 iters): Baseline or batch of obvious fixes from pre-training analysis
2. **Iterations 2–4** (300–500 iters each): Single-variable hypothesis testing
3. **Iteration 5** (500 iters): Confirmation run with best config so far — verify metrics are on track
4. **Iterations 6–10+**: Continue if metrics are not yet satisfactory. There is no rush to production.
5. **Production**: Only after passing the Production Readiness Checklist (see below)

**Do NOT start production after only 2–3 tuning iterations.** Even if metrics look promising, you need enough iterations to:
- Confirm the robot is actually walking (not hip-walking or belly-sliding)
- Verify velocity tracking is functional (not just reward-maximizing through other terms)
- Test sensitivity to at least 3 different parameter changes
- Have one "confirmation" run where you change nothing and verify stability

### What counts as a "tuning iteration"

A tuning iteration is a run where you changed at least one parameter AND analyzed the results. Runs that crashed, failed validation, or were aborted due to bugs do NOT count toward the minimum 5.

## Production Readiness Checklist (ALL must pass)

Before starting a production run, every item below must be explicitly evaluated and logged in the journal as PASS or FAIL with supporting evidence. If ANY item is FAIL, do NOT start production — continue tuning.

### 1. Body Coverage (PASS/FAIL)
- All non-foot bodies are covered by termination OR penalty lists
- Evidence: Body coverage table from pre-training audit (or updated version)

### 2. Velocity Tracking (PASS/FAIL)
- Velocity tracking reward (exp-kernel) is above **0.6** for linear velocity, above **0.3** for angular velocity in the most recent tuning run
- OR: `error_vel_xy` < **0.5** and `error_vel_yaw` < **1.5** if error metrics are available
- If these metrics are not directly available, compute from the velocity reward terms: a velocity tracking reward close to 1.0 (for exp-kernel rewards) indicates low error
- Evidence: Exact metric values from phase_report.json

### 3. Visual Gait Quality (PASS/FAIL)
- Robot walks on its FEET (not hips, belly, shanks, or thighs)
- Robot maintains upright posture (base roughly horizontal)
- Legs show clear swing/stance phases (not dragging or shuffling)
- Evidence: Side-view frames from play_for_inspection.py. State explicitly what you CAN and CANNOT verify from the available camera angle.

### 4. No Reward Hacking Detected (PASS/FAIL)
- `suspicious_patterns` list in phase_report.json is empty, OR all patterns are addressed
- Total reward is not dominated by a single easy-to-exploit term
- Velocity tracking reward is a significant contributor (not drowned out by gait or penalty terms)
- Evidence: Per-term reward breakdown and suspicious_patterns from phase_report.json

### 5. Sufficient Tuning Iterations (PASS/FAIL)
- At least 5 successful tuning iterations completed
- At least 3 different parameters/hypotheses tested
- Evidence: Journal iteration count

### 6. Metric Convergence (PASS/FAIL)
- Key metrics (total reward, velocity tracking, gait reward) show convergence or positive trend in the latest tuning run — not still climbing steeply at the end
- If `curve_shape` is `"still_improving"` for key metrics, the tuning run was too short — run longer or note the risk
- Evidence: convergence_summary from phase_report.json

### 7. Numerical Eval Passes (PASS/FAIL) — auto_train v3

- `evaluation.tracking.lin_vel_xy_rmse` < **0.25** m/s
- `evaluation.tracking.yaw_rate_rmse` < **0.3** rad/s
- `evaluation.survival.alive_at_1000_steps` > **0.9**
- `evaluation.gait.arrhythmic` == **false**
- `evaluation.posture.upright_frac` > **0.95**
- Evidence: `eval_report.json` (merged into phase_report.json under `evaluation`). These thresholds are calibrated against prior-art baselines on AYG flat; revise if needed after first run on a known-good checkpoint and document the revision in the journal.

### 8. Distribution Shift Within Bounds (PASS/FAIL) — auto_train v3

- `evaluation.distribution_shift.obs_shift_magnitude` < **1.0** (eval observations stay within ~1 σ of training-time normalizer stats)
- OR: explicitly document which specific dimension(s) shifted and why it is acceptable (e.g., known terrain difference between train and eval).
- If the checkpoint has no obs normalizer, this criterion is N/A — log it as N/A in the journal, not as PASS.
- Evidence: `evaluation.distribution_shift.top_shifted_dims` from `eval_report.json`.

### Logging the checklist

In the journal, before the production run entry, write:

```
## Production Readiness Assessment

| Check                      | Status | Evidence                                    |
|---------------------------|--------|---------------------------------------------|
| Body coverage             | PASS   | All bodies covered (see iter 0 audit)       |
| Velocity tracking (xy)    | PASS   | track_lin_vel_xy_exp = 0.65 (iter 7)        |
| Velocity tracking (yaw)   | PASS   | track_ang_vel_z_exp = 0.35 (iter 7)         |
| Visual gait quality       | PASS   | Side-view shows foot contact, upright stance |
| No reward hacking         | PASS   | suspicious_patterns: empty                   |
| Sufficient iterations     | PASS   | 8 tuning iterations completed                |
| Metric convergence        | PASS   | Reward plateaued at iter ~350 in last run    |
| Numerical eval            | PASS   | lin_xy_rmse=0.18, yaw_rmse=0.22, alive_1000=1.0, arrhythmic=false, upright=0.99 |
| Distribution shift        | PASS   | obs_shift_magnitude=0.41 (well under 1.0)    |

**Decision:** PROCEED with production run / CONTINUE TUNING because [reason]
```

If you must declare a check as "UNCERTAIN" (e.g., camera angle insufficient for visual confirmation), that counts as FAIL. You must either find a way to verify or continue tuning with explicit acknowledgment of the risk.

## Auto-Train Loop

For each iteration:

1. **Reason** — Based on previous results (or initial analysis for iteration 1), decide what to change and why. Think like an expert RL engineer. **Change ONE variable per iteration** (or a clearly independent group). State your hypothesis and which variable is being tested. Exception: iteration 1 can batch obvious fixes identified in the pre-training analysis. For iteration counts: use **300–500 iterations for tuning runs**. State in the journal WHY you chose the specific count. See Tuning Strategy section.
2. **Write overrides** — Save override JSON to `.claude/skills/auto_train/experiments/.scratch/`.

3. **Run phase** — Launch training as a detached process so it survives any timeout.

   **Local mode:**
   ```bash
   nohup setsid .venv/bin/python .claude/skills/auto_train/resources/run_phase.py \
     --task=<TASK> --max-iterations=<N> --num-envs=<N> --headless \
     --overrides-file=.claude/skills/auto_train/experiments/.scratch/<file>.json \
     --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
     --monitor-interval=60 --abort-min-reward-at=300:-10 --abort-plateau-patience=500 \
     > .claude/skills/auto_train/experiments/.scratch/current_phase.log 2>&1 &
   ```

   **Remote mode:** First push local changes to server (overrides, scripts, and any Level 2 source edits):
   ```bash
   rsync -avz --progress -e "ssh -p $SSH_PORT" \
       --exclude='.git' --exclude='__pycache__' --exclude='.venv' --exclude='logs' \
       $LOCAL_CF_LAB/ \
       $SSH_USER_HOST:$REMOTE_CF_LAB/
   ```
   Then SSH to server and launch `run_phase.py` with `--skip-play --skip-eval --skip-parse-env` (server has no rendering, and parse_env / evaluate_policy run locally):
   ```bash
   ssh -p $SSH_PORT $SSH_USER_HOST "cd $REMOTE_CF_LAB && \
     export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
     nohup setsid .venv/bin/python .claude/skills/auto_train/resources/run_phase.py \
       --task=<TASK> --max-iterations=<N> --num-envs=<N> --headless \
       --skip-play --skip-eval --skip-parse-env \
       --overrides-file=.claude/skills/auto_train/experiments/.scratch/<file>.json \
       --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
       --monitor-interval=60 --abort-min-reward-at=300:-10 --abort-plateau-patience=500 \
       > .claude/skills/auto_train/experiments/.scratch/current_phase.log 2>&1 &"
   ```
   The `ssh` command returns immediately because the process is detached on the server.

   **IMPORTANT:** Always use `.venv/bin/python`, never bare `python`. Each Bash call starts a fresh shell — venv activation from earlier steps does not persist. The `.venv/bin/python` ensures the correct Python (with Isaac Lab, tbparse, etc.) is used. `run_phase.py` uses `sys.executable` for all subprocesses (training, play, metrics), so the venv propagates automatically.

   **Abort criteria:** Always include `--monitor-interval=60` and at least `--abort-min-reward-at=300:-10` for tuning runs. This catches obviously broken runs early and saves compute. For production runs, use looser thresholds or omit abort criteria.

4. **Wait for completion** — Use the blocking wait script instead of polling.

   **Local mode:**
   ```bash
   .venv/bin/python .claude/skills/auto_train/resources/wait_for_phase.py \
     --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
     --poll-interval=30 --timeout=7200
   ```

   **Remote mode:**
   ```bash
   ssh -p $SSH_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=120 $SSH_USER_HOST \
     "cd $REMOTE_CF_LAB && .venv/bin/python .claude/skills/auto_train/resources/wait_for_phase.py \
       --report-path=.claude/skills/auto_train/experiments/.scratch/current_phase_report.json \
       --poll-interval=30 --timeout=7200"
   ```
   The `-o ServerAliveInterval=30 -o ServerAliveCountMax=120` keeps the SSH connection alive during long training runs.

   Use `timeout: 7500000` on the Bash tool call (slightly over 2 hours). The script blocks internally and prints progress to stderr. When training completes, it prints the full report JSON to stdout and exits.

   **First check at 60s:** Before using `wait_for_phase.py`, do one quick check after 60 seconds to catch fast failures (validation errors, import errors, OOM):

   Local mode:
   ```bash
   sleep 55 && cat .claude/skills/auto_train/experiments/.scratch/current_phase_report.json
   ```
   Remote mode:
   ```bash
   sleep 55 && ssh -p $SSH_PORT $SSH_USER_HOST "cat $REMOTE_CF_LAB/.claude/skills/auto_train/experiments/.scratch/current_phase_report.json"
   ```
   If status is `"validation_error"`, `"crashed"`, or `"failed"`, skip the wait and proceed to analysis. If `"running"`, use `wait_for_phase.py` for the rest.

   **Terminal statuses:** `"completed"`, `"early_stopped"`, `"failed"`, `"crashed"`, `"validation_error"` — proceed to analysis.

   **Dead process detection:** The report file contains a `"pid"` field. If the file says `"running"` but `kill -0 <PID>` fails (remote mode: `ssh -p $SSH_PORT $SSH_USER_HOST "kill -0 <PID>"`), treat it as a crash.

   **SSH drop recovery (remote mode):** If SSH drops during the wait, the training process on the server survives (detached via `nohup setsid`). Reconnect and check the report file:
   ```bash
   ssh -p $SSH_PORT $SSH_USER_HOST "cat $REMOTE_CF_LAB/.claude/skills/auto_train/experiments/.scratch/current_phase_report.json"
   ```
   If status is still `"running"`, re-run the blocking wait command. If terminal, proceed to step 5.

5. **Analyze** — Get the results and perform analysis.

   **Remote mode only — pull logs from server before analysis:**
   ```bash
   rsync -avz --progress -e "ssh -p $SSH_PORT" \
       $SSH_USER_HOST:$REMOTE_CF_LAB/logs/ \
       $LOCAL_CF_LAB/logs/
   ```
   This brings checkpoints, TensorBoard events, metrics.json, and phase_report.json to local.

   **Remote mode only — run play and extract frames locally:**
   The phase_report.json from the server will have `"frame_paths": []` because `--skip-play` was used. Determine the local log directory from the report's `"log_dir"` field (convert the server absolute path to local by replacing `$REMOTE_CF_LAB` with `$LOCAL_CF_LAB`). Then run play and frame extraction locally:
   ```bash
   nohup setsid .venv/bin/python .claude/skills/auto_train/resources/play_for_inspection.py \
     --task=<PLAY_TASK> \
     --checkpoint=$LOCAL_LOG_DIR/<latest_model>.pt \
     --video --video_length=300 --headless --num_envs=4 \
     > /dev/null 2>&1 &
   ```
   Wait for play to finish (it takes ~1-2 minutes), then extract frames:
   ```bash
   .venv/bin/python .claude/skills/auto_train/resources/extract_frames.py \
     --video=$LOCAL_LOG_DIR/videos/play/<video>.mp4 \
     --output-dir=$LOCAL_LOG_DIR/frames \
     --num-frames=12
   ```
   The play task ID is derived from the training task: replace `-v0` with `-Play-v0` (e.g., `Isaac-Velocity-Flat-Ayg-v0` → `Isaac-Velocity-Flat-Ayg-Play-v0`).

   **Then run the numerical eval locally** (auto_train v3 — server had `--skip-eval`):
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     .venv/bin/python .claude/skills/auto_train/resources/evaluate_policy.py \
       --task=<PLAY_TASK> \
       --checkpoint=$LOCAL_LOG_DIR/<latest_model>.pt \
       --report-path=$LOCAL_LOG_DIR/eval_report.json \
       --eval-steps=1000 --num-envs=4 --headless --video
   ```
   This produces `eval_report.json` with tracking RMSE, gait pattern, survival, posture, distribution-shift signals (Numerical Inspection Protocol). Read it alongside `phase_report.json` (the server-produced `phase_report.evaluation` field is empty in remote mode — use `eval_report.json` directly).

   **Both modes:** Read the full `phase_report.json` (from the local `log_dir`) for detailed metrics following the TensorBoard Analysis Protocol. Read extracted PNG frames following the Visual Inspection Protocol. Read the `evaluation` key of `phase_report.json` (or `eval_report.json` directly) following the Numerical Inspection Protocol — this is the rollout-based ground truth for tracking RMSE, gait pattern, survival, and distribution shift. Then check:

   **Velocity Tracking Gate:**
   - Velocity tracking reward (exp-kernel) below **0.3** after 300+ iterations → **BLOCKING problem**. Do not proceed to other tuning — your next iteration MUST address velocity tracking (increase weight, decrease competing penalties, check command ranges).
   - Between 0.3–0.6 → note as "needs improvement" and prioritize in the next 1–2 iterations.
   - Above 0.6 → velocity tracking is acceptable — proceed with other tuning goals.

   **Suspicious Patterns:** Check the `suspicious_patterns` field in phase_report.json. If any `"critical"` patterns are detected, they must be addressed before continuing.

6. **Log** — Append iteration entry following the Enhanced Journal Format (see below). Include: Hypothesis (specific and falsifiable), Variable under test, Iteration count with reasoning, Changes (exact old→new values), Result, Detailed Metrics table, Velocity Tracking Gate assessment, Visual Assessment (with CAN/CANNOT framework), Reasoning (why did this result occur), Conclusion and next step, Log dir. After iteration 3+, update the Cross-Iteration Comparison Table.
7. **Repeat** — Go to step 1 with new insights.

## Visual Inspection Protocol

### Camera Setup

The auto-train system uses `play_for_inspection.py` — a custom play wrapper that provides a **side-view camera at robot height** following a single robot, with only 2–4 environments. This replaces the default overhead view of 50+ robots which is useless for gait quality assessment.

Camera defaults: distance=2.0m, height=0.4m, azimuth=90° (side view), following robot 0 via `origin_type="asset_root"`.

### What to look for in each frame

Check for these specific failure modes:
- **Hip-walking**: Hips touching ground, legs splayed. Side-view: robot is too low, no visible gap between hip and ground.
- **Belly-sliding**: Base touching ground. Side-view: robot is flat on the ground.
- **Shuffling**: Feet never leave the ground. Side-view: no visible foot clearance during swing phase.
- **Spider-walking**: Legs extended outward, base too low. Side-view: abnormally wide stance.
- **Actual walking**: Clear swing/stance phases, base at normal height, feet lifting and placing.

### Honest Assessment Rules

- **NEVER say "robots walking upright with refined dynamic gaits" unless you can actually verify it.** If you cannot clearly see foot contact patterns, say so: "Side-view shows forward movement and upright posture. Foot clearance appears present but details are limited by resolution."
- **State the camera angle and its limitations** in every visual assessment.
- **Distinguish between what you CAN see and what you are INFERRING.** Example:
  - CAN SEE: robots moving forward, upright posture, legs cycling
  - INFER: feet are making ground contact (based on reward terms being positive)
  - CANNOT VERIFY: whether hip joints are also touching ground
- **If all robots look similar from an angle, that is NOT evidence of good gait.** They could all be doing the same exploit.

## TensorBoard Analysis Protocol

After each training run, perform ALL of the following analyses — do not just read final values from metrics.json:

### 1. Convergence Analysis

For each key metric (total reward, each reward term, episode length), determine:
- **Where did it plateau?** Check `converged_at_iteration` and `percent_of_training_at_convergence` fields. If a metric plateaued at iteration 200 in a 500-iteration run, the last 300 iterations were wasted.
- **Is it still climbing?** If `curve_shape` is `"still_improving"`, the run was too short to draw conclusions. Note this explicitly.
- **Did it plateau too early?** If `curve_shape` is `"converged_early"` (before 50% of training), the reward signal may be too weak or the learning rate too low.

### 2. Per-Term Reward Breakdown

Do not just look at total reward. For each reward term in `phase_report.json -> metrics -> reward_terms`:
- **Which terms are the largest contributors?** If one penalty term dominates total reward, the policy is spending all its capacity avoiding that penalty rather than tracking velocity.
- **Which terms are near-zero?** A reward term that stays at zero throughout training is either misconfigured (impossible to achieve) or irrelevant (weight too low to matter).
- **Are task rewards (velocity tracking, gait) actually improving?** Or is total reward climbing only because penalties are decreasing?

### 3. Velocity Tracking Specifically

Velocity tracking is the PRIMARY task objective. Treat it with special attention:
- A velocity tracking reward near **0.0** means the robot is NOT tracking velocity at all. For exp-kernel rewards, 0.0 = maximum error, 1.0 = perfect tracking.
- **Below 0.3 after 300+ iterations is a BLOCKING problem.** Do not tune other things — fix velocity first.
- If velocity tracking is poor but total reward is high, **the robot is cheating**. It found a way to maximize other reward terms without moving as commanded.

### 4. Trend Field Interpretation

The `trend` field in metrics.json (`"improving"`, `"stable"`, `"degrading"`) uses linear regression on the last 20% of data points. This is a CRUDE indicator:
- `"stable"` does NOT mean "good" — it means "not changing." A metric can be stably terrible.
- `"improving"` does NOT mean "sufficient" — it means the slope is positive.
- **Always look at actual values, not just trends.** Report both: "track_lin_vel_xy: 0.15 (trend: stable) — this is too low despite being stable."

### 5. Suspicious Patterns

Check the `suspicious_patterns` field in phase_report.json. The analysis detects:
- **gait_gaming_tracking** (critical): Gait reward is strong but tracking terms converged early — robot may be synchronizing legs without following velocity commands
- **high_reward_low_tracking** (critical): High total reward but tracking is <15% of total — robot is exploiting non-tracking rewards
- **tracking_flatlined_early** (warning): Tracking term converged before 50% of training
- **body_contact_nonzero** (warning): Robot body is touching ground

### 6. Cross-Iteration Comparison

After iteration 3+, maintain a comparison table in the journal (see Enhanced Journal Format). This makes it immediately visible if velocity tracking is stuck while total reward climbs — a classic sign of reward hacking.

## Numerical Inspection Protocol (auto_train v3)

After each training run, `run_phase.py` automatically invokes `evaluate_policy.py` and merges its `eval_report.json` into `phase_report.json` under the `evaluation` key. This rollout-based evaluation catches policies that look fine in training metrics but behave poorly at inference.

### What to read

1. **`evaluation.tracking`** — RMSE of commanded vs achieved base velocity.
   - `lin_vel_xy_rmse`: must be < **0.25** m/s for "ship-ready" flat. < **0.4** is recoverable; > **0.4** means tracking is broken at the rollout level even if `track_lin_vel_xy_exp` looks decent.
   - `yaw_rate_rmse`: must be < **0.3** rad/s.
   - `lin_vel_z_rmse`: advisory. High z-velocity = bouncing.
2. **`evaluation.gait`** — duty cycles per foot + regularity.
   - Each `duty_cycles.{LF,RF,LH,RH}` should be ~0.4–0.7. Below 0.15 or above 0.85 → the foot is either always swinging or always planted → `arrhythmic == true`.
   - `regularity_std` > 10 steps between touchdowns → arrhythmic.
3. **`evaluation.survival`** — `alive_at_1000_steps` > 0.9 is the bar. Lower means robot is falling or terminating early at inference even if it survived in training (a distribution / corruption gap).
4. **`evaluation.posture.upright_frac`** — fraction of steps with projected gravity z < -0.85 (upright). Must be > 0.95.
5. **`evaluation.energy_proxy`** — advisory. Sudden spikes between iterations suggest the policy found a high-effort exploit.
6. **`evaluation.action_smoothness`** — independent of training `action_rate_l2` reward. Spikes here indicate the policy is brittle on slightly different observations.
7. **`evaluation.distribution_shift`** — most important new signal:
   - `obs_shift_magnitude` < 1.0 → eval observation distribution within ~1 σ of training-time normalizer stats. **Healthy.**
   - 1.0–3.0 → the policy is operating outside its training distribution. Common when the play env differs from train env in subtle ways (no DR, different terrain, different num_envs). Read `top_shifted_dims` to identify which observations diverged.
   - \> 3.0 → policy is being asked to extrapolate hard. Numerical metrics above are not trustworthy — fix this before reading the rest.

### How to use eval signals in tuning

- **Tracking RMSE high + `track_lin_vel_xy_exp` low** → Velocity Tracking Gate problem. Same root cause; fix it.
- **Tracking RMSE high + `track_lin_vel_xy_exp` HIGH** → reward gaming. The exp-kernel reward saturates but actual error is high. Inspect σ in the exp kernel (it may be too forgiving). Cross-reference QUADRUPED_PRIOR_ART.md §1 typical σ values.
- **Survival low + posture upright_frac low** → robot falls at inference. Often a randomization mismatch (training with DR, eval without). Compare `*_PLAY` env config to training env.
- **Eval warnings empty + training looks great + final iteration → ship.** The numerical eval validates the visual + tensorboard story.

### Cross-signal warnings

`phase_report.json` includes a `cross_signal_warnings` array combining training and eval signals:
- `training_anomaly_with_distribution_shift` — `suspicious_patterns` has a critical entry AND `obs_shift_magnitude > 1.0`. Strong evidence the training-time reward landscape is being exploited.
- `clean_training_metrics_but_failed_eval` — no `suspicious_patterns` but ≥3 eval warnings. The policy is brittle in ways the training metrics didn't catch. Investigate.

## User controls during auto-train

- `stop` — stop after current phase
- `level 1` / `level 2` — switch privilege level mid-session
- `focus on X` — redirect Claude's priority (e.g., "focus on foot clearance")

## Enhanced Journal Format

Each auto-train session logs to `.claude/skills/auto_train/experiments/<experiment_name>/journal.md`. The journal is the single source of truth for what was tried, what happened, and what to do next.

**Header** (written once at the top):
```
# Auto-Train: <task_name>
Started: <date>  |  Level: <1|2>  |  Device: <device_info>
```
In remote mode, add the server info:
```
# Auto-Train: <task_name>
Started: <date>  |  Level: <1|2>  |  Device: <device_info>
Mode: Remote  |  Server: <user@host:port>  |  Remote cf_lab: <remote_path>
```

**Body Coverage Audit** (written once, before iteration 1):
```
## Body Coverage Audit

| Body          | Foot? | Termination? | Penalty? | Status   |
|---------------|-------|-------------|----------|----------|
| Base          | No    | Yes         | No       | COVERED  |
| LF_Hip        | No    | No          | Yes      | COVERED  |
| ...           |       |             |          |          |

**Gaps found:** [None / List of uncovered bodies and action taken]
```

**Per-iteration entry** (appended after each phase):
```
## Iteration N — <short goal>

**Hypothesis:** What we expect to happen and why. Be specific and falsifiable:
"Doubling track_lin_vel_xy weight from 1.0 to 2.0 should make velocity tracking the dominant
reward signal, causing the tracking reward to rise above 0.4 within 300 iterations."

**Variable under test:** The single parameter/group being changed
**Iteration count and reasoning:** 400 iterations — enough to see velocity tracking trend;
previous run showed signal by iter 200

**Changes:** Parameter changes with exact old → new values:
  - `rewards.track_lin_vel_xy_exp.weight`: 1.0 → 2.0

**Result:** success / early_stopped / failed / crashed
**Training time:** Xs (Y iterations completed)

**Detailed Metrics:**
| Metric                    | Final  | Mean (last 100) | Trend      | Curve Shape     | Converged@iter |
|--------------------------|--------|-----------------|------------|-----------------|----------------|
| Total reward              | 12.3   | 11.8            | improving  | still_improving | —              |
| track_lin_vel_xy_exp      | 0.45   | 0.42            | improving  | still_improving | —              |
| track_ang_vel_z_exp       | 0.31   | 0.28            | stable     | converged       | ~200           |
| gait                      | 5.2    | 4.9             | improving  | still_improving | —              |
| Episode length            | 18.5   | 17.2            | stable     | converged       | ~100           |

**Suspicious Patterns:** [None / List from phase_report.json]

**Velocity Tracking Gate:**
  - Linear velocity tracking: 0.42 — NEEDS IMPROVEMENT (target: >0.6)
  - Angular velocity tracking: 0.28 — BLOCKING (target: >0.3)

**Numerical Eval (from evaluation key in phase_report.json):**
  - lin_vel_xy_rmse: 0.31 m/s (target: <0.25)
  - yaw_rate_rmse: 0.22 rad/s (target: <0.3) ✓
  - survival.alive_at_1000_steps: 1.0 ✓
  - gait.arrhythmic: false ✓, duty_cycles ≈ {LF:0.52, RF:0.51, LH:0.53, RH:0.50}
  - posture.upright_frac: 0.98 ✓
  - distribution_shift.obs_shift_magnitude: 0.41 ✓
  - Eval warnings: [lin_vel_xy_rmse_above_threshold]

**Visual Assessment:**
  - Camera: side-view at robot level (play_for_inspection.py, distance=2.0m, height=0.4m)
  - CAN VERIFY: [e.g., robots moving forward, upright posture, leg swing visible]
  - CANNOT VERIFY: [e.g., exact foot placement details at this resolution]
  - Failure modes checked: hip-walking: not detected / belly-sliding: not detected / shuffling: uncertain

**Reasoning:** Why did this result occur? What does it tell us about the reward landscape?
  - "Velocity tracking improved from 0.15 to 0.42 after doubling weight, confirming the
    reward signal was too weak. However, yaw tracking is still poor — may need separate attention."

**Conclusion and Next Step:** What to try next and why
  - "Linear velocity responding well. Next: increase track_ang_vel_z weight to address yaw tracking."

**Log dir:** `logs/rsl_rl/.../<timestamp>/`
```

**Cross-Iteration Comparison Table** (updated after iteration 3+):
```
## Progress Summary (updated after iteration N)

| Metric              | Iter 1 | Iter 2 | Iter 3 | ... | Best | Target |
|--------------------|--------|--------|--------|-----|------|--------|
| Total reward        | 5.2    | 8.1    | 12.3   |     | 12.3 | >15    |
| Vel tracking xy     | 0.12   | 0.15   | 0.42   |     | 0.42 | >0.6   |
| Vel tracking yaw    | 0.08   | 0.09   | 0.28   |     | 0.28 | >0.3   |
| Gait reward         | 2.1    | 3.5    | 5.2    |     | 5.2  | >5.0   |
| Episode length      | 10.2   | 14.5   | 18.5   |     | 18.5 | >18    |
```

**Production Readiness Assessment** (before production run): See Production Readiness Checklist section.

**Post-Production Summary** (after production run completes):
```
## Production Run — Final Results

**Config:** [list of all override values used]
**Iterations:** X completed (target was Y)
**Training time:** Z hours

**Final Metrics (detailed):**
| Metric                    | Final  | Mean (last 100) | Peak   | Converged@iter |
|--------------------------|--------|-----------------|--------|----------------|
| Total reward              | ...    | ...             | ...    | ...            |
| [every reward term]       | ...    | ...             | ...    | ...            |
| Episode length            | ...    | ...             | ...    | ...            |

**Visual Assessment (production):**
  - [Same CAN/CANNOT format as tuning iterations]

**What CAN be verified from this run:**
  - [List specific claims supported by data]

**What CANNOT be verified and requires further review:**
  - [List aspects that need sim-to-real validation, different camera angles, etc.]

**Recommended next steps:**
  - [Deploy to Gazebo for sim-to-sim validation, test on hardware, etc.]
```

**Resuming after disconnect:** Start a new conversation and say: *"Continue auto-train, read journal at `.claude/skills/auto_train/experiments/<name>/journal.md`"*. Claude will read the journal, pick up from the last iteration, and continue.

## Scripts

All scripts are in `.claude/skills/auto_train/resources/`:
- `run_phase.py` — orchestrator: parse_env → train → metrics → play → frames → evaluate_policy → report
- `train_with_overrides.py` — modified train.py with `--overrides-file` JSON support
- `analyze_metrics.py` — TensorBoard events → JSON with convergence detection and suspicious pattern analysis
- `play_for_inspection.py` — side-view play wrapper for visual gait inspection (replaces overhead play.py)
- `extract_frames.py` — MP4 → PNG frames for visual inspection (uses OpenCV)
- `wait_for_phase.py` — blocks until training completes, prints final report (replaces sleep-poll loop)
- `parse_env.py` — *(v3)* dynamic env-config introspection: walks the registered task's @configclass tree (rewards / observations / actions / terminations / events / curriculum) and emits `env_schema.{json,md}` for tuner context injection
- `evaluate_policy.py` — *(v3)* numerical + distribution-shift evaluation: tracking RMSE, gait pattern, energy, survival, posture, action smoothness, and obs-shift vs training-time normalizer stats. Emits `eval_report.json` + terrain-aware `eval_video.mp4`

Prior-art reference docs in `.claude/skills/auto_train/resources/prior_art/`:
- `QUADRUPED_PRIOR_ART.md` — distilled cheatsheet (reward vocabulary, obs conventions, DR ranges, PPO hparams, failure modes). Auto-injected at iteration 1.
- `repos.md` — full catalog of ~100 Isaac-based quadruped locomotion repos. Grep on demand when a tuning hypothesis benefits from a specific reference.

## Device scaling

Use GPU model and VRAM to choose appropriate `num_envs` and iteration counts:
- Laptop (RTX 3060 12GB): ~2048 envs
- Desktop (RTX 4090 24GB): ~4096 envs
- Cloud (A100 80GB): ~4096-8192 envs

## Key Rules

- **Always visually inspect honestly** — The robot can maximize reward while hip-walking or crawling. Follow the Visual Inspection Protocol. State what you CAN and CANNOT verify from the camera angle. Never claim good gait quality you cannot actually confirm.
- **Never skip play** — In local mode, do not use `--skip-play`. In remote mode, use `--skip-play` on the server (no rendering capability), then run play LOCALLY after pulling logs. Play must always happen — just not necessarily on the same machine as training.
- **Scale to device** — Use the device info to choose `num_envs` and iteration counts appropriately.
- **Start conservative with short runs** — First iteration should be a baseline or small change at 300–500 iterations. Do NOT use 2000 iterations for tuning.
- **One variable per iteration** — Change one thing at a time so you can attribute results. State your hypothesis in the journal.
- **Pre-analyze rewards AND body coverage** — Mathematically verify reward terms are feasible AND verify every non-foot body is covered by termination or penalty before training.
- **Velocity tracking is the primary objective** — A robot that maximizes total reward but does not track commanded velocity is useless. If velocity tracking metrics are poor (reward < 0.3 or error > 1.0), treat this as a blocking issue and fix it before tuning anything else. Never declare a run "production-ready" with poor velocity tracking.
- **State uncertainty explicitly** — If you cannot verify something, say "UNCERTAIN" or "CANNOT VERIFY", never "looks good." Distinguish observation from inference. "Reward is 12.5" is an observation. "The robot is walking well" is an inference that requires evidence.
- **Never declare production-ready based on metrics alone** — Metrics can be gamed. Visual confirmation of actual walking (not reward-hacking) is required, and the confirmation must be honest about what the camera angle can actually show.
- **Journal everything** — If the conversation disconnects, the journal is the only way to resume.
- **Final production run** — Only after the Production Readiness Checklist passes. Bake winning params into source config.
- **Level 1** — Only JSON overrides (reward weights, PPO hyperparams). No source file edits.
- **Level 2** — Full autonomy: can edit source configs, add/remove rewards, write new reward functions.

## Unattended Operation (CRITICAL — read carefully)

Auto-train is designed to run unattended (e.g., overnight with `claude --dangerously-skip-permissions`). **ANY blocking operation is a critical bug.** These rules ensure it never hangs or wastes compute:

- **NEVER use AskUserQuestion** — There is no human watching. If you are uncertain, make a reasonable decision and log your reasoning in the journal. Never block waiting for user input. No exceptions.
- **NEVER ask for confirmation for ANY file operation** — Creating journal.md, writing override JSON, editing source configs at Level 2, creating experiment directories — just do it. Do not present options and wait for selection. All file writes (Write, Edit) within the auto-train workflow are pre-authorized.
- **If uncertain about a decision, choose the safer option and document why** — Do not block. Log your reasoning in the journal and continue.
- **Max 15 tuning iterations** — After 15 iterations (not counting the final production run), stop the loop. However, do NOT start production just because you hit 15. If the Production Readiness Checklist does not pass after 15 iterations, log a summary of what is still failing and stop WITHOUT a production run. A bad production run wastes more compute than stopping early.
- **Abort after 3 consecutive failures** — If 3 iterations in a row produce `status: "failed"` or `status: "crashed"` in the phase report, stop the loop. Log the failure pattern and suspected root cause in the journal. If at Level 2, check whether your source edits introduced a syntax or import error before retrying.
- **Detached launch** — Always launch `run_phase.py` via `nohup setsid ... &` (as shown in the Auto-Train Loop section). NEVER use `run_in_background: true` — it has a timeout that kills long training runs. The `nohup setsid` approach fully detaches the process.
- **Disk awareness** — Each iteration produces checkpoints (~50-200 MB) and videos. If running many iterations, note cumulative disk usage in the journal. If `df -h` shows <10 GB free on the training partition at any check, stop and log a warning.
- **Level 2 safety** — Before starting a new training after editing source files, verify the edit is syntactically valid: `.venv/bin/python -c "import ast; ast.parse(open('<edited_file>').read()); print('syntax OK')"`. If this fails, fix the error before training. Note: do NOT use `import cf_lab.tasks` as a check — it requires Isaac Sim runtime and will always fail in a plain shell.
- **SSH resilience (remote mode)** — Always use `-p $SSH_PORT` on every SSH and rsync command. Use `-o ServerAliveInterval=30 -o ServerAliveCountMax=120` on long-running SSH sessions (wait_for_phase.py). Training on the server is detached via `nohup setsid` and survives SSH disconnects. Never consider a dropped SSH connection as a training failure — always reconnect and check the actual report file status.
- **Rsync verification (remote mode)** — After pulling logs from the server, verify that `phase_report.json` and at least one checkpoint file exist locally before proceeding to play. If rsync was interrupted, retry once. Do not run play on incomplete data.

## Environment Variables (always set before training)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Porting to other branches/devices

The auto-train system is self-contained in a single folder. To enable it on any branch/device:
1. Copy `.claude/skills/auto_train/` folder into that branch's `.claude/skills/`

That's it. The skill handles setup (dependencies, directories) automatically on first run.

For remote mode, the push rsync command syncs auto_train scripts to the server automatically before each training run. No manual setup on the server is needed beyond having cf_lab + `.venv` working.

---

Begin now. Create the experiment folder and journal, run the pre-training body coverage audit and reward analysis, and start iteration 1.
