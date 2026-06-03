# Quadruped Locomotion Prior-Art Cheatsheet

Distilled from the Isaac-based quadruped repo survey in `repos.md`. Synthesises
canonical reward terms, observation conventions, domain-randomization ranges,
and PPO hyperparameters that the broader Isaac Lab / Isaac Gym quadruped
community has converged on.

**Purpose:** auto-injected at iteration 1 of `/auto-train` so the tuner reasons
about hypothesis space grounded in what's known to work, not from scratch.

**Anchor repos:** `IsaacLab` (NVIDIA velocity task), `legged_gym` (ETH RSL, the
template every fork descends from), `robot_lab` (broadest morphology coverage),
`LeggedLab` (direct workflow, sim-to-real validated), `unitree_rl_lab`
(production deployment pipeline), `basic-locomotion-isaaclab` (IIT-DLS,
flat/rough/vision variants), `walk-these-ways` (gait-conditioned MoB),
`HIMLoco` (internal model + 4096 envs), `extreme-parkour` /
`Isaaclab_Parkour` (high-entropy parkour), `Isaac-Velocity-Flat-Anymal-D-v0`
(canonical Isaac Lab baseline numbers).

---

## 1. Reward Vocabulary

These are the reward terms that appear in nearly every successful Isaac Lab /
Isaac Gym quadruped locomotion repo. Weights below are the **canonical range**
observed across repos for flat-terrain velocity tracking on an ANYmal-class
quadruped (mass ~10–50 kg, leg ~0.3–0.6 m, target ~1 m/s). Scale roughly by
mass/dimension when porting to a different morphology.

### Load-bearing terms (always present)

| Term | Function | Typical weight | Notes |
|------|----------|---------------:|-------|
| `track_lin_vel_xy_exp` | exp(-‖cmd − v_xy‖² / σ²) | **+1.0 to +2.0** | Primary task reward. σ commonly 0.25–0.5. Below 0.6 final value ⇒ tracking failed. |
| `track_ang_vel_z_exp` | exp(-(cmd_yaw − ω_z)² / σ²) | **+0.5 to +1.0** | σ commonly 0.25. Below 0.3 final value ⇒ yaw tracking failed. |
| `lin_vel_z_l2` | (v_z)² | **−1.0 to −2.0** | Discourages bouncing / hopping. |
| `ang_vel_xy_l2` | ‖ω_xy‖² | **−0.02 to −0.1** | Discourages roll/pitch oscillation. |
| `action_rate_l2` | ‖a_t − a_{t−1}‖² | **−0.005 to −0.02** | Smoothness. Going below −0.05 over-constrains and stalls learning. |
| `dof_acc_l2` | ‖q̈‖² | **−2.5e-7 to −1e-6** | Energy / smoothness; tiny weight because magnitudes are large. |
| `dof_torques_l2` | ‖τ‖² | **−1e-4 to −5e-5** | Energy. Same scaling caveat. |
| `feet_air_time` | (t_air − threshold) on first contact | **+0.1 to +1.0** | Threshold ~0.4 s for ~1 m/s on 0.3 m legs; reduce for shorter legs / slower targets. |

### Stability / safety terms (usually present)

| Term | Function | Typical weight | Notes |
|------|----------|---------------:|-------|
| `undesired_contacts` | sum 1{contact on body_names} | **−1.0** | Body names exclude feet; commonly target shanks/thighs as `.*_Shank`, `.*_Thigh`. |
| `flat_orientation_l2` | ‖projected_gravity_xy‖² | **−5.0** (flat) / **0.0** (rough) | Disabled on rough so the robot can lean into terrain. |
| `base_height_l2` | (h − h_target)² | **−5.0** (flat) / **0.0** (rough) | h_target ≈ resting stand height (0.30–0.40 m for ANYmal-class). |
| `dof_pos_limits` | clamp violation² | **−5.0 to 0.0** | Most repos leave 0 unless joint-limit hits are observed. |
| `joint_deviation_l1` | ‖q − q_default‖ | **0.0 to −0.2** | Pulls posture toward nominal; over-strong → stiff gait. |

### Quality / shaping terms (task-dependent)

| Term | Function | Typical weight | Notes |
|------|----------|---------------:|-------|
| `foot_clearance` | exp-kernel on swing-phase height | **+0.1 to +0.5** | Target height 0.05–0.15 m. Critical for visual gait quality on flat. |
| `feet_regulation` | penalize foot vel at touchdown / non-zero in stance | **−0.05 to −0.2** | Used by IIT-DLS, fan-ziqi forks. Helps gait regularity. |
| `gait` (WTW-style) | match commanded duty/frequency | **+0.5 to +2.0** | Only when `gait_freq` / `gait_phase` are commanded inputs. |
| `feet_slide` | foot xy-velocity while in contact | **−0.05 to −0.25** | Stops feet from sliding in stance — common in robot_lab. |

### Things that bite if mis-weighted (failure modes — see §6)

- **`feet_air_time` too high** → exaggerated bounding / hop-gaits that maximize air time but break tracking.
- **`gait` >> tracking** → "gait_gaming_tracking" pattern (see `analyze_metrics.py:detect_suspicious_patterns`).
- **`action_rate` too punitive** → robot freezes at default pose, tracking flatlines near zero.
- **`base_height_l2` too low h_target** → robot crouches and crawls.
- **No `undesired_contacts` on shank/thigh** → hip-walking / belly-sliding (always pair the body-coverage audit with these penalties).

---

## 2. Observation Conventions

Canonical proprioceptive policy observation vector for blind locomotion (used by
`legged_gym`, `robot_lab`, `LeggedLab`, IsaacLab velocity task, `unitree_rl_lab`):

| Index range | Field | Dim | Noise (Unoise n_min/n_max) | Notes |
|-------------|-------|----:|----------------------------|-------|
| 0–2 | `base_lin_vel` | 3 | ±0.1 | In body frame. |
| 3–5 | `base_ang_vel` | 3 | ±0.2 | In body frame. |
| 6–8 | `projected_gravity` | 3 | ±0.05 | Body-frame gravity; encodes pitch+roll. |
| 9–11 | `velocity_commands` (lin_x, lin_y, ang_z) | 3 | 0 | Sampled from `commands.base_velocity.ranges`. |
| 12–23 | `joint_pos_rel` (q − q_default) | 12 | ±0.01 | 12 DoF for quadrupeds. |
| 24–35 | `joint_vel` | 12 | ±1.5 | Velocities can be noisy on real hardware. |
| 36–47 | `last_actions` | 12 | 0 | Feedback from previous control step. |

Total blind-policy obs ≈ **48 dims** (matches `IsaacLab` and most forks).

**Rough terrain** adds a `height_scan` term — typically a 11×17 = 187-dim
grid-pattern ray-cast (resolution 0.1 m, size 1.6 × 1.0 m) attached to the base.
Total rough obs ≈ **235–236 dims** (matches AygRoughEnvCfg via 0.1m grid pattern, 49 base + 187 scan).

**Critic asymmetry:** several recent repos (`robot_lab`, `LeggedLab`, IsaacLab
flat) feed the critic privileged observations (true `base_lin_vel`, friction,
terrain heights even on flat) while the policy gets noisy proprio only. This
helps sim-to-real value-bootstrapping but adds plumbing — only consider at
Level 2.

**Noise:** disable `enable_corruption` for `*_PLAY` variants (matches Isaac Lab
default — see `flat_env_cfg.py:AygFlatEnvCfg_PLAY.__post_init__`).

---

## 3. Domain Randomization Ranges That Converge

These are the DR ranges across surveyed repos that produce policies which
**transfer** without further tuning. Stronger ranges are common in parkour /
sim-to-real focused repos; gentler ranges in vanilla velocity tasks.

| Event | Mode | Range | Conservative / Aggressive |
|-------|------|-------|---------------------------|
| Friction | startup | static 0.4–2.0, dynamic 0.4–2.0 | Both bounds widely used (IsaacLab, robot_lab, IIT-DLS). Going below 0.3 destabilizes early training. |
| Restitution | startup | 0.0–0.0 (flat) up to 0.0–0.5 (parkour) | Default 0 is fine for flat. |
| Base mass | startup | (−1.0, +3.0) kg add (AYG default) ; (±1.5) kg for ~10 kg robots | Add-mode common; multiply-mode rarer. |
| Base COM offset | startup | x,y: ±0.05 m; z: ±0.01 m | Tighter z because most quadrupeds are z-stable. |
| Joint stiffness/damping | startup | ±20% multiply | Optional but improves sim-to-real (LeggedLab, robot_lab). |
| Reset base | reset | pose: x,y ±0.5 m, yaw ±π; velocity: zeros | Yaw full-range randomization is universal. |
| Reset joints | reset | position scale (1.0, 1.0) = exactly default; widen to (0.5, 1.5) for harder init | AYG uses (1.0, 1.0). |
| Push robot | interval (7–15 s) | velocity ±0.5–1.0 m/s xy | Disabled in `*_PLAY` variants. |
| External force/torque | reset | force ±2–5 N, torque ±1 Nm (when used) | Used heavily in HIMLoco / robust-locomotion repos. |
| Actuator gain perturbation | startup | ±20% (rarely on flat, common in robot_lab) | Skip on flat baseline. |

**Rule of thumb:** the AYG default (`AygRoughEnvCfg.__post_init__` lines 82–96)
sits in the conservative range — friction 0.4–2.0, base mass (−1, +3), COM ±5 cm,
yaw ±π. That matches IsaacLab's Anymal-D velocity baseline closely. Don't
randomize more aggressively until you've validated a clean flat policy.

---

## 4. PPO Hyperparameters

RSL-RL defaults that the surveyed repos rarely deviate from. These are the
`agent.*` overrides the tuner can adjust:

| Param | RSL-RL default | Typical override range | When to deviate |
|-------|----------------|------------------------|-----------------|
| `learning_rate` | 1.0e-3 (adaptive) | 5e-4 – 1.5e-3 | RSL-RL adapts based on KL; usually leave alone. Force lower when training is unstable. |
| `num_steps_per_env` | 24 | 24 (standard) – 48 (longer horizon for parkour) | Increase for tasks with sparse reward. |
| `num_mini_batches` | 4 | 4 – 8 | More batches → smaller minibatches → less noisy updates. |
| `num_learning_epochs` | 5 | 5 (standard) – 8 | Too high → overfitting on stale data. |
| `clip_param` | 0.2 | 0.2 (default) | Rarely touched. |
| `entropy_coef` | 0.01 | 0.005 – 0.02 | Higher (0.01–0.02) for parkour / exploration-heavy tasks. Lower (0.005) once gait is found. |
| `value_loss_coef` | 1.0 | 1.0 (default) | Rarely touched. |
| `gamma` | 0.99 | 0.99 (standard) | Lower (0.95) only for very short horizon tasks. |
| `lam` | 0.95 | 0.95 (default) | Rarely touched. |
| `desired_kl` | 0.01 | 0.005 – 0.02 | Adaptive lr target. Tighter (0.005) for more conservative updates. |
| `max_grad_norm` | 1.0 | 1.0 (default) | Rarely touched. |

**Network architecture:** most surveyed repos use `MLP(256, 128, 128)` or
`MLP(512, 256, 128)` actor and critic with `elu` activation. Increasing capacity
rarely helps blind locomotion; reduce only if VRAM-constrained.

**Note on `num_envs`:** scale with VRAM. Surveyed defaults: legged_gym 4096
(Preview-4 Isaac Gym), IsaacLab 4096 (RTX-class), HIMLoco 4096, parkour repos
often 6144–8192. RSL-RL is throughput-bound by GPU memory, not by quality —
2048 envs converge to the same policy as 4096 with more wallclock.

---

## 5. Task-Specific Notes

### Flat velocity tracking (the AYG baseline)

- Disable `height_scanner` and remove `height_scan` from policy obs.
- Disable `terrain_levels` curriculum.
- `flat_orientation_l2 = -5.0` is canonical — keeps the base level.
- `feet_air_time.weight` lower than rough (AYG: 0.25 on flat vs 0.01 on rough,
  see `flat_env_cfg.py:18` and `rough_env_cfg.py:106`).
- Expected production-quality numbers: `track_lin_vel_xy_exp` > 0.7,
  `track_ang_vel_z_exp` > 0.4, total reward 15+, episode length near max.

### Rough terrain

- Enable `height_scanner` (11×17 grid pattern), add `height_scan` to policy obs.
- Enable `terrain_levels` curriculum — robot starts on easy patches, progresses
  to harder.
- Disable / reduce `flat_orientation_l2` (the robot must lean into slopes).
- Disable / reduce `base_height_l2` (height varies with terrain).
- Expect `track_lin_vel_xy_exp` ~0.5–0.6 (lower than flat is normal).
- Episode length is the key health indicator — if it drops, robot is falling.

### Walk-These-Ways (WTW) — gait-conditioned

- Adds commanded `gait_freq`, `gait_phase` (trot, pace, bound, pronk), and
  `footswing_height` to obs.
- Reward includes a `gait` term that matches commanded duty cycles per foot.
- WTW failure mode: `gait` > `tracking` ⇒ robot maintains gait but doesn't
  follow velocity command. This is exactly the `gait_gaming_tracking` critical
  pattern. Counter by keeping `track_lin_vel_xy_exp.weight ≥ gait.weight`.
- No canonical WTW port to Isaac Lab as of 2026-05; AYG's
  `Isaac-WTW-*-Ayg-*` envs are first-party.

### Curriculum

- `terrain_levels` is the standard rough-terrain curriculum (level up if robot
  walks far, level down if it falls).
- WTW uses `sigma_exp_neg_anneal` to anneal exp-kernel σ for gait reward over
  ~48k steps — keeps early training tractable.
- Don't introduce custom curricula at Level 1 — they require source edits.

---

## 6. Known Failure Modes

Cross-reference with `analyze_metrics.py:detect_suspicious_patterns` and the
Visual Inspection Protocol in `SKILL.md`.

| Failure mode | Symptom | Root cause | Counter |
|--------------|---------|------------|---------|
| **Gait gaming tracking** (critical) | `gait` ↑, `track_lin_vel_xy_exp` flatlines early at low value | `gait.weight ≥ track_lin_vel_xy.weight`; robot maximizes legs cycling without going where commanded. | Reduce `gait.weight` OR raise `track_lin_vel_xy.weight`. Keep tracking dominant. |
| **High reward, low tracking** (critical) | Total reward climbing; tracking term < 15% of total | Penalty terms drowning out signal; robot does anything to avoid penalty (often stand still). | Increase tracking weight; reduce penalty weights; verify reward feasibility per Pre-Training Reward Analysis. |
| **Hip-walking / belly-sliding** | Visual: hips touch ground, legs splayed. Metrics: `body_contact_*` non-zero. | Body not in `terminations.base_contact.body_names` or `undesired_contacts.body_names`. | Run Body Coverage Audit. Add `.*_Hip`, `.*_Thigh`, `.*_Shank` to penalty list. |
| **Shuffling / no foot clearance** | Visual: feet drag, no swing visible | `feet_air_time` weight too low OR `foot_clearance` reward absent | Raise `feet_air_time.weight`; add/raise `foot_clearance` reward. |
| **Action explosion on terminations** | Episode length ↓ quickly, action_rate spikes | Early terminations + no `action_rate_l2` penalty; the policy "escapes" through actions on reset | Add `action_rate_l2` if missing; check termination triggers. |
| **Premature convergence (`converged_early`)** | Reward plateau before 50% of training | Reward signal too weak, OR `entropy_coef` too low, OR `desired_kl` too tight | Raise `entropy_coef` (0.01 → 0.015), loosen `desired_kl` (0.01 → 0.02), or raise weight of stuck term. |
| **Spider-walking** (legs splayed) | Visual: legs extended outward, base low | No `joint_deviation_l1` OR `dof_pos_limits` not enforced | Add `joint_deviation_l1` at small negative weight; check joint limits. |
| **Velocity tracking ≈ 0** (blocking) | `track_lin_vel_xy_exp < 0.3` after 300 iters | Reward weight too low, command range too wide, OR competing penalty dominates | This is the Velocity Tracking Gate. Fix this before tuning anything else. |
| **Bouncing / hopping** | Visual: all four legs leave ground together; high `lin_vel_z_l2` | `lin_vel_z_l2.weight` too low | Increase magnitude to −2.0 to −5.0. |

---

## 7. Cross-References

- Full repo catalog: `resources/prior_art/repos.md`
- Analyzer-detected patterns: `resources/analyze_metrics.py:detect_suspicious_patterns`
- Body coverage audit procedure: `SKILL.md` § "Pre-Training Body Coverage Audit"
- Production thresholds: `SKILL.md` § "Production Readiness Checklist"
- AYG current weights (anchor): `cf_lab/source/cf_lab/cf_lab/tasks/manager_based/velocity/rough_env_cfg.py:98-119` and `flat_env_cfg.py:17-19`

When proposing a tuning change, **cite the prior-art range** from this doc in
the journal hypothesis ("legged_gym + IsaacLab use `action_rate_l2` in
[−0.02, −0.005]; current AYG is −0.01 — within range, leave alone"). That keeps
hypotheses grounded.
