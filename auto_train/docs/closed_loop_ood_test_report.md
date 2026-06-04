# auto_train OOD Closed-Loop Test Report

Acceptance criterion for issue #2 §4: validate the upgraded tuner on
`Isaac-Velocity-Flat-Ayg-OOD-v0`, a deliberately-sabotaged AYG flat baseline.
The tuner should pull the sabotaged weights back to within prior-art ranges
(`resources/prior_art/QUADRUPED_PRIOR_ART.md` §1) without manual intervention.

This template is filled by the **User** after running the closed-loop overnight.
This Claude session does not execute /auto-train.

---

## Setup

- **Date run:** _YYYY-MM-DD_
- **Branch / commit:** _git rev_
- **Hardware:** _e.g., RTX 4090 (server) / RTX 3060 (laptop)_
- **Command:**
  ```bash
  /auto-train Isaac-Velocity-Flat-Ayg-OOD-v0 level 2 on <device>
  ```
- **Iterations the tuner ran:** _N tuning iters + final production_

## Sabotaged starting weights (baseline)

Sourced from `cf_lab/source/cf_lab/cf_lab/tasks/manager_based/velocity/ood_test_env_cfg.py`.

| Reward term | Sabotaged value | Prior-art range (QUADRUPED_PRIOR_ART.md) |
|-------------|----------------:|------------------------------------------|
| `track_lin_vel_xy_exp.weight` | 0.1 | +1.0 to +2.0 |
| `action_rate_l2.weight` | -10.0 | -0.02 to -0.005 |
| `feet_air_time.weight` | 0.0 | +0.1 to +1.0 |
| `dof_acc_l2.weight` | -1e-2 | -1e-6 to -2.5e-7 |
| `foot_clearance.weight` | 0.0 | +0.1 to +0.5 |
| Domain randomization | DISABLED (physics_material, add_base_mass, push_robot, base_com all None) | enabled with default ranges |

## Before — vanilla 500-iter run on the sabotaged env

Run before any tuning. Captures baseline numbers.

| Metric | Value |
|--------|------:|
| Total reward (last 100) | _fill_ |
| `track_lin_vel_xy_exp` final | _fill_ |
| `track_ang_vel_z_exp` final | _fill_ |
| Episode length (last 100) | _fill_ |
| `lin_vel_xy_rmse` (from eval_report.json) | _fill_ |
| `yaw_rate_rmse` | _fill_ |
| `survival.alive_at_1000_steps` | _fill_ |
| `gait.arrhythmic` | _fill_ |
| `posture.upright_frac` | _fill_ |
| `distribution_shift.obs_shift_magnitude` | _fill_ |
| Visual gait quality (CAN VERIFY) | _description_ |

Phase report path: _`logs/rsl_rl/.../phase_report.json`_

## After — /auto-train run

### Final weights chosen by the tuner

Pulled from the journal's final production-readiness section.

| Reward term | Final value | Within prior-art range? |
|-------------|------------:|:-----------------------:|
| `track_lin_vel_xy_exp.weight` | _fill_ | _Y/N_ |
| `action_rate_l2.weight` | _fill_ | _Y/N_ |
| `feet_air_time.weight` | _fill_ | _Y/N_ |
| `dof_acc_l2.weight` | _fill_ | _Y/N_ |
| `foot_clearance.weight` | _fill_ | _Y/N_ |
| _other terms tuned_ | _fill_ | _Y/N_ |
| Domain randomization | _re-enabled? Y/N_ | _Y/N_ |

### Final metrics

| Metric | After | Before | Δ |
|--------|------:|------:|--:|
| Total reward (last 100) | _fill_ | _fill_ | _fill_ |
| `track_lin_vel_xy_exp` final | _fill_ | _fill_ | _fill_ |
| `track_ang_vel_z_exp` final | _fill_ | _fill_ | _fill_ |
| `lin_vel_xy_rmse` | _fill_ | _fill_ | _fill_ |
| `yaw_rate_rmse` | _fill_ | _fill_ | _fill_ |
| `survival.alive_at_1000_steps` | _fill_ | _fill_ | _fill_ |
| `gait.arrhythmic` | _fill_ | _fill_ | _fill_ |
| `posture.upright_frac` | _fill_ | _fill_ | _fill_ |
| `obs_shift_magnitude` | _fill_ | _fill_ | _fill_ |
| Visual gait quality | _description_ | _description_ | _description_ |

### Tuner trajectory summary

Reference the journal at `.claude/skills/auto_train/experiments/<name>/journal.md`.

- **Iteration 1 hypothesis:** _quote_
- **Number of single-variable iterations:** _N_
- **Parameters touched:** _list_
- **Did the tuner use QUADRUPED_PRIOR_ART.md citations in hypotheses?** _Y/N + examples_
- **Did Production Readiness Checklist all 8 criteria pass?** _Y/N — which failed if any_

## Verdict

Pass criteria (issue #2 §4 acceptance):

- [ ] Final `lin_vel_xy_rmse < 0.25` AND `yaw_rate_rmse < 0.3`
- [ ] Final `survival.alive_at_1000_steps > 0.9`
- [ ] Final `gait.arrhythmic == false`
- [ ] Final `posture.upright_frac > 0.95`
- [ ] Final `obs_shift_magnitude < 1.0` (or justified deviation)
- [ ] Sabotaged weights pulled back into prior-art ranges (≥3 of 5)
- [ ] Tuner completed without manual edits to the env_cfg

**Overall:** _PASS / FAIL_

**Notes / unexpected behavior:** _free text_

**Follow-ups:** _e.g., open issues for any signals the tuner couldn't recover from._
