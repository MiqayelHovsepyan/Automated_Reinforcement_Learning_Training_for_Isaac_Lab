# Isaac-based Quadrupedal Locomotion Repositories

A catalog of GitHub/GitLab repositories implementing quadrupedal (and hybrid quadruped+humanoid) locomotion using NVIDIA Isaac frameworks (Isaac Lab, Isaac Gym, Isaac Sim, Orbit).

Last compiled: 2026-05.

Inclusion criteria: repos that (a) use an Isaac framework and (b) support at least one quadruped (Unitree A1/Go1/Go2/B1/B2/Aliengo, ANYmal-B/C/D, Boston Dynamics Spot, MIT Mini Cheetah, Deeprobotics Lite3, Solo, Cassie-adjacent hybrid platforms, custom 3D-printed quadrupeds, etc.). Humanoid-only repos are excluded; hybrid (quad+humanoid) repos are included.

---

## Official / Foundational

### [legged_gym](https://github.com/leggedrobotics/legged_gym)
**Org:** ETH RSL  ·  **Framework:** Isaac Gym (Preview 4)  ·  **Robots:** ANYmal-B/C, Unitree A1, Cassie
The original Rudin et al. "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021) codebase. Reference implementation for GPU-parallelised legged locomotion that essentially every subsequent quadruped RL repo forks or borrows from. Includes actuator nets, terrain curricula, friction/mass randomisation, and noisy observation augmentation needed for sim-to-real.

### [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
**Org:** ETH RSL  ·  **Framework:** Isaac Gym / Isaac Lab compatible  ·  **Robots:** (algorithmic library)
Fast, minimal PPO/teacher-student/distillation library designed to run fully on GPU. Default learner backend for legged_gym, Isaac Lab locomotion environments, walk-these-ways, parkour, HIMLoco, and most other quadruped RL projects in this list.

### [IsaacLab](https://github.com/isaac-sim/IsaacLab)
**Org:** NVIDIA (isaac-sim)  ·  **Framework:** Isaac Lab (on Isaac Sim)  ·  **Robots:** ANYmal-B/C/D, Unitree A1/Go1/Go2, Boston Dynamics Spot, Unitree H1/G1, Cassie, Digit, plus manipulators
The current canonical NVIDIA framework (successor to Orbit). Ships ready-to-train velocity-tracking + rough-terrain curricula for 11 morphologies including A1, Go1, Go2, ANYmal-B/C/D, Cassie, Digit, Spot, H1, G1. RSL-RL, RL-Games, SKRL, Stable Baselines integration. The "default starting point" for new quadruped RL work as of 2025-2026.

### [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) (formerly NVIDIA-Omniverse/IsaacGymEnvs)
**Org:** NVIDIA (isaac-sim)  ·  **Framework:** Isaac Gym Preview 4  ·  **Robots:** ANYmal, A1, Cassie, plus manipulators/humanoids
NVIDIA's official RL examples accompanying the NeurIPS 2021 Isaac Gym paper. Includes ANYmal terrain locomotion, AMP humanoid (template for quadruped AMP forks), and shadow-hand tasks. Now in maintenance-mode in favour of Isaac Lab, but still the reference codebase many older papers target.

### [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) (formerly NVIDIA-Omniverse/OmniIsaacGymEnvs)
**Org:** NVIDIA (isaac-sim)  ·  **Framework:** Omniverse Isaac Gym  ·  **Robots:** ANYmal, A1, plus shadow hand etc.
Bridge release that ported the IsaacGymEnvs tasks (incl. ANYmal terrain and AnymalTerrain) into Omniverse-based Isaac Sim before the Orbit/Isaac Lab unification. Useful if you need the legacy task definitions; deprecated in favour of Isaac Lab.

### [orbit](https://github.com/isaac-sim/IsaacLab) (formerly isaac-orbit/orbit, NVIDIA-Omniverse/orbit)
**Org:** NVIDIA  ·  **Framework:** Isaac Orbit (predecessor to Isaac Lab)  ·  **Robots:** ANYmal, A1, etc.
The original Orbit repo from Mittal et al. has been folded into Isaac Lab. Historical legged tasks (Isaac-Velocity-Flat/Rough-Anymal-D-v0, etc.) originated here. Many 2023-era forks still reference orbit branches; follow the redirect to IsaacLab for current versions.

### [IsaacLabEureka](https://github.com/isaac-sim/IsaacLabEureka)
**Org:** NVIDIA Research  ·  **Framework:** Isaac Lab  ·  **Robots:** any Isaac Lab task incl. quadrupeds
LLM-driven reward-function search pipeline for Isaac Lab. Used for automatic shaping of quadruped locomotion rewards (e.g., Spot, ANYmal). Worth including if you want auto-tuned reward engineering rather than hand-crafted reward terms.

---

## Quadruped-Focused (Isaac Lab)

### [robot_lab](https://github.com/fan-ziqi/robot_lab)
**Org:** fan-ziqi (independent)  ·  **Framework:** Isaac Lab v2  ·  **Robots:** ANYmal-D, Unitree Go2/B2/A1, Go2W/B2W (wheeled), Deeprobotics Lite3, Zsibot ZSL1, Magiclab MagicDog, Agibot D1
Probably the most actively maintained third-party Isaac Lab extension for quadrupeds. Adds many robots and wheel-legged variants on top of stock Isaac Lab, plus extensive domain-randomisation hooks (mass/inertia/COM/actuator-gain/external-force randomisation) and an asymmetric actor-critic structure tuned for sim-to-real. Also hosts a "BeyondMimic" motion-imitation pipeline.

### [LeggedLab](https://github.com/Hellod035/LeggedLab) (canonical "Hellod026/legged_lab" variant)
**Org:** Hellod035 (Wandong Sun)  ·  **Framework:** Isaac Lab (direct workflow)  ·  **Robots:** tested on real Unitree G1 + H1; quadruped tasks present
Provides a direct (non-manager-based) Isaac Lab workflow for legged robots, including multi-GPU/multi-node RSL-RL training. Has been validated sim-to-real on Unitree G1 and H1, with quadruped configs available. The "legged_lab" name the user remembered corresponds to this repo.

### [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
**Org:** Unitree Robotics (official)  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree Go2, H1, G1-29dof
Unitree's official Isaac-Lab RL training + deployment pipeline. Trains with RSL-RL, validates sim-to-sim in MuJoCo, exports to ONNX, and ships a C++ deployment controller using Unitree SDK2. Currently the cleanest "out-of-box" Isaac Lab → real Go2 path.

### [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
**Org:** Unitree Robotics (official)  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree quadrupeds + humanoids
Unitree's Isaac-Lab-based simulation environment focused on data collection, playback, generation, and model validation. Companion to unitree_rl_lab; useful for evaluation pipelines and dataset replay.

### [basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)
**Org:** IIT Dynamic Legged Systems Lab  ·  **Framework:** Isaac Lab v2.3+  ·  **Robots:** Unitree Aliengo, Go2, B2, HyQReal2
IIT-DLS's IsaacLab extension covering four quadrupeds, each with flat-blind, rough-blind, and rough-vision variants. Implements concurrent state estimation, rapid motor adaptation (RMA-style), and morphological-symmetry losses. Includes sim-to-sim and sim-to-real scripts.

### [legged-loco](https://github.com/yang-zj1026/legged-loco)
**Org:** Zhejiang U / yang-zj1026  ·  **Framework:** Isaac Lab 1.1  ·  **Robots:** Unitree Go2 (quadruped) + Unitree H1 (humanoid)
Hybrid quad + humanoid low-level locomotion policy training, often used as a base for hierarchical / VLM-driven controller stacks. Cited by several follow-up navigation works.

### [isaac-quad-loco](https://github.com/dyumanaditya/isaac-quad-loco)
**Org:** dyumanaditya  ·  **Framework:** Isaac Sim Orbit / Isaac Lab  ·  **Robots:** Anymal-D
Combines Isaac-PPO learning with MPC for quadruped locomotion. Useful as a clean reference for learning + classical-control hybrids inside the Isaac ecosystem.

### [isaaclab-anymal-locomotion](https://github.com/mturan33/isaaclab-anymal-locomotion)
**Org:** mturan33  ·  **Framework:** Isaac Lab  ·  **Robots:** ANYmal-C/D
From-scratch PPO implementation that reaches ~96% of RSL-RL performance on Isaac-Velocity tasks. Pedagogically valuable if you want a transparent PPO loop without the RSL-RL abstractions.

### [IsaacLab-Quadruped-Locomotion](https://github.com/huangfq07/IsaacLab-Quadruped-Locomotion)
**Org:** huangfq07  ·  **Framework:** Isaac Lab  ·  **Robots:** ANYmal-C
Enhanced PPO for Isaac-Velocity-Flat-Anymal-C-v0 with adaptive learning-rate scheduling. Small, focused fork good for studying PPO hyperparameter tweaks.

### [IsaacLab-for-Go2-RL](https://github.com/timbojones91/IsaacLab-for-Go2-RL)
**Org:** timbojones91  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree Go2
Go2-specific Isaac Lab fork with custom velocity/rough tasks. A simpler entry point than the full unitree_rl_lab for Go2-only experimentation.

### [Anymal_Navigation](https://github.com/LucaFrat/Anymal_Navigation)
**Org:** LucaFrat (ETH thesis)  ·  **Framework:** Isaac Lab  ·  **Robots:** ANYmal-C
Hierarchical RL for ANYmal-C: trains a high-level planner on top of a pre-trained low-level locomotion policy to navigate cluttered rough terrain. Demonstrates HRL composition cleanly inside Isaac Lab.

### [RL_Dog](https://github.com/pietrodardano/RL_Dog)
**Org:** pietrodardano  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree AlienGo
Walk + stop policies for the Unitree AlienGo in Isaac Lab. One of the few public AlienGo-on-Isaac-Lab references.

### [isaac-wild-go2](https://github.com/Charlescai123/isaac-wild-go2)
**Org:** Charlescai123 (2025)  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree Go2
"Runtime Learning Framework" letting a Go2 explore and adapt to open / wild environments safely. Focus on safe online adaptation rather than pure sim training.

### [DeepTransition](https://github.com/MiladShafiee/DeepTransition)
**Org:** MiladShafiee  ·  **Framework:** Isaac Gym (build is migratable)  ·  **Robots:** ANYmal-class quadruped
Companion code for "Viability leads to the emergence of gait transitions in learning agile quadrupedal locomotion on challenging terrains" (Nature Communications, 2024). Studies emergent gait switching on rough terrain.

### [Reinforcement-Learning-Isaac-Lab-Projects](https://github.com/nicolaloi/Reinforcement-Learning-Isaac-Lab-Projects)
**Org:** nicolaloi  ·  **Framework:** Isaac Lab  ·  **Robots:** mixed incl. quadruped tasks
Collection of small experimental Isaac Lab RL environments (quadruped + manipulator). Useful as a teaching/recipe repository.

### [IsaacLab-Tutorial](https://github.com/Lab-of-AI-and-Robotics/IsaacLab-Tutorial)
**Org:** Lab of AI and Robotics (Korea)  ·  **Framework:** Isaac Lab  ·  **Robots:** Unitree Go2 (quad), Unitree H1 (humanoid)
A ten-chapter tutorial codebase walking from baseline Go2 quadruped to H1 humanoid RL. Best entry point for new users wanting both morphologies covered pedagogically.

---

## Quadruped-Focused (Isaac Gym)

### [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
**Org:** Improbable AI (MIT, Margolis & Agrawal)  ·  **Framework:** Isaac Gym Preview 4  ·  **Robots:** Unitree Go1
The canonical CoRL 2022 implementation of MoB (Multiplicity of Behaviour) for Go1. Trains a single policy that exposes commandable gait parameters (frequency, duty, footswing height, body height/pose) and deploys via unitree_legged_sdk. Still the most-cited Go1 sim-to-real reference.

### [walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2)
**Org:** Teddy Liao  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go2
Direct Go2 port of walk-these-ways, swapping the legacy unitree_legged_sdk (UDP) for unitree-sdk2 (DDS). The de facto answer to "how do I run WTW on a Go2?"

### [rapid-locomotion-rl](https://github.com/Improbable-AI/rapid-locomotion-rl)
**Org:** Improbable AI (MIT)  ·  **Framework:** Isaac Gym  ·  **Robots:** MIT Mini Cheetah, Unitree Go1
RSS 2022 paper code (Margolis et al.). Implements the Grid Adaptive Curriculum and teacher–student distillation pipeline used to push Mini Cheetah / Go1 to high commanded velocities.

### [dribblebot](https://github.com/Improbable-AI/dribblebot)
**Org:** Improbable AI (MIT)  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go1
ICRA 2023 "Dynamic Legged Manipulation in the Wild": Go1 soccer-ball dribbling policy. Builds on the walk-these-ways training infrastructure. Useful if you need loco-manipulation reference code rather than pure locomotion.

### [learning-compliance](https://github.com/Improbable-AI/learning-compliance)
**Org:** Improbable AI (MIT)  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree B1 + Z1 arm
Trains a compliant whole-body controller for B1+arm. One of the few public references for the Unitree B1 quadruped on Isaac Gym.

### [extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
**Org:** Cheng et al. (CMU)  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree A1/Go1-class
ICRA 2024. "Train your parkour robot in less than 20 hours." Single end-to-end vision-conditioned policy from depth → torque for high-jumps, gap-leaps, handstands, tilted ramps. Hugely influential 2024 result.

### [Isaaclab_Parkour](https://github.com/CAI23sbP/Isaaclab_Parkour)
**Org:** CAI23sbP (community port)  ·  **Framework:** Isaac Lab  ·  **Robots:** quadruped (Go1/A1-class)
Community port of extreme-parkour to Isaac Lab, endorsed by the original authors. Lets you run parkour training on the modern Isaac Lab stack instead of Preview-4 Isaac Gym.

### [parkour (ZiwenZhuang)](https://github.com/ZiwenZhuang/parkour)
**Org:** Tsinghua / Stanford / CMU  ·  **Framework:** Isaac Gym + rsl_rl  ·  **Robots:** A1-class quadruped
"Robot Parkour Learning" (CoRL 2023, Best Systems Paper finalist). Trains specialist skills (climb, leap, crawl, squeeze, run) then distills via DAgger into one vision-conditioned policy. Often paired with extreme-parkour as the two reference 2023-24 quadruped parkour codebases.

### [HIMLoco](https://github.com/OpenRobotLab/HIMLoco) (also mirrored at [InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco))
**Org:** OpenRobotLab / Shanghai AI Lab (Junfeng Long et al.)  ·  **Framework:** Isaac Gym Preview 4 (4096 envs)  ·  **Robots:** Unitree A1, Aliengo
Implements both "Hybrid Internal Model" (ICLR 2024) and "H-Infinity Locomotion Control" (2024). Adds an internal-model branch to PPO that predicts robot response to inputs, giving robust real-world performance on stairs, slopes, slippery surfaces.

### [DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ)
**Org:** Manaro-Alpha (community)  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree A1
Re-implementation of "DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep RL." Uses a context encoder + variational world model for blind robust locomotion.

### [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware) ([escontra fork](https://github.com/escontra/AMP_for_hardware))
**Org:** Alejandro Escontrela (UC Berkeley)  ·  **Framework:** Isaac Gym (fork of legged_gym)  ·  **Robots:** Unitree A1
"Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions" (2022). Grounds skills with ~4.5 s of mocap reference data instead of hand-engineered rewards. The most-cited AMP-for-quadruped reference.

### [MetalHead](https://github.com/inspirai/MetalHead)
**Org:** Inspire AI  ·  **Framework:** Isaac Gym (AMP variant)  ·  **Robots:** Unitree A1
AMP-style natural locomotion, jumping, and recovery for A1, building on AMP_for_hardware. Notable for the recovery/jumping skills.

### [rl_amp](https://github.com/fan-ziqi/rl_amp)
**Org:** fan-ziqi  ·  **Framework:** Isaac Gym (legged_gym + rsl_rl)  ·  **Robots:** legged_gym-compatible quadrupeds
Minimal AMP implementation on top of legged_gym + rsl_rl. Easier to read than AMP_for_hardware if you just want the algorithmic delta.

### [rsl_rl_AMP](https://github.com/osudrl/rsl_rl_AMP)
**Org:** Oregon State University Dynamic Robotics Lab  ·  **Framework:** GPU rsl_rl extension  ·  **Robots:** legged_gym-compatible
RSL-RL fork with AMP added; useful drop-in replacement learner for legged_gym-style training.

### [quadrupedal-agility](https://github.com/NJU-RLC/quadrupedal-agility)
**Org:** Nanjing U RL+Control Lab  ·  **Framework:** Isaac Gym  ·  **Robots:** quadruped (Aliengo-class)
"Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots." Combines RL with imitation to expand skill repertoire.

### [GenLoco](https://github.com/HybridRobotics/GenLoco)
**Org:** Berkeley Hybrid Robotics Lab (Feng et al.)  ·  **Framework:** Isaac Gym  ·  **Robots:** randomised quadruped morphologies (A1, Go1, Mini Cheetah, AlienGo, etc.)
"Generalized Locomotion Controllers for Quadrupedal Robots" (CoRL 2022). Trains on randomised morphologies so a single policy transfers across many quadruped platforms.

### [Deep-Tracking-Control](https://github.com/priest-yang/Deep-Tracking-Control)
**Org:** priest-yang  ·  **Framework:** Isaac Gym + MPC  ·  **Robots:** quadruped
Implements "Deep Tracking Control" combining RL policy with MPC tracking. Reference for hybrid learning + optimal-control pipelines.

### [rl-mpc-locomotion](https://github.com/silvery107/rl-mpc-locomotion)
**Org:** silvery107  ·  **Framework:** Isaac Gym + Cheetah-MPC  ·  **Robots:** Unitree AlienGo (Mini Cheetah controllers ported)
Hierarchical RL high-level + MIT-Cheetah MPC low-level. Common didactic reference for layered control.

### [legged_env (generalroboticslab)](https://github.com/generalroboticslab/legged_env)
**Org:** General Robotics Lab (Duke)  ·  **Framework:** Isaac Gym + rl_games  ·  **Robots:** template for arbitrary legged robots
Drop-in Isaac Gym template designed so you can plug in any URDF and train. Used as the base for "Text2Robot" zero-shot sim-to-real demos.

### [legged_gym_isaac (chengxuxin)](https://github.com/chengxuxin/legged_gym_isaac)
**Org:** Xuxin Cheng  ·  **Framework:** Isaac Gym  ·  **Robots:** quadrupeds incl. A1/Go1
Cheng's personal fork/predecessor that fed into extreme-parkour. Light fork of legged_gym with extra training utilities.

### [quadruped_isaacgym](https://github.com/ZaneCodeJourney/quadruped_isaacgym)
**Org:** ZaneCodeJourney  ·  **Framework:** Isaac Gym  ·  **Robots:** quadruped
"Model-free End-to-end Learning of Agile Quadrupedal Locomotion" — proprioceptive-only blind locomotion across rough terrain. Compact reference for the blind-baseline approach.

### [go2gym](https://github.com/X-Noname-X/go2gym)
**Org:** X-Noname-X  ·  **Framework:** Isaac Gym + legged_gym  ·  **Robots:** Unitree Go2
Go2 adaptation of legged_gym (without the Improbable AI WTW additions). Lightweight if you only need stock PPO velocity tracking on Go2 in Isaac Gym.

### [go2_rl_gym](https://github.com/gabearod2/go2_rl_gym)
**Org:** gabearod2  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go2
Another Go2-focused Isaac Gym fork with custom reward shaping. Includes deployment helpers.

### [wtw_legged_gym](https://github.com/GuoPingPan/wtw_legged_gym)
**Org:** GuoPingPan  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go1
Walk-These-Ways re-organised on top of stock legged_gym, simplifying merging WTW changes into other legged_gym derivatives.

### [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
**Org:** Unitree Robotics (official)  ·  **Framework:** Isaac Gym Preview 4  ·  **Robots:** Unitree Go2, H1, H1_2, G1
Unitree's official legged_gym-style training repo (the Isaac Gym counterpart to unitree_rl_lab). Currently the simplest "official Unitree" starting point if you are still on Isaac Gym.

### [ai3603_legged_gym](https://github.com/Bireflection/ai3603_legged_gym)
**Org:** SJTU course (AI3603)  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go1
Course-companion fork of legged_gym for Go1 — pedagogically useful, well-commented.

---

## Hybrid (Quadruped + Humanoid)

### [robot_lab](https://github.com/fan-ziqi/robot_lab) (also above)
Eight quadrupeds + ten humanoids in one Isaac-Lab extension. The widest morphology coverage of any third-party Isaac Lab repo.

### [LeggedLab](https://github.com/Hellod035/LeggedLab) (also above)
Quadruped and humanoid (G1/H1 confirmed sim-to-real) under one direct-workflow Isaac Lab codebase.

### [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) (also above)
Officially covers Go2 quadruped + H1/G1 humanoids with one training stack.

### [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (also above)
Isaac Gym counterpart, same hybrid coverage.

### [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse)
**Org:** CMU LeCAR Lab  ·  **Framework:** Isaac Gym + Isaac Sim + Genesis (multi-sim)  ·  **Robots:** primarily Unitree H1/G1 humanoids
Despite the name, the framework abstraction is morphology-agnostic and is regularly used alongside quadruped tasks in the LeCAR ecosystem. Worth tracking because its multi-simulator backend pattern is replicated in newer quadruped + humanoid hybrid repos. (Include only if you specifically want LeCAR's plumbing — it does *not* ship quadruped tasks by default, so it's borderline against your "must support a quadruped" rule.)

### [ABS](https://github.com/LeCAR-Lab/ABS)
**Org:** CMU LeCAR Lab  ·  **Framework:** Isaac Gym (legged_gym subdir)  ·  **Robots:** Unitree Go1
"Agile But Safe: Collision-Free High-Speed Legged Locomotion" (RSS 2024). Quadruped-specific paper from a humanoid-heavy lab — train + deploy code under training/legged_gym.

### [renanmb/Omniverse_legged_robotics](https://github.com/renanmb/Omniverse_legged_robotics)
**Org:** renanmb (community)  ·  **Framework:** Isaac Sim USDs  ·  **Robots:** large catalogue of legged URDF/USD models (quadruped + biped)
Not training code — a curated asset library to bring extra legged morphologies into Omniverse / Isaac Sim. Useful when robot_lab/Isaac Lab don't ship the robot you want.

### [hit_omniverse](https://github.com/shaosb/hit_omniverse)
**Org:** shaosb (HIT)  ·  **Framework:** Isaac Lab  ·  **Robots:** humanoid focus, but framework is hybrid-capable
Decoupled imitation + locomotion in Isaac Lab. Listed because it is often forked for hybrid quad/humanoid setups; check whether the latest commits add quadruped configs.

---

## Walk-These-Ways Implementations & Derivatives

(Most direct WTW work is in Isaac Gym since the original predates Isaac Lab.)

### [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
Canonical MIT WTW for Go1, Isaac Gym. (See above.)

### [Teddy-Liao/walk-these-ways-go2](https://github.com/Teddy-Liao/walk-these-ways-go2)
Direct Go2 port. (See above.)

### [Iniationware/unitree-go-walk-these-ways](https://github.com/Iniationware/unitree-go-walk-these-ways)
**Org:** Iniationware  ·  **Framework:** Isaac Gym  ·  **Robots:** Unitree Go-series
Another deployment-focused WTW fork with updated SDK plumbing.

### [GuoPingPan/wtw_legged_gym](https://github.com/GuoPingPan/wtw_legged_gym)
WTW re-organised on top of stock legged_gym. (See above.)

### "WTW in Isaac Lab"
No fully-featured, canonical port of WTW to Isaac Lab exists as of mid-2026; the closest equivalents are (a) the WTW actuator model that was upstreamed into Isaac Lab's Go1 actuator config, and (b) feature-equivalent gait-parameter conditioning recipes in robot_lab and unitree_rl_lab. Track Improbable-AI's org for any future port.

---

## Sim-to-Real / Deployment-Focused

### [go2_omniverse](https://github.com/abizovnuralem/go2_omniverse)
**Org:** abizovnuralem  ·  **Framework:** Isaac Sim + Orbit/Isaac Lab  ·  **Robots:** Unitree Go2, G1
Brings Go2/G1 into Isaac Sim with ROS2, multi-robot, and VR-teleop integration. Very popular as a starting point for sim-side data collection and visualisation, less so for raw RL training.

### [isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)
**Org:** Zhefan Xu (CMU)  ·  **Framework:** Isaac Sim + Isaac Lab + ROS 2  ·  **Robots:** Unitree Go2
Go2 simulation platform aimed at navigation, decision-making, and autonomy testing rather than low-level RL. Pairs with the NavRL repo from the same author.

### [go2_isaac_ros2](https://github.com/CLeARoboticsLab/go2_isaac_ros2)
**Org:** UT Austin CLeAR Lab  ·  **Framework:** Isaac Sim + Isaac Lab + ROS 2  ·  **Robots:** Unitree Go2
ROS2 low-level (joint-level) control bridge for a Go2 in Isaac Sim, emulating the /lowcmd interface. Excellent if you want to drop policies trained elsewhere into a ROS2 control stack identical to the real robot.

### [sim2real-3d-printed-quadruped](https://github.com/shaheenbharwani/sim2real-3d-printed-quadruped)
**Org:** Shaheen Bharwani  ·  **Framework:** Isaac Lab → PyTorch → ROS 2 → Arduino  ·  **Robots:** custom 12-DoF 3D-printed quadruped
Full open-source pipeline from Isaac Lab training to a real 3D-printed quadruped via ROS2 and Arduino. Rare end-to-end example for non-commercial / hobby quadruped hardware.

### [master_kin](https://github.com/ChristophKin/master_kin)
**Org:** ChristophKin  ·  **Framework:** Isaac Sim  ·  **Robots:** Unitree Go1, Go2
Master's-thesis codebase bringing both Go1 and Go2 into Isaac Sim with low-level controllers.

### [boston-dynamics/spot-rl-example](https://github.com/boston-dynamics/spot-rl-example)
**Org:** Boston Dynamics + The AI Institute  ·  **Framework:** Isaac Lab (training side)  ·  **Robots:** Spot
Official BD/AII deployment-side code for the Spot RL Research Kit. Pairs with Isaac Lab's Spot velocity-tracking tasks for the canonical Isaac Lab → real Spot pipeline (the "Closing the Sim-to-Real Gap" NVIDIA blog post). Together they form NVIDIA + BD's "default" Spot reference stack.

### [QuadrupedRobotSimulator](https://github.com/AuTURBO/QuadrupedRobotSimulator)
**Org:** AuTURBO  ·  **Framework:** Isaac Sim  ·  **Robots:** generic quadruped
Lightweight Isaac-Sim-based simulator template for quadruped control experiments (mostly classical control, but plumbing is useful for sim-side deployment testing).

### [SKYBIRDSGP/Quadruped-Isaac-Sim](https://github.com/SKYBIRDSGP/Quadruped-Isaac-Sim)
**Org:** SKYBIRDSGP  ·  **Framework:** Isaac Sim  ·  **Robots:** generic 12-DoF quadruped
Gait-control + simulation testbed for a quadruped in Isaac Sim. Useful as a minimal scaffolding example.

---

## Research / Paper Code Releases

### [constraints-as-terminations](https://github.com/Gepetto/constraints-as-terminations)
**Org:** LAAS-CNRS Gepetto  ·  **Framework:** Isaac Gym + Isaac Lab  ·  **Robots:** Solo, Anymal-class quadrupeds
Reference implementation of "Integrating Constraints in PPO by Treating Them as Terminations." Particularly nice for the Solo robot (rarely represented in Isaac repos).

### [MimicKit](https://github.com/xbpeng/MimicKit)
**Org:** Xue Bin "Jason" Peng (SFU)  ·  **Framework:** Isaac Gym + Isaac Lab backends  ·  **Robots:** humanoid + quadruped imitation tasks
Lightweight suite of motion-imitation methods (DeepMimic, AMP, ASE) with a clean Isaac Gym/Lab backend abstraction. Maintained replacement for the older DeepMimic + motion_imitation codebases.

### [MimicKit_IsaacLab](https://github.com/NathanWu7/MimicKit_IsaacLab)
**Org:** NathanWu7  ·  **Framework:** Isaac Lab  ·  **Robots:** any MimicKit-supported character incl. quadrupeds
Isaac-Lab-only port of MimicKit. Combine with robot_lab's "BeyondMimic" to do reference-motion imitation directly inside Isaac Lab.

### [LocoMan](https://github.com/linchangyi1/LocoMan)
**Org:** CMU (Lin Chang-Yi et al.)  ·  **Framework:** Isaac Gym + Unitree Go1  ·  **Robots:** Unitree Go1 with custom loco-manipulators
IROS 2024 loco-manipulation paper code. Useful if you care about extending quadruped locomotion with foot-mounted manipulation hardware.

### [NeuralIMC-Quadruped](https://github.com/UltronAI/NeuralIMC-Quadruped)
**Org:** UltronAI  ·  **Framework:** Isaac Gym  ·  **Robots:** quadruped
Neural Internal Model Control extension for quadruped locomotion. Related in spirit to HIMLoco.

### [NavRL](https://github.com/Zhefan-Xu/NavRL)
**Org:** Zhefan Xu (CMU)  ·  **Framework:** NVIDIA Isaac + ROS1/ROS2  ·  **Robots:** UAV primarily, with quadruped variants
"NavRL: Learning Safe Flight in Dynamic Environments" (RA-L 2025). Included because the framework is regularly adapted for Go2 navigation in the author's companion repo (isaac-go2-ros2).

### [Improbable-AI/rapid-locomotion-rl](https://github.com/Improbable-AI/rapid-locomotion-rl) (also above)
RSS 2022 reference for grid adaptive curricula.

### [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) (also above)
ICRA 2024 parkour reference.

### [ZiwenZhuang/parkour](https://github.com/ZiwenZhuang/parkour) (also above)
CoRL 2023 parkour reference.

### [OpenRobotLab/HIMLoco](https://github.com/OpenRobotLab/HIMLoco) (also above)
ICLR 2024 + 2024 H-Infinity reference.

### [LeCAR-Lab/ABS](https://github.com/LeCAR-Lab/ABS) (also above)
RSS 2024 safe high-speed quadruped locomotion reference.

---

## Curated Lists / Awesome Repos

(Not training code, but high-signal entry points to find more Isaac quadruped work.)

### [awesome-isaac-gym (robotlearning123)](https://github.com/robotlearning123/awesome-isaac-gym)
Most-cited Isaac Gym list; includes most of the seminal quadruped papers + code links.

### [awesome-isaac-gym (wangcongrobot)](https://github.com/wangcongrobot/awesome-isaac-gym)
Alternate Isaac Gym list; complementary coverage.

### [awesome-isaac-sim](https://github.com/sjtuyinjie/awesome-isaac-sim)
Isaac Sim-specific list (more breadth into manipulation & navigation but covers quadruped too).

### [awesome-legged-locomotion-learning](https://github.com/gaiyi7788/awesome-legged-locomotion-learning)
Broad legged-locomotion list, tagged by simulator.

### [awesome-rl-for-legged-locomotion (apexrl)](https://github.com/apexrl/awesome-rl-for-legged-locomotion)
Sim-to-real-flavoured legged locomotion list.

### [awesome-rl-robotics (fan-ziqi)](https://github.com/fan-ziqi/awesome-rl-robotics)
Curated mostly around Isaac Gym + Isaac Lab.

### [awesome-loco-manipulation](https://github.com/aCodeDog/awesome-loco-manipulation)
Loco-manipulation list (quadruped + arm). Many entries use Isaac.

### [awesome-unitree-robots (shaoxiang)](https://github.com/shaoxiang/awesome-unitree-robots)
Unitree-specific aggregator across Isaac Sim, MuJoCo, Gazebo, PyBullet.

### [Awesome_Quadrupedal_Robots](https://github.com/curieuxjy/Awesome_Quadrupedal_Robots)
Quadruped-focused, simulator-agnostic but tagged.

---

## GitLab

Coverage of Isaac-based quadruped projects on GitLab is sparse compared to GitHub. The major public ones found:

### ETH RSL GitLab (private)
ETH Robotic Systems Lab has historically hosted private GitLab mirrors of `legged_gym`, `rsl_rl`, and downstream papers (perceptive locomotion, ANYmal field-deployment code). Public mirrors are at github.com/leggedrobotics. No public-access GitLab quadruped Isaac codebase from RSL was located.

### IIT / Other European labs
Most IIT, KAIST, and other lab Isaac-quadruped repos live on GitHub (e.g., iit-DLSLab/basic-locomotion-isaaclab). No first-party GitLab equivalents were located.

If you specifically need a GitLab home, your best bet is to (a) check the institutional GitLab of the paper's authors (often gated), and (b) check Hugging Face Spaces, which has emerged as a third Isaac-Lab hosting target alongside GitHub.

---

## Notes on Excluded Repos

The following repos came up in searches but were excluded:

- **google-deepmind/barkour_robot**: Barkour is MuJoCo/MJX/Brax-based, not Isaac.
- **roboterax/humanoid-gym**, **LeCAR-Lab/human2humanoid**, **LeCAR-Lab/ASAP**, **LeCAR-Lab/SoFTA**, **mturan33/isaac-g1-***: humanoid-only.
- **xbpeng/DeepMimic**, **erwincoumans/motion_imitation**: predate Isaac integration; use PyBullet/custom sims.
- **google-deepmind/mujoco_playground**, **mujocolab/mjlab**, **robfiras/loco-mujoco**: MuJoCo/MJX-based (mjlab borrows Isaac Lab's API but does not use Isaac Sim).
- **jaykorea/Isaac-(gym|RL)-Two-wheel-Legged-Bot**, **jaykorea/Isaac-RL-Two-wheel-Legged-Bot**: two-wheeled bipedal balancers — borderline; included if you consider "wheel-legged" a quadruped variant, otherwise exclude.
- Most generic `IsaacLab` topic repos that are pure manipulation/arm projects (UR, Kinova, Franka, SO-ARM).
