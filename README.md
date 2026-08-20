# Robots Imitation Learning

This repository contains two independently runnable robot imitation-learning projects. The history on `final-code-history` is a transparent, dependency-ordered reconstruction from completed local sources; commit dates and authors were not backdated or fabricated.

## Repository layout

```text
task1/ManiSkill/   ManiSkill PickCube simulation and Diffusion Policy workflow
task2/lerobot/     LeRobot SO-101 real-robot workflow
```

## Source provenance

| Project | Baseline | Local additions |
|---|---|---|
| Task 1 | `haosulab/ManiSkill` commit `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` | PickCube expert demonstrations, wrapped-environment teleoperation fix, evaluation metric compatibility |
| Task 2 | `JoyandAI/lerobot` commit `aa8f59362b2897623a2f48fefe08a973bf52f020` | Persistent SO-101 device paths, dataset column materialization, Rerun scalar API compatibility |

The original project licenses and third-party notices remain in each project directory.

## Task 1 data flow

```text
ManiSkill motion-planning teacher
              |
              v
teacher_pickcube_mp_100.h5 + matching JSON
              |
              v
trajectory replay and observation/control conversion
              |
              v
state observations + pd_ee_delta_pos actions
              |
              v
Diffusion Policy training, evaluation, and checkpointing
```

The committed pair below is the raw expert-demonstration input, not a trained model or evaluation result:

- `task1/ManiSkill/PickCube-v1/motionplanning/teacher_pickcube_mp_100.h5`
- `task1/ManiSkill/PickCube-v1/motionplanning/teacher_pickcube_mp_100.json`

It contains 100 successful `PickCube-v1` motion-planning trajectories recorded with `obs_mode=none` and `control_mode=pd_joint_pos`. Replay is therefore required before state-based Diffusion Policy training.

### Replay the raw demonstrations

From `task1/ManiSkill`:

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path PickCube-v1/motionplanning/teacher_pickcube_mp_100.h5 \
  --use-first-env-state \
  --target-control-mode pd_ee_delta_pos \
  --obs-mode state \
  --save-traj \
  --sim-backend cpu
```

Expected generated dataset:

```text
PickCube-v1/motionplanning/teacher_pickcube_mp_100.state.pd_ee_delta_pos.physx_cpu.h5
PickCube-v1/motionplanning/teacher_pickcube_mp_100.state.pd_ee_delta_pos.physx_cpu.json
```

### Run a short training smoke test

From `task1/ManiSkill/examples/baselines/diffusion_policy`:

```bash
python train.py \
  --env-id PickCube-v1 \
  --demo-path ../../../PickCube-v1/motionplanning/teacher_pickcube_mp_100.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode pd_ee_delta_pos \
  --sim-backend physx_cpu \
  --num-demos 10 \
  --batch-size 256 \
  --max-episode-steps 100 \
  --total-iters 3000 \
  --log-freq 100 \
  --eval-freq 1000 \
  --save-freq 1000 \
  --num-eval-episodes 20 \
  --num-eval-envs 10 \
  --exp-name teacher_fullflow_smoke \
  --no-capture-video
```

The smoke test verifies that loading, training, evaluation, and checkpoint saving execute. It does not by itself prove successful autonomous PickCube behavior.

## Task 2 notes

Task 2 keeps the LeRobot source tree under `task2/lerobot`. The local SO-101 configuration uses persistent Linux `/dev/serial/by-id` and `/dev/v4l/by-id` paths so device enumeration order does not change robot or camera assignment.

The configured camera names are:

- `fixed`: fixed external camera
- `handeye`: wrist/hand-eye camera

Device paths are machine-specific. Update them for a different workstation before connecting hardware. See `task2/lerobot/examples/12_use_so101.md` for the baseline SO-101 workflow.

## Deliberately excluded artifacts

The reconstructed history does not include locally generated or machine-specific artifacts:

- training `runs`, checkpoints, logs, and evaluation/result videos;
- Task 1 result evidence and locally generated replay datasets;
- the truncated 96-byte StackCube teleoperation HDF5 file;
- backup files and file-mode-only changes;
- SO-101 calibration caches and calibration backups;
- accidentally created CLI-option filenames;
- work-in-progress HybridCV and target-recognition design documents.

Official media and test fixtures that are part of the upstream baseline snapshots remain intact.
