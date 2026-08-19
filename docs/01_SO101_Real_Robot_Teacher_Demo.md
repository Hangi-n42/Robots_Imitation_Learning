# Experiment 2: SO-101 Real-Robot Imitation Learning — Complete Teacher Demonstration

> **Document type: Teacher Demo / Classroom Demonstration Manual**
>
> Goal: demonstrate the complete real-robot data path from a clean dataset:
>
> **teleoperation → collect 5 demonstrations → inspect the LeRobot Dataset → validate the data → decode video with PyAV → start Diffusion Policy training → observe step/loss logs**.
>
> Five demonstrations are used only for a classroom smoke test. They are not enough to claim reliable grasping performance.

---

# 0. Fixed experiment configuration

Task:

```text
Grasp the cube and lift it.
```

Pipeline:

```text
Human operates Leader
        ↓
Follower executes
        ↓
fixed + handeye RGB
+ follower joint state
+ expert teleoperation action
        ↓
control_robot.py --control.type=record
        ↓
LeRobot Dataset
├── Parquet: state / action / timestamp / indices
├── MP4: fixed / handeye
└── Metadata: info / episodes / stats / tasks
        ↓
data validation
        ↓
LeRobotDataset / DataLoader
        ↓
PyAV: MP4 → RGB Tensor
Parquet → state/action Tensor
        ↓
Diffusion Policy
        ↓
forward → loss → backward → optimizer.step()
```

| Item | Value |
|---|---|
| Robot | SO-101 Leader + Follower |
| Task | `Grasp the cube and lift it.` |
| Cameras | `fixed` + `handeye` |
| FPS | 30 |
| Demonstrations | 5 |
| Teacher repo_id | `dseg/demo_teacher_pick_cube_20260817_v1` |
| Policy | Diffusion Policy |
| Video backend | PyAV |
| Device | CUDA |
| Completion criterion | training log shows `step:... loss:...` |

---

# 1. Start a fresh terminal

```bash
cd ~/code/lerobot
conda activate lerobot
```

Verify:

```bash
which python
```

The path should contain:

```text
envs/lerobot
```

Define paths:

```bash
export HF_USER=dseg
export DEMO_ROOT="$HOME/code/lerobot/real_demo_20260817"

export TEACHER_REPO=dseg/demo_teacher_pick_cube_20260817_v1
export TEACHER_ROOT="$HOME/.cache/huggingface/lerobot/dseg/demo_teacher_pick_cube_20260817_v1"

export TEACHER_TRAIN_OUT="$DEMO_ROOT/teacher_demo/model/demo_teacher_pick_cube_5_diffusion"
```

Create the teacher workspace:

```bash
mkdir -p "$DEMO_ROOT/teacher_demo/data"
mkdir -p "$DEMO_ROOT/teacher_demo/model"
mkdir -p "$DEMO_ROOT/teacher_demo/logs"
```

---

# 2. Verify a clean starting state

```bash
test ! -e "$TEACHER_ROOT" \
  && echo "Teacher dataset path clean" \
  || echo "WARNING: Teacher dataset already exists"
```

```bash
test ! -e "$TEACHER_TRAIN_OUT" \
  && echo "Teacher training output path clean" \
  || echo "WARNING: Teacher training output already exists"
```

If these are only temporary teacher-test artifacts and you have verified the paths:

```bash
rm -f "$DEMO_ROOT/teacher_demo/data/dataset"
rm -rf "$TEACHER_ROOT"
rm -rf "$TEACHER_TRAIN_OUT"
```

Never remove student datasets, historical formal datasets, or calibration files.

---

# 3. Hardware checks

## Leader / Follower

```bash
ls -l /dev/serial/by-id/
```

Previously verified IDs on this machine:

```text
Leader:
usb-1a86_USB_Single_Serial_5A7A054767-if00

Follower:
usb-1a86_USB_Single_Serial_5A7A056628-if00
```

Prefer stable `/dev/serial/by-id/` paths over dynamic `/dev/ttyACM*` numbering.

## Cameras

```bash
ls -l /dev/video*
ls -l /dev/v4l/by-id/
```

The course setup uses:

```text
fixed:
GENERAL_WEBCAM

handeye:
DSJ-250318-J
```

## Calibration

```bash
ls -lh \
.cache/calibration/so101/main_follower.json \
.cache/calibration/so101/main_leader.json
```

Validate JSON:

```bash
python -m json.tool .cache/calibration/so101/main_follower.json >/dev/null \
  && echo "Follower calibration OK"

python -m json.tool .cache/calibration/so101/main_leader.json >/dev/null \
  && echo "Leader calibration OK"
```

---

# 4. Teleoperation rehearsal

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

Check:

```text
Human → Leader → Follower
```

In Rerun verify:

```text
observation.images.fixed   = external view
observation.images.handeye = gripper view
```

Also verify joint directions, gripper motion, and absence of collisions or abnormal high-speed motion.

Exit with:

```text
Ctrl+C
```

---

# 5. Record 5 teacher demonstrations

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${TEACHER_REPO} \
  --control.tags='["teacher","expert","real_demo","pick_cube","demo5"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=5 \
  --control.push_to_hub=false \
  --control.display_data=true
```

Aim for successful, smooth demonstrations:

```text
approach → grasp → lift 5–10 cm → stabilize → end episode
```

Common controls in the course build:

```text
Left Arrow   redo current episode
Right Arrow  finish current episode early
Esc          stop recording
```

Reset the cube and robot between episodes and keep the cameras fixed.

---

# 6. Inspect the generated Dataset

```bash
ls -lh "$TEACHER_ROOT"
```

Expected:

```text
data/
meta/
videos/
```

List everything:

```bash
find "$TEACHER_ROOT" -maxdepth 4 -type f | sort
```

A typical dataset contains:

```text
data/chunk-000/episode_XXXXXX.parquet

videos/chunk-000/observation.images.fixed/episode_XXXXXX.mp4
videos/chunk-000/observation.images.handeye/episode_XXXXXX.mp4

meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/tasks.jsonl
```

---

# 7. Explain where data conversion happens

In ManiSkill:

```text
raw motion-planning trajectory
→ replay_trajectory.py
→ explicit conversion
→ training dataset
```

On the SO-101 real robot:

```text
camera + state + teleoperation action
→ control_robot.py record
→ directly structured as a LeRobot Dataset
→ Parquet + MP4 + Metadata
```

There is no mandatory separate `convert.py`.

Training-time processing still performs:

```text
MP4 → PyAV → RGB Tensor
Parquet → state/action Tensor
timestamps → temporal windows
samples → Batch
```

---

# 8. Validate metadata

```bash
cat "$TEACHER_ROOT/meta/info.json"
```

Focus on:

```text
total_episodes
total_frames
total_tasks
total_videos
fps
features
```

For this demo:

```text
total_episodes = 5
total_videos   = 10
fps            = 30
```

Check JSONL counts:

```bash
wc -l \
"$TEACHER_ROOT/meta/episodes.jsonl" \
"$TEACHER_ROOT/meta/episodes_stats.jsonl" \
"$TEACHER_ROOT/meta/tasks.jsonl"
```

Expected:

```text
5
5
1
```

---

# 9. Inspect one Parquet episode

```bash
python - <<'PY'
import glob
import pandas as pd

root = "/home/dseg/.cache/huggingface/lerobot/dseg/demo_teacher_pick_cube_20260817_v1"
f = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))[0]
df = pd.read_parquet(f)

print("File:", f)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nAction:", df["action"].iloc[0])
print("\nState :", df["observation.state"].iloc[0])
PY
```

Explain:

```text
observation.state = measured follower state
action            = expert control target
```

They do not need to be identical because the real robot has tracking error and dynamics.

---

# 10. Validate all episodes

```bash
python - <<'PY'
import glob, os
import pandas as pd
import numpy as np

root = "/home/dseg/.cache/huggingface/lerobot/dseg/demo_teacher_pick_cube_20260817_v1"
files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))

all_df = []
for f in files:
    df = pd.read_parquet(f)
    all_df.append(df)
    state = np.stack(df["observation.state"].apply(np.asarray))
    action = np.stack(df["action"].apply(np.asarray))
    print(
        os.path.basename(f),
        "frames =", len(df),
        "state =", state.shape,
        "action =", action.shape,
        "nan_state =", np.isnan(state).any(),
        "nan_action =", np.isnan(action).any(),
    )

df = pd.concat(all_df, ignore_index=True)
state = np.stack(df["observation.state"].apply(np.asarray))
action = np.stack(df["action"].apply(np.asarray))

print("total frames:", len(df))
print("state range :", np.ptp(state, axis=0))
print("action range:", np.ptp(action, axis=0))
PY
```

Pass criteria:

```text
5 episodes
(N, 6) state/action
no NaNs
reasonable trajectory lengths
non-degenerate joint ranges
```

---

# 11. Validate PyAV and video decoding

```bash
python - <<'PY'
import av
print("PyAV version:", av.__version__)
PY
```

Decode one video from each camera:

```bash
python - <<'PY'
import av, glob

root = "/home/dseg/.cache/huggingface/lerobot/dseg/demo_teacher_pick_cube_20260817_v1"

for key in ["observation.images.fixed", "observation.images.handeye"]:
    files = sorted(glob.glob(root + f"/videos/**/{key}/*.mp4", recursive=True))
    video = files[0]

    container = av.open(video)
    stream = container.streams.video[0]

    n = 0
    shape = None
    for frame in container.decode(stream):
        if n == 0:
            shape = frame.to_ndarray(format="rgb24").shape
        n += 1

    print(key, "files=", len(files), "frames=", n, "shape=", shape)
PY
```

Expected RGB shape:

```text
(480, 640, 3)
```

---

# 12. Link the dataset into the teacher workspace

```bash
ln -sfn \
"$TEACHER_ROOT" \
"$DEMO_ROOT/teacher_demo/data/dataset"
```

---

# 13. Start Diffusion Policy training

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${TEACHER_REPO} \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --output_dir="$TEACHER_TRAIN_OUT" \
  --job_name=demo_teacher_pick_cube_5_diffusion \
  --policy.device=cuda \
  --wandb.enable=false
```

Internal data path:

```text
LeRobotDataset
→ Parquet + decoded MP4
→ temporal sample
→ DataLoader Batch
→ Diffusion Policy
→ loss
→ backward
→ optimizer.step()
```

---

# 14. Completion criterion

These messages alone are not enough:

```text
Creating dataset
Creating policy
Start offline training
```

Wait until the log contains something like:

```text
step:200 ... loss:...
```

That proves:

```text
forward
→ loss
→ backward
→ optimizer update
```

The classroom demonstration is complete. You may stop with `Ctrl+C`.

Do not claim that five demonstrations have produced a reliable real-world policy.

---

# 15. Reset after the teacher demo

Only if these are temporary teacher-demo artifacts:

```bash
rm -f "$DEMO_ROOT/teacher_demo/data/dataset"
rm -rf "$TEACHER_ROOT"
rm -rf "$TEACHER_TRAIN_OUT"
```

Do not delete student data or calibration files.
