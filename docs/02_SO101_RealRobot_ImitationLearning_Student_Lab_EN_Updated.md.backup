# Experiment 2: SO-101 Real-Robot Imitation Learning — Complete Student Lab Guide

> **Document type: Student Lab Manual**
>
> This is a complete end-to-end procedure for one student. It does not assume team rotation or handoff.
>
> The experiment has two stages. Stage A uses 5 SO-101 demonstrations to understand the LeRobot Dataset and verify that Diffusion Policy training starts correctly. Stage B collects 50 high-quality demonstrations and completes 100K-step training, checkpoint saving, real-robot inference, autonomous rollout, and success-rate evaluation.
>
> **The 5-demo dataset is only a teaching smoke test. The 50-demo dataset is the formal training and deployment experiment.**

---

# 0. What you will complete

```text
Stage A: 5-demo Smoke Test
Environment check
↓
Leader / Follower / dual-camera check
↓
Teleoperation
↓
Collect 5 successful demonstrations
↓
Check Parquet + MP4 + Metadata
↓
Check state / action
↓
PyAV decoding
↓
Start Diffusion Policy
↓
Observe step + loss
↓
Smoke Test complete

Stage B: 50-demo Formal Experiment
Collect 50 high-quality demonstrations
↓
Complete Dataset Quality Check
↓
Diffusion Policy 100K Training
↓
Checkpoint
↓
1 safety Inference episode
↓
10 Autonomous Rollouts
↓
Success Rate
↓
Formal experiment complete
```

Task:

```text
Grasp the cube and lift it.
```

Network and account requirements:

```text
Hugging Face login: NOT required
W&B login: NOT required
Dataset upload to Hub: disabled
Policy upload to Hub: disabled
```

This lab is intentionally local. Students do not need to run `hf auth login` or `wandb login`.

The fixed 5-demo student dataset is:

```text
repo_id:
dseg/student_pick_cube_demo5_v1
```

Student workspace:

```text
~/code/lerobot/real_demo_20260817/student_demo/
├── data/
├── model/
├── logs/
└── results/
```

---

# 1. Enter the environment

```bash
cd ~/code/lerobot
conda activate lerobot
```

Check:

```bash
which python
```

It should point to:

```text
.../envs/lerobot/...
```

---

# 2. Define the 5-demo paths

```bash
export HF_USER=dseg

export STUDENT_REPO=dseg/student_pick_cube_demo5_v1

export STUDENT_ROOT="$HOME/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1"

export STUDENT_WORK="$HOME/code/lerobot/real_demo_20260817/student_demo"

export STUDENT_TRAIN_OUT="$STUDENT_WORK/model/student_pick_cube_demo5_diffusion"
```

Create the directories:

```bash
mkdir -p "$STUDENT_WORK/data"
mkdir -p "$STUDENT_WORK/model"
mkdir -p "$STUDENT_WORK/logs"
mkdir -p "$STUDENT_WORK/results"
```

Check:

```bash
printf "STUDENT_REPO=%s\nSTUDENT_ROOT=%s\nSTUDENT_WORK=%s\nSTUDENT_TRAIN_OUT=%s\n" \
"$STUDENT_REPO" "$STUDENT_ROOT" "$STUDENT_WORK" "$STUDENT_TRAIN_OUT"
```

---

# 3. Make sure you are not overwriting an old Dataset

```bash
test ! -e "$STUDENT_ROOT" \
  && echo "Student dataset path clean" \
  || echo "WARNING: dataset already exists"
```

Training directory:

```bash
test ! -e "$STUDENT_TRAIN_OUT" \
  && echo "Student training path clean" \
  || echo "WARNING: training output already exists"
```

If this is not your first run, do not overwrite a previous experiment.

For example, use a new version:

```bash
export STUDENT_REPO=dseg/student_pick_cube_demo5_v2
export STUDENT_ROOT="$HOME/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v2"
export STUDENT_TRAIN_OUT="$STUDENT_WORK/model/student_pick_cube_demo5_diffusion_v2"
```

---

# 4. Check both robot arms

```bash
ls -l /dev/serial/by-id/
```

You should see both Leader and Follower.

The course machine has used:

```text
Leader:
usb-1a86_USB_Single_Serial_5A7A054767-if00

Follower:
usb-1a86_USB_Single_Serial_5A7A056628-if00
```

If only one device appears:

```text
Do not continue recording.
```

Check first:

- USB cable;
- controller power;
- loose connections;
- `dmesg` disconnect messages;
- device permissions.

---

# 5. Check both cameras

```bash
ls -l /dev/video*
ls -l /dev/v4l/by-id/
```

Both devices should be present:

```text
GENERAL_WEBCAM
DSJ-250318-J
```

Course naming:

```text
fixed
= external/environment camera

handeye
= gripper camera
```

Linux `/dev/videoX` numbering may change after unplugging/replugging a camera. Do not rely only on remembered numbers such as `video0` and `video2`.

---

# 6. Check calibration

```bash
ls -lh \
.cache/calibration/so101/main_follower.json \
.cache/calibration/so101/main_leader.json
```

Validate the JSON files:

```bash
python -m json.tool .cache/calibration/so101/main_follower.json >/dev/null \
  && echo "Follower calibration OK"

python -m json.tool .cache/calibration/so101/main_leader.json >/dev/null \
  && echo "Leader calibration OK"
```

If calibration is already valid, do not recalibrate.

---

# 7. Teleoperation test

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

Confirm:

```text
Your hand
↓
Leader
↓
Follower
```

The directions should match.

In Rerun:

```text
observation.images.fixed
= external camera image

observation.images.handeye
= gripper camera image
```

If either camera is black or frozen, fix it before recording.

When finished:

```text
Ctrl+C
```

---

# 8. Collect 5 demonstrations

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${STUDENT_REPO} \
  --control.root="$STUDENT_ROOT" \
  --control.tags='["student","real_demo","pick_cube","demo5"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=5 \
  --control.push_to_hub=false \
  --control.display_data=true
```

Demonstration quality:

```text
Move continuously
↓
Avoid long hesitation
↓
Do not keep empty grasps
↓
Do not keep drop failures
↓
Grasp the cube and visibly lift it
```

Controls:

```text
←   Current Episode is bad: rerecord it
→   Current Episode succeeded: finish it early
Esc End recording
```

---

# 9. Understand where the Dataset is stored

After recording:

```bash
ls -lh "$STUDENT_ROOT"
```

You should see:

```text
data
meta
videos
```

List all files:

```bash
find "$STUDENT_ROOT" -maxdepth 4 -type f | sort
```

A recorded episode is more than a video:

```text
One Episode
├── Parquet
│   ├── observation.state
│   ├── action
│   ├── timestamp
│   ├── frame_index
│   └── episode_index
│
├── fixed MP4
├── handeye MP4
└── metadata
```

---

# 10. Check Dataset metadata

```bash
cat "$STUDENT_ROOT/meta/info.json"
```

Confirm:

```text
total_episodes = 5
fps = 30
total_videos = 10
```

Also inspect `features`.

It should include:

```text
action
observation.state
observation.images.fixed
observation.images.handeye
timestamp
frame_index
episode_index
index
task_index
```

---

# 11. Check metadata counts

```bash
wc -l \
"$STUDENT_ROOT/meta/episodes.jsonl" \
"$STUDENT_ROOT/meta/episodes_stats.jsonl" \
"$STUDENT_ROOT/meta/tasks.jsonl"
```

Expected:

```text
5
5
1
```

Meaning:

```text
5 Episodes
5 Episode statistics records
1 task
```

---

# 12. Check 5 Parquet files and 10 videos

Parquet:

```bash
find "$STUDENT_ROOT/data" \
  -type f \
  -name '*.parquet' \
  | sort
```

There should be 5.

Videos:

```bash
find "$STUDENT_ROOT/videos" \
  -type f \
  -name '*.mp4' \
  | sort
```

There should be:

```text
5 fixed
+
5 handeye
=
10 videos
```

---

# 13. Inspect one Episode

```bash
python - <<'PY'
import glob
import pandas as pd
import os

root = os.path.expanduser(
    "~/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1"
)

f = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))[0]
df = pd.read_parquet(f)

print("File:", f)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nAction example:")
print(df["action"].iloc[0])

print("\nState example:")
print(df["observation.state"].iloc[0])
PY
```

Interpretation:

```text
observation.state
= current robot state

action
= expert/human control target
```

SO-101 uses 6 dimensions:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

---

# 14. Check all Episode lengths and NaNs

```bash
python - <<'PY'
import glob
import os
import pandas as pd
import numpy as np

root = os.path.expanduser(
    "~/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1"
)

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
        "nan_action =", np.isnan(action).any()
    )

df = pd.concat(all_df, ignore_index=True)
state = np.stack(df["observation.state"].apply(np.asarray))
action = np.stack(df["action"].apply(np.asarray))

print("\nTOTAL FRAMES:", len(df))
print("STATE RANGE :", np.ptp(state, axis=0))
print("ACTION RANGE:", np.ptp(action, axis=0))
PY
```

Pass conditions:

```text
5 parquet files
No extremely abnormal episode length
state/action shape = (N, 6)
No NaN
Joint ranges show actual motion
```

---

# 15. Do not manually delete only one bad Parquet file

Wrong:

```bash
rm episode_000003.parquet
```

One Episode is linked to:

```text
Parquet
fixed MP4
handeye MP4
episodes.jsonl
episodes_stats.jsonl
info.json
global index
```

Deleting only one file breaks Dataset consistency.

If a demonstration is bad during recording:

```text
Use ← to rerecord it.
```

If you discover a problem only after recording, the simplest classroom approach is:

```text
Keep the old Dataset
Use a new repo_id
Record a clean replacement Dataset
```

---

# 16. Check PyAV

```bash
python - <<'PY'
import av
print("PyAV version:", av.__version__)
PY
```

If import fails, do not start training.

---

# 17. Decode one fixed and one handeye video

```bash
python - <<'PY'
import av
import glob
import os

root = os.path.expanduser(
    "~/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1"
)

for key in [
    "observation.images.fixed",
    "observation.images.handeye",
]:
    videos = sorted(
        glob.glob(root + f"/videos/**/{key}/*.mp4", recursive=True)
    )

    video = videos[0]
    container = av.open(video)
    stream = container.streams.video[0]

    n = 0
    shape = None

    for frame in container.decode(stream):
        if n == 0:
            rgb = frame.to_ndarray(format="rgb24")
            shape = rgb.shape
        n += 1

    print(key)
    print("  video count:", len(videos))
    print("  decoded frames:", n)
    print("  RGB shape:", shape)
PY
```

Expected:

```text
video count = 5
RGB shape = (480, 640, 3)
```

---

# 18. Link the Dataset into the student workspace

```bash
ln -sfn \
"$STUDENT_ROOT" \
"$STUDENT_WORK/data/dataset"
```

Check:

```bash
ls -lh "$STUDENT_WORK/data"
```

You should see:

```text
dataset -> ~/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1
```

The symlink is only for organization. The actual Dataset remains under:

```text
~/.cache/huggingface/lerobot/...
```

---

# 19. Start the 5-demo Diffusion Policy Smoke Test

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${STUDENT_REPO} \
  --dataset.root="$STUDENT_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --output_dir="$STUDENT_TRAIN_OUT" \
  --job_name=student_pick_cube_demo5_diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

No separate `convert.py` step is required.

The Dataset Loader handles:

```text
Parquet
→ state/action

MP4
→ PyAV
→ RGB frame

timestamp
→ temporal alignment

multiple samples
→ Batch
```

---

# 20. How to know the 5-demo pipeline is really running

Seeing only:

```text
Creating dataset
Creating policy
Start offline training
```

is not enough.

You need to see:

```text
step:...
loss:...
```

For example:

```text
step:200 ... loss:0.xxx ...
```

That means:

```text
Dataset
↓
DataLoader
↓
forward
↓
loss
↓
backward
↓
optimizer.step()
```

all executed successfully.

At this point, **Stage A: 5-demo Smoke Test is complete**.

You may stop the smoke training:

```text
Ctrl+C
```

If your assignment is only to verify the teaching pipeline, you can stop here. For the complete real-robot imitation learning experiment, continue to **Step 24: 50-demo Formal Experiment**.

---

# 21. Why the 5-demo model does not need to finish training

Five demonstrations are very small.

They are useful for:

```text
Verifying data collection
Verifying Dataset structure
Verifying video decoding
Verifying DataLoader
Verifying that the model can train
```

They are not sufficient evidence for:

```text
"Diffusion Policy has learned reliable real-robot grasping."
```

The formal 50-demo stage below will perform:

```text
Collect more high-quality demos
↓
Check / clean the Dataset
↓
Longer training
↓
Save checkpoint
↓
Real-robot rollout
↓
Repeated success-rate evaluation
```

---

# 22. What you should be able to explain after Stage A

1. What the Leader and Follower are.
2. Why the Leader is only used to generate demonstrations.
3. Why both `fixed` and `handeye` are observations.
4. What `observation.state` means.
5. What `action` means.
6. Why one Episode contains both Parquet and MP4 data.
7. Why metadata is necessary.
8. Why the real robot does not use ManiSkill's `replay_trajectory.py`.
9. Where PyAV works in the pipeline.
10. What the DataLoader does.
11. Why Diffusion Policy learns an `observation → action distribution`.
12. Why lower training loss does not guarantee rollout success.
13. Why 5 demonstrations are only a smoke test.

---

# 23. Shortest 5-demo checklist

```bash
cd ~/code/lerobot
conda activate lerobot

export HF_USER=dseg
export STUDENT_REPO=dseg/student_pick_cube_demo5_v1
export STUDENT_ROOT="$HOME/.cache/huggingface/lerobot/dseg/student_pick_cube_demo5_v1"
export STUDENT_WORK="$HOME/code/lerobot/real_demo_20260817/student_demo"
export STUDENT_TRAIN_OUT="$STUDENT_WORK/model/student_pick_cube_demo5_diffusion"
```

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${STUDENT_REPO} \
  --control.root="$STUDENT_ROOT" \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=5 \
  --control.push_to_hub=false \
  --control.display_data=true
```

```bash
cat "$STUDENT_ROOT/meta/info.json"
find "$STUDENT_ROOT" -maxdepth 4 -type f | sort
```

```bash
python - <<'PY'
import av
print("PyAV:", av.__version__)
PY
```

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${STUDENT_REPO} \
  --dataset.root="$STUDENT_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --output_dir="$STUDENT_TRAIN_OUT" \
  --job_name=student_pick_cube_demo5_diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

When you see:

```text
step + loss
```

the 5-demo Smoke Test is complete. Continue to Step 24 for the formal 50-demo experiment.

---

# 24. Stage B: 50-demo Formal Experiment overview

> **Formal Imitation Learning Experiment**
>
> This stage completes the real-robot imitation learning loop rather than only checking the software pipeline:

```text
50 Demonstrations
↓
Dataset Quality Check
↓
Diffusion Policy Training
↓
100K Steps
↓
Checkpoint
↓
Real-Robot Inference
↓
Autonomous Rollout
↓
Success Rate
```

Final goal:

> **Without operating the Leader, use the trained Diffusion Policy to read the dual-camera observations and robot state, predict actions, and autonomously control the Follower to grasp and lift the cube.**

---

# 25. Define the 50-demo formal experiment paths

```bash
cd ~/code/lerobot
conda activate lerobot

export HF_USER=dseg

export FORMAL_REPO=dseg/student_pick_cube_50_v1
export FORMAL_ROOT="$HOME/.cache/huggingface/lerobot/dseg/student_pick_cube_50_v1"

export STUDENT_WORK="$HOME/code/lerobot/real_demo_20260817/student_demo"
export FORMAL_TRAIN_OUT="$STUDENT_WORK/model/student_pick_cube_50_diffusion"
export FORMAL_LOG="$STUDENT_WORK/logs/student_pick_cube_50_training.log"
```

Create directories:

```bash
mkdir -p "$STUDENT_WORK/data"
mkdir -p "$STUDENT_WORK/model"
mkdir -p "$STUDENT_WORK/logs"
mkdir -p "$STUDENT_WORK/results"
```

Check:

```bash
printf "FORMAL_REPO=%s\nFORMAL_ROOT=%s\nFORMAL_TRAIN_OUT=%s\nFORMAL_LOG=%s\n" \
"$FORMAL_REPO" "$FORMAL_ROOT" "$FORMAL_TRAIN_OUT" "$FORMAL_LOG"
```

---

# 26. Make sure you will not overwrite an old formal run

```bash
test ! -e "$FORMAL_ROOT" \
  && echo "50-demo dataset path clean" \
  || echo "WARNING: 50-demo dataset already exists"

test ! -e "$FORMAL_TRAIN_OUT" \
  && echo "Training output path clean" \
  || echo "WARNING: training output already exists"
```

If an old experiment already exists, do not overwrite it. Use a new version:

```bash
export FORMAL_REPO=dseg/student_pick_cube_50_v2
export FORMAL_ROOT="$HOME/.cache/huggingface/lerobot/dseg/student_pick_cube_50_v2"
export FORMAL_TRAIN_OUT="$STUDENT_WORK/model/student_pick_cube_50_diffusion_v2"
```

---

# 27. Recheck hardware before formal recording

Robot arms:

```bash
ls -l /dev/serial/by-id/
```

Cameras:

```bash
ls -l /dev/video*
ls -l /dev/v4l/by-id/
```

Calibration:

```bash
ls -lh \
.cache/calibration/so101/main_follower.json \
.cache/calibration/so101/main_leader.json
```

All of the following must be valid:

```text
Leader OK
Follower OK
fixed camera OK
handeye camera OK
Follower calibration OK
Leader calibration OK
```

Do not start the 50-demo collection if any of these checks fail.

---

# 28. Teleoperation test before formal recording

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

Confirm:

```text
Your hand
↓
Leader
↓
Follower
```

Also confirm in Rerun:

```text
observation.images.fixed
= environment camera

observation.images.handeye
= gripper camera
```

When done:

```text
Ctrl+C
```

---

# 29. Formal demonstration quality requirements

Task:

```text
Grasp the cube and lift it.
```

Each demonstration should approximately follow:

```text
Normal start pose
↓
Smoothly approach the cube
↓
Align the gripper
↓
Close the gripper
↓
Successfully grasp
↓
Clearly lift the cube
```

Principles:

- use continuous, stable motion;
- avoid long hesitation;
- do not keep empty grasps;
- do not keep episodes where the cube is dropped;
- avoid obvious table collisions;
- allow small changes in cube start position, but keep them near the main training distribution.

**Data quality has priority over raw episode count.**

---

# 30. Collect 50 formal demonstrations

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${FORMAL_REPO} \
  --control.root="$FORMAL_ROOT" \
  --control.tags='["student","formal","pick_cube","diffusion","train50"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=50 \
  --control.push_to_hub=false \
  --control.display_data=true
```

During recording:

```text
←   Bad Episode: rerecord it
→   Successful Episode: end it early
Esc Stop recording
```

The goal is not simply "50 files". The goal is **50 high-quality successful demonstrations**.

---

# 31. Check the 50-demo Dataset metadata

```bash
cat "$FORMAL_ROOT/meta/info.json"
```

Confirm:

```text
total_episodes = 50
fps = 30
```

With two cameras, you normally expect:

```text
total_videos = 100
```

The features should include at least:

```text
action
observation.state
observation.images.fixed
observation.images.handeye
timestamp
frame_index
episode_index
index
task_index
```

---

# 32. Check metadata counts

```bash
wc -l \
"$FORMAL_ROOT/meta/episodes.jsonl" \
"$FORMAL_ROOT/meta/episodes_stats.jsonl" \
"$FORMAL_ROOT/meta/tasks.jsonl"
```

Expected:

```text
50
50
1
```

---

# 33. Check Parquet and video counts

Parquet:

```bash
find "$FORMAL_ROOT/data" \
  -type f \
  -name '*.parquet' \
  | wc -l
```

Expected:

```text
50
```

Videos:

```bash
find "$FORMAL_ROOT/videos" \
  -type f \
  -name '*.mp4' \
  | wc -l
```

With two cameras:

```text
100
```

---

# 34. Check every Episode for state / action shape and NaNs

```bash
python - <<'PY'
import glob
import os
import pandas as pd
import numpy as np

root = os.path.expanduser(
    "~/.cache/huggingface/lerobot/dseg/student_pick_cube_50_v1"
)

files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))

print("Total parquet files:", len(files))

total_frames = 0

for i, f in enumerate(files):
    df = pd.read_parquet(f)

    state = np.stack(df["observation.state"].apply(np.asarray))
    action = np.stack(df["action"].apply(np.asarray))

    total_frames += len(df)

    print(
        f"episode={i:02d}",
        "frames=", len(df),
        "state=", state.shape,
        "action=", action.shape,
        "nan_state=", np.isnan(state).any(),
        "nan_action=", np.isnan(action).any(),
    )

print("\nTOTAL FRAMES:", total_frames)
PY
```

Pass conditions:

```text
Total parquet files = 50
state shape = (N, 6)
action shape = (N, 6)
nan_state = False
nan_action = False
```

---

# 35. Inspect Episode lengths and video quality

Watch the reported:

```text
frames=
```

If most trajectories are in a reasonable range but one Episode is extremely short, inspect the corresponding video manually.

Do not delete an Episode only because its frame count is shorter.

List sample videos:

```bash
find "$FORMAL_ROOT/videos" \
  -type f \
  -name '*.mp4' \
  | sort \
  | head -20
```

Confirm:

```text
fixed video is valid
handeye video is valid
grasp action is complete
no obvious corruption or severe frame loss
```

---

# 36. Recheck PyAV decoding

```bash
python - <<'PY'
import av
import glob
import os

root = os.path.expanduser(
    "~/.cache/huggingface/lerobot/dseg/student_pick_cube_50_v1"
)

for key in [
    "observation.images.fixed",
    "observation.images.handeye",
]:
    videos = sorted(
        glob.glob(root + f"/videos/**/{key}/*.mp4", recursive=True)
    )

    print("\n", key)
    print("video count:", len(videos))

    video = videos[0]
    container = av.open(video)
    stream = container.streams.video[0]

    n = 0
    for frame in container.decode(stream):
        n += 1

    print("decoded frames:", n)
PY
```

Target:

```text
fixed video count = 50
handeye video count = 50
```

---

# 37. Final pass conditions before formal training

All of these must pass:

```text
total_episodes = 50
Parquet = 50
fixed videos = 50
handeye videos = 50
state = 6 dimensions
action = 6 dimensions
no NaN
PyAV decodes successfully
manual video sampling looks correct
```

If any critical item fails, fix the Dataset before starting the 100K formal training.

---

# 38. Start formal 50-demo Diffusion Policy training

Formal settings:

```text
Dataset: 50 demonstrations
Policy: Diffusion Policy
Batch size: 8
Training steps: 100000
Seed: 1000
Device: CUDA
W&B: Disabled
Hugging Face upload: Disabled
```

Run:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${FORMAL_REPO} \
  --dataset.root="$FORMAL_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$FORMAL_TRAIN_OUT" \
  --job_name=student_pick_cube_50_diffusion \
  --batch_size=8 \
  --steps=100000 \
  --seed=1000 \
  --log_freq=200 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --wandb.enable=false \
  2>&1 | tee "$FORMAL_LOG"
```

This experiment does not require:

```bash
hf auth login
wandb login
```

---

# 39. Confirm that formal training has actually started

You should first see messages similar to:

```text
Creating dataset
Creating policy
Creating optimizer and scheduler
Start offline training
```

Then you should continuously see:

```text
step:...
loss:...
grdn:...
lr:...
```

For example:

```text
step:200 ... loss:0.xxx ...
```

That means:

```text
Dataset
↓
DataLoader
↓
Forward
↓
Loss
↓
Backward
↓
Optimizer Step
```

are all executing.

---

# 40. Formal training must finish

For the 5-demo smoke test:

```text
See step + loss
→ stopping is acceptable
```

For the 50-demo formal experiment:

```text
Train to
100000 / 100000
```

In another terminal you may monitor the GPU:

```bash
watch -n 1 nvidia-smi
```

Follow the log:

```bash
tail -f "$FORMAL_LOG"
```

or:

```bash
grep "step:" "$FORMAL_LOG" | tail
```

---

# 41. Check checkpoints

```bash
find "$FORMAL_TRAIN_OUT/checkpoints" \
  -maxdepth 2 \
  -type d \
  -name pretrained_model \
  | sort
```

After training finishes, define the final policy:

```bash
export POLICY_PATH="$FORMAL_TRAIN_OUT/checkpoints/last/pretrained_model"
```

Check it:

```bash
echo "$POLICY_PATH"
ls -lah "$POLICY_PATH"
```

Do not proceed to real-robot inference until this directory exists and contains a complete saved policy.

---

# 42. Understand the control flow during inference

During demonstration collection:

```text
Human
↓
Leader
↓
Action
↓
Follower
```

During autonomous inference:

```text
fixed camera ───┐
handeye camera ─┼→ Observation
robot state ────┘
        ↓
Diffusion Policy
        ↓
Predicted Action
        ↓
Follower
```

The **Leader is no longer the action source** during autonomous inference.

---

# 43. Safety preparation before inference

Before the first autonomous rollout:

- clear obstacles around the arm;
- route USB, camera, and power cables safely;
- keep hands outside the robot workspace;
- be ready to press `Ctrl+C`;
- be ready to cut actuator power if necessary;
- place the cube near a common training position for the first test.

The first objective is to verify:

```text
Checkpoint loads
+
Policy performs inference
+
Follower executes autonomously
```

Do not begin with extreme out-of-distribution cube positions.

---

# 44. Run only 1 safety inference Episode first

Define a separate smoke-evaluation Dataset:

```bash
export EVAL_SMOKE_REPO=dseg/eval_student_pick_cube_50_smoke_v1
export EVAL_SMOKE_ROOT="$HOME/.cache/huggingface/lerobot/dseg/eval_student_pick_cube_50_smoke_v1"
```

Check the path:

```bash
test ! -e "$EVAL_SMOKE_ROOT" \
  && echo "Inference smoke path clean" \
  || echo "WARNING: eval dataset already exists"
```

Run:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${EVAL_SMOKE_REPO} \
  --control.root="$EVAL_SMOKE_ROOT" \
  --control.tags='["student","eval","diffusion","pick_cube"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --control.policy.path="$POLICY_PATH" \
  --control.display_data=true
```

The key parameter is:

```text
--control.policy.path="$POLICY_PATH"
```

It changes the control source from the Leader to the trained Diffusion Policy.

---

# 45. How to know inference is working

Observe:

```text
Checkpoint loads successfully
↓
Both cameras provide observations
↓
observation.state is available
↓
Policy continuously produces action
↓
Follower moves without anyone operating the Leader
```

If the policy loads and controls the Follower autonomously, the **real-robot inference pipeline is working**.

A failed grasp does not automatically mean that the inference program failed.

---

# 46. How to interpret the first failed grasp

Possible outcomes:

```text
Approaches the cube but misses
Gripper is slightly misaligned
Grasps but drops the cube
Motion is generally correct but incomplete
```

First check:

```text
Did the policy load?
Did it keep producing actions?
Did the Follower move autonomously?
```

If yes, deployment/inference is working. The remaining issue is policy performance, data quality, training quality, or generalization.

---

# 47. Run 10 formal autonomous rollouts

After the 1-Episode safety inference works:

```bash
export EVAL_REPO=dseg/eval_student_pick_cube_50_v1
export EVAL_ROOT="$HOME/.cache/huggingface/lerobot/dseg/eval_student_pick_cube_50_v1"
```

Check:

```bash
test ! -e "$EVAL_ROOT" \
  && echo "Formal eval path clean" \
  || echo "WARNING: formal eval dataset already exists"
```

Run:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp the cube and lift it." \
  --control.repo_id=${EVAL_REPO} \
  --control.root="$EVAL_ROOT" \
  --control.tags='["student","formal_eval","diffusion","pick_cube"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path="$POLICY_PATH" \
  --control.display_data=true
```

Between Episodes, restore the scene to a normal initial condition. Move the cube only within a small range near the training distribution for the initial evaluation.

---

# 48. Unified SUCCESS / FAIL definition

An Episode is `SUCCESS` only if all conditions are met:

```text
Gripper grasps the cube
+
Cube leaves the table
+
Cube remains clearly lifted
```

Otherwise record:

```text
FAIL
```

Examples of failure:

- touches the cube but does not grasp it;
- grasps and immediately drops it;
- never lifts the cube from the table;
- does not complete the task;
- clearly moves in the wrong direction.

---

# 49. Record 10 rollout results and compute success rate

Record:

```text
Episode 01:
Episode 02:
Episode 03:
Episode 04:
Episode 05:
Episode 06:
Episode 07:
Episode 08:
Episode 09:
Episode 10:
```

Compute:

```text
Success Rate
=
Number of successful Episodes / 10
```

Example:

```text
7 / 10 = 70%
```

---

# 50. Check the Evaluation Dataset and videos

```bash
cat "$EVAL_ROOT/meta/info.json"
```

Confirm:

```text
total_episodes = 10
```

List rollout videos:

```bash
find "$EVAL_ROOT/videos" \
  -type f \
  -name '*.mp4' \
  | sort
```

With two cameras:

```text
10 Episodes × 2 Cameras = 20 MP4 files
```

---

# 51. Save one representative successful rollout

```bash
mkdir -p "$STUDENT_WORK/results"
```

Find the fixed-camera video for a successful Episode and copy it:

```bash
cp "/actual/path/to/successful/fixed_video.mp4" \
"$STUDENT_WORK/results/student_pick_cube_50_success.mp4"
```

Recommended fixed result path:

```text
~/code/lerobot/real_demo_20260817/student_demo/results/
student_pick_cube_50_success.mp4
```

---

# 52. Formal experiment result record

```text
Student:

Task:
Grasp the cube and lift it.

Robot:
SO-101

Training demonstrations:
50

Observation:
observation.state
observation.images.fixed
observation.images.handeye

Action dimension:
6

Policy:
Diffusion Policy

Batch size:
8

Training steps:
100000

Seed:
1000

Training Dataset:
dseg/student_pick_cube_50_v1

Training Dataset path:
~/.cache/huggingface/lerobot/dseg/student_pick_cube_50_v1

Training output:
~/code/lerobot/real_demo_20260817/student_demo/model/student_pick_cube_50_diffusion

Final checkpoint:
checkpoints/last/pretrained_model

Evaluation episodes:
10

Successful episodes:

Failed episodes:

Success Rate:

Successful rollout video:
```

---

# 53. Formal experiment pass criteria

The formal experiment is complete only after all of the following:

```text
1. Leader / Follower work correctly
2. fixed / handeye cameras work correctly
3. 50 high-quality demonstrations are collected
4. Dataset metadata is valid
5. Parquet data is valid
6. MP4 data is valid
7. state / action data is valid
8. PyAV decoding works
9. Diffusion Policy formal training starts
10. 100000 training steps complete
11. Checkpoint is saved
12. Checkpoint is loaded successfully
13. Real-robot Diffusion Policy inference works
14. Follower acts autonomously with no Leader operation
15. 10 Autonomous Rollouts are completed
16. Success Rate is calculated
17. At least one successful rollout video is saved
```

---

# 54. Final imitation-learning loop

```text
Human Demonstration
↓
Leader
↓
Follower
↓
50 Successful Demonstrations
↓
Parquet + MP4 + Metadata
↓
LeRobot Dataset
↓
DataLoader
↓
Diffusion Policy
↓
100K Training Steps
↓
Checkpoint
↓
Real-Time Observation
↓
Policy Inference
↓
Predicted Action
↓
Follower
↓
Autonomous Grasp
↓
10 Rollouts
↓
Success Rate
```

The student should clearly understand the difference:

```text
5-demo
=
Smoke Test
=
Verify recording, Dataset, PyAV, DataLoader, and training pipeline
```

versus:

```text
50-demo
=
Formal Experiment
=
Formal training + Checkpoint + real-robot inference
+ Autonomous Rollout + Success Rate
```

This completes the full:

```text
Data
→ Training
→ Deployment
→ Evaluation
```

real-robot imitation-learning loop.
