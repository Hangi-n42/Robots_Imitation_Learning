# Experiment 2: SO-101 Real-Robot Imitation Learning — Concepts, Data Structure, and Source Code Guide

> **Document type: Concept & Code Guide**
>
> This guide explains the system rather than repeating terminal commands.
>
> Core questions:
>
> - How does a human demonstration become training data?
> - How does real-robot data differ from ManiSkill trajectories?
> - What are Parquet, MP4, and metadata?
> - What are Observation, State, Action, Episode, Dataset, and DataLoader?
> - What does Diffusion Policy learn?
> - Where do ResNet18, PyAV, DDPM, horizon, and action chunks fit?
> - Which LeRobot source files should you read first?

---

# 1. System map

Training:

```text
Human
↓
Leader
↓
Follower
↓
fixed RGB + handeye RGB
+ follower state
+ expert action
↓
LeRobot Dataset
↓
Offline Imitation Learning
↓
Diffusion Policy
```

Future inference:

```text
Leader removed
↓
live cameras + current state
↓
trained policy
↓
action
↓
Follower
↓
new observation
↓
policy ...
```

The five-demo classroom lab stops once training produces `step + loss`.

---

# 2. Imitation Learning

Imitation learning uses:

```text
Expert Demonstrations
↓
Observation → Expert Action pairs
↓
learn π(a | o)
```

The robot is not controlled by handwritten task rules. The policy learns a mapping from observations to an action distribution.

---

# 3. Leader and Follower

**Leader**: the human-operated arm used to capture expert intent.

**Follower**: the real robot that interacts with the object.

Collection:

```text
Human → Leader → Follower
```

Inference:

```text
Policy → Follower
```

The Leader is therefore a data-collection interface, not part of the final policy.

---

# 4. Observation, State, and Action

Observation includes:

```text
observation.images.fixed
observation.images.handeye
observation.state
```

The SO-101 state vector is 6-D:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

The action vector is also 6-D and records the expert control target.

The learning problem is:

```text
observation → expert action
```

---

# 5. Why action and state are not identical

`action_t` is a command or target. `state_t` is a measured physical state.

Real hardware introduces:

```text
servo dynamics
load
control latency
communication latency
tracking error
```

Therefore:

```text
action_t != state_t
```

is expected.

---

# 6. Episode

An episode is a complete temporal trajectory:

```text
t0: image + state + action
t1: image + state + action
...
tN: image + state + action
```

Five demonstrations mean five episodes, not five training samples.

---

# 7. Why Parquet and MP4 are both needed

Structured numerical data is stored in Parquet:

```text
state
action
timestamp
frame_index
episode_index
```

Images are stored as MP4:

```text
fixed video
handeye video
```

This avoids millions of small image files and allows timestamp-based frame retrieval during training.

---

# 8. Metadata

`meta/info.json` describes the full dataset:

```text
robot_type
total_episodes
total_frames
fps
features
data_path
video_path
```

`episodes.jsonl` tracks episodes.

`episodes_stats.jsonl` stores per-episode statistics.

`tasks.jsonl` maps task text to task indices.

Metadata is part of the dataset indexing system, not optional documentation.

---

# 9. Why real SO-101 data does not need ManiSkill replay

ManiSkill:

```text
Motion Planner
→ raw simulation trajectory
→ replay_trajectory.py
→ converted observation/control representation
```

SO-101:

```text
camera + follower state + leader action
→ control_robot.py record
→ final LeRobot training representation
```

The real-robot recorder already writes the observation/action representation needed by the training pipeline.

---

# 10. Where data processing actually happens

There are two stages.

**Record-time processing**

```text
camera frames
robot state
teleoperation actions
→ episode organization
→ timestamps and indices
→ video encoding
→ Parquet
→ metadata
```

**Training-time processing**

```text
LeRobot Dataset
→ timestamp queries
→ video decoding
→ RGB Tensor
→ state/action loading
→ temporal windows
→ normalization
→ Batch
→ Policy
```

No extra `convert.py` is required, but substantial processing still happens.

---

# 11. DataLoader

The DataLoader pipeline is:

```text
Dataset
→ sample
→ Batch
→ GPU
→ Model
```

In practice:

```text
LeRobotDataset.__getitem__()
→ read numeric data
→ query matching video frames
→ build sample
→ collate into Batch
```

Many "training errors" occur before model forward, for example video decoder failures or inconsistent dataset metadata.

---

# 12. PyAV

PyAV is a Python binding for FFmpeg.

Its role:

```text
MP4
→ decoded RGB frame
→ Tensor
```

It is not the policy, the visual backbone, or the robot controller.

The option:

```text
--dataset.video_backend=pyav
```

changes how video is read, not what learning algorithm is used.

---

# 13. TorchCodec vs PyAV

Both can implement:

```text
MP4 → RGB
```

The course uses PyAV for engineering stability.

```text
           MP4
            │
     ┌──────┴──────┐
     ↓             ↓
 TorchCodec       PyAV
     │             │
     └──────┬──────┘
            ↓
        RGB Tensor
            ↓
      Vision Encoder
```

---

# 14. Diffusion Policy

A deterministic controller might be written as:

```text
a = π(o)
```

Diffusion Policy instead models something closer to:

```text
p(action sequence | observation)
```

Conceptually:

```text
noisy action trajectory
→ condition on observation
→ iterative denoising
→ smooth action trajectory
```

This is useful for continuous, multimodal robot behavior.

---

# 15. Vision backbone: ResNet18

The verified course configuration has used:

```text
vision_backbone = resnet18
```

Role:

```text
RGB image
→ ResNet18
→ visual feature
```

If:

```text
use_separate_rgb_encoder_per_camera = False
```

then fixed and handeye share encoder weights.

---

# 16. pretrained_backbone_weights=None

This means the ResNet18 does not load ImageNet pretrained weights.

Do not confuse it with:

```text
use_imagenet_stats = True
```

which only refers to image normalization statistics.

---

# 17. Conditional 1D U-Net

A simplified view:

```text
observation features
+ noisy action trajectory
+ diffusion timestep
→ Conditional 1D U-Net
→ denoising prediction
→ action trajectory
```

The previously verified course configuration included large U-Net channel sizes such as:

```text
down_dims = [512, 1024, 2048]
```

which explains the large parameter count of the visual Diffusion Policy.

---

# 18. Horizon

A configuration such as:

```text
horizon = 16
```

means the policy reasons over an action sequence of length 16 rather than predicting only one immediate action.

---

# 19. n_obs_steps

For example:

```text
n_obs_steps = 2
```

uses a short history:

```text
observation at t-1
+
observation at t
```

which provides temporal context.

---

# 20. n_action_steps

For example:

```text
n_action_steps = 8
```

means the policy may plan a longer horizon but executes only a chunk before observing again.

This is:

```text
action chunking
+
receding-horizon execution
```

---

# 21. DDPM timestep is not robot time

Diffusion denoising steps are internal generative-model iterations.

They are not physical robot control cycles.

Always distinguish:

```text
diffusion timestep
```

from:

```text
robot/environment timestep
```

---

# 22. Offline training vs rollout

Offline training:

```text
fixed dataset
→ repeated sampling
→ imitate expert
```

Rollout:

```text
policy action
→ changes real world
→ new observation
→ next policy action
```

Rollout errors accumulate. Therefore low training loss does not guarantee real-world success.

---

# 23. Distribution shift

If all demonstrations place the cube in region A and evaluation places it in region B outside the demonstrated distribution, the policy may fail.

This is a central imitation-learning issue:

```text
distribution shift / covariate shift
```

---

# 24. Why five demos are only a smoke test

Five demonstrations can validate:

```text
recording
dataset structure
video decode
data loading
policy forward
loss
backward
```

They cannot establish:

```text
robustness
generalization
real-world success rate
```

---

# 25. Do not delete a single Parquet file

One episode links:

```text
Parquet
fixed MP4
handeye MP4
episodes.jsonl
episodes_stats.jsonl
global indices
```

Deleting one component breaks consistency.

Redo the episode during recording or create a new clean dataset.

---

# 26. Source map 1: real-robot control and recording

Entry point:

```text
lerobot/scripts/control_robot.py
```

Search:

```bash
grep -n \
  -E 'teleoperate|record|calibrate|policy|resume' \
  lerobot/scripts/control_robot.py
```

Study how teleoperation, recording, calibration, and policy rollout are connected.

---

# 27. Source map 2: training entry point

```text
lerobot/scripts/train.py
```

Search:

```bash
grep -n \
  -E 'Creating dataset|Creating policy|DataLoader|loss|backward|optimizer|checkpoint' \
  lerobot/scripts/train.py
```

Read along the data flow:

```text
dataset
→ policy
→ optimizer
→ DataLoader
→ training loop
→ loss
→ backward
```

---

# 28. Source map 3: LeRobotDataset

```text
lerobot/common/datasets/lerobot_dataset.py
```

Search:

```bash
grep -n \
  -E '__getitem__|_query_videos|timestamp|episode|video' \
  lerobot/common/datasets/lerobot_dataset.py
```

Conceptual path:

```text
__getitem__()
→ numeric state/action
→ timestamp-based video query
→ sample
```

---

# 29. Source map 4: video decoding

```text
lerobot/common/datasets/video_utils.py
```

Search:

```bash
grep -n \
  -E 'torchcodec|pyav|decode_video_frames' \
  lerobot/common/datasets/video_utils.py
```

This is where:

```text
dataset timestamp
→ MP4 frame
→ RGB Tensor
```

becomes concrete code.

---

# 30. Source map 5: Diffusion Policy

Because LeRobot versions differ, first locate files:

```bash
find lerobot \
  -type f \
  -path '*diffusion*' \
  | sort
```

Look for files similar to:

```text
configuration_diffusion.py
modeling_diffusion.py
```

and components for:

```text
normalization
scheduler
RGB encoder
U-Net
DDPM
```

---

# 31. Verify runtime configuration

Do not rely only on online defaults.

Search the local code:

```bash
grep -R -n \
  -E 'vision_backbone|pretrained_backbone_weights|use_separate_rgb_encoder_per_camera|horizon|n_obs_steps|n_action_steps' \
  lerobot \
  | head -100
```

The training startup configuration is the final source of truth.

---

# 32. Robot configuration

Search:

```bash
grep -R -n \
  -E 'So101|SO101|so101' \
  lerobot/common/robot_devices/robots \
  | head -100
```

Focus on:

```text
Leader port
Follower port
motor configuration
camera configuration
```

---

# 33. Camera source

```text
lerobot/common/robot_devices/cameras/
```

Inspect:

```bash
find lerobot/common/robot_devices/cameras \
  -maxdepth 2 \
  -type f \
  | sort
```

Understand:

```text
/dev/videoX
→ OpenCV capture
→ RGB frame
→ Dataset recorder
```

---

# 34. Calibration

Files:

```text
.cache/calibration/so101/main_leader.json
.cache/calibration/so101/main_follower.json
```

They describe the mapping between physical servo coordinates and robot logical coordinates.

A learning model cannot compensate for fundamentally incorrect actuator calibration.

---

# 35. Why stable by-id paths matter

`/dev/ttyACM0` is dynamically assigned.

`/dev/serial/by-id/` is based on device identity and is much more stable.

The same idea applies to:

```text
/dev/v4l/by-id/
```

for cameras.

---

# 36. ManiSkill ↔ SO-101 mapping

| ManiSkill | SO-101 real robot |
|---|---|
| Simulator | Real world |
| Panda | SO-101 Follower |
| Motion Planner | Human + Leader |
| Sim state | Cameras + joint state |
| Automatic reset | Manual reset |
| `.h5` demo | Parquet + MP4 |
| Replay/convert | Record-time structuring + Loader |
| Environment success | Human-defined success |
| Diffusion Policy | Diffusion Policy |
| Sim rollout | Real-robot rollout |

The common learning pipeline is:

```text
Demonstration
→ Dataset
→ Policy
→ Rollout
→ Evaluation
```

---

# 37. End-to-end source/data table

| Stage | Input | Output | Key code |
|---|---|---|---|
| Teleoperation | human motion | follower action | `control_robot.py` |
| Recording | camera/state/action | LeRobot Dataset | `control_robot.py` |
| Numeric storage | state/action/time | Parquet | dataset recording |
| Visual storage | RGB stream | MP4 | camera/video writer |
| Dataset loading | Parquet + MP4 + metadata | sample | `lerobot_dataset.py` |
| Video decoding | MP4 + timestamp | RGB Tensor | `video_utils.py` |
| Training | Batch | updated parameters | `train.py` |
| Policy model | observation | action sequence | diffusion implementation |

---

# 38. Questions you should be able to answer

1. Why is Human + Leader analogous to the ManiSkill Motion Planner?
2. Why is Leader not part of the deployed policy?
3. Why are state and action different?
4. Why does each episode have both Parquet and MP4?
5. Why are timestamps necessary?
6. Why is metadata part of the dataset?
7. Why is there no ManiSkill-style replay step for this real-robot dataset?
8. Where does preprocessing actually happen?
9. What does PyAV do?
10. What does DataLoader do before model forward?
11. What does ResNet18 do?
12. What does `pretrained_backbone_weights=None` mean?
13. Why does `use_imagenet_stats=True` not imply ImageNet pretraining?
14. How do horizon, observation steps, and action steps differ?
15. Why is a DDPM timestep not a robot timestep?
16. Why can low training loss still produce failed rollouts?
17. Why are five demonstrations only a smoke test?
18. Why must you not delete a single Parquet file manually?
19. Why are `/dev/serial/by-id` paths preferable to `/dev/ttyACM0`?
20. Which LeRobot source files should you read first?
