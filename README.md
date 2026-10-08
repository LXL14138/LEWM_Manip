# S1 — a LeWM world model for the three-block grasp-and-carry task

`S1` is a latent world model (LeWM: ViT-tiny encoder + autoregressive latent predictor) trained **from scratch with the
plain dynamics objective only** — no tilt / danger labels were used while training it — on ~50 k simulated episodes and
branches of a robot arm that grasps a green block and carries it past two red pillars ("guards") that must not be
toppled. Given the last three camera frames and a sequence of commanded actions, it predicts the future latent state.
Small read-out heads fitted afterwards on its frozen latents recover the block / gripper positions to ~0.5 cm and the
red-pillar tilt to ~2°, and a latent-RRT planner that uses it as its imagination reached the goal from 16 / 30
near-pillar starts (the previously delivered fine-tuned model B: 11 / 30).

Everything needed to run the model is in this directory. **No training data is included or required.**

## Contents

| path | what it is |
|---|---|
| `model_state.pt` | the self-contained model bundle (72 MB): weights, architecture config, pixel / action normalisation, resize rule, upstream-code fingerprints. sha256 `bd7bc284…e98e` |
| `lewm_s1/` | a standalone loader (`load_world_model`) that depends only on the upstream LeWM code and the packages in `requirements.txt`; verified **bit-identical** to the project's own evaluation code |
| `demo.py` | loads the model, encodes three frames, rolls out 24 actions, prints the predicted red-pillar tilt per step and checks the rollout against `examples/expected_rollout.npz` (`PASS` / `FAIL`) |
| `examples/` | one 3-frame window from a **training** episode with its 24 following actions and the true tilt; the expected rollout latents for the self-check |
| `heads/tilt_head.pt` | the tilt / unsafe read-out head fitted on S1's frozen real latents (row 0 × 30 = tilt in degrees, row 1 = unsafe logit) |
| `provenance/` | the training run's `run.json` / `summary.json` / `environment.json`, the head's records, and the physical-comparison report (Chinese); machine-specific absolute paths in these records are replaced by placeholders (`<project>`, `<cluster>`, `<le-wm>`) |
| `RUNNING.md` | the exact commands that were run to verify this package, with their complete outputs, parity numbers and known limitations |
| `tools/` | the scripts that built the example and ran the parity check — they need the training repository and are **not** part of the runnable API |
| `env/` | the simulator add-on: the MuJoCo three-block scene the model was trained on (`three_block_scene.py`, `scene_config.py`, `task_progress.py`, verbatim copies of the project's scene, hashes in `env/ENV_SOURCES.json`), `env_demo.py` (scripted push of a pillar with the model's predictions next to the simulator's truth) and `requirements-env.txt` (the simulator packages). See "Running the model in the simulator" below |
| `examples/env_demo_*` | our run of `env/env_demo.py`: `env_demo_reference.npz` (its commands and truth, for `--compare`), `env_demo_expected_summary.json` (its numbers), `env_demo_strip.png` / `env_demo_rollout.gif` (what it looks like) |
| `requirements.txt`, `SHA256SUMS`, `LICENSE` | pinned dependencies; checksums of every file; license |

## Quick start

```bash
# 1. the upstream LeWM model code; the bundle records this exact commit and the loader refuses any other
git clone https://github.com/lucas-maes/le-wm.git
git -C le-wm checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac

# 2. python environment (Python 3.10 is required; CPU is enough for inference)
#    if your distribution has no python3.10 (Ubuntu 24.04 ships 3.12), get one with uv or pyenv, e.g.
#    uv python install 3.10 && uv venv -p 3.10 .venv    (Python 3.12 was not tested)
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
pip install -r requirements.txt

# 3. verify the files and run the demo
sha256sum -c SHA256SUMS
python demo.py --upstream ./le-wm
```

The demo ends with `PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)` when its rollout matches the shipped
reference (CPU, float32). Its complete expected output is in `RUNNING.md` §3. Options:
`--encoder-builder transformers` builds the same encoder without importing `stable-pretraining` and its heavy
dependencies (verified bit-identical; needs only torch, numpy, hydra-core, omegaconf, antlr4-python3-runtime, einops,
transformers); `--head none` skips the tilt head; `--device cuda` runs on a GPU (numbers then differ at the ~1e-5
level, outside the PASS contract); `--threads N`.

Note: the shipped example comes from a training episode that never touches a pillar, so its true tilt is 0° at every
step and the demo shows the head reading ≈ 0° — it demonstrates the pipeline and the numerical parity, not the head's
sensitivity. The real evaluations are the reports in `provenance/`.

## Using the model in your own code

```python
from lewm_s1 import load_world_model, letterbox

wm = load_world_model("model_state.pt", upstream_dir="./le-wm", device="cpu", head_path="heads/tilt_head.pt")

frames = ...   # uint8 [3, 224, 224, 3] RGB: the last three camera frames (t-2, t-1, t), letterboxed like the training data
actions = ...  # float [2 + K, 5] simulator commands in [-1, 1]: the two executed before frame t, then the K to imagine

z_ctx = wm.encode(frames)                        # [3, 192] latents (post-projector interface)
a = wm.normalize_actions(actions)                # [2 + K, 4] world-model actions (columns dx, dy, dz, dgripper)
z_hat = wm.rollout(z_ctx[None], a[None])[0]      # [K, 192] predicted latents for t+1 .. t+K
tilt = wm.tilt_deg(z_hat)                        # [K] predicted worst-guard tilt in degrees
unsafe = wm.unsafe_prob(z_hat)                   # [K] sigmoid(unsafe logit); alarm line wm.head_unsafe_threshold
```

- One control step is 0.05 s (`wm.control_dt`); the planner used K = 20 (1 s). The model always needs 3 consecutive
  frames as context, plus the two commands executed before the newest frame.
- `encode` expects frames prepared exactly like the training data: the 640 × 480 front-camera image LANCZOS-scaled to
  224 × 168 and centred on an ImageNet-mean canvas → 224 × 224 × 3 uint8 RGB. `lewm_s1.letterbox(image)` does this.
- Raw actions are the simulator's normalised commands `[dx, dy, dz, yaw, dgripper]` in [-1, 1] (±1 = ±0.05 m
  end-effector displacement per step; yaw must be 0 — it is fixed in this task). `normalize_actions` drops the yaw
  column and applies the bundle's recorded mean / scale. Details and units: `RUNNING.md` §7.
- `rollout` uses the same sliding 3-frame context as the project's evaluators; `predict_next(z_hist, a_hist)` is the
  one-step call.

## Running the model in the simulator

`env/env_demo.py` shows the model working inside the scene it was trained on, with the simulator's own truth next to
every prediction. Nothing recorded is replayed: the robot, the camera frames, the commands and the truth are all produced
live by MuJoCo in the same run.

**The scene** (`env/three_block_scene.py`, a byte-identical copy of the project's scene built on the UR5e + Robotiq 2F-85
of OGBench / stable-worldmodel): three 4 × 4 × 10 cm boxes on a table — the green target at [0.425, 0] m and two red
pillars ("guards") at y = −0.05 m and y = +0.05 m, 1 cm from the green block; a fixed front camera at [0.70125, 0, 0.324] m
looking at [0.425, 0, 0.12] m, fovy 48°, 640 × 480, shadows off; the gripper starts at [0.375, 0, 0.16] m with its yaw
fixed at 90°; a state is *unsafe* when a guard's tilt exceeds 30°. Pillars: 0.198 kg, friction 1.0; they tip once
past about 22° (atan(2/5) for a 4 × 10 cm box), so a push that stops at 30° still topples the pillar unless the carried
block holds it (which is what happens in our run).

**What the script does.** A plain waypoint controller with simulator-truth waypoints (no learned policy, no planner;
the same controller form as the project's scripted collectors) runs SETTLE (1 s, gripper open) → APPROACH above the
green block → DESCEND → CLOSE (until the two finger pads bear force on the block for 6 consecutive steps) → LIFT the
block to a pinch height of 10.5 cm (its centre ends at z = 9.3 cm, so the block's lower 5–6 cm overlap the top of
the 10 cm pillars) → PUSH sideways towards the left
pillar at 3 cm/s until the **true** worst-guard tilt exceeds `--stop-tilt-deg` (30°) → HOLD for 1 s. At every control
step t ≥ 2 the world model receives exactly what it received in training — the last three letterboxed camera frames
(t−2, t−1, t) and the two commands executed before frame t — plus the commands the robot is about to execute, and its
tilt read-out k = 1, 5, 10 and 20 steps ahead (0.05 s per step) is printed next to the simulator's true tilt of frame
t+k. The tilt head is also read on the real latents of every frame (k = 0).

**Install** (same Python 3.10 venv as above; both requirement files in one pip call — if `requirements.txt` was already
installed, the gymnasium 1.4.0 → 1.3.0 downgrade message is expected):

```bash
pip install -r requirements.txt -r env/requirements-env.txt
```

`env/requirements-env.txt` pins mujoco 3.11.0, ogbench 1.2.1, dm_control, gymnasium, pygame / pymunk / shapely (imported
by stable-worldmodel), glfw / PyOpenGL and **stable-worldmodel at git commit 4821c8e6** — the PyPI 0.1.1 release is an
older commit of the same package and is not what the scene was collected with. MuJoCo needs a rendering backend,
chosen with the environment variable `MUJOCO_GL` (`--gl` sets it, overriding an exported value with a note; without
`--gl` the demo keeps an exported `MUJOCO_GL` and defaults to `egl` when there is none):
`egl` — headless, needs a GPU driver that exposes EGL (what we used); `glfw` — needs a display (WSLg / X11 / macOS /
Windows); `osmesa` — CPU Mesa, headless anywhere, slow.

**Run** (from the package root; measured in a fresh single venv on a 2-thread CPU: 24 s wall, peak RSS 1.48 GB; as two
stages 7 s + 22 s):

```bash
MUJOCO_GL=egl python env/env_demo.py --upstream ./le-wm                # [--encoder-builder transformers] [--head none]
MUJOCO_GL=egl python env/env_demo.py --upstream ./le-wm --out run2 --compare examples/env_demo_reference.npz
# two environments (simulator here, torch elsewhere):
MUJOCO_GL=egl python env/env_demo.py --stage sim --out run_sim         # MuJoCo + torch, no LeWM stack; writes run_sim/record.npz
python env/env_demo.py --stage wm --record run_sim/record.npz --upstream ./le-wm --out run_wm   # torch only
```

`--out` must be a new directory (an existing one gives a one-line `FAIL`). In a single process the model is loaded
before the simulation starts, so a Python that lacks the torch / LeWM packages fails at once with a `FAIL` line that
points to the two-stage split and to `--encoder-builder transformers`. The terminal shows the phase table, then one row per step from 10 steps before the
push to the end (`--print-all` for every step):

```text
  t | phase    | cmd dy | true(t) | head(t) | pred t+1  / true  | pred t+5  / true  | pred t+10 / true  | pred t+20 / true  | p_unsafe(t+10)
128 | PUSH     | -0.111 |     8.5 |     2.9 |     6.1 /     9.7 |    16.4 /    14.4 |    19.9 /    22.3 |    34.7 /    34.7 | 0.453
129 | PUSH     | -0.116 |     9.7 |     5.6 |     8.3 /    10.8 |    19.2 /    15.8 |    22.8 /    24.2 |    44.0 /    34.8 | 0.896
130 | PUSH     | -0.120 |    10.8 |     8.3 |     9.5 /    11.8 |    21.7 /    17.3 |    29.6 /    26.3 |    45.6 /    34.9 | 0.976*
```

`cmd dy` is the raw sideways command of step t (−1 … 1; −0.12 ≈ a 6 mm target displacement), `true(t)` the simulator's
worst-guard tilt of frame t, `head(t)` the tilt head read on the real latent of frame t, `pred t+k / true` the tilt
read on the latent the model predicts k steps ahead from the context ending at t, next to the true tilt of frame t+k,
and `p_unsafe(t+10)` the model's unsafe score 10 steps ahead (`*` = above the head's alarm line
`head_unsafe_threshold` = 0.9608). Then a summary: the steps at which the truth first exceeds 15°, 22° (the project's
alarm line) and 30°; the mean absolute error per horizon over all rows, over the rows whose true tilt lies in [15°, 30°)
and over the rows where the truth is growing (the endpoints of the provenance report); for every horizon the first
step at which the prediction reached 22° / 30° and the lead time it would have given before the true 30° crossing; a
false-alarm check over the contact-free prefix (where the truth is exactly 0°); a long open-loop rollout from the first
push step that is told the whole push in advance; and the verdict line. `DEMO OK` means: the run completed, the truth
crossed 30°, the head reads real latents to < 5° MAE, the 10-step-ahead prediction reached 22° no later than the
truth reached 30° (an alarm would have fired in time), and no prediction at any horizon reached 22° over the
contact-free prefix (no false alarm). `DEMO RAN (model missed the event or raised a false alarm)` means the run
completed but the model did not meet that — the demo reports the model's quality, it never grades it as PASS / FAIL.
Written to `--out`:
`summary.json` (all numbers), `steps.csv` (one row per step with every horizon), `record.npz` (frames, commands, truth,
latents, predictions, ~2 MB), `reference.npz` (commands and truth without frames, 16 kB), `strip.png` (the three context
frames of the first push step and eight frames of the push, each with the true tilt and the prediction made 10 steps
earlier), `rollout.gif` (every frame in real time, 1 MB; `--no-gif` to skip) and `tilt_chart.png` (truth vs the k = 1 /
5 / 10 predictions). `--compare examples/env_demo_reference.npz` prints the difference between your simulator's commands
/ tilt trace / phase step counts and ours (0 on the machine that produced the reference).

**Exactness.** The frames go through the training pipeline: `env.render(camera="front_pixels")` (640 × 480) →
`lewm_s1.letterbox` (LANCZOS to 224 × 168 on the ImageNet-mean canvas), verified to be bit-identical (max |diff| = 0) to
the project's cache converter `wm_training.prepare_data.letterbox` and to its PNG round trip on live frames
(`RUNNING.md` §9.4). The commands are the simulator's own normalised actions (±1 = ±5 cm target displacement of the
pinch site per 0.05 s step, yaw 0, gripper ±1), normalised for the model with the bundle's training statistics; the
model's context is formed exactly as the project's closed-loop planner forms it (three frames + the two previous
commands; the first prediction is made at step 2). The truth is `info["safety"]` of the same simulator step.

**Our run** (`examples/env_demo_expected_summary.json`, 2 CPU threads, MuJoCo 3.11.0, EGL): 162 steps = 8.1 s of
simulation — SETTLE 0–19, APPROACH 20–41, DESCEND 42–86, CLOSE 87–93, LIFT 94–113, PUSH 114–141, HOLD 142–161. The true
tilt leaves 0° at frame 123, exceeds 15° at 134, 22° at 138 and 30° at 142, and ends at 35.4° (the leaning pillar rests
against the held block instead of falling flat). Tilt MAE on real latents 0.87° (0.60° where the truth ≤ 30°); MAE of the
predictions 0.83° (k = 1), 0.59° (k = 5), 1.18° (k = 10), 2.17° (k = 20) over all rows, and 2.6° / 2.7° / 3.0° / 6.3° over the
rows whose true tilt lies in [15°, 30°). The 10-step-ahead prediction reached 22° at step 129, 13 steps = 0.65 s before the
truth crossed 30° (k = 20: 21 steps = 1.05 s; k = 5: 7 steps; k = 1: 6 steps), and the unsafe score crossed the alarm line
at step 130. Over the 123 contact-free frames the largest predicted tilt was 0.85° and the largest unsafe score 0.000.
The long open-loop rollout from the first push step (48 steps, told the whole push in advance) does **not** reproduce the
event — it predicts at most 11° where the truth reaches 35° — which is the under-prediction of tilt growth in contact
described under "What it is good at"; what works is the sliding window that re-anchors on real frames every step.

**Limitations of the demo.** One scripted scenario (one start, one push direction and speed; `--lift-z`, `--push-speed`,
`--push-dir +y`, `--stop-tilt-deg` change it). MuJoCo is deterministic only within one build and CPU: on another
machine the contact events can shift by a few steps and the tilt trace will differ slightly, so the phase bounds and
numbers above are ours, not a specification — the demo always reports its own live truth, and `--compare` quantifies the
difference. Rendering backends differ at the pixel level (EGL vs glfw/d3d12 here: ≤ 9 of 255 on 1.1 % of the 224-px
values; the tilt read-out moved by ≤ 0.21° on real latents and ≤ 0.66° on predicted ones; the physics is unaffected).
The multi-step predictions use the commands that are actually executed later (open-loop evaluation "with the recorded
actions", the protocol of the provenance report); the one-step column is the only quantity that is online in the strict
sense. The head is calibrated for 0–30°; readings beyond that (here up to 38.7° for a 35.4° truth) are extrapolation.
Importing the scene pulls torch, torchvision, pygame, pymunk and shapely through stable-worldmodel, and a harmless
`ale-py` UserWarning is printed. The script is ~500 lines; `--stage sim` / `--stage wm` split it for machines whose
simulator and torch environments are separate (that is how it was verified here, see `RUNNING.md` §9).

## Model

| | |
|---|---|
| encoder | ViT-tiny, 12 layers, hidden 192, 3 heads, patch 14, 224 px, trained from scratch (no pretraining) + projector |
| predictor | autoregressive transformer over the 3 past latents and actions: depth 6, dim 192, 16 heads, MLP 2048, dropout 0.1 |
| action encoder | 4 → 192 embedding |
| parameters | 18.0 M |
| latent | 192-d post-projector vector; history of 3 frames; one step = 0.05 s |
| objective | LeWM: 0.91 × one-step prediction MSE (teacher-forced) + 0.09 × SIGReg; nothing else |

## Training

| | |
|---|---|
| data | four sources of 224-px simulation episodes (same scene, same camera): `wm_full_001` train split (4 500 random-exploration episodes, 42.7 % of windows), `wm_train_bench_001` training rows (961 targeted near-pillar episodes, 6.0 %), `wm_rrt_branches_001` (39 389 planner-action-law branches, 27.1 %), `wm_cf_branches_001` (5 400 counterfactual stop / retreat / push branches, 24.2 %) — 4.64 M four-frame windows per epoch |
| recipe | batch 128, AdamW, lr 5e-5 with 1 000 warm-up steps and cosine decay to 0.1 ×, weight decay 1e-3, bf16-mixed, 10 epochs = 362 850 optimizer steps, seed 20260921 |
| hardware | 1 × NVIDIA H100 80 GB (WashU RIS Compute2), 13.1 h of training (13 h 37 min wall, 68 GB host RAM) |
| validation, every epoch | `val_pred_loss / val_persistence_mse` on the held-out validation split: 0.156 → 0.123 → 0.109 → 0.097 → 0.087 → 0.075 → 0.071 → 0.063 → 0.061 → **0.0597** (the earlier base model, 3 epochs on random exploration only: 0.0852) |
| held-out data | the 250 reserved TEST episodes were never opened; validation / dev sets were never trained on |

Full records: `provenance/run.json` (every source with window counts, guard checks and hashes), `provenance/summary.json`
(per-epoch validation), `provenance/environment.json` (packages, driver).

## What it is good at — and not

Independent read-out probes, fitted on each model's own real latents and read on its predicted latents, compared with
the base model on identical anchors (`provenance/compare_wm_physical_s1_001_REPORT_ZH.md`; 15 pre-declared endpoints
with episode-bootstrap intervals, **all 15 better**):

| open-loop rollout with the recorded actions | base | **S1** |
|---|---|---|
| green-block position error, 16 steps ahead | 0.88 cm | **0.64 cm** |
| gripper position error, 16 steps ahead | 1.31 cm | **1.04 cm** |
| red-pillar tilt error, 10 steps ahead, true tilt in [15°, 30°) | 13.1° | **4.5°** |
| tilt error, 10 steps ahead, rows where the tilt is growing | 16.0° | **4.9°** |
| recall of "crosses 22° within 10 steps" (false-alarm rate) | 21 % (2.8 %) | **72 %** (1.4 %) |

Reading the frozen latents directly (no rollout), the tilt head reads the red-pillar tilt with 2.0° MAE on the held-out
near-pillar dev set.

Closed-loop latent-RRT pilot (60 starts, the planner configuration of the delivered model B, simulator truth as judge;
project-internal report, not included here): near-pillar starts — success 16 / 30 (B: 11), toppled 3 (B: 1),
timeout 9 (B: 16); ordinary starts 29 / 30 (B: 29). The remaining failures come from the model under-predicting the
tilt growth while the block is in contact with a pillar (predicted 5–13° where the truth reached 37°) and from the
direction signal under-estimating progress on the left side of the pillar.

Limitations: single training seed; the evaluation sets come from the same simulator and generator as the training data;
the model has only been used at 224 px with this camera and this scene; `unsafe_prob` is not a calibrated probability.

## Lineage

`S1` ← trained from scratch with the project's `train_scratch.py` (bit-identical to the upstream LeWM `train.py`
objective on a single source; not included here). Related models in the project: `base` (3 epochs, random exploration
only), `S0b` (10 epochs, random exploration only), `B` (base fine-tuned with a tilt loss; the previous deliverable),
`S1T` (S1 fine-tuned with a frozen tilt head: better offline, no gain in the planner).

## License

The weights and the code in this directory are released under the MIT License (`LICENSE`). The upstream LeWM code
(https://github.com/lucas-maes/le-wm, Lucas Maes, 2026) is MIT-licensed and is not redistributed here — clone it at the
pinned commit as described above.
