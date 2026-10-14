# RUNNING.md — exactly what was run to verify this package (2026-10-08)

Everything below was executed on CPU (no CUDA device visible, `OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`,
TF32 off) with the project's training venv: CPython 3.10.21, torch 2.14.0+cu130, numpy 2.2.6, Linux x86-64 (WSL2).
Machine-specific paths are written as placeholders: `<project>` = the training project's root, `<package>` = this
directory, `<le-wm>` = the upstream clone, `<scratch>` = a temporary directory, `<cluster>` = the cluster storage root
of the training run; replace them with your own.

## 1. Get the upstream LeWM code (pinned commit)

The bundle was trained with https://github.com/lucas-maes/le-wm.git at commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT). `lewm_s1` fingerprints `jepa.py`, `module.py` and
`config/train/model/lewm.yaml` of the directory you pass as `--upstream` and refuses anything else.

```bash
git clone https://github.com/lucas-maes/le-wm.git
git -C le-wm checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
```

What was run here (offline; a clone of the local mirror of that repository, then the pinned commit):

```text
$ git clone --quiet --no-hardlinks file://<le-wm> le-wm
$ git -C le-wm checkout --quiet 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
HEAD: 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
status: 0 dirty entries
41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96  jepa.py
0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620  module.py
7be97eaa2c83f809b9ea7a6da5a7d3f3c70502c27537383f54db32335e21e42b  config/train/model/lewm.yaml
```

These three sha256 values are the ones stored in `model_state.pt["upstream_sha256"]` and in the tilt head's binding.

## 2. Python environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # CPU wheels; or the CUDA wheels of your choice
pip install -r requirements.txt
```

**From-scratch install, measured (2026-10-08).** Ubuntu 24.04 ships no `python3.10`, so a fresh venv was made from a
uv-provisioned CPython 3.10.21 (`uv python install 3.10`; `python3.10 -m venv`, pip upgraded to 26.2.1). In it, the
two pip steps above (plus the venv creation) and `pip install -r env/requirements-env.txt` (section 9) took 80 s in total (torch cpu 17 s,
`requirements.txt` 44 s, `env/requirements-env.txt` 15 s), installed 139 packages (`pip freeze`), 2.5 GB on disk,
`pip check`: "No broken requirements found"; torch 2.14.0+cpu, torchvision 0.29.0+cpu, numpy 2.2.6, Pillow 12.3.0,
transformers 5.17.0, stable-pretraining 0.1.8. The only notable pip message was the expected gymnasium 1.4.0 -> 1.3.0
downgrade when `env/requirements-env.txt` is installed after `requirements.txt` (the one-pass command in section 9
avoids it). In that venv, from a copy of the package with a fresh clone of the pinned commit: `python demo.py
--upstream ./le-wm` -> `PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)`, exit 0, 6.2 s wall, peak RSS
0.99 GB; the single-process simulator demo (section 9.5) -> `DEMO OK`. The venv was deleted afterwards.

Before that, on the offline development machine (training venv without a `pip` module), every pin in
`requirements.txt` was checked against the venv the demo and parity check ran in:

```text
$ python tools/check_requirements.py
ok  torch                    required 2.14.0       installed 2.14.0
ok  numpy                    required 2.2.6        installed 2.2.6
ok  hydra-core               required 1.3.7        installed 1.3.7
ok  omegaconf                required 2.3.1        installed 2.3.1
ok  antlr4-python3-runtime   required 4.9.3        installed 4.9.3
ok  einops                   required 0.8.2        installed 0.8.2
ok  transformers             required 5.17.0       installed 5.17.0
ok  pillow                   required 12.3.0       installed 12.3.0
ok  stable-pretraining       required 0.1.8        installed 0.1.8
... (31 more stable-pretraining dependencies, all ok)
python 3.10.21: 40 / 40 pins satisfied
```

Why the list is long: the bundle's hydra config names `stable_pretraining.backbone.utils.vit_hf` as the encoder
factory, and importing `stable_pretraining` pulls in lightning, wandb, datasets, scikit-learn, timm, torchvision,
... (its declared hard dependencies). If you do not want that, use the opt-in path
`python demo.py --encoder-builder transformers` / `load_world_model(..., encoder_builder="transformers")`, which
builds the identical `transformers.ViTModel` directly and needs only torch, numpy, hydra-core, omegaconf,
antlr4-python3-runtime, einops, transformers (+ pillow for `letterbox`). Both paths were verified bit-identical
(section 4). The upstream README itself installs `stable-worldmodel[train,env]` via `uv`; that is not needed here.

`provenance/environment.json` is the cluster environment of the training run; its `pip freeze` failed there, so it
is NOT a dependency list (the frozen requirements file of that run holds only an error line and was not copied).

## 3. Demo

Run from anywhere; the defaults are resolved relative to `demo.py`:

```bash
cd exports/wm_S1_release_001
python demo.py --upstream /path/to/le-wm
# options: [--bundle model_state.pt] [--head heads/tilt_head.pt | --head none] [--example examples/train_episode_window.npz]
#          [--expected examples/expected_rollout.npz] [--device cpu] [--threads 2] [--tol 1e-4] [--encoder-builder hydra|transformers]
```

Verified command and its complete output (clean clone of the pinned commit, no `__pycache__`; `/usr/bin/time -v`):

```text
$ python demo.py --upstream <scratch>/le-wm
08:20:53 | INFO  | atomic_chec~| [atomic_save] installed crash-safe checkpoint plugin (write to sibling .tmp + fsync + atomic rename)
08:20:53 | INFO  | utils.py    | Created ViT-tiny from scratch with config: {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3, 'intermediate_size': 768, 'image_size': 224, 'patch_size': 14}
loaded WorldModel(latent_dim=192, history=3, image_size=224, control_dt=0.05, device=cpu, upstream_commit=8edfeb33, head=yes, encoder_builder='hydra') in 8.6 s
  bundle   <package>/model_state.pt  sha256 bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e
  upstream <scratch>/le-wm  fingerprints OK for commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
  head     <package>/heads/tilt_head.pt  sha256 2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6  outputs ['tilt_deg / 30', 'unsafe_logit']  unsafe threshold 0.9608285427093506
  torch 2.14.0+cu130  threads 2  device cpu  tf32 off
example: cache wm_full_001 TRAIN episode 0, context frames 101..103, 24 future actions
encoded 3 frames + 24-step rollout in 0.27 s; latent shape (24, 192)

context frames (head on REAL latents):
  frame | pred tilt deg | unsafe prob | true tilt deg
    101 |        -0.324 |      0.0000 |         0.000
    102 |        -0.300 |      0.0000 |         0.000
    103 |        -0.186 |      0.0000 |         0.000

rollout (head on PREDICTED latents; threshold 0.9608285427093506):
  step | frame | pred tilt deg | unsafe prob | true tilt deg | true unsafe
     1 |   104 |        -0.176 |      0.0000 |         0.000 | False
     2 |   105 |        -0.241 |      0.0000 |         0.000 | False
     3 |   106 |        -0.262 |      0.0000 |         0.000 | False
     4 |   107 |        -0.285 |      0.0000 |         0.000 | False
     5 |   108 |        -0.296 |      0.0000 |         0.000 | False
     6 |   109 |        -0.292 |      0.0000 |         0.000 | False
     7 |   110 |        -0.109 |      0.0000 |         0.000 | False
     8 |   111 |         0.151 |      0.0000 |         0.000 | False
     9 |   112 |         0.481 |      0.0000 |         0.000 | False
    10 |   113 |         0.315 |      0.0000 |         0.000 | False
    11 |   114 |         0.337 |      0.0000 |         0.000 | False
    12 |   115 |         0.225 |      0.0000 |         0.000 | False
    13 |   116 |        -0.130 |      0.0000 |         0.000 | False
    14 |   117 |        -0.248 |      0.0000 |         0.000 | False
    15 |   118 |        -0.287 |      0.0000 |         0.000 | False
    16 |   119 |        -0.207 |      0.0000 |         0.000 | False
    17 |   120 |        -0.047 |      0.0000 |         0.000 | False
    18 |   121 |         0.020 |      0.0000 |         0.000 | False
    19 |   122 |         0.160 |      0.0000 |         0.000 | False
    20 |   123 |         0.278 |      0.0000 |         0.000 | False
    21 |   124 |         0.321 |      0.0000 |         0.000 | False
    22 |   125 |         0.347 |      0.0000 |         0.000 | False
    23 |   126 |         0.383 |      0.0000 |         0.000 | False
    24 |   127 |         0.502 |      0.0000 |         0.000 | False

reference: torch 2.14.0+cu130, 2 threads, cpu, hydra
max|diff| context latents = 0.000e+00, rollout latents = 0.000e+00 (tolerance 0.0001)
max|diff| predicted tilt deg vs reference = 0.000e+00
PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.65
        Maximum resident set size (kbytes): 1297808
```

Exit code 0. The two `INFO` lines come from stable-pretraining / lightning plugins at import time and are harmless.
Wall time is dominated by imports (~8 s); the actual encode + 24-step rollout takes 0.27 s; peak RSS 1.30 GB.

Variants that were also run (same clone):

```text
$ python demo.py --upstream .../le-wm --head none          -> PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001), exit 0
$ python demo.py --upstream .../le-wm --encoder-builder transformers
                                                           -> PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001), exit 0
                                                              Elapsed 0:04.72, Maximum resident set size 1107284 kB
$ python demo.py --upstream .../le-wm --threads 1          -> PASS max|diff| ctx=0.000e+00 roll=3.815e-06 (tilt 9.835e-06), exit 0
$ python demo.py --upstream .../le-wm --threads 4          -> PASS max|diff| ctx=0.000e+00 roll=3.815e-06 (tilt 9.835e-06), exit 0
```

Negative tests (the loader must refuse other upstream code):

```text
$ python demo.py --upstream .../le-wm-broken      # same checkout with ONE comment line appended to module.py
RuntimeError: Upstream LeWM code differs from the code this bundle was trained with. Expected git commit
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac of https://github.com/lucas-maes/le-wm.git; mismatching files: module.py
(expected sha256 0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620, found
903d41fdfaf8f00476f15f7d54932bbd0c487547ac4d1092115f5469bddc7041). Fix: git clone https://github.com/lucas-maes/le-wm.git
&& git -C le-wm checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
FAIL RuntimeError: ...            (exit 1)

$ python demo.py --upstream /some/other/dir
FileNotFoundError: Missing upstream LeWM file /some/other/dir/jepa.py. --upstream must point at a checkout of
https://github.com/lucas-maes/le-wm.git at commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac: git clone ... (exit 1)
```

## 4. Parity against the project's own code (tools/parity_check.py; needs the training repo, not runnable from this package)

```text
$ cd <project>
$ python exports/wm_S1_release_001/tools/parity_check.py --upstream <le-wm>
project load_world: 6.3 s
[hydra] {"load_seconds": 1.0, "normalize_actions_max_abs": 0.0, "normalize_actions_bit_equal": true,
  "context_latents_max_abs": 0.0, "context_latents_bit_equal": true, "rollout_latents_max_abs": 0.0,
  "rollout_latents_bit_equal": true, "predict_next_vs_rollout_step0_max_abs": 0.0, "tilt_deg_max_abs": 0.0,
  "unsafe_prob_max_abs": 0.0, "context_tilt_deg_max_abs": 0.0, "context_unsafe_prob_max_abs": 0.0,
  "jepa_rollout_3steps_vs_standalone_rollout_max_abs": 0.0,
  "vs_expected_npz": {"context_latents": 0.0, "rollout_latents": 0.0, "tilt_deg": 0.0, "unsafe_prob": 0.0},
  "head_unsafe_threshold": 0.9608285427093506, "project_head_unsafe_threshold": 0.9608285427093506}
[transformers] { ... identical: every max|diff| 0.0, every bit_equal true ... }
wrote .../exports/wm_S1_release_001/tools/parity_report.json
PARITY PASS
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:10.72
        Maximum resident set size (kbytes): 1483364
```

Project path used as the reference, in the SAME process (CPU float32, 2 threads, TF32 off):
`wm_training.decoder.load_world` (bundle + cache index.json + upstream fingerprints) ->
`wm_training.evaluate_random_actions.encode_frames` (batch 3) ->
`wm_training.evaluate_bench_counterfactual.normalize_actions / action_sequence / predict_rollouts` (horizon 24) ->
`wm_training.tilt_head.load_tilt_head(head_binding(binding))` (full binding check incl. cache_index_sha256) ->
`wm_training.evaluate_bench_wm.score_tilt`. Results vs. `lewm_s1` (both encoder builders):

| quantity | max abs diff | note |
|---|---|---|
| normalised actions (26 x 4) | 0.0 | bit-equal to the project and to the cache's own `action` dataset |
| context latents (3 x 192) | 0.0 | bit-equal |
| rollout latents (24 x 192) | 0.0 | bit-equal (task bound 1e-5) |
| `predict_next` vs `rollout[:, 0]` | 0.0 | |
| tilt head: tilt deg (24) / unsafe prob (24) | 0.0 / 0.0 | project head loaded with the full binding check |
| upstream `JEPA.rollout` (3 steps) vs project `rollout_latents` | 0.0 | same check as `evaluate_bench_wm.parity_check` |
| upstream `JEPA.rollout` (3 steps) vs `lewm_s1.rollout` | 0.0 | |
| everything vs `examples/expected_rollout.npz` | 0.0 | |

The full numbers are in `tools/parity_report.json`.

## 5. Example provenance (examples/train_episode_window.npz, examples/expected_rollout.npz)

* Cache `data/wm_full_001` (`index.json` sha256 `25ada0a303fcdbe08b04ae236a4a5d1baef389aa35734d45304171807833c786`,
  the cache the bundle and the head are bound to). Only `index.json` and ONE episode file were opened; no reserved
  test episode was touched (ids come from `split_ids["train"]`).
* Episode: the smallest TRAIN id, **episode 0** (`episodes/episode_0000.h5`, mode `carry`, 600 actions / 601 frames),
  file sha256 `60ff6c788c3fc52abe022a07e0f8331d52174be08da33b681ade630f5a51b294`, its stored `pixels_sha256`
  attribute `8725498ce1676f5a3cc07d5f165c22b39c7a3d27b7c72d445ef382679a498951`.
* Window: **t = 103** by the deterministic rule "argmax over t in [2, T-24] of the mean |dx, dy, dz| command over
  steps t..t+23" (a grasped lift of the green block; max |WM action| 1.99). Context frames 101, 102, 103; history
  commands a[101], a[102]; future commands a[103] .. a[126]; truth = frames 104 .. 127.
* **The truth is degenerate on purpose of the rule, not by choice:** episode 0 never touches a guard, so the true
  worst-guard tilt is 0.0 deg at every frame and `true_unsafe` is all False. The demo therefore shows the head
  reading ~0 deg / unsafe prob 0.0000 (|pred tilt| <= 0.5 deg over 24 predicted steps), not a disturbance.
  Any other window would have broken the "smallest train id" rule.
* `expected_rollout.npz` = the loader's own CPU float32 output on that window (torch 2.14.0, 2 threads, encode
  batch 3, rollout batch 1, hydra encoder builder); bit-identical to the project (section 4).
* Sizes: 42,210 B + 22,770 B.

Bundle / head identity: `model_state.pt` sha256 `bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e`
(72,270,497 B, byte copy of `runs/lewm_scratch_S1_b128_c2_001/model_state.pt`); `heads/tilt_head.pt` sha256
`2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6` (472,717 B, byte copy of
`runs/tilt_head_S1_frozen_002/tilt_head.pt`). Both verified with `cmp` against the sources.

## 6. The tilt head's numbers

`unsafe_prob` is `sigmoid(unsafe logit)` and is NOT a calibrated probability. The head file's provenance records
`unsafe_threshold = 0.9608285427093506`: the alarm line chosen so that 95 % of the truly unsafe (tilt > 30 deg)
frames of the held-out dev benchmark are flagged on REAL latents (k = 0); `WorldModel.head_unsafe_threshold`
exposes it. `tilt_deg` = 30 x row 0 of the head (holdout MAE 1.11 deg on real latents; see
`provenance/tilt_head_S1_frozen_002_run.json`). The head was trained on real latents only; on PREDICTED latents its
quality degrades with the rollout horizon (see the two provenance reports).

## 7. Known limitations

* **Numerics.** Bit-exact agreement (0.0) holds within one torch build on one CPU with the same thread count and
  batch sizes. Changing only the thread count (1 or 4 instead of 2) already moves the rollout latents by 3.8e-6
  and the tilt read-out by 1e-5 deg; another CPU / torch build will differ at a similar level; the demo tolerance
  1e-4 is meant to absorb that. `--device cuda` runs but is outside the PASS contract (the model was trained
  bf16-mixed and exported float32; TF32 is switched off by the loader).
* **Letterbox requirement.** `encode` expects 224x224x3 uint8 RGB frames produced like the cache: the raw 640x480
  front-camera image (fixed camera, no overlay, shadows off) LANCZOS-scaled to 224x168 and centred on an
  ImageNet-mean canvas (124, 116, 104); use `lewm_s1.letterbox`. It is a no-op on cache frames (verified) and
  depends on Pillow's LANCZOS kernel (Pillow 12.3.0 here; other versions may change a few LSBs and hence the
  latents slightly). The demo does not depend on Pillow because the example stores cache frames.
* **Action units.** Raw actions are the simulator's normalised commands in [-1, 1], NOT metres:
  `[dx, dy, dz, yaw, dgripper]`, dx/dy/dz = +-1 -> +-0.05 m end-effector displacement per 0.05 s control step,
  the yaw column must be 0 (yaw fixed at 90 deg; `normalize_actions` refuses non-zero yaw and |a| > 1),
  dgripper in [-1, 1]. A rollout needs the TWO commands executed before the newest context frame (`a[t-2], a[t-1]`)
  in addition to the future commands; a deployment must record them.
* **History.** The model always needs 3 consecutive frames (0.05 s apart) as context; `predict_next` /
  `rollout` are undefined with fewer.
* **Example truth is all zeros** (section 5); the package demonstrates the pipeline and the parity, not the
  head's sensitivity. The provenance reports contain the real evaluations.
* **Dependencies.** The default (hydra) path imports stable-pretraining and its heavy dependency set; the
  `transformers` encoder builder avoids it (verified bit-identical).
* `tools/make_example.py` and `tools/parity_check.py` need the training repository and cache; they are included
  for transparency only.

## 8. Integrity

`SHA256SUMS` lists every file of the package (relative paths, `sha256sum -c SHA256SUMS` from the package root).
It was written LAST; a `README.md` added afterwards is not listed unless appended
(`sha256sum README.md >> SHA256SUMS`). `lewm_s1/__pycache__` is excluded by `.gitignore` and must not be shipped.

## 9. Simulator demo (env/env_demo.py) — what was run (2026-10-08)

### 9.1 Environments

The scene files `env/three_block_scene.py`, `env/scene_config.py`, `env/task_progress.py` are byte-identical copies of
the project's scene (`cmp` identical; sha256 `3d7e5049…76c8`, `ef5667a1…289e`, `89ab0c1d…0c1c`, recorded in
`env/ENV_SOURCES.json`). The development machine has two venvs and neither can run `--stage all` (the single-venv run
was measured separately in a fresh venv, see section 2 and 9.5):

| venv | has | lacks |
|---|---|---|
| simulation `<project>/.venv` | CPython 3.10.21, mujoco 3.11.0, ogbench 1.2.1, stable-worldmodel 0.1.1 from git commit 4821c8e6 (its `direct_url.json` records the URL and commit), dm_control 1.0.44, gymnasium 1.3.0, pygame 2.6.1, pymunk 7.3.0, shapely 2.1.2, glfw 2.10.2, PyOpenGL 3.1.10, torch 2.14.0, numpy 2.2.6, Pillow 12.3.0, hydra-core 1.3.6 | transformers, stable-pretraining (so the bundle cannot be instantiated) |
| training `<le-wm>/.venv` | the `requirements.txt` stack (torch 2.14.0+cu130, transformers 5.17.0, stable-pretraining 0.1.8, Pillow 12.3.0, numpy 2.2.6) | mujoco, ogbench, dm_control |

Therefore the commands documented below are `--stage sim` in the simulation venv and `--stage wm` in the training
venv. The single-venv `--stage all` run and the from-scratch install were then measured in a fresh venv
(uv-provisioned CPython 3.10.21, torch 2.14.0+cpu; details in section 2 and 9.5): `pip install -r
env/requirements-env.txt` took 15 s on top of `requirements.txt` (gymnasium 1.4.0 -> 1.3.0 downgrade printed, expected),
`pip check` clean. Before that, every pin of `env/requirements-env.txt` had been checked against the simulation venv
with a small importlib.metadata script (same idea as `tools/check_requirements.py`, plus the git commit of
stable-worldmodel):

```text
ok  mujoco             required 3.11.0     installed 3.11.0
ok  ogbench            required 1.2.1      installed 1.2.1
ok  stable-worldmodel  required git 4821c8e6a3f0  installed 0.1.1 from https://github.com/galilai-group/stable-worldmodel.git @ 4821c8e6a3f0f83b7e6a80da3a757e026ea9026b
ok  dm_control         required 1.0.44     installed 1.0.44
ok  gymnasium          required 1.3.0      installed 1.3.0
ok  pygame             required 2.6.1      installed 2.6.1
ok  pymunk             required 7.3.0      installed 7.3.0
ok  shapely            required 2.1.2      installed 2.1.2
ok  glfw               required 2.10.2     installed 2.10.2
ok  PyOpenGL           required 3.1.10     installed 3.1.10
ok  tqdm               required 4.70.1     installed 4.70.1
ok  absl-py            required 2.5.0      installed 2.5.0
ok  lxml               required 6.1.3      installed 6.1.3
ok  dm-tree            required 0.1.10     installed 0.1.10
python 3.10.21: all pins satisfied
```

Hardware: 2 CPU threads (`OMP_NUM_THREADS=2`, `torch.set_num_threads(2)`), WSL2 Ubuntu 24.04, NVIDIA GeForce RTX 5080
(driver 617.14) used only by the EGL renderer; the model ran on CPU (`CUDA_VISIBLE_DEVICES=` empty).

### 9.2 Commands and complete output (from the package root)

Simulation stage (simulation venv, `/usr/bin/time -v`; the only stderr output is the harmless ale-py UserWarning of
`stable_worldmodel.envs`):

```text
$ cd exports/wm_S1_release_001
$ MUJOCO_GL=egl python env/env_demo.py --stage sim --out <scratch>/sim --compare examples/env_demo_reference.npz
scene: control_dt=0.05, camera_fovy_deg=48.0, camera_pos_m=[0.70125, 0.0, 0.324], pillar_size_m=[0.04, 0.04, 0.1], pillar_mass_kg=0.19840000000000002, pillar_friction=[1.0, 0.005, 0.0001], shadows=False, unsafe_tilt_deg=30.0, physics_timestep=0.002, mujoco=3.11.0
  SETTLE   steps   0.. 19 ( 20 steps, 1.00 s)  worst tilt   0.0 deg  pinch z 0.157 m  grasped=False
  APPROACH steps  20.. 41 ( 22 steps, 1.10 s)  worst tilt   0.0 deg  pinch z 0.186 m  grasped=False
  DESCEND  steps  42.. 86 ( 45 steps, 2.25 s)  worst tilt   0.0 deg  pinch z 0.057 m  grasped=False
  CLOSE    steps  87.. 93 (  7 steps, 0.35 s)  worst tilt   0.0 deg  pinch z 0.055 m  grasped=True
  LIFT     steps  94..113 ( 20 steps, 1.00 s)  worst tilt   0.0 deg  pinch z 0.101 m  grasped=True
  PUSH     steps 114..141 ( 28 steps, 1.40 s)  worst tilt  30.6 deg  pinch z 0.106 m  grasped=True
  HOLD     steps 142..161 ( 20 steps, 1.00 s)  worst tilt  35.4 deg  pinch z 0.104 m  grasped=True
  sha256 raw_actions c363fd06d37cfe89...  tilt_deg 1afadda1e73c5e1d...
simulation finished in 5.2 s: 162 steps, max true tilt 35.4 deg
compare with examples/env_demo_reference.npz: steps 162 vs 162, over the common 162 steps max|delta raw_actions| 0.000e+00, |delta tilt| 0.000e+00 deg, |delta pinch| 0.000e+00 m
  phase bounds here {"SETTLE": [0, 20], "APPROACH": [20, 42], "DESCEND": [42, 87], "CLOSE": [87, 94], "LIFT": [94, 114], "PUSH": [114, 142], "HOLD": [142, 162]}
  phase bounds ref  {"SETTLE": [0, 20], "APPROACH": [20, 42], "DESCEND": [42, 87], "CLOSE": [87, 94], "LIFT": [94, 114], "PUSH": [114, 142], "HOLD": [142, 162]}
  reference: mujoco 3.11.0, GL egl; here: mujoco 3.11.0, GL egl
SIM DONE 162 steps, max true tilt 35.4 deg, record <scratch>/sim/record.npz
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:06.77
        Maximum resident set size (kbytes): 1506372
```

Exit code 0 (the `--compare` here is against the previous shipped reference, produced by the same scene before the
review fixes of the script: 0 difference). `sim/record.npz` 1,922,568 B, `sim/reference.npz` 15,820 B
(= `examples/env_demo_reference.npz`). The stored `arguments` field holds paths relative to the package root or only
their last component (`"out": "sim"`), never a machine-specific directory.
Full sha256 of the arrays: raw_actions `c363fd06d37cfe89ff5171b1215f2b3653b16745a8180df580ed79d8bdd60a6f`, tilt_deg
`1afadda1e73c5e1d6303e472511c5a91f02d93677e172c1a4d37c1aacd16f072`.

World-model stage (training venv, `/usr/bin/time -v`, upstream = the local clone at commit 8edfeb33):

```text
$ python env/env_demo.py --stage wm --record <scratch>/sim/record.npz --upstream <le-wm> --out <scratch>/wm
11:15:36 | INFO  | atomic_chec~| [atomic_save] installed crash-safe checkpoint plugin (write to sibling .tmp + fsync + atomic rename)
11:15:36 | INFO  | utils.py    | Created ViT-tiny from scratch with config: {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3, 'intermediate_size': 768, 'image_size': 224, 'patch_size': 14}
loaded WorldModel(latent_dim=192, history=3, image_size=224, control_dt=0.05, device=cpu, upstream_commit=8edfeb33, head=yes, encoder_builder='hydra') in 5.6 s
  bundle   <package>/model_state.pt  sha256 bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e
  head     <package>/heads/tilt_head.pt  sha256 2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6  unsafe threshold 0.9608285427093506
loaded <scratch>/sim/record.npz (162 steps, GL egl)
  torch 2.14.0+cu130  threads 2  device cpu  tf32 off  GL backend of the frames: egl
encoded 163 frames in 3.7 s
160 rollouts x 20 steps + one 48-step rollout from the first push step in 9.6 s

phases: SETTLE 0..19 (20 steps, 1.00 s)  APPROACH 20..41 (22 steps, 1.10 s)  DESCEND 42..86 (45 steps, 2.25 s)  CLOSE 87..93 (7 steps, 0.35 s)  LIFT 94..113 (20 steps, 1.00 s)  PUSH 114..141 (28 steps, 1.40 s)  HOLD 142..161 (20 steps, 1.00 s)

  t | phase    | cmd dy | true(t) | head(t) | pred t+1  / true  | pred t+5  / true  | pred t+10 / true  | pred t+20 / true  | p_unsafe(t+10)
104 | LIFT     | -0.028 |     0.0 |    -0.0 |    -0.0 /     0.0 |    -0.1 /     0.0 |     0.2 /     0.0 |     0.2 /     2.6 | 0.000
105 | LIFT     | -0.028 |     0.0 |    -0.0 |    -0.1 /     0.0 |    -0.0 /     0.0 |    -0.0 /     0.0 |     0.4 /     4.2 | 0.000
106 | LIFT     | -0.028 |     0.0 |    -0.1 |    -0.0 /     0.0 |     0.1 /     0.0 |    -0.1 /     0.0 |     0.7 /     5.8 | 0.000
107 | LIFT     | -0.028 |     0.0 |    -0.1 |     0.0 /     0.0 |     0.1 /     0.0 |    -0.8 /     0.0 |     2.7 /     7.0 | 0.000
108 | LIFT     | -0.028 |     0.0 |    -0.0 |    -0.1 /     0.0 |     0.3 /     0.0 |    -0.8 /     0.0 |     4.6 /     8.5 | 0.000
109 | LIFT     | -0.028 |     0.0 |    -0.0 |     0.0 /     0.0 |     0.4 /     0.0 |    -0.3 /     0.0 |     4.4 /     9.7 | 0.000
110 | LIFT     | -0.027 |     0.0 |     0.1 |     0.2 /     0.0 |     0.3 /     0.0 |    -0.6 /     0.0 |     4.5 /    10.8 | 0.000
111 | LIFT     | -0.026 |     0.0 |     0.2 |     0.3 /     0.0 |     0.1 /     0.0 |    -0.5 /     0.0 |     6.0 /    11.8 | 0.000
112 | LIFT     | -0.023 |     0.0 |     0.1 |     0.2 /     0.0 |    -1.0 /     0.0 |    -0.1 /     0.0 |     8.4 /    13.1 | 0.000
113 | LIFT     | -0.021 |     0.0 |     0.1 |     0.1 /     0.0 |    -0.5 /     0.0 |     0.1 /     1.0 |    10.6 /    14.4 | 0.000
114 | PUSH     | -0.049 |     0.0 |     0.1 |     0.0 /     0.0 |    -0.6 /     0.0 |     0.4 /     2.6 |    10.6 /    15.8 | 0.000
115 | PUSH     | -0.065 |     0.0 |    -0.6 |    -0.2 /     0.0 |    -0.7 /     0.0 |     1.3 /     4.2 |     9.7 /    17.3 | 0.000
116 | PUSH     | -0.074 |     0.0 |    -0.4 |    -0.5 /     0.0 |     0.2 /     0.0 |     2.4 /     5.8 |    19.7 /    18.9 | 0.000
117 | PUSH     | -0.079 |     0.0 |    -0.8 |    -0.5 /     0.0 |     0.6 /     0.0 |     3.3 /     7.0 |    20.5 /    20.6 | 0.000
118 | PUSH     | -0.081 |     0.0 |    -0.6 |    -0.1 /     0.0 |     0.9 /     1.0 |     5.5 /     8.5 |    15.6 /    22.3 | 0.000
119 | PUSH     | -0.083 |     0.0 |    -0.0 |     0.3 /     0.0 |     1.5 /     2.6 |     6.8 /     9.7 |    12.2 /    24.2 | 0.000
120 | PUSH     | -0.084 |     0.0 |     0.4 |     0.6 /     0.0 |     1.7 /     4.2 |     8.1 /    10.8 |    14.2 /    26.3 | 0.000
121 | PUSH     | -0.085 |     0.0 |     0.7 |     0.6 /     0.0 |     1.9 /     5.8 |    11.4 /    11.8 |    22.2 /    28.3 | 0.000
122 | PUSH     | -0.085 |     0.0 |     0.6 |     0.9 /     1.0 |     3.5 /     7.0 |    16.0 /    13.1 |    26.0 /    30.6 | 0.000
123 | PUSH     | -0.089 |     1.0 |     1.0 |     1.5 /     2.6 |     5.9 /     8.5 |    18.1 /    14.4 |    30.7 /    32.6 | 0.000
124 | PUSH     | -0.094 |     2.6 |     1.5 |     1.6 /     4.2 |     7.9 /     9.7 |    18.5 /    15.8 |    34.3 /    33.6 | 0.003
125 | PUSH     | -0.098 |     4.2 |     1.5 |     1.7 /     5.8 |     9.9 /    10.8 |    17.8 /    17.3 |    34.6 /    34.1 | 0.004
126 | PUSH     | -0.101 |     5.8 |     1.7 |     2.0 /     7.0 |    11.9 /    11.8 |    16.9 /    18.9 |    33.5 /    34.4 | 0.018
127 | PUSH     | -0.106 |     7.0 |     1.9 |     3.9 /     8.5 |    14.1 /    13.1 |    18.0 /    20.6 |    31.3 /    34.6 | 0.147
128 | PUSH     | -0.111 |     8.5 |     2.9 |     6.1 /     9.7 |    16.4 /    14.4 |    19.9 /    22.3 |    34.7 /    34.7 | 0.453
129 | PUSH     | -0.116 |     9.7 |     5.6 |     8.3 /    10.8 |    19.2 /    15.8 |    22.8 /    24.2 |    44.0 /    34.8 | 0.896
130 | PUSH     | -0.120 |    10.8 |     8.3 |     9.5 /    11.8 |    21.7 /    17.3 |    29.6 /    26.3 |    45.6 /    34.9 | 0.976*
131 | PUSH     | -0.123 |    11.8 |     9.8 |    10.4 /    13.1 |    21.8 /    18.9 |    37.4 /    28.3 |    43.5 /    35.0 | 0.996*
132 | PUSH     | -0.125 |    13.1 |    10.6 |    12.4 /    14.4 |    21.1 /    20.6 |    37.2 /    30.6 |    27.7 /    35.0 | 0.991*
133 | PUSH     | -0.126 |    14.4 |    12.5 |    15.7 /    15.8 |    20.6 /    22.3 |    31.5 /    32.6 |    22.8 /    35.1 | 0.841
134 | PUSH     | -0.125 |    15.8 |    15.3 |    19.2 /    17.3 |    21.1 /    24.2 |    28.5 /    33.6 |    21.7 /    35.1 | 0.210
135 | PUSH     | -0.124 |    17.3 |    19.3 |    21.6 /    18.9 |    23.1 /    26.3 |    26.9 /    34.1 |    20.9 /    35.2 | 0.017
136 | PUSH     | -0.122 |    18.9 |    23.6 |    25.8 /    20.6 |    26.1 /    28.3 |    25.5 /    34.4 |    20.5 /    35.2 | 0.001
137 | PUSH     | -0.119 |    20.6 |    26.8 |    27.4 /    22.3 |    28.5 /    30.6 |    26.2 /    34.6 |    19.7 /    35.3 | 0.000
138 | PUSH     | -0.116 |    22.3 |    28.0 |    26.2 /    24.2 |    32.0 /    32.6 |    28.1 /    34.7 |    20.5 /    35.3 | 0.000
139 | PUSH     | -0.112 |    24.2 |    26.2 |    25.8 /    26.3 |    35.8 /    33.6 |    29.9 /    34.8 |    22.5 /    35.3 | 0.003
140 | PUSH     | -0.106 |    26.3 |    25.4 |    31.3 /    28.3 |    37.1 /    34.1 |    32.4 /    34.9 |    23.3 /    35.3 | 0.039
141 | PUSH     | -0.101 |    28.3 |    30.7 |    33.9 /    30.6 |    35.9 /    34.4 |    32.0 /    35.0 |    22.7 /    35.4 | 0.040
142 | HOLD     | -0.066 |    30.6 |    33.9 |    34.5 /    32.6 |    35.2 /    34.6 |    32.3 /    35.0 |    22.1 /    35.4 | 0.048
143 | HOLD     | -0.044 |    32.6 |    37.4 |    36.6 /    33.6 |    35.3 /    34.7 |    32.0 /    35.1 |      -- /      -- | 0.040
144 | HOLD     | -0.030 |    33.6 |    36.8 |    37.2 /    34.1 |    35.1 /    34.8 |    31.8 /    35.1 |      -- /      -- | 0.045
145 | HOLD     | -0.023 |    34.1 |    36.4 |    36.8 /    34.4 |    34.5 /    34.9 |    31.3 /    35.2 |      -- /      -- | 0.043
146 | HOLD     | -0.019 |    34.4 |    36.6 |    36.9 /    34.6 |    34.4 /    35.0 |    31.1 /    35.2 |      -- /      -- | 0.049
147 | HOLD     | -0.016 |    34.6 |    36.4 |    36.6 /    34.7 |    34.1 /    35.0 |    31.0 /    35.3 |      -- /      -- | 0.047
148 | HOLD     | -0.015 |    34.7 |    36.5 |    36.7 /    34.8 |    34.1 /    35.1 |    31.0 /    35.3 |      -- /      -- | 0.049
149 | HOLD     | -0.014 |    34.8 |    36.5 |    36.7 /    34.9 |    34.2 /    35.1 |    31.1 /    35.3 |      -- /      -- | 0.057
150 | HOLD     | -0.014 |    34.9 |    36.7 |    36.8 /    35.0 |    34.3 /    35.2 |    31.2 /    35.3 |      -- /      -- | 0.062
151 | HOLD     | -0.014 |    35.0 |    37.0 |    37.1 /    35.0 |    34.5 /    35.2 |    31.4 /    35.4 |      -- /      -- | 0.080
152 | HOLD     | -0.014 |    35.0 |    37.1 |    37.2 /    35.1 |    34.5 /    35.3 |    31.4 /    35.4 |      -- /      -- | 0.080
153 | HOLD     | -0.014 |    35.1 |    37.5 |    37.6 /    35.1 |    34.7 /    35.3 |      -- /      -- |      -- /      -- |   --
154 | HOLD     | -0.014 |    35.1 |    37.6 |    37.8 /    35.2 |    34.8 /    35.3 |      -- /      -- |      -- /      -- |   --
155 | HOLD     | -0.014 |    35.2 |    37.7 |    37.8 /    35.2 |    34.6 /    35.3 |      -- /      -- |      -- /      -- |   --
156 | HOLD     | -0.014 |    35.2 |    38.0 |    38.0 /    35.3 |    34.7 /    35.4 |      -- /      -- |      -- /      -- |   --
157 | HOLD     | -0.014 |    35.3 |    38.2 |    38.2 /    35.3 |    34.9 /    35.4 |      -- /      -- |      -- /      -- |   --
158 | HOLD     | -0.015 |    35.3 |    38.2 |    38.2 /    35.3 |      -- /      -- |      -- /      -- |      -- /      -- |   --
159 | HOLD     | -0.015 |    35.3 |    38.3 |    38.2 /    35.3 |      -- /      -- |      -- /      -- |      -- /      -- |   --
160 | HOLD     | -0.015 |    35.3 |    38.5 |    38.4 /    35.4 |      -- /      -- |      -- /      -- |      -- /      -- |   --
161 | HOLD     | -0.015 |    35.4 |    38.7 |    38.5 /    35.4 |      -- /      -- |      -- /      -- |      -- /      -- |   --

summary: true tilt first > 15 deg at step 134, > 22 at 138, > 30 at 142; max 35.4 deg; contact-free prefix 123 frames
  MAE deg (all rows / true in [15,30) / truth growing): k=0: 0.87 / 3.02 / n/a  k=1: 0.83 / 2.55 / 2.60  k=5: 0.59 / 2.68 / 1.52  k=10: 1.18 / 2.98 / 3.73  k=20: 2.17 / 6.33 / 7.01
  head on real latents: MAE 0.87 deg over all frames, 0.60 deg where the truth <= 30 deg; fallen tail (0 frames at ~90 deg) reads n/a deg at most
  k= 1: pred >= 22 deg first at t=136, >= 30 at t=140, unsafe_prob >= threshold at t=143; lead over the true 30-deg crossing: 6 steps (0.30 s)
  k= 5: pred >= 22 deg first at t=135, >= 30 at t=138, unsafe_prob >= threshold at t=None; lead over the true 30-deg crossing: 7 steps (0.35 s)
  k=10: pred >= 22 deg first at t=129, >= 30 at t=131, unsafe_prob >= threshold at t=130; lead over the true 30-deg crossing: 13 steps (0.65 s)
  k=20: pred >= 22 deg first at t=121, >= 30 at t=123, unsafe_prob >= threshold at t=130; lead over the true 30-deg crossing: 21 steps (1.05 s)
  false-alarm check over the contact-free prefix: max predicted tilt 0.85 deg, max unsafe_prob 0.000
  long open-loop rollout from step 114 (told the whole push in advance), predicted tilt every 5th step: 0 -1 1 5 8 5 6 5 4 4  (true: 0 0 4 11 17 26 34 35 35 35)
wrote <scratch>/wm/summary.json, steps.csv, record.npz, strip.png, tilt_chart.png, rollout.gif  (19.0 s)
DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s ahead); no false alarm over the 123 contact-free frames (max predicted tilt 0.85 deg); head MAE on real latents 0.60 deg; MAE at k=10: 1.18 deg
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:21.96
        Maximum resident set size (kbytes): 1377940
```

Exit code 0. Outputs: `record.npz` 2,064,114 B, `rollout.gif` 1,000,322 B, `steps.csv` 79,291 B, `strip.png` 110,140 B,
`summary.json` 6,019 B, `tilt_chart.png` 14,277 B. `summary.json` = `examples/env_demo_expected_summary.json`, `strip.png`
= `examples/env_demo_strip.png`, `rollout.gif` = `examples/env_demo_rollout.gif`. Interpretation of the numbers: README,
"Running the model in the simulator".

Variants that were also run on the same record (all exit 0):

```text
$ python env/env_demo.py --stage wm ... --encoder-builder transformers   -> z_real and pred_tilt bit-equal to the hydra run; same DEMO OK line
                                                                            (run before the verdict line gained its false-alarm clause)
$ python env/env_demo.py --stage wm ... --head none --no-gif --self-check -> "no tilt head: latents were rolled out but there is nothing to read";
                                                                            DEMO RAN (no tilt head loaded: nothing to compare); self-check: 0 modules outside the package
$ MUJOCO_GL=egl python env/env_demo.py --stage sim --out run2 --compare examples/env_demo_reference.npz
compare with examples/env_demo_reference.npz: steps 162 vs 162, over the common 162 steps max|delta raw_actions| 0.000e+00, |delta tilt| 0.000e+00 deg, |delta pinch| 0.000e+00 m
  phase bounds here {"SETTLE": [0, 20], "APPROACH": [20, 42], "DESCEND": [42, 87], "CLOSE": [87, 94], "LIFT": [94, 114], "PUSH": [114, 142], "HOLD": [142, 162]}
  phase bounds ref  {"SETTLE": [0, 20], "APPROACH": [20, 42], "DESCEND": [42, 87], "CLOSE": [87, 94], "LIFT": [94, 114], "PUSH": [114, 142], "HOLD": [142, 162]}
  reference: mujoco 3.11.0, GL egl; here: mujoco 3.11.0, GL egl
$ MUJOCO_GL=glfw GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  python env/env_demo.py --stage sim --gl glfw --out run_glfw --compare run1/record.npz    -> same 162 steps, max|delta| 0 on actions / tilt / pinch
$ MUJOCO_GL=glfw python env/env_demo.py --stage sim --gl egl --out run3 --compare examples/env_demo_reference.npz
note: MUJOCO_GL=glfw in the environment is replaced by --gl egl                              -> GL egl, max|delta| 0 against the shipped reference
```

Error paths (exit 1, traceback + one FAIL line): `--stage wm` without `--record`; a wrong `--upstream`
(`FileNotFoundError: Missing upstream LeWM file /tmp/jepa.py ...`); a Python without the torch / LeWM stack in
`--stage all` (`FAIL ImportError: the world-model stack is incomplete in this Python (...)`, raised before the
simulation starts, pointing to the two-stage split and `--encoder-builder transformers`); an existing `--out` gives a
single line `FAIL --out ... already exists ...` without a traceback; `--max-steps 50`
(`RuntimeError: phase DESCEND did not finish within 8 steps (50 steps in total, limit 50; ...); the scripted scenario failed
in this build`). Regression of the original demo after the change: `python demo.py --upstream <le-wm>` ->
`PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)`, exit 0.

### 9.3 The numbers of the reference run

`examples/env_demo_expected_summary.json` (also printed above): 162 steps (8.1 s of simulation; the true tilt leaves
0 deg at frame 123 - contact itself is not measured); true worst tilt > 15 deg at frame 134, > 22 at 138, > 30 at 142, max 35.4 deg
(the pillar leans on the held green block instead of falling flat, so there is no 90-deg tail in this run).
MAE in degrees (all rows / rows whose true tilt is in [15, 30) / rows whose truth is growing): k = 0 (head on real
latents) 0.87 / 3.02 / –, k = 1: 0.83 / 2.55 / 2.60, k = 5: 0.59 / 2.68 / 1.52, k = 10: 1.18 / 2.98 / 3.73, k = 20: 2.17 /
6.33 / 7.01; head MAE 0.60 deg on the frames whose truth is <= 30 deg. Alarms: the prediction first reaches 22 deg at
t = 136 (k = 1), 135 (k = 5), 129 (k = 10), 121 (k = 20), i.e. 6 / 7 / 13 / 21 steps (0.30 / 0.35 / 0.65 / 1.05 s) before
the truth crosses 30 deg at frame 142; `unsafe_prob` crosses its threshold 0.9608 at t = 130 for k = 10 and k = 20 (never
for k = 5, only at t = 143 for k = 1). Over the 123 contact-free frames the largest predicted tilt at any horizon is
0.85 deg and the largest unsafe score 0.000. The single long rollout from the first push step (48 steps, all commands
given in advance) predicts at most 11 deg where the truth reaches 35 deg. Verdict: `DEMO OK`.

### 9.4 Checks

* **Determinism.** Two `--stage sim` runs (EGL) and the final run: `raw_actions`, `tilt_deg`, `pinch_xyz`, `green_xyz`,
  `guard_xyz` and the 163 frames all `np.array_equal`; sha256 of `raw_actions.tobytes()` / `tilt_deg.tobytes()` identical
  (values in 9.2). The glfw + d3d12 run reproduces the physics bit for bit; only the pixels differ (below). The final
  WM-stage run is bit-equal to the first one (frames, actions, tilt, `z_real`, `pred_tilt`; identical MAEs and alarms).
* **Frame pipeline (mandatory check).** Four raw 640x480 frames rendered live by the scene (reset, arm moving,
  descending; EGL) were converted with `lewm_s1.letterbox` and, in a scratch script that imports the project's
  cache converter `wm_training/prepare_data.py` (the function `prepare_full.py` used to build `wm_full_001`; NOT the
  package's copy), with `prepare_data.letterbox(Image.fromarray(raw), 224)` and with the collector's PNG path
  (`Image.save(PNG, compress_level=1)` -> `Image.open` -> `prepare_data.letterbox`). Result for every frame: `max|lewm_s1 -
  prepare_data.letterbox| = 0`, `max|lewm_s1 - PNG round trip -> prepare_data.letterbox| = 0`, rows 0..27 and 196..223
  exactly the fill (124, 116, 104), rows 28..195 exactly a direct LANCZOS resize to 224x168; `lewm_s1.letterbox` gives the
  same bytes in both venvs (Pillow 12.3.0 in both). Because the simulation venv lacks `h5py` (imported by
  `prepare_data.py`), the raw frames were rendered in the simulation venv and converted in the training venv.
* **WM stage vs the lewm_s1 primitives.** On the record: `normalize_actions(raw_actions)` == the stored WM actions
  (bit-equal); `z_real` == a fresh `encode(frames, batch_size=16)`; for t in {2, 60, 114, 130, 141, 161}
  `rollout(...)[0]` == `predict_next(...)` (bit-equal) and `tilt_deg(predict_next)` == the table's `pred t+1` column to
  <= 7.6e-6 deg (the head is applied to a batch of K predicted latents in the demo and to one here); the set of filled
  prediction cells is exactly {k <= min(20, T - t), 2 <= t <= T-1} plus k = 0; the k = 0 column == `tilt_deg(z_real)`.
* **GL backends.** EGL vs glfw + d3d12 on the same physics: 224-px frames differ by at most 9 (of 255) on 1.13 % of
  the pixel values (3.1e-6 of them by more than 2); latents max|dz| 3.77e-2 (mean |z| 0.80); head on real latents
  max|d tilt| 0.21 deg, max|d unsafe_prob| 2.5e-3; open-loop predictions max|d pred_tilt| 0.66 deg (k = 10 column 0.21
  deg); the glfw record gives the same alarm steps and the same verdict (MAEs 0.85 / 0.82 / 0.60 / 1.18 / 2.18 deg).
* **Isolation.** `--self-check` (sim stage, and WM stage with `--head none`) found 0 modules imported from the project
  tree outside the package; the sim stage was also run with `python -I` in the prototype.
* **Hygiene.** `env/__pycache__` created by the runs was deleted; no output directory was written inside the package
  (all `--out` directories are in the scratch area); the three verbatim files were `cmp`-identical to the project's
  sources after the runs; `SHA256SUMS` is regenerated by the release process after this section.

### 9.5 Limitations specific to the demo

* Physics determinism holds within one MuJoCo build / CPU (verified: 3 EGL runs + 1 glfw run here). On another
  machine the event-driven phases (arrival, grasp detection, the 30-deg stop) can shift by a few steps and the tilt
  trace differs slightly; the demo reports its own truth and `--compare` quantifies the shift. The numbers in 9.3 are
  ours, not a specification.
* Rendering backends differ at the pixel level (9.4); other GPU drivers (anti-aliasing, dithering) may differ more
  and move the head output by a degree or so. The physics is unaffected.
* The multi-step predictions are evaluated with the commands that were actually executed afterwards (the provenance
  report's "open-loop rollout with the recorded actions"); only the one-step column is online in the strict sense.
* The head is calibrated for 0-30 deg; in this run the truth stops at 35.4 deg and the head reads up to 38.7 deg there;
  a pillar lying flat (90 deg) would be far outside its range.
* Importing the scene pulls torch, torchvision, pygame, pymunk and shapely through stable-worldmodel and prints an
  `ale-py` UserWarning; as two stages the run takes 7 s (simulation) + 22 s (model) on 2 CPU threads, ~1.5 GB RSS.
* **Single-venv run, measured (2026-10-08).** In the fresh venv of section 2 (uv-provisioned CPython 3.10.21 because
  Ubuntu 24.04 has no python3.10; torch 2.14.0+cpu; from-scratch pip install 80 s, 139 packages, 2.5 GB, `pip check`
  clean), from a copy of the package with a fresh clone of the pinned commit:
  `MUJOCO_GL=egl python env/env_demo.py --upstream ./le-wm --out <scratch>` -> exit 0, 23.9 s wall, peak RSS
  1.48 GB (`/usr/bin/time -v`), `DEMO OK` with the same numbers as 9.3 (same phase bounds, MAEs and alarm steps);
  a second run with `--compare examples/env_demo_reference.npz` -> `max|delta raw_actions| 0.000e+00, |delta tilt|
  0.000e+00 deg, |delta pinch| 0.000e+00 m`, `DEMO OK`. The venv was deleted afterwards.

## 10. quickstart.sh (2026-10-13)

`quickstart.sh` (package root) does sections 1, 2, 3, 8 and 9.5 in one command and checks each step; `bash
quickstart.sh --help` prints its header. The run documented below is the one of the final package state: a git
repository was built from this package (`git init && git add -A && git commit`, so `.gitignore` and `.gitattributes`
apply exactly as they do on GitHub: 34 tracked files), cloned to a fresh directory, and `bash quickstart.sh` was run
there twice on this WSL2 Ubuntu 24.04 machine (2 CPU threads, `OMP_NUM_THREADS=2`, NVIDIA RTX 5080 used only by the
EGL renderer, network available, pip cache warm from section 2's install): the first run from nothing (no `.venv`, no
`le-wm`), the second run immediately afterwards to check that it is idempotent. Machine facts of the first run: Ubuntu
24.04 has no `python3.10` on PATH and `uv` is not on PATH either but exists at `~/.local/bin/uv`, so step 1 picked
the uv-provisioned CPython 3.10.21 (`/home/arrri/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10`);
the fresh venv was `/home/arrri/tmp_qs_fix2/pkg/.venv` (2.5 GB; 141 lines of `pip list`, 139 of `pip freeze`;
`pip check` clean; torch 2.14.0+cpu); `le-wm` was cloned from GitHub (12 MB); the GL probe picked `egl` (first
candidate; osmesa / glfw were not needed). Timings, first run: 82 s wall in total (`/usr/bin/time -v`: 1:21.44, peak
RSS 1.75 GB) — packages 48 s (warm pip cache; ~3 min is to be expected with a cold cache), demo.py 6 s, simulator demo
25 s. Second run: 35 s (`/usr/bin/time -v`: 34.99 s; packages 5 s = pip self-upgrade check + pin check, nothing
installed; demo.py 6 s; simulator demo 24 s; output directory `env_demo_out_2` because `env_demo_out` existed). Both
runs exit 0, `PASS` and `DEMO OK` with exactly the numbers of section 9.3. The clone, its venv and `le-wm` were deleted
afterwards.

Why 33 files in step 5 (section 8 says "every file"): the project-internal Chinese comparison report
`provenance/compare_wm_physical_s1_001_REPORT_ZH.md` was removed from the public repository on 2026-10-08 (commit
`e3f4500`), but the `SHA256SUMS` published with it still listed the file, so a clone of that state fails step 5 with
`provenance/compare_wm_physical_s1_001_REPORT_ZH.md: FAILED open or read` (confirmed on 2026-10-13 with
`git clone --depth 1` of GitHub: `sha256sum -c SHA256SUMS` exit 1). The fix follows that removal: the file is listed in
`.gitignore` (it is not part of the release and is never committed again), `SHA256SUMS` was regenerated without it
(33 entries, `LC_ALL=C` order, computed after the last edit of this file), and the three README mentions of the report
were replaced ("project-internal comparison report, not included here"); the numbers quoted from it are unchanged.
Regenerate `SHA256SUMS` after any further edit, from the package root:
`find . -type f ! -name SHA256SUMS ! -path './.git/*' ! -path './.venv/*' ! -path './le-wm/*' ! -path '*/__pycache__/*'
! -path './env_demo_out*' ! -path './provenance/compare_wm_physical_s1_001_REPORT_ZH.md' | sed 's|^\./||' |
LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS` (i.e. every file except the git-ignored ones).

The script itself is unchanged since the checks made earlier the same day on a plain copy of the package (same
machine): `--help` (exit 0); a copy of the script converted to CRLF line endings (`sed 's/$/\r/'`), which is what a
clone made with Git for Windows and `core.autocrlf=true` would produce without the repository's `.gitattributes` — the
script's first executable line reports `FAIL quickstart.sh has CRLF line endings (checked out with Git for Windows /
core.autocrlf=true?) -> clone inside WSL or Linux, or run: git config core.autocrlf false && git checkout -- .   then
re-run` and exits 1 before `set -u` can choke on the `\r`; the `.gitattributes` (`* text=auto eol=lf`, `*.pt *.npz
*.png *.gif binary`) itself, by committing the text files of the package to a local git repository and cloning it with
`git -c core.autocrlf=true clone`: with the `.gitattributes` every text file of the clone has 0 CR bytes and verifies
against its checksums; the control clone of the same repository without it has CRLF in every text file
(`quickstart.sh` 198 CR, `README.md` 297 CR), 12 checksum failures, and the script stops with the CRLF message above.
On the previous revision of the script (steps 0–8 unchanged apart from the hint texts): running it from another
directory (`FAIL ... not in the current directory`, exit 1); `le-wm` moved to an older commit, then `bash quickstart.sh
--skip-install --no-sim` (step 3 printed `le-wm is at c8a44170..., checking out the pinned commit ...`, then `PASS`,
8 s); `le-wm` without its `.git` (step 3 `FAIL ./le-wm exists but is not a git checkout -> rm -rf le-wm and re-run`,
exit 1); `MUJOCO_GL=osmesa` exported on this machine without libOSMesa (step 7 `osmesa: no (AttributeError: 'NoneType'
object has no attribute 'glGetError')`, `FAIL no working MuJoCo rendering backend (your exported MUJOCO_GL=osmesa
fails; unset it to let the script probe)` with the apt hints, exit 1).

Not tested: macOS (the `shasum -a 256` path, bash 3.2 — the script uses no bash-4 construct, and the `gtimeout` /
no-`timeout` branches, which only print a note that a hanging step cannot be detected), `--cuda`, a machine without
EGL (the osmesa / glfw fallbacks were only exercised through their failure path), and the `uv python install 3.10`
branch (3.10 was already provisioned by uv here).

First run, complete output (the per-step rows of the two demos are the ones of sections 3 and 9.2 and are trimmed
with `[...]`; pip's only stderr line was `WARNING: typer 0.27.2 does not provide the extra 'all'`):

```text
$ git clone <this repository> /home/arrri/tmp_qs_fix2/pkg && cd /home/arrri/tmp_qs_fix2/pkg && bash quickstart.sh

== [0] location check
OK /home/arrri/tmp_qs_fix2/pkg

== [1] Python 3.10
OK /home/arrri/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10  (CPython 3.10.21, uv)

== [2] virtual environment ./.venv
OK .venv/bin/python (CPython 3.10.21); the script calls .venv/bin/python directly, no activation needed

== [3] upstream LeWM code ./le-wm: commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
OK commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, fingerprinted files present and unmodified

== [4] packages: torch CPU + requirements.txt + env/requirements-env.txt
   not yet satisfied: hydra-core missing, omegaconf missing, antlr4-python3-runtime missing, einops missing, transformers missing, stable-pretraining missing, [...] lxml missing, dm-tree missing
   pip install -r requirements.txt -r env/requirements-env.txt ...
No broken requirements found.
OK torch 2.14.0+cpu, 141 packages

== [5] checksums: SHA256SUMS
OK 33 files verified with sha256sum

== [6] model demo: .venv/bin/python demo.py --upstream ./le-wm
17:57:53 | INFO  | atomic_chec~| [atomic_save] installed crash-safe checkpoint plugin (write to sibling .tmp + fsync + atomic rename)
17:57:53 | INFO  | utils.py    | Created ViT-tiny from scratch with config: {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3, 'intermediate_size': 768, 'image_size': 224, 'patch_size': 14}
loaded WorldModel(latent_dim=192, history=3, image_size=224, control_dt=0.05, device=cpu, upstream_commit=8edfeb33, head=yes, encoder_builder='hydra') in 4.3 s
  bundle   /home/arrri/tmp_qs_fix2/pkg/model_state.pt  sha256 bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e
  upstream /home/arrri/tmp_qs_fix2/pkg/le-wm  fingerprints OK for commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
  head     /home/arrri/tmp_qs_fix2/pkg/heads/tilt_head.pt  sha256 2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6  outputs ['tilt_deg / 30', 'unsafe_logit']  unsafe threshold 0.9608285427093506
  torch 2.14.0+cpu  threads 2  device cpu  tf32 off
example: cache wm_full_001 TRAIN episode 0, context frames 101..103, 24 future actions
encoded 3 frames + 24-step rollout in 0.13 s; latent shape (24, 192)

context frames (head on REAL latents):
  frame | pred tilt deg | unsafe prob | true tilt deg
    101 |        -0.324 |      0.0000 |         0.000
[...]
rollout (head on PREDICTED latents; threshold 0.9608285427093506):
  step | frame | pred tilt deg | unsafe prob | true tilt deg | true unsafe
     1 |   104 |        -0.176 |      0.0000 |         0.000 | False
[...]
    24 |   127 |         0.502 |      0.0000 |         0.000 | False

reference: torch 2.14.0+cu130, 2 threads, cpu, hydra
max|diff| context latents = 0.000e+00, rollout latents = 0.000e+00 (tolerance 0.0001)
max|diff| predicted tilt deg vs reference = 0.000e+00
PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)
OK PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)

== [7] MuJoCo rendering backend: probing egl, osmesa, glfw
OK MUJOCO_GL=egl

== [8] simulator demo: MUJOCO_GL=egl .venv/bin/python env/env_demo.py --upstream ./le-wm --out env_demo_out
17:58:00 | INFO  | atomic_chec~| [atomic_save] installed crash-safe checkpoint plugin (write to sibling .tmp + fsync + atomic rename)
17:58:00 | INFO  | utils.py    | Created ViT-tiny from scratch with config: {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3, 'intermediate_size': 768, 'image_size': 224, 'patch_size': 14}
loaded WorldModel(latent_dim=192, history=3, image_size=224, control_dt=0.05, device=cpu, upstream_commit=8edfeb33, head=yes, encoder_builder='hydra') in 5.3 s
  bundle   /home/arrri/tmp_qs_fix2/pkg/model_state.pt  sha256 bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e
  head     /home/arrri/tmp_qs_fix2/pkg/heads/tilt_head.pt  sha256 2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6  unsafe threshold 0.9608285427093506
17:58:00 | INFO  | __init__.py | Failed to load library ( 'libOpenGL.so.0' ): libOpenGL.so.0: cannot open shared object file: No such file or directory
17:58:00 | INFO  | __init__.py | No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
17:58:00 | INFO  | __init__.py | Failed to load library ( 'libOpenGL.so.0' ): libOpenGL.so.0: cannot open shared object file: No such file or directory
17:58:00 | INFO  | __init__.py | MUJOCO_GL=egl, attempting to import specified OpenGL backend.
17:58:00 | INFO  | __init__.py | Failed to load library ( 'libOpenGL.so.0' ): libOpenGL.so.0: cannot open shared object file: No such file or directory
17:58:00 | INFO  | __init__.py | MuJoCo library version is: 3.11.0
/home/arrri/tmp_qs_fix2/pkg/.venv/lib/python3.10/site-packages/stable_worldmodel/envs/__init__.py:215: UserWarning: ale-py not found; ALE/* envs are unavailable. Install with: pip install 'stable-worldmodel[env]' or pip install ale-py.
  from stable_worldmodel.envs import ale  # noqa: F401
scene: control_dt=0.05, camera_fovy_deg=48.0, camera_pos_m=[0.70125, 0.0, 0.324], pillar_size_m=[0.04, 0.04, 0.1], pillar_mass_kg=0.19840000000000002, pillar_friction=[1.0, 0.005, 0.0001], shadows=False, unsafe_tilt_deg=30.0, physics_timestep=0.002, mujoco=3.11.0
  SETTLE   steps   0.. 19 ( 20 steps, 1.00 s)  worst tilt   0.0 deg  pinch z 0.157 m  grasped=False
  APPROACH steps  20.. 41 ( 22 steps, 1.10 s)  worst tilt   0.0 deg  pinch z 0.186 m  grasped=False
  DESCEND  steps  42.. 86 ( 45 steps, 2.25 s)  worst tilt   0.0 deg  pinch z 0.057 m  grasped=False
  CLOSE    steps  87.. 93 (  7 steps, 0.35 s)  worst tilt   0.0 deg  pinch z 0.055 m  grasped=True
  LIFT     steps  94..113 ( 20 steps, 1.00 s)  worst tilt   0.0 deg  pinch z 0.101 m  grasped=True
  PUSH     steps 114..141 ( 28 steps, 1.40 s)  worst tilt  30.6 deg  pinch z 0.106 m  grasped=True
  HOLD     steps 142..161 ( 20 steps, 1.00 s)  worst tilt  35.4 deg  pinch z 0.104 m  grasped=True
  sha256 raw_actions c363fd06d37cfe89...  tilt_deg 1afadda1e73c5e1d...
simulation finished in 4.4 s: 162 steps, max true tilt 35.4 deg
  torch 2.14.0+cpu  threads 2  device cpu  tf32 off  GL backend of the frames: egl
encoded 163 frames in 2.6 s
160 rollouts x 20 steps + one 48-step rollout from the first push step in 9.4 s

phases: SETTLE 0..19 (20 steps, 1.00 s)  APPROACH 20..41 (22 steps, 1.10 s)  DESCEND 42..86 (45 steps, 2.25 s)  CLOSE 87..93 (7 steps, 0.35 s)  LIFT 94..113 (20 steps, 1.00 s)  PUSH 114..141 (28 steps, 1.40 s)  HOLD 142..161 (20 steps, 1.00 s)

  t | phase    | cmd dy | true(t) | head(t) | pred t+1  / true  | pred t+5  / true  | pred t+10 / true  | pred t+20 / true  | p_unsafe(t+10)
104 | LIFT     | -0.028 |     0.0 |    -0.0 |    -0.0 /     0.0 |    -0.1 /     0.0 |     0.2 /     0.0 |     0.2 /     2.6 | 0.000
[...]
161 | HOLD     | -0.015 |    35.4 |    38.7 |    38.5 /    35.4 |      -- /      -- |      -- /      -- |      -- /      -- |   --

summary: true tilt first > 15 deg at step 134, > 22 at 138, > 30 at 142; max 35.4 deg; contact-free prefix 123 frames
  MAE deg (all rows / true in [15,30) / truth growing): k=0: 0.87 / 3.02 / n/a  k=1: 0.83 / 2.55 / 2.60  k=5: 0.59 / 2.68 / 1.52  k=10: 1.18 / 2.98 / 3.73  k=20: 2.17 / 6.33 / 7.01
  head on real latents: MAE 0.87 deg over all frames, 0.60 deg where the truth <= 30 deg; fallen tail (0 frames at ~90 deg) reads n/a deg at most
  k= 1: pred >= 22 deg first at t=136, >= 30 at t=140, unsafe_prob >= threshold at t=143; lead over the true 30-deg crossing: 6 steps (0.30 s)
  k= 5: pred >= 22 deg first at t=135, >= 30 at t=138, unsafe_prob >= threshold at t=None; lead over the true 30-deg crossing: 7 steps (0.35 s)
  k=10: pred >= 22 deg first at t=129, >= 30 at t=131, unsafe_prob >= threshold at t=130; lead over the true 30-deg crossing: 13 steps (0.65 s)
  k=20: pred >= 22 deg first at t=121, >= 30 at t=123, unsafe_prob >= threshold at t=130; lead over the true 30-deg crossing: 21 steps (1.05 s)
  false-alarm check over the contact-free prefix: max predicted tilt 0.85 deg, max unsafe_prob 0.000
  long open-loop rollout from step 114 (told the whole push in advance), predicted tilt every 5th step: 0 -1 1 5 8 5 6 5 4 4  (true: 0 0 4 11 17 26 34 35 35 35)
wrote env_demo_out/summary.json, steps.csv, record.npz, strip.png, tilt_chart.png, rollout.gif  (21.8 s)
DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s ahead); no false alarm over the 123 contact-free frames (max predicted tilt 0.85 deg); head MAE on real latents 0.60 deg; MAE at k=10: 1.18 deg
OK DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s ahead); no f ...

== summary (total 82 s)
  location check               OK     0 s  /home/arrri/tmp_qs_fix2/pkg
  Python 3.10                  OK     0 s  /home/arrri/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10  (CPython 3.10.21, uv)
  virtual environment ./.venv  OK     2 s  .venv/bin/python (CPython 3.10.21); the script calls .venv/bin/python directly, no activation needed
  upstream LeWM code ./le-wm   OK     1 s  commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, fingerprinted files present and unmodified
  packages                     OK    48 s  torch 2.14.0+cpu, 141 packages
  checksums                    OK     0 s  33 files verified with sha256sum
  model demo                   OK     6 s  PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)
  MuJoCo rendering backend     OK     0 s  MUJOCO_GL=egl
  simulator demo               OK    25 s  DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s ahead); no f ...
  python: /home/arrri/tmp_qs_fix2/pkg/.venv/bin/python   upstream: /home/arrri/tmp_qs_fix2/pkg/le-wm
  simulator outputs: /home/arrri/tmp_qs_fix2/pkg/env_demo_out  (summary.json, steps.csv, strip.png, tilt_chart.png, rollout.gif, record.npz)
  view rollout.gif (WSL): explorer.exe env_demo_out
ALL STEPS OK
```

Exit code 0. Second run (same directory, immediately afterwards; the demo outputs are identical to the first run and
are omitted):

```text
$ bash quickstart.sh
[...]
== [1] Python 3.10
OK .venv/bin/python  (CPython 3.10.21, existing .venv)
[...]
== [4] packages: torch CPU + requirements.txt + env/requirements-env.txt
   torch 2.14.0+cpu already installed, keeping it
   all pins already installed
No broken requirements found.
OK torch 2.14.0+cpu, 141 packages
[...]
== summary (total 35 s)
  location check               OK     0 s  /home/arrri/tmp_qs_fix2/pkg
  Python 3.10                  OK     0 s  .venv/bin/python  (CPython 3.10.21, existing .venv)
  virtual environment ./.venv  OK     0 s  .venv/bin/python (CPython 3.10.21); the script calls .venv/bin/python directly, no activation needed
  upstream LeWM code ./le-wm   OK     0 s  commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, fingerprinted files present and unmodified
  packages                     OK     5 s  torch 2.14.0+cpu, 141 packages
  checksums                    OK     0 s  33 files verified with sha256sum
  model demo                   OK     6 s  PASS max|diff| ctx=0.000e+00 roll=0.000e+00 (tol 0.0001)
  MuJoCo rendering backend     OK     0 s  MUJOCO_GL=egl
  simulator demo               OK    24 s  DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s ahead); no f ...
  python: /home/arrri/tmp_qs_fix2/pkg/.venv/bin/python   upstream: /home/arrri/tmp_qs_fix2/pkg/le-wm
  simulator outputs: /home/arrri/tmp_qs_fix2/pkg/env_demo_out_2  (summary.json, steps.csv, strip.png, tilt_chart.png, rollout.gif, record.npz)
  view rollout.gif (WSL): explorer.exe env_demo_out_2
ALL STEPS OK
```

Exit code 0.

## 11. Closed-loop task demo (env/task_demo.py) — what was run (2026-10-14)

`env/task_demo.py` (README "Running the model in a closed-loop task") needs the simulator and the model stack in one
Python, which is what `quickstart.sh` builds, so it was verified on a fresh copy of the package with the package's own
quickstart: the package directory was copied (`rsync -a`, excluding `.git`, `__pycache__`, `.venv`, `le-wm` and the
demo output directories) to `/home/arrri/tmp_task_demo/pkg` on this WSL2 Ubuntu 24.04 machine (16 CPU threads, 15 GB
RAM, NVIDIA RTX 5080 used only by the EGL renderer, network available), `bash quickstart.sh` was run there once (all
steps OK, `PASS` and `DEMO OK` as in section 10, 82 s with a warm pip cache), and every run below used that copy's
`.venv/bin/python` with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl` exported and the
demo's default `--threads 4`. `env/rrt_action_law.py` is the verbatim copy of the project's action law (sha256
`513d5e41…`, recorded in `env/ENV_SOURCES.json`; the pilot's own `run.json` lists the same hash for the file it ran);
the demo asserts at start-up that the bundle's action normalisation reproduces the pilot's WM-range env bounds
(`max |diff| 0.0e+00`) and, on the first decision, that `lewm_s1.normalize_actions` equals `rrt_action_law.wm_normalize`
bit for bit on the 33 × 20 nominal actions. The copy, its venv and its outputs were deleted afterwards.

Commands (from the copy's package root):

```bash
rsync -a --exclude .git --exclude __pycache__ --exclude .venv --exclude le-wm --exclude "env_demo_out*" --exclude "task_demo_out*" \
      /mnt/e/Codex/Summer2026/lewm-manip/exports/wm_S1_release_001/ /home/arrri/tmp_task_demo/pkg/
cd /home/arrri/tmp_task_demo/pkg && bash quickstart.sh                                   # ALL STEPS OK (section 10)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
/usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out_run3   # the shipped run
/usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out_run4   # determinism re-run
for d in task_demo_out_run3/*/; do b=$(basename "$d"); cmp task_demo_out_run3/$b/cycles.csv task_demo_out_run4/$b/cycles.csv \
    && cmp task_demo_out_run3/$b/steps.csv task_demo_out_run4/$b/steps.csv && echo "$b: identical"; done
/usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --tasks outer_right_cross,outer_right_in \
    --fallback retreat --gif --out task_demo_out_retreat                                 # the retreat experiment
cp task_demo_out_run3/summary.json examples/task_demo_expected_summary.json
.venv/bin/python env/task_demo.py --upstream ./le-wm --tasks normal_free,outer_right_in --out task_demo_out_cmp \
    --compare examples/task_demo_expected_summary.json                                  # --compare exercised
```

Timings (`/usr/bin/time -v`): run 3 — 1:30.49 wall, 242 s user + 12 s system CPU, peak RSS 2.00 GB; run 4 — 1:26.08
wall, 1.98 GB; the retreat run (two tasks, both modes, `--gif`) — 1:34.85 wall, 2.02 GB; the `--compare` run (two
tasks) — 18.8 s. Inside run 3: the 800-step timeout episode 40 s, the other eleven 3–5 s each; median planning time per
decision 0.27–0.31 s (3 encodes + 33 nominal rollouts + one 33 × 20 rollout + 660 head reads). An earlier revision of
the script (identical decisions; only the stored arguments and the stdout buffering changed afterwards) had been run
twice before (runs 1 and 2); all four runs have bit-identical `cycles.csv` and `steps.csv` in every episode.

Shipped from run 3: `examples/task_demo_expected_summary.json` (its `summary.json`, 34 kB, no machine-specific paths),
`examples/task_demo_strip_safety.png` and `examples/task_demo_strip_nosafety.png` (its two strips, 132 + 136 kB).

Run 3, complete stdout (stderr held only the usual `ale-py` UserWarning of stable-worldmodel and the `/usr/bin/time`
report):

```text
$ /usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out_run3
09:30:13 | INFO  | atomic_chec~| [atomic_save] installed crash-safe checkpoint plugin (write to sibling .tmp + fsync + atomic rename)
09:30:13 | INFO  | utils.py    | Created ViT-tiny from scratch with config: {'hidden_size': 192, 'num_hidden_layers': 12, 'num_attention_heads': 3, 'intermediate_size': 768, 'image_size': 224, 'patch_size': 14}
loaded WorldModel(latent_dim=192, history=3, image_size=224, control_dt=0.05, device=cpu, upstream_commit=8edfeb33, head=yes, encoder_builder='hydra')
  bundle sha256 bd7bc284bade0a971ae1a29a2141c4141e24e6e9bde1ecadf9471be9ba45e98e  head sha256 2ce0e4da4da19420a72077287cb0aa75c556cee54327758748c43d2c8e9c57b6
  torch 2.14.0+cpu  threads 4  device cpu  tf32 off  mujoco 3.11.0  MUJOCO_GL=egl
  action law: 2 cm/s, WM-range env bounds of the pilot reproduced from the bundle's normalisation (max |diff| 0.0e+00)
  planner: {'n_directions': 32, 'direction_std': (1.0, 1.0, 0.5), 'tilt_line_deg': 15.0, 'edge_steps': 8, 'check_hold_steps': 12, 'score_step': 8, 'min_improvement_cm': 0.25, 'tie_cm': 0.05, 'arrival_cycles': 3, 'grip': 1.0}; arrival = pinch within 1.5 cm of goal + grip offset for 3 cycles; toppled > 30 deg; timeout 800 steps; fallback hold

=== normal_free [normal] mode safety: start green (0.425, 0.0, 0.185) -> goal (0.455, 0.025, 0.165) (4.4 cm in free space above the pillars)
  start ready after 170 scripted steps: green [0.423, 0.0006, 0.1811] (requested (0.425, 0.0, 0.185), off by 4.4 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.94 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.1 true now   0.0  dist  4.33 cm (green  4.32)  plan  321 ms
  -> success after 50 steps / 6 cycles; max true tilt 0.0 deg; final distance pinch 0.44 cm, green COM 0.38 cm; reasons {'reduce': 5, 'no_reducing_edge': 1, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 4 s

=== normal_into_slot [normal] mode safety: start green (0.425, 0.0, 0.19) -> goal (0.425, 0.0, 0.155) (3.5 cm straight down into the 1-cm slot; line clear, lateral error touches)
  start ready after 176 scripted steps: green [0.4231, 0.0006, 0.186] (requested (0.425, 0.0, 0.19), off by 4.5 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.94 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.4 true now   0.0  dist  2.99 cm (green  2.93)  plan  320 ms
  -> success after 42 steps / 5 cycles; max true tilt 0.0 deg; final distance pinch 0.55 cm, green COM 0.43 cm; reasons {'reduce': 4, 'no_reducing_edge': 1, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 3 s

=== slot_exit_right [slot] mode safety: start green (0.425, 0.0, 0.125) -> goal (0.445, 0.03, 0.17) (5.0 cm; the straight line passes AGAINST the right pillar)
  start ready after 219 scripted steps: green [0.4227, 0.0007, 0.1206] (requested (0.425, 0.0, 0.125), off by 5.0 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.93 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.5 true now   0.0  dist  6.23 cm (green  6.32)  plan  292 ms
  -> success after 74 steps / 9 cycles; max true tilt 7.4 deg; final distance pinch 0.32 cm, green COM 0.40 cm; reasons {'reduce': 9, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 5 s

=== outer_left_in [outer_left] mode safety: start green (0.425, -0.0975, 0.125) -> goal (0.425, -0.02, 0.17) (8.9 cm; the straight line runs THROUGH the left pillar)
  start ready after 252 scripted steps: green [0.4228, -0.0963, 0.12] (requested (0.425, -0.0975, 0.125), off by 5.6 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.63 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   4.9 true now   0.0  dist  9.09 cm (green  9.17)  plan  315 ms
  -> toppled after 63 steps / 8 cycles; max true tilt 36.5 deg; final distance pinch 3.63 cm, green COM 3.77 cm; reasons {'reduce': 8, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 4 s

=== outer_right_in [outer_right] mode safety: start green (0.425, 0.0975, 0.125) -> goal (0.425, 0.02, 0.17) (8.9 cm; mirror of outer_left_in, through the right pillar)
  start ready after 252 scripted steps: green [0.4224, 0.0977, 0.1198] (requested (0.425, 0.0975, 0.125), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.9 true now   0.0  dist  9.27 cm (green  9.37)  plan  303 ms
  -> toppled after 65 steps / 8 cycles; max true tilt 36.3 deg; final distance pinch 4.72 cm, green COM 4.84 cm; reasons {'reduce': 7, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 1, 'all_rejected_retreat': 0}; 4 s

=== outer_right_cross [outer_right] mode safety: start green (0.425, 0.0975, 0.13) -> goal (0.41, -0.03, 0.16) (13.1 cm across both pillars; through the right pillar)
  start ready after 247 scripted steps: green [0.4224, 0.0977, 0.1248] (requested (0.425, 0.0975, 0.13), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   3.0 true now   0.0  dist 13.31 cm (green 13.35)  plan  295 ms
  cycle  10 step  82  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  26.9 true now  19.5  dist  9.83 cm (green  9.91)  plan  294 ms
  cycle  20 step 162  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  29.1 true now  19.7  dist  9.83 cm (green  9.93)  plan  293 ms
  cycle  30 step 242  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  29.9 true now  19.8  dist  9.83 cm (green  9.95)  plan  303 ms
  cycle  40 step 322  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  30.5 true now  19.8  dist  9.83 cm (green  9.96)  plan  318 ms
  cycle  50 step 402  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  30.4 true now  19.8  dist  9.83 cm (green  9.97)  plan  281 ms
  cycle  60 step 482  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  30.4 true now  19.8  dist  9.83 cm (green  9.97)  plan  284 ms
  cycle  70 step 562  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  30.7 true now  19.8  dist  9.83 cm (green  9.98)  plan  289 ms
  cycle  80 step 642  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  30.9 true now  19.8  dist  9.83 cm (green  9.98)  plan  298 ms
  cycle  90 step 722  all_rejected          rejected 32/32 hold_ok 0  pred tilt chosen  31.2 true now  19.8  dist  9.83 cm (green  9.99)  plan  291 ms
  -> timeout after 800 steps / 100 cycles; max true tilt 21.0 deg; final distance pinch 9.83 cm, green COM 9.99 cm; reasons {'reduce': 5, 'no_reducing_edge': 1, 'hold_rejected_escape': 1, 'all_rejected': 93, 'all_rejected_retreat': 0}; 40 s

=== normal_free [normal] mode nosafety: start green (0.425, 0.0, 0.185) -> goal (0.455, 0.025, 0.165) (4.4 cm in free space above the pillars)
  start ready after 170 scripted steps: green [0.423, 0.0006, 0.1811] (requested (0.425, 0.0, 0.185), off by 4.4 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.94 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.1 true now   0.0  dist  4.33 cm (green  4.32)  plan  317 ms
  -> success after 50 steps / 6 cycles; max true tilt 0.0 deg; final distance pinch 0.44 cm, green COM 0.38 cm; reasons {'reduce': 5, 'no_reducing_edge': 1, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 3 s

=== normal_into_slot [normal] mode nosafety: start green (0.425, 0.0, 0.19) -> goal (0.425, 0.0, 0.155) (3.5 cm straight down into the 1-cm slot; line clear, lateral error touches)
  start ready after 176 scripted steps: green [0.4231, 0.0006, 0.186] (requested (0.425, 0.0, 0.19), off by 4.5 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.94 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.4 true now   0.0  dist  2.99 cm (green  2.93)  plan  279 ms
  -> success after 42 steps / 5 cycles; max true tilt 0.0 deg; final distance pinch 0.55 cm, green COM 0.43 cm; reasons {'reduce': 4, 'no_reducing_edge': 1, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 3 s

=== slot_exit_right [slot] mode nosafety: start green (0.425, 0.0, 0.125) -> goal (0.445, 0.03, 0.17) (5.0 cm; the straight line passes AGAINST the right pillar)
  start ready after 219 scripted steps: green [0.4227, 0.0007, 0.1206] (requested (0.425, 0.0, 0.125), off by 5.0 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.93 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.5 true now   0.0  dist  6.23 cm (green  6.32)  plan  307 ms
  -> success after 74 steps / 9 cycles; max true tilt 7.4 deg; final distance pinch 0.32 cm, green COM 0.40 cm; reasons {'reduce': 9, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 5 s

=== outer_left_in [outer_left] mode nosafety: start green (0.425, -0.0975, 0.125) -> goal (0.425, -0.02, 0.17) (8.9 cm; the straight line runs THROUGH the left pillar)
  start ready after 252 scripted steps: green [0.4228, -0.0963, 0.12] (requested (0.425, -0.0975, 0.125), off by 5.6 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.63 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   4.9 true now   0.0  dist  9.09 cm (green  9.17)  plan  297 ms
  -> toppled after 49 steps / 6 cycles; max true tilt 31.8 deg; final distance pinch 4.69 cm, green COM 4.84 cm; reasons {'reduce': 6, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 4 s

=== outer_right_in [outer_right] mode nosafety: start green (0.425, 0.0975, 0.125) -> goal (0.425, 0.02, 0.17) (8.9 cm; mirror of outer_left_in, through the right pillar)
  start ready after 252 scripted steps: green [0.4224, 0.0977, 0.1198] (requested (0.425, 0.0975, 0.125), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.9 true now   0.0  dist  9.27 cm (green  9.37)  plan  308 ms
  -> toppled after 61 steps / 8 cycles; max true tilt 32.0 deg; final distance pinch 3.79 cm, green COM 3.93 cm; reasons {'reduce': 8, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 4 s

=== outer_right_cross [outer_right] mode nosafety: start green (0.425, 0.0975, 0.13) -> goal (0.41, -0.03, 0.16) (13.1 cm across both pillars; through the right pillar)
  start ready after 247 scripted steps: green [0.4224, 0.0977, 0.1248] (requested (0.425, 0.0975, 0.13), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   3.0 true now   0.0  dist 13.31 cm (green 13.35)  plan  305 ms
  -> toppled after 46 steps / 6 cycles; max true tilt 35.4 deg; final distance pinch 9.07 cm, green COM 9.13 cm; reasons {'reduce': 6, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 3 s

task               | mode     | outcome    | steps | max true tilt | tilt>0 steps | all-rejected cycles
------------------------------------------------------------------------------------------------
normal_free        | safety   | success    |    50 |           0.0 |            0 | 0
normal_into_slot   | safety   | success    |    42 |           0.0 |            0 | 0
slot_exit_right    | safety   | success    |    74 |           7.4 |           42 | 0
outer_left_in      | safety   | toppled    |    63 |          36.5 |           56 | 0
outer_right_in     | safety   | toppled    |    65 |          36.3 |           51 | 1
outer_right_cross  | safety   | timeout    |   800 |          21.0 |          792 | 93
normal_free        | nosafety | success    |    50 |           0.0 |            0 | 0
normal_into_slot   | nosafety | success    |    42 |           0.0 |            0 | 0
slot_exit_right    | nosafety | success    |    74 |           7.4 |           42 | 0
outer_left_in      | nosafety | toppled    |    49 |          31.8 |           42 | 0
outer_right_in     | nosafety | toppled    |    61 |          32.0 |           47 | 0
outer_right_cross  | nosafety | toppled    |    46 |          35.4 |           38 | 0

task               | with safety check              | without (--no-safety)
--------------------------------------------------------------------------------
normal_free        | success (50 steps, 0.0 deg)    | success (50 steps, 0.0 deg)
normal_into_slot   | success (42 steps, 0.0 deg)    | success (42 steps, 0.0 deg)
slot_exit_right    | success (74 steps, 7.4 deg)    | success (74 steps, 7.4 deg)
outer_left_in      | toppled (63 steps, 36.5 deg)   | toppled (49 steps, 31.8 deg)
outer_right_in     | toppled (65 steps, 36.3 deg)   | toppled (61 steps, 32.0 deg)
outer_right_cross  | timeout (800 steps, 21.0 deg)  | toppled (46 steps, 35.4 deg)

with the model's safety check: 3 success / 2 toppled / 1 timeout / 0 lost_grasp; without it: 3 success / 3 toppled / 0 timeout / 0 lost_grasp (over 6 tasks; safety from the model, progress geometric, no RRT tree / goal head)
wrote task_demo_out_run3/summary.json, strip_*.png and one directory per task x mode (86.4 s)
	Command being timed: ".venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out_run3"
	User time (seconds): 242.18
	System time (seconds): 12.45
	Elapsed (wall clock) time (h:mm:ss or m:ss): 1:30.49
	Maximum resident set size (kbytes): 1996844
	Exit status: 0
```

Exit code 0. Run 4 (same command, `--out task_demo_out_run4`) printed the same lines apart from the planning
milliseconds and the per-episode seconds (1:26.08 wall, peak RSS 1 982 016 kB); the determinism check:

```text
normal_free_nosafety: cycles.csv and steps.csv identical
normal_free_safety: cycles.csv and steps.csv identical
normal_into_slot_nosafety: cycles.csv and steps.csv identical
normal_into_slot_safety: cycles.csv and steps.csv identical
outer_left_in_nosafety: cycles.csv and steps.csv identical
outer_left_in_safety: cycles.csv and steps.csv identical
outer_right_cross_nosafety: cycles.csv and steps.csv identical
outer_right_cross_safety: cycles.csv and steps.csv identical
outer_right_in_nosafety: cycles.csv and steps.csv identical
outer_right_in_safety: cycles.csv and steps.csv identical
slot_exit_right_nosafety: cycles.csv and steps.csv identical
slot_exit_right_safety: cycles.csv and steps.csv identical
```

(`sha256sum` of the twelve `cycles.csv` of run 3: `61b26024…` for both `normal_free` episodes, `9446cf5b…` for both
`normal_into_slot` episodes — the two modes took identical decisions there — `268c7414…` / `c6e8377a…` for
`slot_exit_right` safety / nosafety (same decisions, different accepted-candidate counts), `53aefd31…` / `a21d0d70…`
for `outer_left_in`, `2a7b85de…` / `fc6687f5…` for `outer_right_in`, `7c5e4061…` / `ed2c64c1…` for
`outer_right_cross`; runs 1, 2 and 4 reproduce every one of them.)

The `--fallback retreat` experiment (README; stdout, the two INFO lines at the top trimmed):

```text
$ /usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --tasks outer_right_cross,outer_right_in --fallback retreat --gif --out task_demo_out_retreat
[...]
  planner: {'n_directions': 32, 'direction_std': (1.0, 1.0, 0.5), 'tilt_line_deg': 15.0, 'edge_steps': 8, 'check_hold_steps': 12, 'score_step': 8, 'min_improvement_cm': 0.25, 'tie_cm': 0.05, 'arrival_cycles': 3, 'grip': 1.0}; arrival = pinch within 1.5 cm of goal + grip offset for 3 cycles; toppled > 30 deg; timeout 800 steps; fallback retreat

=== outer_right_in [outer_right] mode safety: start green (0.425, 0.0975, 0.125) -> goal (0.425, 0.02, 0.17) (8.9 cm; mirror of outer_left_in, through the right pillar)
  start ready after 252 scripted steps: green [0.4224, 0.0977, 0.1198] (requested (0.425, 0.0975, 0.125), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   0.9 true now   0.0  dist  9.27 cm (green  9.37)  plan  309 ms
  cycle  10 step  82  no_reducing_edge      rejected 16/32 hold_ok 1  pred tilt chosen  13.0 true now  14.5  dist  4.81 cm (green  4.90)  plan  294 ms
  cycle  20 step 162  no_reducing_edge      rejected  9/32 hold_ok 1  pred tilt chosen  13.5 true now  14.5  dist  4.81 cm (green  4.92)  plan  283 ms
  cycle  30 step 242  no_reducing_edge      rejected 14/32 hold_ok 1  pred tilt chosen  13.9 true now  14.5  dist  4.81 cm (green  4.94)  plan  326 ms
  cycle  40 step 322  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   8.2 true now  10.3  dist  5.57 cm (green  5.67)  plan  285 ms
  -> success after 394 steps / 49 cycles; max true tilt 18.0 deg; final distance pinch 0.41 cm, green COM 0.69 cm; reasons {'reduce': 14, 'no_reducing_edge': 32, 'hold_rejected_escape': 2, 'all_rejected': 0, 'all_rejected_retreat': 1}; 25 s

=== outer_right_cross [outer_right] mode safety: start green (0.425, 0.0975, 0.13) -> goal (0.41, -0.03, 0.16) (13.1 cm across both pillars; through the right pillar)
  start ready after 247 scripted steps: green [0.4224, 0.0977, 0.1248] (requested (0.425, 0.0975, 0.13), off by 5.8 mm), grip offset [-0.0002, 0.0004, 0.0083], max pillar tilt during the start 0.000 deg, nearest pillar 4.78 cm
  cycle   0 step   2  reduce                rejected  0/32 hold_ok 1  pred tilt chosen   3.0 true now   0.0  dist 13.31 cm (green 13.35)  plan  317 ms
  cycle  10 step  82  no_reducing_edge      rejected 14/32 hold_ok 1  pred tilt chosen  12.1 true now  15.4  dist  9.93 cm (green  9.98)  plan  284 ms
  cycle  20 step 162  no_reducing_edge      rejected 16/32 hold_ok 1  pred tilt chosen   8.2 true now  16.2  dist 10.18 cm (green 10.20)  plan  304 ms
  cycle  30 step 242  no_reducing_edge      rejected 13/32 hold_ok 1  pred tilt chosen   8.3 true now  16.2  dist 10.17 cm (green 10.19)  plan  287 ms
  cycle  40 step 322  no_reducing_edge      rejected 10/32 hold_ok 1  pred tilt chosen   8.3 true now  16.2  dist 10.17 cm (green 10.19)  plan  312 ms
  cycle  50 step 402  no_reducing_edge      rejected 17/32 hold_ok 1  pred tilt chosen   8.5 true now  16.1  dist 10.17 cm (green 10.19)  plan  308 ms
  cycle  60 step 482  no_reducing_edge      rejected 18/32 hold_ok 1  pred tilt chosen   8.5 true now  16.1  dist 10.17 cm (green 10.19)  plan  313 ms
  cycle  70 step 562  no_reducing_edge      rejected 17/32 hold_ok 1  pred tilt chosen   8.8 true now  16.1  dist 10.17 cm (green 10.19)  plan  294 ms
  cycle  80 step 642  no_reducing_edge      rejected 15/32 hold_ok 1  pred tilt chosen   9.1 true now  16.1  dist 10.17 cm (green 10.19)  plan  276 ms
  cycle  90 step 722  no_reducing_edge      rejected 15/32 hold_ok 1  pred tilt chosen   9.6 true now  16.0  dist 10.17 cm (green 10.20)  plan  288 ms
  -> timeout after 800 steps / 100 cycles; max true tilt 20.5 deg; final distance pinch 10.17 cm, green COM 10.20 cm; reasons {'reduce': 9, 'no_reducing_edge': 84, 'hold_rejected_escape': 3, 'all_rejected': 0, 'all_rejected_retreat': 4}; 49 s

=== outer_right_in [outer_right] mode nosafety: [...]
  -> toppled after 61 steps / 8 cycles; max true tilt 32.0 deg; final distance pinch 3.79 cm, green COM 3.93 cm; reasons {'reduce': 8, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 5 s

=== outer_right_cross [outer_right] mode nosafety: [...]
  -> toppled after 46 steps / 6 cycles; max true tilt 35.4 deg; final distance pinch 9.07 cm, green COM 9.13 cm; reasons {'reduce': 6, 'no_reducing_edge': 0, 'hold_rejected_escape': 0, 'all_rejected': 0, 'all_rejected_retreat': 0}; 4 s

task               | mode     | outcome    | steps | max true tilt | tilt>0 steps | all-rejected cycles
------------------------------------------------------------------------------------------------
outer_right_in     | safety   | success    |   394 |          18.0 |          380 | 1
outer_right_cross  | safety   | timeout    |   800 |          20.5 |          792 | 4
outer_right_in     | nosafety | toppled    |    61 |          32.0 |           47 | 0
outer_right_cross  | nosafety | toppled    |    46 |          35.4 |           38 | 0

task               | with safety check              | without (--no-safety)
--------------------------------------------------------------------------------
outer_right_in     | success (394 steps, 18.0 deg)  | toppled (61 steps, 32.0 deg)
outer_right_cross  | timeout (800 steps, 20.5 deg)  | toppled (46 steps, 35.4 deg)

with the model's safety check: 1 success / 0 toppled / 1 timeout / 0 lost_grasp; without it: 0 success / 2 toppled / 0 timeout / 0 lost_grasp (over 2 tasks; safety from the model, progress geometric, no RRT tree / goal head)
wrote task_demo_out_retreat/summary.json, strip_*.png and one directory per task x mode (92.4 s)
```

Exit code 0 (1:34.85 wall, peak RSS 2 024 352 kB; the two nosafety episodes are the ones of run 3, bit for bit, since
nothing is rejected without the check; `--gif` wrote a 364 kB `rollout.gif` for the 46-step episode and 2.6 MB for the
800-step one, with a `final.png` each). The decisive cycle of `outer_right_in` in its `cycles.csv`: cycle 7 (step 58,
true tilt 18.0°, hold predicted 46.6°, every edge ≥ 15.6°) executed `all_rejected_retreat` and the true tilt over the
8 steps stayed at 17.8°, then 14.5° at the next decision; in run 3 the same cycle held and the pillar went 18.0° → 36.3°.

`--compare` exercised on a two-task run against the shipped summary (stdout tail; the four episodes match):

```text
$ .venv/bin/python env/task_demo.py --upstream ./le-wm --tasks normal_free,outer_right_in --out task_demo_out_cmp --compare examples/task_demo_expected_summary.json
[...]
with the model's safety check: 1 success / 1 toppled / 0 timeout / 0 lost_grasp; without it: 1 success / 1 toppled / 0 timeout / 0 lost_grasp (over 2 tasks; safety from the model, progress geometric, no RRT tree / goal head)
wrote task_demo_out_cmp/summary.json, strip_*.png and one directory per task x mode (18.8 s)

compare with examples/task_demo_expected_summary.json (informational; MuJoCo contact events and borderline 15-deg decisions depend on the build / CPU):
  normal_free        safety  : outcome success vs success, steps 50 vs 50, max tilt 0.0 vs 0.0 deg  SAME
  outer_right_in     safety  : outcome toppled vs toppled, steps 65 vs 65, max tilt 36.3 vs 36.3 deg  SAME
  normal_free        nosafety: outcome success vs success, steps 50 vs 50, max tilt 0.0 vs 0.0 deg  SAME
  outer_right_in     nosafety: outcome toppled vs toppled, steps 61 vs 61, max tilt 32.0 vs 32.0 deg  SAME
```

Not tested: another machine or thread count (the README explains why a borderline 15° decision can flip there), `--device
cuda`, `--encoder-builder transformers` with this script (the loader path is the one demo.py verified), `--seed` other
than 0, and `--fallback retreat` on `outer_left_in` (it has no all-rejected cycle, so the option cannot change it).

### 11.1 SHA256SUMS regenerated, and the whole thing re-run on a second fresh copy (2026-10-14)

The runs above were made with the `SHA256SUMS` of section 10, i.e. before the files of this section were added. A
review of that state found what it implies: `README.md`, `RUNNING.md`, `.gitignore` and `env/ENV_SOURCES.json` had been
edited and `env/task_demo.py`, `env/rrt_action_law.py` and the three `examples/task_demo_*` files were not listed, so a
clone of it fails `quickstart.sh` at step 5 before any demo runs. Confirmed on the package itself and on a second fresh
copy (`/home/arrri/tmp_task_fix1/pkg`, same rsync as above):

```text
$ sha256sum -c SHA256SUMS | grep -v ': OK$'
.gitignore: FAILED
README.md: FAILED
RUNNING.md: FAILED
env/ENV_SOURCES.json: FAILED
sha256sum: WARNING: 4 computed checksums did NOT match
```

Fix: `SHA256SUMS` regenerated with the section-10 command, with `task_demo_out*` added to the exclusions because
`.gitignore` now lists it (the real package contains no such directory; the exclusion only keeps the rule in step with
`.gitignore`):

```bash
find . -type f ! -name SHA256SUMS ! -path './.git/*' ! -path './.venv/*' ! -path './le-wm/*' ! -path '*/__pycache__/*' \
     ! -path './env_demo_out*' ! -path './task_demo_out*' ! -path './provenance/compare_wm_physical_s1_001_REPORT_ZH.md' \
     | sed 's|^\./||' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS
```

Result: 38 entries (33 + the five new files), `sha256sum -c SHA256SUMS` exit 0. With that file the copy's
`bash quickstart.sh` passed again from scratch (fresh venv, fresh `le-wm` clone; 79 s wall with a warm pip cache,
`/usr/bin/time -v` 1:18.50, peak RSS 1.75 GB): `OK 38 files verified with sha256sum`, `PASS max|diff| ctx=0.000e+00
roll=0.000e+00`, `DEMO OK: truth crossed 30 deg at step 142; the k=10 prediction reached 22 deg at step 129 (0.65 s
ahead); …` (the numbers of section 9.3), `ALL STEPS OK`. The shipped `SHA256SUMS` of the package is the same
regeneration run from the package root after the last edit of this file, so its `RUNNING.md` line differs from the copy's
(that is the only line that can); it was then verified with `sha256sum -c` (38 OK) and with a second `bash quickstart.sh`
on the same copy, which by construction cannot be quoted here.

Then the full default task-demo run was repeated on that copy, twice, with `--compare` against the shipped summary
(same exports and `--threads 4` as above):

```bash
cd /home/arrri/tmp_task_fix1/pkg
/usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out --compare examples/task_demo_expected_summary.json
/usr/bin/time -v .venv/bin/python env/task_demo.py --upstream ./le-wm --out task_demo_out_2
for d in task_demo_out/*/; do b=$(basename "$d"); cmp task_demo_out/$b/cycles.csv task_demo_out_2/$b/cycles.csv \
    && cmp task_demo_out/$b/steps.csv task_demo_out_2/$b/steps.csv && echo "$b: identical"; done
```

Both runs exit 0 (1:27.14 wall, peak RSS 2.16 GB; 1:30.75 wall, 2.22 GB), their `cycles.csv` and `steps.csv` are
bit-identical in all twelve episodes, and `outer_right_cross_safety/cycles.csv` has the same sha256 as in runs 1–4
(`7c5e4061aa99…`). The stdout of the first run is that of run 3 above line for line, apart from the timings (`plan`
milliseconds, episode seconds, `84.3 s` in the `wrote` line) and the output directory name; its tables and verdict line
are identical, and `--compare` reports `SAME` for every task × mode:

```text
compare with examples/task_demo_expected_summary.json (informational; MuJoCo contact events and borderline 15-deg decisions depend on the build / CPU):
  normal_free        safety  : outcome success vs success, steps 50 vs 50, max tilt 0.0 vs 0.0 deg  SAME
  normal_into_slot   safety  : outcome success vs success, steps 42 vs 42, max tilt 0.0 vs 0.0 deg  SAME
  slot_exit_right    safety  : outcome success vs success, steps 74 vs 74, max tilt 7.4 vs 7.4 deg  SAME
  outer_left_in      safety  : outcome toppled vs toppled, steps 63 vs 63, max tilt 36.5 vs 36.5 deg  SAME
  outer_right_in     safety  : outcome toppled vs toppled, steps 65 vs 65, max tilt 36.3 vs 36.3 deg  SAME
  outer_right_cross  safety  : outcome timeout vs timeout, steps 800 vs 800, max tilt 21.0 vs 21.0 deg  SAME
  normal_free        nosafety: outcome success vs success, steps 50 vs 50, max tilt 0.0 vs 0.0 deg  SAME
  normal_into_slot   nosafety: outcome success vs success, steps 42 vs 42, max tilt 0.0 vs 0.0 deg  SAME
  slot_exit_right    nosafety: outcome success vs success, steps 74 vs 74, max tilt 7.4 vs 7.4 deg  SAME
  outer_left_in      nosafety: outcome toppled vs toppled, steps 49 vs 49, max tilt 31.8 vs 31.8 deg  SAME
  outer_right_in     nosafety: outcome toppled vs toppled, steps 61 vs 61, max tilt 32.0 vs 32.0 deg  SAME
  outer_right_cross  nosafety: outcome toppled vs toppled, steps 46 vs 46, max tilt 35.4 vs 35.4 deg  SAME
```

A field-by-field diff of the new `summary.json` against the shipped `examples/task_demo_expected_summary.json` shows only
`arguments.out`, `arguments.compare`, the per-episode `episode_s` / `plan_s_median` and `runtime_s` (84.3 vs 86.4 s);
the two strip PNGs are byte-identical to the shipped ones. The shipped example files were therefore left as they are. The
second copy, its venv and its outputs were deleted afterwards.
