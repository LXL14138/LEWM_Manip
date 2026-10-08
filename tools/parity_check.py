"""Parity of the standalone lewm_s1 loader against the project's own code paths (project-side tool, NOT for users).

Run from the project root with the training venv (needs wm_training, data/wm_full_001/index.json, runs/...):

    python exports/wm_S1_release_001/tools/parity_check.py --upstream /path/to/le-wm

Project path: wm_training.decoder.load_world + evaluate_random_actions.encode_frames + evaluate_bench_counterfactual
{normalize_actions, action_sequence, predict_rollouts} + tilt_head.load_tilt_head(head_binding) + evaluate_bench_wm.score_tilt.
Standalone path: lewm_s1 (hydra encoder AND the opt-in transformers encoder). Both in ONE process, CPU float32, TF32 off,
the same thread count. Also cross-checks the first 3 rollout steps against upstream JEPA.rollout and compares everything
with examples/expected_rollout.npz. Writes tools/parity_report.json (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--run", type=Path, default=Path("runs/lewm_scratch_S1_b128_c2_001"))
parser.add_argument("--cache", type=Path, default=Path("data/wm_full_001"))
parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

package = args.package.resolve()
out_path = args.out or package / "tools" / "parity_report.json"
if out_path.exists():
    raise SystemExit(f"Refusing to overwrite {out_path}")
repo_root = Path.cwd().resolve()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(package))

torch.set_num_threads(args.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cpu")

from wm_training.decoder import load_world                                              # noqa: E402
from wm_training.evaluate_bench_counterfactual import action_sequence, normalize_actions, predict_rollouts  # noqa: E402
from wm_training.evaluate_bench_wm import score_tilt                                     # noqa: E402
from wm_training.evaluate_random_actions import encode_frames                            # noqa: E402
from wm_training.evaluate_short import rollout_latents                                   # noqa: E402
from wm_training.tilt_head import head_binding, load_tilt_head                           # noqa: E402

t0 = time.time()
world, cfg, index, binding = load_world(args.run, args.cache, args.upstream, device)
print(f"project load_world: {time.time() - t0:.1f} s")
mean32 = np.asarray(index["normalization"]["mean"], np.float32)
scale32 = np.asarray(index["normalization"]["scale"], np.float32)
pixel_mean = torch.tensor(index["pixel_mean"], dtype=torch.float32).view(1, 3, 1, 1)
pixel_std = torch.tensor(index["pixel_std"], dtype=torch.float32).view(1, 3, 1, 1)

with np.load(package / "examples" / "train_episode_window.npz", allow_pickle=False) as ex:
    frames, raw_hist, raw_future = ex["frames"], ex["raw_actions_hist"], ex["raw_actions_future"]
    wm_hist, wm_future = ex["wm_actions_hist"], ex["wm_actions_future"]
horizon = len(raw_future)

# ---- project path
a_hist_p = normalize_actions(raw_hist, mean32, scale32)
a_future_p = normalize_actions(raw_future, mean32, scale32)
seq_p = action_sequence(a_hist_p, a_future_p)
z_ctx_p = encode_frames(world, frames, pixel_mean, pixel_std, device, batch_size=3)
z_roll_p = predict_rollouts(world, z_ctx_p[None], seq_p[None], horizon, device)[0]
head_p, prov_p = load_tilt_head(package / "heads" / "tilt_head.pt", head_binding(binding), "cpu")
tilt_p, unsafe_p = score_tilt(head_p, z_roll_p, device)
ctx_tilt_p, ctx_unsafe_p = score_tilt(head_p, z_ctx_p, device)

# ---- upstream JEPA.rollout cross-check (first 3 steps), as wm_training.evaluate_bench_wm.parity_check does
pix = (torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255)[:, None]
pix = (pix - pixel_mean[None]) / pixel_std[None]
with torch.no_grad():
    ref_roll3 = world.rollout({"pixels": pix[:3, 0][None, None]}, torch.from_numpy(seq_p[:5])[None, None], history_size=3)["predicted_emb"][:, 0, -3:]
    ours3 = rollout_latents(world, torch.from_numpy(z_ctx_p[None, :3]), torch.from_numpy(seq_p[None, :5]), 3)
jepa_vs_project = float((ours3 - ref_roll3).abs().max())

report = {"threads": torch.get_num_threads(), "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
          "machine": platform.machine(), "horizon": int(horizon), "example_t": int(np.load(package / "examples" / "train_episode_window.npz")["t"]),
          "project": {"run": str(args.run), "cache_index_sha256": binding["cache_index_sha256"], "wm_checkpoint_sha256": binding["wm_checkpoint_sha256"],
                      "jepa_rollout_vs_rollout_latents_3steps_max_abs": jepa_vs_project},
          "standalone": {}}

def maxabs(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())

with np.load(package / "examples" / "expected_rollout.npz", allow_pickle=False) as ref:
    ref_ctx, ref_roll, ref_tilt, ref_unsafe = ref["context_latents"], ref["rollout_latents"], ref["tilt_deg"], ref["unsafe_prob"]
report["project_vs_expected_npz"] = {"context_latents": maxabs(z_ctx_p, ref_ctx), "rollout_latents": maxabs(z_roll_p, ref_roll),
                                     "tilt_deg": maxabs(tilt_p, ref_tilt), "unsafe_prob": maxabs(unsafe_p, ref_unsafe)}

import lewm_s1  # noqa: E402

for builder in ("hydra", "transformers"):
    t1 = time.time()
    wm = lewm_s1.load_world_model(package / "model_state.pt", args.upstream, device="cpu",
                                  head_path=package / "heads" / "tilt_head.pt", encoder_builder=builder)
    a_hist_s, a_future_s = wm.normalize_actions(raw_hist), wm.normalize_actions(raw_future)
    seq_s = wm.action_sequence(a_hist_s, a_future_s)
    z_ctx_s = wm.encode(frames)
    z_roll_s = wm.rollout(z_ctx_s[None], seq_s[None])[0]
    z_next_s = wm.predict_next(z_ctx_s[None], seq_s[None, :3])[0]
    tilt_s, unsafe_s = wm.tilt_deg(z_roll_s), wm.unsafe_prob(z_roll_s)
    ctx_tilt_s, ctx_unsafe_s = wm.tilt_deg(z_ctx_s), wm.unsafe_prob(z_ctx_s)
    with torch.no_grad():
        ref3_s = wm.model.rollout({"pixels": pix[:3, 0][None, None]}, torch.from_numpy(seq_s[:5])[None, None], history_size=3)["predicted_emb"][:, 0, -3:]
    entry = {
        "load_seconds": round(time.time() - t1, 2),
        "normalize_actions_max_abs": max(maxabs(a_hist_s, a_hist_p), maxabs(a_future_s, a_future_p)),
        "normalize_actions_bit_equal": bool(np.array_equal(a_hist_s, a_hist_p) and np.array_equal(a_future_s, a_future_p)
                                            and np.array_equal(a_hist_s, wm_hist) and np.array_equal(a_future_s, wm_future)),
        "context_latents_max_abs": maxabs(z_ctx_s, z_ctx_p), "context_latents_bit_equal": bool(np.array_equal(z_ctx_s, z_ctx_p)),
        "rollout_latents_max_abs": maxabs(z_roll_s, z_roll_p), "rollout_latents_bit_equal": bool(np.array_equal(z_roll_s, z_roll_p)),
        "predict_next_vs_rollout_step0_max_abs": maxabs(z_next_s, z_roll_s[0]),
        "tilt_deg_max_abs": maxabs(tilt_s, tilt_p), "unsafe_prob_max_abs": maxabs(unsafe_s, unsafe_p),
        "context_tilt_deg_max_abs": maxabs(ctx_tilt_s, ctx_tilt_p), "context_unsafe_prob_max_abs": maxabs(ctx_unsafe_s, ctx_unsafe_p),
        "jepa_rollout_3steps_vs_standalone_rollout_max_abs": float((ref3_s.numpy() - z_roll_s[:3]).__abs__().max()),
        "vs_expected_npz": {"context_latents": maxabs(z_ctx_s, ref_ctx), "rollout_latents": maxabs(z_roll_s, ref_roll),
                            "tilt_deg": maxabs(tilt_s, ref_tilt), "unsafe_prob": maxabs(unsafe_s, ref_unsafe)},
        "head_unsafe_threshold": wm.head_unsafe_threshold, "project_head_unsafe_threshold": prov_p.get("unsafe_threshold"),
        "tilt_deg_standalone": [round(float(x), 4) for x in tilt_s], "tilt_deg_project": [round(float(x), 4) for x in tilt_p],
        "unsafe_prob_standalone": [round(float(x), 6) for x in unsafe_s],
    }
    report["standalone"][builder] = entry
    print(f"[{builder}] " + json.dumps({k: v for k, v in entry.items() if not isinstance(v, list)}, indent=None))
    del wm

bound = 1e-5
checks = []
for builder, e in report["standalone"].items():
    checks += [e["context_latents_max_abs"] <= bound, e["rollout_latents_max_abs"] <= bound, e["tilt_deg_max_abs"] <= 1e-3,
               e["unsafe_prob_max_abs"] <= 1e-5, e["jepa_rollout_3steps_vs_standalone_rollout_max_abs"] <= bound,
               e["normalize_actions_bit_equal"], e["vs_expected_npz"]["rollout_latents"] <= 1e-4]
checks.append(jepa_vs_project <= bound)
report["passed"] = bool(all(checks))
report["bound_latents"] = bound
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
print("PARITY", "PASS" if report["passed"] else "FAIL")
sys.exit(0 if report["passed"] else 1)
