"""Build examples/train_episode_window.npz and examples/expected_rollout.npz (project-side tool, NOT needed by users).

Needs the training cache data/wm_full_001 (index.json + ONE train episode) and h5py; run from the project root:

    python exports/wm_S1_release_001/tools/make_example.py --cache data/wm_full_001 --upstream /path/to/le-wm

Rules: only index.json and the smallest-id TRAIN episode are opened (never a test id). Window rule (deterministic):
t = argmax over t in [2, T-24] of the mean |dx, dy, dz| command over steps t .. t+23. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

HORIZON = 24

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--threads", type=int, default=2)
args = parser.parse_args()

package = args.package.resolve()
sys.path.insert(0, str(package))
import lewm_s1  # noqa: E402

torch.set_num_threads(args.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

cache = args.cache.resolve()
index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
train_ids = index["split_ids"]["train"]
episode_id = min(train_ids)
rows = [r for r in index["episodes"]["train"] if r["id"] == episode_id]
if len(rows) != 1 or episode_id not in set(train_ids):
    raise SystemExit(f"Episode {episode_id} is not a selected TRAIN episode")
row = rows[0]
if index["source_action_columns"] != list(lewm_s1.ACTION_COLUMNS) or index["frameskip"] != 1:
    raise SystemExit("Unexpected action/time convention in the cache")
episode_path = (cache / row["path"]).resolve()
if not episode_path.is_relative_to(cache / "episodes"):
    raise SystemExit("Episode path escapes the cache")
episode_sha256 = lewm_s1.sha256_file(episode_path)
print(f"TRAIN episode {episode_id}: {row['path']} mode={row['mode']} num_actions={row['num_actions']} sha256={episode_sha256}")

mean = np.asarray(index["normalization"]["mean"], np.float32)
scale = np.asarray(index["normalization"]["scale"], np.float32)

with h5py.File(episode_path, "r") as f:
    if f.attrs["format"] != "three_blocks_lewm_v1" or not f.attrs["complete"] or int(f.attrs["episode_id"]) != episode_id:
        raise SystemExit("Corrupt / unexpected cache episode")
    if f.attrs["alignment"] != "pixels[t] -- action[t] --> pixels[t+1]":
        raise SystemExit("Unexpected alignment attribute")
    meta = json.loads(f.attrs["metadata_json"])
    pixels_sha256 = str(f.attrs["pixels_sha256"])
    diagnostics = f["diagnostics"]
    raw5 = np.asarray(diagnostics["actions"])                        # [T, 5] float64 normalised commands
    cache4 = np.asarray(f["action"], np.float32)                     # [T, 4] = raw5[:, (0,1,2,4)] as float32
    guard_tilt = np.asarray(diagnostics["guard_tilt_deg"])           # [T+1, 2]
    unsafe = np.asarray(diagnostics["unsafe"], bool)                 # [T+1]
    n_actions = len(raw5)
    if raw5.shape != (n_actions, 5) or cache4.shape != (n_actions, 4) or guard_tilt.shape != (n_actions + 1, 2):
        raise SystemExit("Unexpected episode layout")
    if not np.array_equal(cache4, raw5[:, list(lewm_s1.ACTION_COLUMNS)].astype(np.float32)):
        raise SystemExit("cache action dataset is not the float32 cast of the raw command columns")
    magnitude = np.abs(raw5[:, :3]).mean(1)
    candidates = range(2, n_actions - HORIZON + 1)
    scores = np.array([magnitude[t:t + HORIZON].mean() for t in candidates])
    t = int(list(candidates)[int(np.argmax(scores))])
    print(f"window t={t} (rule: argmax mean |dx,dy,dz| over steps t..t+{HORIZON - 1}; score {scores.max():.4f})")
    frames = np.asarray(f["pixels"][t - 2:t + 1])                    # frames t-2, t-1, t
    if frames.shape != (3, 224, 224, 3) or frames.dtype != np.uint8:
        raise SystemExit("Unexpected frame block")

raw_hist = raw5[t - 2:t].astype(np.float32)
raw_future = raw5[t:t + HORIZON].astype(np.float32)
wm_hist = lewm_s1.normalize_actions(raw_hist, mean, scale)
wm_future = lewm_s1.normalize_actions(raw_future, mean, scale)
ref_hist = ((cache4[t - 2:t] - mean) / scale).astype(np.float32)
ref_future = ((cache4[t:t + HORIZON] - mean) / scale).astype(np.float32)
if not (np.array_equal(wm_hist, ref_hist) and np.array_equal(wm_future, ref_future)):
    raise SystemExit("normalize_actions is not bit-identical to the cache normalisation")
true_tilt = guard_tilt[t + 1:t + 1 + HORIZON].max(1)
context_tilt = guard_tilt[t - 2:t + 1].max(1)
true_unsafe = unsafe[t + 1:t + 1 + HORIZON]

examples = package / "examples"
examples.mkdir(exist_ok=True)
window_path = examples / "train_episode_window.npz"
expected_path = examples / "expected_rollout.npz"
for path in (window_path, expected_path):
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")

window = dict(
    frames=frames, frame_indices=np.array([t - 2, t - 1, t], np.int64),
    raw_actions_hist=raw_hist, raw_actions_future=raw_future, wm_actions_hist=wm_hist, wm_actions_future=wm_future,
    true_tilt_deg=true_tilt.astype(np.float64), true_unsafe=true_unsafe, context_tilt_deg=context_tilt.astype(np.float64),
    episode_id=np.int64(episode_id), t=np.int64(t), horizon=np.int64(HORIZON), control_dt=np.float64(index["control_dt"]),
    cache=np.str_(cache.name), episode_path=np.str_(row["path"]), episode_sha256=np.str_(episode_sha256),
    pixels_sha256=np.str_(pixels_sha256), action_columns=np.array(lewm_s1.ACTION_COLUMNS, np.int64),
    action_order=np.array(meta["action_order"]), action_low=np.asarray(meta["action_low"], np.float64),
    action_high=np.asarray(meta["action_high"], np.float64), fixed_yaw_deg=np.float64(meta["fixed_yaw_deg"]),
    action_units=np.str_(json.dumps(meta.get("action_units"))),
    window_rule=np.str_(f"t = argmax over t in [2, T-{HORIZON}] of mean |dx,dy,dz| of the raw commands over steps t..t+{HORIZON - 1}"),
    episode_mode=np.str_(row["mode"]),
)
np.savez_compressed(window_path, **window)
print(f"wrote {window_path} ({window_path.stat().st_size} bytes)")

# ---- the loader's own CPU float32 rollout = the reference the demo compares against
wm = lewm_s1.load_world_model(package / "model_state.pt", args.upstream, device="cpu", head_path=package / "heads" / "tilt_head.pt")
z_ctx = wm.encode(frames)
seq = wm.action_sequence(wm_hist, wm_future)
z_roll = wm.rollout(z_ctx[None], seq[None])[0]
expected = dict(context_latents=z_ctx.astype(np.float32), rollout_latents=z_roll.astype(np.float32),
                tilt_deg=wm.tilt_deg(z_roll), unsafe_prob=wm.unsafe_prob(z_roll),
                context_tilt_deg_pred=wm.tilt_deg(z_ctx), context_unsafe_prob_pred=wm.unsafe_prob(z_ctx),
                torch_version=np.str_(torch.__version__), num_threads=np.int64(torch.get_num_threads()),
                device=np.str_("cpu"), tf32=np.bool_(False), dtype=np.str_("float32"),
                bundle_sha256=np.str_(wm.bundle_sha256), head_sha256=np.str_(wm.head_info["sha256"]),
                encoder_builder=np.str_(wm.encoder_builder), encode_batch_size=np.int64(3))
np.savez_compressed(expected_path, **expected)
print(f"wrote {expected_path} ({expected_path.stat().st_size} bytes)")
print("predicted tilt deg:", np.round(expected["tilt_deg"], 3).tolist())
print("unsafe prob:", np.round(expected["unsafe_prob"], 4).tolist())
print("context tilt pred:", np.round(expected["context_tilt_deg_pred"], 3).tolist(), "true:", context_tilt.tolist())
print("true tilt deg (frames t+1..t+24):", true_tilt.tolist())
print("|wm action| max:", float(np.abs(seq).max()))
