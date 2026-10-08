#!/usr/bin/env python
"""Smoke test / usage demo of the S1 world-model release.

    python demo.py --upstream /path/to/le-wm [--bundle model_state.pt] [--head heads/tilt_head.pt]
                   [--example examples/train_episode_window.npz] [--expected examples/expected_rollout.npz]
                   [--device cpu] [--threads 2] [--tol 1e-4] [--encoder-builder hydra|transformers]

Loads the bundle through lewm_s1, encodes the 3 context frames of the example window, rolls the 24 recorded
actions out in latent space, prints the tilt head's read-out per step next to the simulator's true tilt, and
compares the latents with examples/expected_rollout.npz (CPU float32 reference). Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True, help="checkout of lucas-maes/le-wm at commit 8edfeb33")
    parser.add_argument("--bundle", type=Path, default=HERE / "model_state.pt")
    parser.add_argument("--head", default=str(HERE / "heads" / "tilt_head.pt"), help="tilt head file; pass '' or none to skip")
    parser.add_argument("--example", type=Path, default=HERE / "examples" / "train_episode_window.npz")
    parser.add_argument("--expected", type=Path, default=HERE / "examples" / "expected_rollout.npz")
    parser.add_argument("--device", default="cpu", help="cpu (the PASS contract) or cuda (numerics may differ)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default min(2, cpu_count))")
    parser.add_argument("--tol", type=float, default=1e-4, help="max |diff| allowed vs expected_rollout.npz")
    parser.add_argument("--encoder-builder", default="hydra", choices=("hydra", "transformers"))
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:  # a clear one-line verdict, the traceback above it
        traceback.print_exc()
        print(f"FAIL {type(error).__name__}: {error}")
        return 1


def run(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(HERE))
    import lewm_s1

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    threads = args.threads if args.threads else min(2, os.cpu_count() or 1)
    torch.set_num_threads(int(threads))
    head_path = None if str(args.head).strip().lower() in ("", "none") else Path(args.head)

    t0 = time.time()
    wm = lewm_s1.load_world_model(args.bundle, args.upstream, device=args.device, head_path=head_path,
                                  encoder_builder=args.encoder_builder)
    print(f"loaded {wm!r} in {time.time() - t0:.1f} s")
    print(f"  bundle   {args.bundle.resolve()}  sha256 {wm.bundle_sha256}")
    print(f"  upstream {wm.upstream_dir}  fingerprints OK for commit {wm.upstream_commit}")
    if wm.head is not None:
        print(f"  head     {wm.head_info['path']}  sha256 {wm.head_info['sha256']}  outputs {wm.head_info['outputs']}  "
              f"unsafe threshold {wm.head_unsafe_threshold}")
    print(f"  torch {torch.__version__}  threads {torch.get_num_threads()}  device {wm.device}  tf32 off")

    with np.load(args.example, allow_pickle=False) as ex:
        frames = ex["frames"]
        wm_hist, wm_future = ex["wm_actions_hist"], ex["wm_actions_future"]
        raw_hist, raw_future = ex["raw_actions_hist"], ex["raw_actions_future"]
        true_tilt, true_unsafe, context_tilt = ex["true_tilt_deg"], ex["true_unsafe"], ex["context_tilt_deg"]
        episode_id, t, horizon = int(ex["episode_id"]), int(ex["t"]), int(ex["horizon"])
        cache = str(ex["cache"])
    print(f"example: cache {cache} TRAIN episode {episode_id}, context frames {t - 2}..{t}, {horizon} future actions")

    # the recorded raw commands must map to the stored WM actions bit for bit
    a_hist = wm.normalize_actions(raw_hist)
    a_future = wm.normalize_actions(raw_future)
    if not (np.array_equal(a_hist, wm_hist) and np.array_equal(a_future, wm_future)):
        raise RuntimeError("normalize_actions disagrees with the WM actions stored in the example")

    t1 = time.time()
    z_ctx = wm.encode(frames)
    z_roll = wm.rollout(z_ctx[None], wm.action_sequence(a_hist, a_future)[None])[0]
    print(f"encoded 3 frames + {horizon}-step rollout in {time.time() - t1:.2f} s; latent shape {z_roll.shape}")

    if wm.head is not None:
        tilt, prob = wm.tilt_deg(z_roll), wm.unsafe_prob(z_roll)
        ctx_tilt, ctx_prob = wm.tilt_deg(z_ctx), wm.unsafe_prob(z_ctx)
        print("\ncontext frames (head on REAL latents):")
        print("  frame | pred tilt deg | unsafe prob | true tilt deg")
        for i in range(3):
            print(f"  {t - 2 + i:5d} | {ctx_tilt[i]:13.3f} | {ctx_prob[i]:11.4f} | {context_tilt[i]:13.3f}")
        print(f"\nrollout (head on PREDICTED latents; threshold {wm.head_unsafe_threshold}):")
        print("  step | frame | pred tilt deg | unsafe prob | true tilt deg | true unsafe")
        for k in range(horizon):
            print(f"  {k + 1:4d} | {t + 1 + k:5d} | {tilt[k]:13.3f} | {prob[k]:11.4f} | {true_tilt[k]:13.3f} | {str(bool(true_unsafe[k])):11s}")
    else:
        print("no tilt head given: skipping the tilt / unsafe columns")

    with np.load(args.expected, allow_pickle=False) as ref:
        ref_ctx, ref_roll = ref["context_latents"], ref["rollout_latents"]
        note = f"reference: torch {ref['torch_version']}, {int(ref['num_threads'])} threads, {ref['device']}, {ref['encoder_builder']}"
        ref_tilt = ref["tilt_deg"] if "tilt_deg" in ref.files else None
    diff_ctx = float(np.abs(z_ctx - ref_ctx).max())
    diff_roll = float(np.abs(z_roll - ref_roll).max())
    print(f"\n{note}")
    print(f"max|diff| context latents = {diff_ctx:.3e}, rollout latents = {diff_roll:.3e} (tolerance {args.tol:g})")
    if wm.head is not None and ref_tilt is not None:
        print(f"max|diff| predicted tilt deg vs reference = {float(np.abs(tilt - ref_tilt).max()):.3e}")
    ok = np.isfinite([diff_ctx, diff_roll]).all() and diff_ctx <= args.tol and diff_roll <= args.tol
    print(f"{'PASS' if ok else 'FAIL'} max|diff| ctx={diff_ctx:.3e} roll={diff_roll:.3e} (tol {args.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
