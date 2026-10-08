#!/usr/bin/env python
"""env_demo.py - watch the S1 world model inside the MuJoCo scene it was trained on.

    MUJOCO_GL=egl python env/env_demo.py --upstream /path/to/le-wm [--out env_demo_out] [--horizons 1,5,10,20] ...

A scripted robot (simulator-truth waypoints; no learned policy, no planner) grasps the green block, lifts it so that
it overlaps the top of the red pillars and pushes the left pillar sideways at --push-speed until the SIMULATOR's true
tilt passes --stop-tilt-deg, then holds. At every control step t >= 2 the world model receives exactly what it
received in training - the last three letterboxed front-camera frames (t-2, t-1, t) and the two commands executed
before frame t - plus the commands the robot executes next, and its tilt read-out k = 1 .. --horizon steps ahead
is compared with the true tilt of frame t+k. Everything (frames, truth, commands) comes from the live simulator.

Two stages, so the halves can run in different environments: SIM (MuJoCo + torch, no LeWM stack; writes record.npz) and WM (torch
only; reads it with --record). --stage all (default) runs both in one process and loads the model BEFORE simulating,
so a missing torch stack fails at once. Exit code 0 when the run completed, 1 on any exception. The verdict line
judges the run, not the model: DEMO OK / DEMO RAN (model missed the event or raised a false alarm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]          # the release package root (lewm_s1/, model_state.pt, ...)
ENV_DIR = PACKAGE / "env"                              # three_block_scene.py + scene_config.py + task_progress.py
PHASE_SPEED_MPS = 0.06                                 # reference speed of the grasp phases (auto_grasp.py default)
PUSH_DISTANCE_M = 0.09                                 # push target = lift waypoint + 9 cm sideways
BAND_DEG = (15.0, 30.0)                                # the provenance report's "[15, 30)" endpoint
ALARM_DEG = 22.0                                       # the project's alarm line
PATH_ARGS = ("upstream", "bundle", "head", "out", "record", "compare")   # the arguments that are paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", type=Path, help="checkout of lucas-maes/le-wm at commit 8edfeb33 (WM stage)")
    p.add_argument("--bundle", type=Path, default=PACKAGE / "model_state.pt")
    p.add_argument("--head", default=str(PACKAGE / "heads" / "tilt_head.pt"), help="tilt head; 'none' to skip")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--encoder-builder", default="hydra", choices=("hydra", "transformers"))
    p.add_argument("--out", type=Path, default=Path("env_demo_out"), help="NEW output directory")
    p.add_argument("--horizon", type=int, default=20, help="longest prediction horizon K (steps of 0.05 s)")
    p.add_argument("--horizons", default="1,5,10,20", help="horizons shown in the table / summary")
    p.add_argument("--stop-tilt-deg", type=float, default=30.0, help="push until the TRUE worst tilt exceeds this")
    p.add_argument("--lift-z", type=float, default=0.105, help="pinch height of the carried block [m]")
    p.add_argument("--push-speed", type=float, default=0.03, help="reference speed of the push [m/s]")
    p.add_argument("--push-dir", default="-y", choices=("-y", "+y"), help="-y pushes guard_left, +y guard_right")
    p.add_argument("--hold-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--gl", default=None, choices=("egl", "glfw", "osmesa"), help="MUJOCO_GL (default egl)")
    p.add_argument("--stage", default="all", choices=("all", "sim", "wm"))
    p.add_argument("--record", type=Path, help="record.npz of a previous --stage sim run (for --stage wm)")
    p.add_argument("--compare", type=Path, help="examples/env_demo_reference.npz: compare this run's physics with it")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--print-all", action="store_true", help="print every step, not only the contact part")
    p.add_argument("--self-check", action="store_true", help="assert that no module outside the package was imported")
    args = p.parse_args()
    if args.out.exists():
        print(f"FAIL --out {args.out} already exists; the demo never overwrites a previous run, pass a new directory")
        return 1
    try:
        return run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"FAIL {type(error).__name__}: {error}")
        return 1


def rel(path):
    """A path as stored in record.npz / summary.json: relative to the package root when it lies inside it, otherwise
    only its last component - no machine-specific directories or user names leave the machine."""
    p = Path(path).resolve()
    return p.relative_to(PACKAGE).as_posix() if p.is_relative_to(PACKAGE) else p.name


def stored_arguments(args):
    """vars(args) and the command line with every path passed through rel()."""
    d = {k: rel(v) if k in PATH_ARGS and v is not None and str(v).lower() not in ("", "none") else v
         for k, v in vars(args).items()}
    d["argv"] = [rel(a) if not a.startswith("-") and ("/" in a or "\\" in a or Path(a).exists()) else a
                 for a in sys.argv[1:]]
    return d


def run(args):
    if args.stage != "wm":                                        # MUJOCO_GL must be set before mujoco is imported
        if args.gl:
            if os.environ.get("MUJOCO_GL", args.gl) != args.gl:
                print(f"note: MUJOCO_GL={os.environ['MUJOCO_GL']} in the environment is replaced by --gl {args.gl}")
            os.environ["MUJOCO_GL"] = args.gl
        else:
            os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    torch.set_num_threads(int(args.threads))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ENV_DIR))                              # the scene imports scene_config by bare name
    sys.path.insert(0, str(PACKAGE))
    import lewm_s1

    horizons = sorted({int(h) for h in args.horizons.split(",")})
    if args.horizon < max(horizons) or min(horizons) < 1:
        raise ValueError("--horizons must lie in [1, --horizon]")
    if args.stage == "wm" and args.record is None:
        raise ValueError("--stage wm needs --record <record.npz of a --stage sim run>")
    t0 = time.time()
    wm = None if args.stage == "sim" else load_model(args, lewm_s1)   # before the simulation: fails fast without torch stack
    args.out.mkdir(parents=True, exist_ok=False)                  # refuse to overwrite a previous run
    if args.stage == "wm":
        with np.load(args.record, allow_pickle=False) as data:
            rec = {k: data[k] for k in data.files}
        print(f"loaded {args.record} ({rec['frames'].shape[0] - 1} steps, GL {rec['gl_backend']})")
    else:
        t1 = time.time()
        rec = simulate(args, lewm_s1.letterbox)
        print(f"simulation finished in {time.time() - t1:.1f} s: {len(rec['raw_actions'])} steps, "
              f"max true tilt {float(rec['tilt_deg'].max()):.1f} deg")
        np.savez_compressed(args.out / "record.npz", **rec)
        # reference.npz = the same record without the frames (~16 kB): what examples/env_demo_reference.npz is
        np.savez_compressed(args.out / "reference.npz", **{k: v for k, v in rec.items() if k != "frames"})
    if args.compare:
        compare(rec, args.compare)
    if args.self_check:
        self_check(args)
    if args.stage == "sim":
        print(f"SIM DONE {len(rec['raw_actions'])} steps, max true tilt {float(rec['tilt_deg'].max()):.1f} deg, "
              f"record {args.out / 'record.npz'}")
        return 0
    print(f"  torch {torch.__version__}  threads {torch.get_num_threads()}  device {wm.device}  tf32 off  "
          f"GL backend of the frames: {rec['gl_backend']}")
    pred = predict(wm, rec, args.horizon)
    summary = report(wm, rec, pred, horizons, args)
    summary["runtime_s"] = round(time.time() - t0, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out / "steps.csv", rec, pred, args.horizon)
    np.savez_compressed(args.out / "record.npz", **rec, **{k: v for k, v in pred.items()})
    draw_outputs(rec, pred, summary, args)
    print(f"wrote {args.out}/summary.json, steps.csv, record.npz, strip.png, tilt_chart.png"
          f"{'' if args.no_gif else ', rollout.gif'}  ({summary['runtime_s']} s)")
    print(summary["verdict"])
    return 0


def load_model(args, lewm_s1):
    """The world model of the WM stage. A package missing from this Python (hydra wraps the ImportError in its own
    exception) is reported with the way out: the two-stage split or the lighter encoder builder."""
    if args.upstream is None:
        raise ValueError("--upstream (checkout of le-wm at commit 8edfeb33) is required for the WM stage")
    head_path = None if str(args.head).strip().lower() in ("", "none") else Path(args.head)
    t1 = time.time()
    try:
        wm = lewm_s1.load_world_model(args.bundle, args.upstream, device=args.device, head_path=head_path,
                                      encoder_builder=args.encoder_builder)
    except Exception as error:
        cause = error
        while cause is not None and not isinstance(cause, ImportError):
            cause = cause.__cause__ or cause.__context__
        if cause is None:
            raise
        raise ImportError(f"the world-model stack is incomplete in this Python ({cause}). Install requirements.txt "
                          "here, or split the demo: --stage sim where MuJoCo is, then --stage wm --record <out>/record.npz "
                          "where torch and the LeWM dependencies are; --encoder-builder transformers avoids the "
                          "stable-pretraining import") from error
    print(f"loaded {wm!r} in {time.time() - t1:.1f} s\n  bundle   {wm.bundle_path}  sha256 {wm.bundle_sha256}")
    if wm.head is not None:
        print(f"  head     {wm.head_info['path']}  sha256 {wm.head_info['sha256']}  unsafe threshold "
              f"{wm.head_unsafe_threshold}")
    return wm


# ---------------------------------------------------------------------------------------------- SIM stage
def simulate(args, letterbox):
    """Scripted grasp -> lift -> push -> hold in the live scene; returns the record (frames, commands, truth)."""
    import numpy as np
    import mujoco
    from three_block_scene import ThreeBlockScene

    env = ThreeBlockScene()
    _, info = env.reset(seed=0)
    dt = float(env.control_timestep())
    if abs(dt - 0.05) > 1e-12:
        raise RuntimeError(f"control step is {dt} s, the model was trained with 0.05 s")
    cam = env.model.camera("front_pixels")
    pillar = env.model.geom("object_1")
    scene = {"control_dt": dt, "camera_fovy_deg": float(cam.fovy[0]), "camera_pos_m": cam.pos.tolist(),
             "pillar_size_m": (2 * pillar.size).tolist(), "pillar_mass_kg": float(env.model.body("object_1").mass[0]),
             "pillar_friction": pillar.friction.tolist(), "shadows": bool(env.shadows_enabled),
             "unsafe_tilt_deg": float(env.unsafe_tilt_deg), "physics_timestep": float(env.model.opt.timestep),
             "mujoco": mujoco.__version__}
    print("scene: " + ", ".join(f"{k}={v}" for k, v in scene.items()))

    def pinch():
        return env.data.site_xpos[env._pinch_site_id].copy()

    def truth(info):
        guards = info["safety"]["guards"]
        return ([g["tilt_deg"] for g in guards], bool(info["safety"]["unsafe"]), pinch(),
                env.data.body("object_0").xipos.copy(),
                np.stack([env.data.body("object_1").xipos, env.data.body("object_2").xipos]).copy(),
                bool(info["task"]["grasped"]))

    raw = env.render(camera="front_pixels")
    if raw.shape != (480, 640, 3) or raw.dtype != np.uint8:
        raise RuntimeError(f"unexpected camera image {raw.shape} {raw.dtype}; the model needs the 640x480 front camera")
    frames, rows, actions, phases = [letterbox(raw)], [truth(info)], [], []
    ref = pinch()
    green0 = env.data.body("object_0").xipos.copy()
    above = np.array([green0[0], green0[1], 0.19])
    grasp = np.array([green0[0], green0[1], green0[2] + 0.005])
    lift = np.array([green0[0], green0[1], args.lift_z])
    push = lift + np.array([0.0, PUSH_DISTANCE_M * (-1 if args.push_dir == "-y" else 1), 0.0])
    low, high = env._workspace_bounds
    for name, wp in (("APPROACH", above), ("DESCEND", grasp), ("LIFT", lift), ("PUSH", push)):
        if np.any(wp < low) or np.any(wp > high):
            raise ValueError(f"{name} waypoint {wp} outside the workspace {low}..{high}")
    state = {"ref": ref, "streak": 0}

    def arrived(wp):
        return np.linalg.norm(pinch() - wp) <= 0.005 and np.linalg.norm(state["ref"] - wp) <= 1e-6

    def worst():
        return max(rows[-1][0])

    # phase: (name, waypoint, gripper closed, reference speed, done(n_steps_in_phase, info), timeout steps)
    plan = [("SETTLE", ref.copy(), False, PHASE_SPEED_MPS, lambda n, i: n >= 20, 20),
            ("APPROACH", above, False, PHASE_SPEED_MPS, lambda n, i: arrived(above), 120),
            ("DESCEND", grasp, False, PHASE_SPEED_MPS, lambda n, i: arrived(grasp), 120),
            ("CLOSE", grasp, True, PHASE_SPEED_MPS, lambda n, i: state["streak"] >= 6, 60),
            ("LIFT", lift, True, PHASE_SPEED_MPS, lambda n, i: arrived(lift), 120),
            ("PUSH", push, True, args.push_speed, lambda n, i: worst() > args.stop_tilt_deg, 100),
            ("HOLD", None, True, args.push_speed, lambda n, i: n >= args.hold_steps, args.hold_steps)]
    bounds = {}
    for name, target, closed, speed, done, timeout in plan:
        target = state["ref"].copy() if target is None else target      # HOLD: reference frozen
        start, n = len(actions), 0
        while True:
            # waypoint controller as in the project's scripted collectors: the reference advances at most
            # speed*dt per step and leads the actual pinch by at most 2 cm; the command is reference - actual
            actual = pinch()
            delta = target - state["ref"]
            dist = float(np.linalg.norm(delta))
            if dist > speed * dt:
                delta *= speed * dt / dist
            cand = state["ref"] + delta
            lead = cand - actual
            if np.linalg.norm(lead) > 0.02:
                cand = actual + lead * (0.02 / np.linalg.norm(lead))
            state["ref"] = np.clip(cand, low, high)
            a = env.normalize_action(np.array([*(state["ref"] - actual), 0.0, 1.0 if closed else -1.0]))
            if a.shape != (5,) or np.any(np.abs(a) > 1) or a[3] != 0:
                raise RuntimeError(f"illegal command {a}")
            time_before = float(env.data.time)
            _, _, _, _, info = env.step(a)
            if abs(float(env.data.time) - time_before - dt) > 1e-9:
                raise RuntimeError("the simulator did not advance by one control step")
            actions.append(np.asarray(a, np.float32))
            phases.append(name)
            frames.append(letterbox(env.render(camera="front_pixels")))
            rows.append(truth(info))
            state["streak"] = state["streak"] + 1 if info["task"]["grasped"] else 0
            n += 1
            if done(n, info):
                break
            if n >= timeout or len(actions) >= args.max_steps:
                raise RuntimeError(f"phase {name} did not finish within {n} steps ({len(actions)} steps in total, limit "
                                   f"{args.max_steps}; worst true tilt {worst():.1f} deg, grasped={info['task']['grasped']}); "
                                   "the scripted scenario failed in this build")
        bounds[name] = [start, len(actions)]
        print(f"  {name:8s} steps {start:3d}..{len(actions) - 1:3d} ({n:3d} steps, {n * dt:4.2f} s)  worst tilt "
              f"{worst():5.1f} deg  pinch z {pinch()[2]:.3f} m  grasped={info['task']['grasped']}")
    env.close()
    raw_actions = np.stack(actions).astype(np.float32)
    tilt = np.array([r[0] for r in rows], np.float64)
    rec = {"frames": np.stack(frames), "raw_actions": raw_actions, "tilt_deg": tilt,
           "unsafe": np.array([r[1] for r in rows]), "pinch_xyz": np.stack([r[2] for r in rows]),
           "green_xyz": np.stack([r[3] for r in rows]), "guard_xyz": np.stack([r[4] for r in rows]),
           "grasped": np.array([r[5] for r in rows]), "phase": np.array(phases),
           "phase_bounds": json.dumps(bounds), "control_dt": dt, "scene": json.dumps(scene),
           "gl_backend": os.environ.get("MUJOCO_GL", ""), "mujoco_version": mujoco.__version__,
           "numpy_version": np.__version__, "arguments": json.dumps(stored_arguments(args), default=str),
           "sha256_raw_actions": hashlib.sha256(raw_actions.tobytes()).hexdigest(),
           "sha256_tilt_deg": hashlib.sha256(tilt.tobytes()).hexdigest()}
    print(f"  sha256 raw_actions {rec['sha256_raw_actions'][:16]}...  tilt_deg {rec['sha256_tilt_deg'][:16]}...")
    return rec


def compare(rec, path):
    """Physics of this run vs a shipped reference run (expected 0 on the machine that produced the reference)."""
    import numpy as np
    with np.load(path, allow_pickle=False) as ref:
        n = min(len(rec["raw_actions"]), len(ref["raw_actions"]))
        d_act = float(np.abs(rec["raw_actions"][:n] - ref["raw_actions"][:n]).max())
        d_tilt = float(np.abs(rec["tilt_deg"][:n + 1] - ref["tilt_deg"][:n + 1]).max())
        d_pinch = float(np.abs(rec["pinch_xyz"][:n + 1] - ref["pinch_xyz"][:n + 1]).max())
        print(f"compare with {path}: steps {len(rec['raw_actions'])} vs {len(ref['raw_actions'])}, over the common "
              f"{n} steps max|delta raw_actions| {d_act:.3e}, |delta tilt| {d_tilt:.3e} deg, |delta pinch| {d_pinch:.3e} m")
        print(f"  phase bounds here {rec['phase_bounds']}\n  phase bounds ref  {str(ref['phase_bounds'])}")
        print(f"  reference: mujoco {ref['mujoco_version']}, GL {ref['gl_backend']}; here: mujoco "
              f"{rec['mujoco_version']}, GL {rec['gl_backend']}")


def self_check(args):
    """Developer check: nothing from the directory tree above the package (the training project) was imported."""
    root = PACKAGE.parents[1]
    skip = tuple(str(Path(s).resolve()) for s in [sys.prefix, sys.base_prefix] + ([args.upstream] if args.upstream else []))
    bad = sorted(m for m, mod in list(sys.modules.items())
                 if (f := getattr(mod, "__file__", None)) and str(Path(f).resolve()).startswith(str(root))
                 and not str(Path(f).resolve()).startswith(str(PACKAGE)) and "site-packages" not in f
                 and not str(Path(f).resolve()).startswith(skip))
    print(f"self-check: {len(bad)} modules imported from {root} outside the package: {bad}")
    if bad:
        raise RuntimeError("the demo imported modules from outside the package")


# ---------------------------------------------------------------------------------------------- WM stage
def predict(wm, rec, horizon):
    """Head on real latents (k = 0) and open-loop predictions k = 1..K from every step t >= 2 (NaN elsewhere)."""
    import numpy as np
    frames, raw = rec["frames"], rec["raw_actions"]
    T = len(raw)
    a_wm = wm.normalize_actions(raw)                              # [T, 4], the training cache's rule
    t1 = time.time()
    z_real = wm.encode(frames, batch_size=16)                     # [T+1, 192]
    print(f"encoded {T + 1} frames in {time.time() - t1:.1f} s")
    has_head = wm.head is not None
    out = {"wm_actions": a_wm, "z_real": z_real,
           "pred_tilt": np.full((T + 1, horizon + 1), np.nan, np.float32),
           "pred_unsafe": np.full((T + 1, horizon + 1), np.nan, np.float32)}
    if has_head:
        out["pred_tilt"][:, 0], out["pred_unsafe"][:, 0] = wm.tilt_deg(z_real), wm.unsafe_prob(z_real)
    t1 = time.time()
    for t in range(2, T):                                         # context frames t-2..t, history a[t-2], a[t-1]
        K = min(horizon, T - t)
        z_hat = wm.rollout(z_real[t - 2:t + 1][None], wm.action_sequence(a_wm[t - 2:t], a_wm[t:t + K])[None])[0]
        if has_head:                                              # column k <-> frame t+k
            out["pred_tilt"][t, 1:K + 1], out["pred_unsafe"][t, 1:K + 1] = wm.tilt_deg(z_hat), wm.unsafe_prob(z_hat)
    push0 = json.loads(str(rec["phase_bounds"]))["PUSH"][0]
    z_long = wm.rollout(z_real[push0 - 2:push0 + 1][None], wm.action_sequence(a_wm[push0 - 2:push0], a_wm[push0:])[None])[0]
    out["long_rollout_start"] = push0
    out["long_rollout_tilt"] = wm.tilt_deg(z_long) if has_head else np.full(len(z_long), np.nan, np.float32)
    print(f"{T - 2} rollouts x {horizon} steps + one {T - push0}-step rollout from the first push step in "
          f"{time.time() - t1:.1f} s")
    return out


def report(wm, rec, pred, horizons, args):
    """Print the per-step table and the summary; return the summary dict."""
    import numpy as np
    T, dt = len(rec["raw_actions"]), float(rec["control_dt"])
    worst = rec["tilt_deg"].max(axis=1)
    bounds = json.loads(str(rec["phase_bounds"]))
    push0 = bounds["PUSH"][0]
    P, U = pred["pred_tilt"], pred["pred_unsafe"]
    thr = wm.head_unsafe_threshold
    print("\nphases: " + "  ".join(f"{n} {b[0]}..{b[1] - 1} ({b[1] - b[0]} steps, {(b[1] - b[0]) * dt:.2f} s)"
                                  for n, b in bounds.items()))
    if wm.head is None:
        print("no tilt head: latents were rolled out but there is nothing to read; pass --head to see tilt columns")
    else:
        head = " | ".join(f"pred t+{k:<2d} / true " for k in horizons)
        ku = min(10, args.horizon)                                # horizon of the unsafe-score column
        print(f"\n  t | phase    | cmd dy | true(t) | head(t) | {head} | p_unsafe(t+{ku})")
        rows = range(2, T) if args.print_all else range(max(2, push0 - 10), T)
        for t in rows:
            cells = " | ".join(f"{P[t, k]:7.1f} / {worst[t + k]:7.1f}" if t + k <= T else "     -- /      --"
                               for k in horizons)
            flag = "*" if t + ku <= T and U[t, ku] >= thr else " "
            pu = f"{U[t, ku]:.3f}{flag}" if t + ku <= T else "  --  "
            print(f"{t:3d} | {rec['phase'][t]:8s} | {rec['raw_actions'][t, 1]:+6.3f} | {worst[t]:7.1f} | "
                  f"{P[t, 0]:7.1f} | {cells} | {pu}")

    def first(cond):
        idx = np.flatnonzero(cond)
        return int(idx[0]) if len(idx) else None

    s = {"steps": T, "control_dt": dt, "phase_bounds": bounds, "max_true_tilt_deg": float(worst.max()),
         "true_first_step_over": {str(th): first(worst > th) for th in (BAND_DEG[0], ALARM_DEG, args.stop_tilt_deg)},
         "contact_free_prefix_frames": int(first(worst > 0) or 0), "horizons": horizons,
         "head_unsafe_threshold": thr, "gl_backend": str(rec["gl_backend"]), "scene": json.loads(str(rec["scene"])),
         "mae_deg": {}, "mae_deg_true_in_band_15_30": {}, "mae_deg_truth_growing": {}, "alarms": {},
         "versions": {"torch": __import__("torch").__version__, "numpy": np.__version__,
                      "mujoco": str(rec["mujoco_version"])},
         "arguments": stored_arguments(args),
         "bundle_sha256": wm.bundle_sha256, "head_sha256": wm.head_info["sha256"] if wm.head else None,
         "sim_arguments": json.loads(str(rec["arguments"])), "sha256_raw_actions": str(rec["sha256_raw_actions"]),
         "sha256_tilt_deg": str(rec["sha256_tilt_deg"]), "long_rollout_from_step": int(pred["long_rollout_start"])}
    if wm.head is None:
        s["verdict"] = "DEMO RAN (no tilt head loaded: nothing to compare)"
        return s
    t30 = s["true_first_step_over"][str(args.stop_tilt_deg)]
    prefix = s["contact_free_prefix_frames"]
    for k in [0] + horizons:
        t_idx = np.arange(0 if k == 0 else 2, T + 1 - k)
        err = np.abs(P[t_idx, k] - worst[t_idx + k])
        band = (worst[t_idx + k] >= BAND_DEG[0]) & (worst[t_idx + k] < BAND_DEG[1])
        growing = worst[t_idx + k] > worst[t_idx]
        s["mae_deg"][str(k)] = float(np.nanmean(err))
        s["mae_deg_true_in_band_15_30"][str(k)] = float(np.nanmean(err[band])) if band.any() else None
        s["mae_deg_truth_growing"][str(k)] = float(np.nanmean(err[growing])) if growing.any() else None
        if k:
            t22, t30p = first(P[t_idx, k] >= ALARM_DEG), first(P[t_idx, k] >= args.stop_tilt_deg)
            tpu = first(U[t_idx, k] >= thr)
            s["alarms"][str(k)] = {
                "first_t_pred_ge_22": None if t22 is None else int(t_idx[t22]),
                "first_t_pred_ge_stop": None if t30p is None else int(t_idx[t30p]),
                "first_t_unsafe_prob_ge_threshold": None if tpu is None else int(t_idx[tpu]),
                "lead_steps_22_vs_true_stop": None if t22 is None or t30 is None else int(t30 - t_idx[t22])}
            s["alarms"][str(k)]["lead_seconds"] = (None if s["alarms"][str(k)]["lead_steps_22_vs_true_stop"] is None
                                                   else s["alarms"][str(k)]["lead_steps_22_vs_true_stop"] * dt)
    s["k0_mae_true_le_30"] = float(np.nanmean(np.abs(P[:, 0] - worst)[worst <= 30]))
    pre = [P[t, k] for t in range(T + 1) for k in range(args.horizon + 1) if t + k < prefix and not np.isnan(P[t, k])]
    pre_u = [U[t, k] for t in range(T + 1) for k in range(args.horizon + 1) if t + k < prefix and not np.isnan(U[t, k])]
    s["false_alarm_check"] = {"contact_free_frames": prefix, "max_pred_tilt_deg": float(max(pre)) if pre else None,
                              "max_unsafe_prob": float(max(pre_u)) if pre_u else None}
    fallen = worst >= 85
    s["fallen_tail"] = {"frames": int(fallen.sum()), "max_head_on_real_latents_deg": float(P[fallen, 0].max()) if fallen.any() else None,
                        "note": "the head is calibrated for 0-30 deg; a pillar lying flat (90 deg) is outside its range"}
    s["long_rollout_tilt_deg"] = [round(float(v), 2) for v in pred["long_rollout_tilt"]]
    s["long_rollout_true_tilt_deg"] = [round(float(v), 2) for v in worst[push0 + 1:]]
    nan_rows = int(np.isnan(P[2:T, 1]).sum())
    print(f"\nsummary: true tilt first > 15 deg at step {s['true_first_step_over']['15.0']}, > 22 at "
          f"{s['true_first_step_over']['22.0']}, > {args.stop_tilt_deg:g} at {t30}; max {worst.max():.1f} deg; "
          f"contact-free prefix {prefix} frames")
    print("  MAE deg (all rows / true in [15,30) / truth growing): " + "  ".join(
        f"k={k}: {s['mae_deg'][str(k)]:.2f} / {fmt(s['mae_deg_true_in_band_15_30'][str(k)])} / "
        f"{fmt(s['mae_deg_truth_growing'][str(k)])}" for k in [0] + horizons))
    print(f"  head on real latents: MAE {s['mae_deg']['0']:.2f} deg over all frames, {s['k0_mae_true_le_30']:.2f} deg "
          f"where the truth <= 30 deg; fallen tail ({s['fallen_tail']['frames']} frames at ~90 deg) reads "
          f"{fmt(s['fallen_tail']['max_head_on_real_latents_deg'])} deg at most")
    for k in horizons:
        a = s["alarms"][str(k)]
        print(f"  k={k:2d}: pred >= 22 deg first at t={a['first_t_pred_ge_22']}, >= {args.stop_tilt_deg:g} at "
              f"t={a['first_t_pred_ge_stop']}, unsafe_prob >= threshold at t={a['first_t_unsafe_prob_ge_threshold']}; "
              f"lead over the true {args.stop_tilt_deg:g}-deg crossing: {a['lead_steps_22_vs_true_stop']} steps "
              f"({fmt(a['lead_seconds'])} s)")
    print(f"  false-alarm check over the contact-free prefix: max predicted tilt "
          f"{fmt(s['false_alarm_check']['max_pred_tilt_deg'])} deg, max unsafe_prob {fmt(s['false_alarm_check']['max_unsafe_prob'], 3)}")
    print(f"  long open-loop rollout from step {push0} (told the whole push in advance), predicted tilt every 5th step: "
          + " ".join(f"{v:.0f}" for v in s["long_rollout_tilt_deg"][::5]) + "  (true: "
          + " ".join(f"{v:.0f}" for v in s["long_rollout_true_tilt_deg"][::5]) + ")")
    k_alarm = str(10 if 10 in horizons else horizons[-1])
    t_alarm = s["alarms"][k_alarm]["first_t_pred_ge_22"]
    fa = s["false_alarm_check"]["max_pred_tilt_deg"]
    quiet = fa is None or fa < ALARM_DEG                          # no alarm before the block touches a pillar
    ok = nan_rows == 0 and worst.max() > args.stop_tilt_deg and s["k0_mae_true_le_30"] < 5 and t30 is not None \
        and t_alarm is not None and t_alarm <= t30 and quiet
    s["verdict"] = (f"DEMO OK: truth crossed {args.stop_tilt_deg:g} deg at step {t30}; the k={k_alarm} prediction reached "
                    f"22 deg at step {t_alarm} ({fmt(s['alarms'][k_alarm]['lead_seconds'])} s ahead); no false alarm over "
                    f"the {prefix} contact-free frames (max predicted tilt {fmt(fa)} deg); head MAE on real latents "
                    f"{s['k0_mae_true_le_30']:.2f} deg; MAE at k={k_alarm}: {s['mae_deg'][k_alarm]:.2f} deg"
                    if ok else
                    f"DEMO RAN (model missed the event or raised a false alarm): truth max {worst.max():.1f} deg crossed "
                    f"{args.stop_tilt_deg:g} at step {t30}; k={k_alarm} prediction first >= 22 deg at t={t_alarm}; max "
                    f"predicted tilt over the {prefix} contact-free frames {fmt(fa)} deg; head MAE "
                    f"{s['k0_mae_true_le_30']:.2f} deg; NaN rows {nan_rows}")
    return s


def fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def write_csv(path, rec, pred, horizon):
    import numpy as np
    T = len(rec["raw_actions"])
    cols = ["t", "phase", "cmd_dx", "cmd_dy", "cmd_dz", "cmd_grip", "true_left_deg", "true_right_deg", "true_worst_deg",
            "unsafe", "grasped", "pinch_z_m", "head_tilt_deg", "head_unsafe_prob"] + \
           [f"pred_tilt_k{k}" for k in range(1, horizon + 1)] + [f"pred_unsafe_k{k}" for k in range(1, horizon + 1)]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for t in range(T + 1):
            a = rec["raw_actions"][t] if t < T else np.full(5, np.nan)
            vals = [t, rec["phase"][min(t, T - 1)], a[0], a[1], a[2], a[4], *rec["tilt_deg"][t], rec["tilt_deg"][t].max(),
                    int(rec["unsafe"][t]), int(rec["grasped"][t]), rec["pinch_xyz"][t, 2], pred["pred_tilt"][t, 0],
                    pred["pred_unsafe"][t, 0], *pred["pred_tilt"][t, 1:], *pred["pred_unsafe"][t, 1:]]
            f.write(",".join(v if isinstance(v, str) else f"{v:.6g}" if isinstance(v, float) or hasattr(v, "dtype")
                             else str(v) for v in vals) + "\n")


# ---------------------------------------------------------------------------------------------- pictures (PIL only)
def draw_outputs(rec, pred, summary, args):
    import numpy as np
    from PIL import Image, ImageDraw
    T = len(rec["raw_actions"])
    worst, P = rec["tilt_deg"].max(axis=1), pred["pred_tilt"]
    push0 = summary["phase_bounds"]["PUSH"][0]
    k = 10 if 10 in summary["horizons"] else summary["horizons"][-1]

    def card(t, label=None):
        im = Image.new("RGB", (224, 254), (30, 30, 30))
        im.paste(Image.fromarray(rec["frames"][t]), (0, 0))
        d = ImageDraw.Draw(im)
        src = t - k
        pred_txt = f"pred@{src}: {P[src, k]:.1f} deg" if src >= 2 and not np.isnan(P[src, k]) else "pred: (no history yet)"
        d.text((4, 226), label or f"step {t}  true {worst[t]:.1f} deg", fill=(255, 255, 255))
        d.text((4, 240), pred_txt, fill=(255, 220, 120))
        return im

    picks = [int(round(v)) for v in np.linspace(push0, T, 8)]
    strip = Image.new("RGB", (8 * 224, 2 * 254 + 20), (30, 30, 30))
    for i, t in enumerate((push0 - 2, push0 - 1, push0)):
        strip.paste(card(t, f"context frame {t} (first push step {push0})"), (i * 224, 0))
    ImageDraw.Draw(strip).text((3 * 224 + 8, 8), f"top: the 3 context frames the model sees at the first push step\n"
                               f"bottom: the push, true tilt vs the prediction made {k} steps earlier", fill=(255, 255, 255))
    for i, t in enumerate(picks):
        strip.paste(card(t), (i * 224, 254 + 20))
    strip.save(args.out / "strip.png")
    if not args.no_gif:
        cards = [card(t) for t in range(T + 1)]
        cards[0].save(args.out / "rollout.gif", save_all=True, append_images=cards[1:], duration=int(1000 * rec["control_dt"]),
                      loop=0)
    # tilt chart: truth and the k = 1 / 5 / 10 predictions against the step index
    W, H, L, B = 900, 400, 50, 40
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    finite = P[~np.isnan(P)]
    ymax = max(95.0, float(finite.max()) if finite.size else 0.0)

    def xy(t, v):
        return L + (W - L - 10) * t / T, H - B - (H - B - 20) * v / ymax

    for v in (0, 30, 60, 90):
        d.line([xy(0, v), xy(T, v)], fill=(220, 220, 220) if v != 30 else (255, 0, 0))
        d.text((4, xy(0, v)[1] - 6), f"{v:3d}", fill=(0, 0, 0))
    for name, (a, b) in summary["phase_bounds"].items():
        d.line([xy(a, 0), xy(a, ymax)], fill=(235, 235, 235))
        d.text((xy(a, 0)[0] + 2, H - B + 4), name, fill=(120, 120, 120))
    series = [("true worst tilt", (0, 0, 0), worst, 0)] + \
             [(f"pred k={kk} (plotted at t+{kk})", c, P[:, kk], kk) for kk, c in ((1, (0, 110, 230)), (5, (0, 160, 60)), (10, (220, 60, 0)))
              if kk <= args.horizon]
    for i, (label, color, y, shift) in enumerate(series):
        pts = [xy(t + shift, y[t]) for t in range(T + 1) if t + shift <= T and not np.isnan(y[t])]
        if len(pts) > 1:
            d.line(pts, fill=color, width=2 if shift == 0 else 1)
        d.text((L + 10 + 190 * i, 4), label, fill=color)
    d.text((L + 10, 18), f"red line = {args.stop_tilt_deg:g} deg (unsafe); x = control step (0.05 s)", fill=(90, 90, 90))
    im.save(args.out / "tilt_chart.png")


if __name__ == "__main__":
    sys.exit(main())
