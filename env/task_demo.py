#!/usr/bin/env python
"""task_demo.py - the S1 world model as a DECISION MAKER in the live simulator (closed loop).

    MUJOCO_GL=egl python env/task_demo.py --upstream ./le-wm [--tasks all|name,name] [--no-safety] [--out task_demo_out]
                                          [--gif] [--max-steps 800] [--seed 0] [--fallback hold|retreat] [--compare ...]

The planner of the project's latent-RRT pilot, reduced to what this package can run: every 8 control steps the model
imagines 32 random candidate moves plus 'hold' (the pilot's nominal actions on its calibrated plant model, 8 edge steps
+ 12 hold steps, 20 predicted latents each, ONE batched rollout) and the tilt head VETOES every candidate whose predicted
worst-pillar tilt reaches 15 deg at any of the 20 steps; among the survivors a GEOMETRIC progress rule (the simulator's
pinch position, 8 mm along the direction, distance to the goal) picks the move, exactly with the pilot's selection rule
(improve on hold by >= 0.25 cm, tie-break by predicted tilt, escape / all-rejected cases). The executed move runs through
the pilot's action law (env/rrt_action_law.py, verbatim); the simulator judges the outcome. What is NOT here: the latent
tree, the distance head and the arrival classifier of the pilot (the package has no goal-image heads), so progress and
arrival are simulator geometry. Six tasks (grasped block at a named start pose -> a goal point), each run with the
safety check and without it (--no-safety only / --tasks restrict). Mirrors latent_planner/planner.py and
closed_loop_eval.py; every mirrored function names its source lines. Exit 0 when the run completed, 1 on any exception.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ENV_DIR = PACKAGE / "env"
sys.path.insert(0, str(ENV_DIR))                                  # scene files + rrt_action_law by bare name (env_demo.py:102)
sys.path.insert(0, str(PACKAGE))
import rrt_action_law as law                                      # noqa: E402  (numpy only; the pilot's law, verbatim copy)

GRIP = 1.0                                                        # closed_loop_eval.py:71
WARM_UP_STEPS = 2                                                 # closed_loop_eval.py:72
ARRIVE_M = 0.015                                                  # the scene judge's hover tolerance GOAL_TOLERANCE_M (harness.py:150)
TOPPLE_DEG, DISTURBED_DEG, START_MAX_TILT_DEG = 30.0, 10.0, 5.0   # harness.py:51; the pilot's start rule is 1 deg (closed_loop_eval.py:73)
LOST_GRASP_STEPS = 10                                             # harness.py:136
STAGING_Z, PHASE_SPEED, DESCENT_SPEED = 0.18, 0.06, 0.02          # harness.py:45; env_demo.py:31; the law's 2 cm/s
GUARD_XY = np.array([[0.425, -0.05], [0.425, 0.05]])              # harness.py:37
# The pilot's WM-range env bounds (runs/rrt_pilot_starts_001/run.json law_config): asserted against the bundle's normalisation
PILOT_ENV_LOW = (-0.3096823899891423, -0.34256465137093534, -0.37271963762219373)
PILOT_ENV_HIGH = (0.1999669656763118, 0.2358150227062972, 0.2618389845573939)


@dataclass(frozen=True)
class Task:
    name: str
    group: str
    start: tuple        # green-COM start pose (m), harness.py start groups
    goal: tuple         # green-COM goal point (m), inside the carry goal box x 0.395-0.46, y +-0.035, z 0.15-0.19
    note: str


# Chosen from the pilot's start groups and goal box BEFORE any model run (never tuned on results).
TASKS = (
    Task("normal_free", "normal", (0.425, 0.000, 0.185), (0.455, 0.025, 0.165), "4.4 cm in free space above the pillars"),
    Task("normal_into_slot", "normal", (0.425, 0.000, 0.190), (0.425, 0.000, 0.155), "3.5 cm straight down into the 1-cm slot; line clear, lateral error touches"),
    Task("slot_exit_right", "slot", (0.425, 0.000, 0.125), (0.445, 0.030, 0.170), "5.0 cm; the straight line passes AGAINST the right pillar"),
    Task("outer_left_in", "outer_left", (0.425, -0.0975, 0.125), (0.425, -0.020, 0.170), "8.9 cm; the straight line runs THROUGH the left pillar"),
    Task("outer_right_in", "outer_right", (0.425, 0.0975, 0.125), (0.425, 0.020, 0.170), "8.9 cm; mirror of outer_left_in, through the right pillar"),
    Task("outer_right_cross", "outer_right", (0.425, 0.0975, 0.130), (0.410, -0.030, 0.160), "13.1 cm across both pillars; through the right pillar"),
)


@dataclass(frozen=True)
class PlannerConfig:          # latent_planner/planner.py:87-110 defaults (RULE 1 / 2 / 3 off)
    n_directions: int = 32
    direction_std: tuple = tuple(law.DIRECTION_STD)      # (1, 1, 0.5)
    tilt_line_deg: float = 15.0
    edge_steps: int = law.EDGE_STEPS                     # 8
    check_hold_steps: int = law.CHECK_HOLD_STEPS         # 12 -> 20 predicted latents
    score_step: int = law.EDGE_STEPS
    min_improvement_cm: float = 0.25
    tie_cm: float = 0.05
    arrival_cycles: int = 3
    grip: float = GRIP

    @property
    def check_steps(self):
        return self.edge_steps + self.check_hold_steps


# ============================================================================= planner pieces (latent_planner/planner.py)
def cycle_seed(base_seed, cycle):                                 # planner.py:235-239
    return int(np.random.SeedSequence([int(base_seed), int(cycle)]).generate_state(1, np.uint32)[0])


def candidate_directions(seed, n, std):                           # planner.py:242-245
    rng = np.random.default_rng(int(seed))
    return np.stack([law.random_direction(rng, np.asarray(std, float)) for _ in range(int(n))])


class ContinuedHold:                                              # planner.py:286-325: consecutive holds = ONE anchored hold
    def __init__(self, carry, steps, grip, config):
        if carry is None or carry["kind"] != "hold":
            raise ValueError("ContinuedHold needs the carry state of an executing hold")
        self.tracker = law.SegmentTracker("hold", carry["reference"], steps=int(steps), grip=grip, config=config)
        self.tracker.integral = np.asarray(carry["integral"], float).copy()

    @property
    def finished(self):
        return self.tracker.finished

    def action(self, pinch):
        return self.tracker.action(pinch)

    def carry_state(self):
        t = self.tracker
        return {"kind": "hold", "reference": t.reference.copy(), "integral": t.integral.copy(),
                "dragged": bool(t.log["reference_dragged"][-1]) if t.log["reference_dragged"] else False}


def make_runner(direction, carry, cfg, pconf, steps, then_hold=0):   # planner.py:328-337
    if direction is not None:
        segments = [law.edge(direction)] + ([law.hold(then_hold)] if then_hold else [])
        return law.LawRunner(segments, pconf.grip, cfg, previous=carry)
    if carry is not None and carry["kind"] == "hold":
        return ContinuedHold(carry, steps, pconf.grip, cfg)
    return law.LawRunner([law.hold(int(steps))], pconf.grip, cfg)


def check_runner(direction, carry, cfg, pconf):                   # planner.py:340-344
    if direction is None:
        return make_runner(None, carry, cfg, pconf, pconf.check_steps)
    return make_runner(direction, carry, cfg, pconf, pconf.edge_steps, then_hold=pconf.check_hold_steps)


def nominal_rollout(pinch, velocity, carry, direction, cfg, pconf):   # planner.py:347-362 (actions + nominal pinch path)
    plant = law.ModelPlant(law.resolve_model("grasped"), pinch, velocity)
    runner = check_runner(None if direction is None else law.unit(direction), carry, cfg, pconf)
    actions, pinches = [], [plant.pinch()]
    while not runner.finished:
        action = runner.action(pinches[-1])
        plant.step(action)
        actions.append(action)
        pinches.append(plant.pinch())
    return np.asarray(actions), np.asarray(pinches)


def expand(pinch, velocity, carry, directions, history_wm, cfg, pconf, normalize):   # planner.py:365-376 (hold LAST)
    rows = [nominal_rollout(pinch, velocity, carry, d, cfg, pconf) for d in [*np.asarray(directions, float), None]]
    actions = np.stack([a for a, _ in rows])                                       # [C, 20, 5]
    C, S = actions.shape[:2]
    future_wm = normalize(actions.reshape(C * S, 5)).reshape(C, S, 4)              # == law.wm_normalize (asserted once)
    sequences = np.concatenate([np.broadcast_to(history_wm, (C, 2, 4)), future_wm], 1).astype(np.float32)
    return actions, sequences, np.stack([p for _, p in rows])


def select_candidate(score, signal8, is_hold, pconf, safety):     # planner.py:390-440: tilt mode, escape 'min_tilt', all-rejected 'hold'
    score, signal8 = np.asarray(score, float), np.asarray(signal8, float)
    if not np.isfinite(score).all() or not np.isfinite(signal8).all():
        raise FloatingPointError("non-finite candidate scores")
    hold = int(np.flatnonzero(is_hold)[0])
    accepted = score < pconf.tilt_line_deg if safety else np.ones(len(score), bool)
    edges = accepted & ~is_hold
    reducing = edges & (signal8[hold] - signal8 >= pconf.min_improvement_cm - 1e-12)
    if reducing.any():
        best = signal8[reducing].min()
        near = np.flatnonzero(reducing & (signal8 <= best + pconf.tie_cm + 1e-12))
        index, reason = int(near[np.lexsort((near, score[near]))[0]]), "reduce"
    elif not accepted.any():
        index, reason = hold, "all_rejected"
    elif accepted[hold]:
        index, reason = hold, "no_reducing_edge"
    else:
        cand = np.flatnonzero(edges)
        index, reason = int(cand[np.lexsort((cand, signal8[cand], score[cand]))[0]]), "hold_rejected_escape"
    return index, reason, accepted, reducing


def retreat_direction(pinch):
    """--fallback retreat (EXPERIMENT, not validated by the project): lift and back away from the nearest pillar."""
    d = np.asarray(pinch[:2], float) - GUARD_XY
    away = d[int(np.argmin(np.linalg.norm(d, axis=1)))]
    away = away / n if (n := float(np.linalg.norm(away))) > 1e-9 else np.array([0.0, np.sign(pinch[1]) or 1.0])
    return law.unit([0.5 * away[0], 0.5 * away[1], 1.0])


# ============================================================================= simulator truth
def pinch_xyz(env):
    return env.data.site_xpos[env._pinch_site_id].copy()


def green_xyz(env):
    return env.data.body("object_0").xipos.copy()


class Episode:
    """Per-step truth of one episode (row 0 = the settled start); mirrors closed_loop_eval.Episode (369-409)."""

    def __init__(self, env, info, pinch_goal, goal):
        self.env, self.pinch_goal, self.goal = env, np.asarray(pinch_goal, float), np.asarray(goal, float)
        self.rows, self.actions, self.ungrasped, self.rendered = [], [], 0, {}
        self.capture(info, -1, "start")

    def capture(self, info, cycle, kind):
        tilts = [float(g["tilt_deg"]) for g in info["safety"]["guards"]]
        p, g = pinch_xyz(self.env), green_xyz(self.env)
        grasped = bool(info["task"]["grasped"])
        self.rows.append({"step": len(self.actions), "cycle": cycle, "kind": kind, "pinch": p, "green": g, "tilt": tilts,
                          "worst": max(tilts), "grasped": grasped, "episode_failed": bool(info["task"]["episode_failed"]),
                          "dist_pinch_cm": 100 * float(np.linalg.norm(p - self.pinch_goal)),
                          "dist_green_cm": 100 * float(np.linalg.norm(g - self.goal))})
        self.ungrasped = 0 if grasped else self.ungrasped + 1

    @property
    def steps(self):
        return len(self.actions)

    @property
    def last(self):
        return self.rows[-1]

    def step(self, action, cycle, kind):
        before = float(self.env.data.time)
        _, _, _, _, info = self.env.step(action)
        if abs(float(self.env.data.time) - before - float(self.env.control_timestep())) > 1e-9:
            raise RuntimeError("env.step did not advance exactly one control step")
        self.actions.append(np.asarray(action, float).copy())
        self.capture(info, cycle, kind)

    def end_reason(self, max_steps):                              # harness.py:157-170 without success / false_arrival
        if self.last["episode_failed"] or self.last["worst"] > TOPPLE_DEG:
            return "toppled"
        if self.ungrasped >= LOST_GRASP_STEPS:
            return "lost_grasp"
        if self.steps >= max_steps:
            return "timeout"
        return None


def scripted_start(env, task, letterbox):
    """Procedural start: env_demo.py:217-281 waypoint controller (SETTLE, APPROACH, DESCEND, CLOSE, LIFT at 6 cm/s), then
    CARRY to STAGING_Z above the start pose (harness.py:19-20), DESCEND_START at the law's 2 cm/s and a 20-step frozen hold.
    The grip offset (pinch - green COM at the end of LIFT, goal_carry.py:258) turns green-COM poses into pinch targets."""
    _, info = env.reset(seed=0)
    dt = float(env.control_timestep())
    if abs(dt - 0.05) > 1e-12:
        raise RuntimeError(f"control step is {dt} s, the model was trained with 0.05 s")
    low, high = env._workspace_bounds
    if not (np.array_equal(low, law.WORKSPACE_LOW) and np.array_equal(high, law.WORKSPACE_HIGH)):
        raise RuntimeError("scene workspace differs from the action law's constants")        # closed_loop_eval.py:114-117
    green0, start = green_xyz(env), np.asarray(task.start, float)
    above, grasp = np.array([green0[0], green0[1], 0.19]), green0 + [0.0, 0.0, 0.005]
    st = {"ref": pinch_xyz(env), "streak": 0, "offset": None, "steps": 0, "max_tilt": 0.0, "info": info}

    def arrived(wp):
        return np.linalg.norm(pinch_xyz(env) - wp) <= 0.005 and np.linalg.norm(st["ref"] - wp) <= 1e-6

    def staging():
        return np.array([start[0], start[1], STAGING_Z]) + st["offset"]

    def start_pinch():
        return start + st["offset"]

    plan = [("SETTLE", None, False, PHASE_SPEED, lambda n: n >= 20, 20),
            ("APPROACH", lambda: above, False, PHASE_SPEED, lambda n: arrived(above), 120),
            ("DESCEND", lambda: grasp, False, PHASE_SPEED, lambda n: arrived(grasp), 120),
            ("CLOSE", lambda: grasp, True, PHASE_SPEED, lambda n: st["streak"] >= 6, 60),
            ("LIFT", lambda: above, True, PHASE_SPEED, lambda n: arrived(above), 120),
            ("CARRY", staging, True, PHASE_SPEED, lambda n: arrived(staging()), 200),
            ("DESCEND_START", start_pinch, True, DESCENT_SPEED, lambda n: arrived(start_pinch()), 200),
            ("SETTLE_START", None, True, DESCENT_SPEED, lambda n: n >= 20, 20)]
    phases = {}
    for name, target_fn, closed, speed, done, timeout in plan:
        n, t0 = 0, st["steps"]
        while True:
            target = st["ref"].copy() if target_fn is None else target_fn()       # None: reference frozen (hold)
            actual = pinch_xyz(env)
            delta = target - st["ref"]
            if (dist := float(np.linalg.norm(delta))) > speed * dt:
                delta *= speed * dt / dist
            cand = st["ref"] + delta
            if np.linalg.norm(lead := cand - actual) > 0.02:
                cand = actual + lead * (0.02 / np.linalg.norm(lead))
            st["ref"] = np.clip(cand, low, high)
            a = env.normalize_action(np.array([*(st["ref"] - actual), 0.0, 1.0 if closed else -1.0]))
            _, _, _, _, info = env.step(a)
            st["steps"] += 1
            n += 1
            st["max_tilt"] = max(st["max_tilt"], *(g["tilt_deg"] for g in info["safety"]["guards"]))
            st["streak"] = st["streak"] + 1 if info["task"]["grasped"] else 0
            if done(n):
                break
            if n >= timeout:
                raise RuntimeError(f"start phase {name} did not finish in {n} steps (worst tilt {st['max_tilt']:.1f} deg, "
                                   f"grasped={info['task']['grasped']})")
        if name == "LIFT":
            st["offset"] = pinch_xyz(env) - green_xyz(env)
        phases[name] = [t0, st["steps"]]
    g = green_xyz(env)
    if not info["task"]["grasped"] or st["max_tilt"] >= START_MAX_TILT_DEG or info["task"]["episode_failed"]:
        raise RuntimeError(f"{task.name}: start not clean (grasped={info['task']['grasped']}, max tilt {st['max_tilt']:.2f} deg)")
    return {"info": info, "grip_offset_m": st["offset"].tolist(), "green_requested_m": start.tolist(), "green_achieved_m": g.tolist(),
            "pose_error_mm": 1000 * float(np.linalg.norm(g - start)), "pinch_m": pinch_xyz(env).tolist(), "steps": st["steps"],
            "max_tilt_deg": float(st["max_tilt"]), "phase_bounds": phases, "grasped": True,
            "nearest_guard_cm": 100 * float(np.linalg.norm(GUARD_XY - g[:2], axis=1).min()),
            "frame": letterbox(env.render(camera="front_pixels"))}


# ============================================================================= one episode
def run_episode(env, wm, task, task_index, safety, args, cfg, pconf, letterbox, out):
    t_ep = time.perf_counter()
    mode = "safety" if safety else "nosafety"
    print(f"\n=== {task.name} [{task.group}] mode {mode}: start green {task.start} -> goal {task.goal} ({task.note})")
    s0 = scripted_start(env, task, letterbox)
    print(f"  start ready after {s0['steps']} scripted steps: green {np.round(s0['green_achieved_m'], 4).tolist()} "
          f"(requested {task.start}, off by {s0['pose_error_mm']:.1f} mm), grip offset {np.round(s0['grip_offset_m'], 4).tolist()}, "
          f"max pillar tilt during the start {s0['max_tilt_deg']:.3f} deg, nearest pillar {s0['nearest_guard_cm']:.2f} cm")
    goal = np.asarray(task.goal, float)
    pinch_goal = goal + np.asarray(s0["grip_offset_m"])
    ep = Episode(env, s0["info"], pinch_goal, goal)
    ep.rendered[0] = s0["frame"]

    def render():
        frame = letterbox(env.render(camera="front_pixels"))
        ep.rendered[ep.steps] = frame
        return frame

    frames = deque([s0["frame"]], maxlen=3)                       # closed_loop_eval.py:588-599: warm-up hold, 3-frame context
    runner = law.LawRunner([law.hold(WARM_UP_STEPS)], GRIP, cfg)
    for _ in range(WARM_UP_STEPS):
        ep.step(runner.action(pinch_xyz(env)), -1, "warm_up")
        frames.append(render())
    carry = runner.carry_state()
    base_seed = int(args.seed) * 1000 + task_index               # closed_loop_eval.py:616
    is_hold = np.zeros(pconf.n_directions + 1, bool)
    is_hold[-1] = True
    cycles, reason, streak, cycle, checked = [], ep.end_reason(args.max_steps), 0, 0, False
    while reason is None:
        t0 = time.perf_counter()
        pinch_now, velocity = ep.last["pinch"], ep.rows[-1]["pinch"] - ep.rows[-2]["pinch"]
        arrived = ep.last["dist_pinch_cm"] <= 100 * ARRIVE_M and ep.last["grasped"]   # planner.py:443-456 ArrivalCounter
        streak = streak + 1 if arrived else 0
        if streak >= pconf.arrival_cycles:
            reason = "success"
            break
        directions = candidate_directions(cycle_seed(base_seed, cycle), pconf.n_directions, pconf.direction_std)
        history_wm = wm.normalize_actions(np.stack(ep.actions[-2:]))
        actions, sequences, nominal_pinch = expand(pinch_now, velocity, carry, directions, history_wm, cfg, pconf, wm.normalize_actions)
        if not checked:                                           # the package recipe == the pilot's law.wm_normalize, bit for bit
            if not np.array_equal(law.wm_normalize(actions, cfg.normalization).astype(np.float32), sequences[:, 2:]):
                raise RuntimeError("wm.normalize_actions differs from rrt_action_law.wm_normalize")
            checked = True
        t1 = time.perf_counter()
        z_ctx = wm.encode(np.stack(frames))                      # [3, 192]
        C, S = sequences.shape[0], pconf.check_steps
        predicted = wm.rollout(np.broadcast_to(z_ctx, (C, 3, z_ctx.shape[1])).copy(), sequences)      # [33, 20, 192], one batch
        tilt = wm.tilt_deg(predicted.reshape(C * S, -1)).reshape(C, S).astype(np.float64)
        max_tilt = tilt.max(1)
        tilt_now = float(wm.tilt_deg(z_ctx[-1:])[0])
        t2 = time.perf_counter()
        # geometric progress signal: pinch + 8 mm * direction, box-clipped like the law's reference; hold = the pinch itself
        p8 = np.array([law.box_step(pinch_now, pconf.edge_steps * law.STEP_M * d, *cfg.box())[0] for d in directions] + [pinch_now])
        signal8 = 100 * np.linalg.norm(p8 - pinch_goal, axis=1)
        nominal8 = 100 * np.linalg.norm(nominal_pinch[:, pconf.score_step] - pinch_goal, axis=1)
        index, why, accepted, reducing = select_candidate(max_tilt, signal8, is_hold, pconf, safety)
        direction = None if is_hold[index] else directions[index]
        if why == "all_rejected" and args.fallback == "retreat":
            direction, why = retreat_direction(pinch_now), "all_rejected_retreat"
        kind = "hold" if direction is None else "edge"
        t3 = time.perf_counter()
        runner = make_runner(direction, carry, cfg, pconf, pconf.edge_steps)   # closed_loop_eval.py:654-669
        executed = 0
        while not runner.finished and reason is None:
            ep.step(runner.action(pinch_xyz(env)), cycle, kind)
            executed += 1
            if args.gif or executed > pconf.edge_steps - 3:       # context frames = the frames after steps 6, 7, 8
                frames.append(render())
            reason = ep.end_reason(args.max_steps)
        carry = runner.carry_state()
        c = {"cycle": cycle, "step": ep.rows[-1 - executed]["step"], "kind": kind, "reason": why, "chosen": index,
             "dir_x": np.nan if direction is None else direction[0], "dir_y": np.nan if direction is None else direction[1],
             "dir_z": np.nan if direction is None else direction[2], "n_accepted": int(accepted.sum()),
             "n_rejected_edges": int((~accepted & ~is_hold).sum()), "hold_accepted": int(accepted[-1]), "n_reducing": int(reducing.sum()),
             "chosen_pred_max_tilt_deg": float(max_tilt[index]), "hold_pred_max_tilt_deg": float(max_tilt[-1]),
             "min_pred_max_tilt_deg": float(max_tilt.min()), "head_tilt_now_deg": tilt_now, "true_tilt_now_deg": ep.rows[-1 - executed]["worst"],
             "dist_pinch_cm": ep.rows[-1 - executed]["dist_pinch_cm"], "dist_green_cm": ep.rows[-1 - executed]["dist_green_cm"],
             "hold_signal8_cm": float(signal8[-1]), "chosen_signal8_cm": float(signal8[index]), "chosen_nominal8_cm": float(nominal8[index]),
             "best_edge_improvement_cm": float((signal8[-1] - signal8[:-1])[accepted[:-1]].max()) if accepted[:-1].any() else np.nan,
             "arrival_streak": streak, "executed_steps": executed, "true_max_tilt_executed_deg": max(r["worst"] for r in ep.rows[-executed:]),
             "plan_s": t3 - t0, "nominal_s": t1 - t0, "model_s": t2 - t1, "sim_s": time.perf_counter() - t3}
        cycles.append(c)
        if cycle % 10 == 0:
            print(f"  cycle {cycle:3d} step {c['step']:3d}  {why:21s} rejected {c['n_rejected_edges']:2d}/32 hold_ok {c['hold_accepted']}  "
                  f"pred tilt chosen {c['chosen_pred_max_tilt_deg']:5.1f} true now {c['true_tilt_now_deg']:5.1f}  dist {c['dist_pinch_cm']:5.2f} cm "
                  f"(green {c['dist_green_cm']:5.2f})  plan {c['plan_s'] * 1000:4.0f} ms")
        cycle += 1
    render()                                                      # the final picture
    worst = np.array([r["worst"] for r in ep.rows])
    reasons = {k: sum(1 for c in cycles if c["reason"] == k) for k in ("reduce", "no_reducing_edge", "hold_rejected_escape", "all_rejected", "all_rejected_retreat")}
    first_topple = int(np.flatnonzero(worst > TOPPLE_DEG)[0]) if (worst > TOPPLE_DEG).any() else None
    summary = {"task": task.name, "group": task.group, "mode": mode, "safety": safety, "fallback": args.fallback, "outcome": reason,
               "steps": ep.steps, "cycles": len(cycles), "max_true_tilt_deg": float(worst.max()), "contact_steps_tilt_gt_0": int((worst > 0).sum()),
               "disturbed_ge_10deg": bool(worst.max() >= DISTURBED_DEG), "all_rejected_cycles": reasons["all_rejected"] + reasons["all_rejected_retreat"],
               "reason_counts": reasons, "toppled_rule": None if reason != "toppled" else
               ("scene episode_failed latch (substep)" if ep.last["episode_failed"] else "control-step worst tilt > 30") +
               (f"; first control step over 30 deg: {first_topple}" if first_topple is not None else "; no control-step reading over 30 deg"),
               "final_dist_pinch_cm": ep.last["dist_pinch_cm"], "final_dist_green_cm": ep.last["dist_green_cm"],
               "green_within_tolerance_at_end": bool(ep.last["dist_green_cm"] <= 100 * ARRIVE_M),
               "geometric_false_arrival": bool(reason == "success" and ep.last["dist_green_cm"] > 100 * ARRIVE_M),
               "min_dist_pinch_cm": float(min(r["dist_pinch_cm"] for r in ep.rows)), "start": {k: v for k, v in s0.items() if k not in ("info", "frame")},
               "goal_green_m": goal.tolist(), "pinch_goal_m": pinch_goal.tolist(), "start_goal_distance_cm": 100 * float(np.linalg.norm(goal - np.asarray(task.start))),
               "base_seed": base_seed, "plan_s_median": float(np.median([c["plan_s"] for c in cycles])) if cycles else None,
               "episode_s": time.perf_counter() - t_ep}
    write_episode(out / f"{task.name}_{mode}", ep, cycles, summary, args)
    summary["cards"] = cards(ep)
    print(f"  -> {reason} after {ep.steps} steps / {len(cycles)} cycles; max true tilt {worst.max():.1f} deg; final distance pinch "
          f"{ep.last['dist_pinch_cm']:.2f} cm, green COM {ep.last['dist_green_cm']:.2f} cm; reasons {reasons}; {summary['episode_s']:.0f} s")
    return summary


# ============================================================================= outputs
def write_episode(d, ep, cycles, summary, args):
    d.mkdir(parents=True, exist_ok=False)
    with (d / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "cycle", "kind", "cmd_dx", "cmd_dy", "cmd_dz", "cmd_grip", "pinch_x", "pinch_y", "pinch_z", "green_x", "green_y",
                    "green_z", "tilt_left_deg", "tilt_right_deg", "worst_tilt_deg", "grasped", "episode_failed", "dist_pinch_cm", "dist_green_cm"])
        for i, r in enumerate(ep.rows):
            a = ep.actions[i - 1] if i else [np.nan] * 5
            w.writerow([r["step"], r["cycle"], r["kind"], *(f"{v:.6g}" for v in (a[0], a[1], a[2], a[4], *r["pinch"], *r["green"], *r["tilt"], r["worst"])),
                        int(r["grasped"]), int(r["episode_failed"]), f"{r['dist_pinch_cm']:.6g}", f"{r['dist_green_cm']:.6g}"])
    with (d / "cycles.csv").open("w", newline="", encoding="utf-8") as f:
        keys = [k for k in cycles[0]] if cycles else ["cycle"]
        w = csv.writer(f)
        w.writerow([k for k in keys if not k.endswith("_s")])     # timings are not deterministic: summary.json only
        for c in cycles:
            w.writerow([c[k] if isinstance(c[k], (str, int)) else f"{c[k]:.6g}" for k in keys if not k.endswith("_s")])
    (d / "summary.json").write_text(json.dumps({**summary, "cycle_timing_s": {k: float(np.median([c[k] for c in cycles])) if cycles else None
                                                                             for k in ("plan_s", "nominal_s", "model_s", "sim_s")}}, indent=2) + "\n", encoding="utf-8")
    if args.gif:
        from PIL import Image
        steps = sorted(ep.rendered)
        ims = [card(ep.rendered[t], f"step {t}  tilt {ep.rows[t]['worst']:.1f} deg", f"{ep.rows[t]['kind']}  dist {ep.rows[t]['dist_pinch_cm']:.1f} cm") for t in steps]
        ims[0].save(d / "rollout.gif", save_all=True, append_images=ims[1:], duration=50, loop=0)
        Image.fromarray(ep.rendered[steps[-1]]).save(d / "final.png")


def card(frame, line1, line2, width=224):
    """A frame (any HxWx3) scaled to ``width`` with two caption lines below."""
    from PIL import Image, ImageDraw
    h = round(width * frame.shape[0] / frame.shape[1])
    im = Image.new("RGB", (width, h + 26), (30, 30, 30))
    im.paste(Image.fromarray(frame).resize((width, h)), (0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((3, h + 1), line1, fill=(255, 255, 255))
    draw.text((3, h + 13), line2, fill=(255, 220, 120))
    return im


def cards(ep):
    """Five small pictures of the episode for the per-mode strip: start, 25 / 50 / 75 % and the end (letterbox bands cropped)."""
    steps = sorted(ep.rendered)
    picks = [0] + [min(steps, key=lambda t: abs(t - q * ep.steps)) for q in (0.25, 0.5, 0.75)] + [steps[-1]]
    return [card(ep.rendered[t][28:196], f"step {t} tilt {ep.rows[t]['worst']:.1f}",
                 "start" if i == 0 else f"{ep.rows[t]['kind']} d {ep.rows[t]['dist_pinch_cm']:.1f} cm", 112) for i, t in enumerate(picks)]


def write_strip(mode, summaries, path):
    from PIL import Image, ImageDraw
    rows = [s for s in summaries if s["mode"] == mode]
    pitch = rows[0]["cards"][0].height + 4 if rows else 120
    W, H = 150 + 5 * 112, 16 + len(rows) * pitch
    im = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(im)
    d.text((4, 2), f"task_demo  mode {mode}: start, 25 %, 50 %, 75 %, end of each episode (true tilt, kind, pinch distance to goal)", fill=(255, 255, 255))
    for i, s in enumerate(rows):
        y = 16 + i * pitch
        d.text((4, y + 20), s["task"], fill=(255, 255, 255))
        d.text((4, y + 34), f"[{s['group']}]", fill=(180, 180, 180))
        d.text((4, y + 52), f"{s['outcome']} {s['steps']} steps", fill=(120, 255, 120) if s["outcome"] == "success" else (255, 120, 120))
        d.text((4, y + 66), f"max tilt {s['max_true_tilt_deg']:.1f} deg", fill=(255, 220, 120))
        for j, c in enumerate(s["cards"]):
            im.paste(c, (150 + j * 112, y))
    im.save(path)


def table(summaries):
    lines = [f"{'task':18s} | {'mode':8s} | {'outcome':10s} | steps | max true tilt | tilt>0 steps | all-rejected cycles",
             "-" * 96]
    for s in summaries:
        lines.append(f"{s['task']:18s} | {s['mode']:8s} | {s['outcome']:10s} | {s['steps']:5d} | {s['max_true_tilt_deg']:13.1f} | "
                     f"{s['contact_steps_tilt_gt_0']:12d} | {s['all_rejected_cycles']:d}")
    return "\n".join(lines)


def comparison(summaries):
    by = {(s["task"], s["mode"]): s for s in summaries}
    tasks = list(dict.fromkeys(s["task"] for s in summaries))
    out = [f"{'task':18s} | with safety check              | without (--no-safety)", "-" * 80]
    for t in tasks:
        cell = lambda m: f"{by[(t, m)]['outcome']} ({by[(t, m)]['steps']} steps, {by[(t, m)]['max_true_tilt_deg']:.1f} deg)" if (t, m) in by else "not run"
        out.append(f"{t:18s} | {cell('safety'):30s} | {cell('nosafety')}")
    counts = lambda m: " / ".join(f"{sum(1 for s in summaries if s['mode'] == m and s['outcome'] == o)} {o}"
                                  for o in ("success", "toppled", "timeout", "lost_grasp"))
    verdict = (f"with the model's safety check: {counts('safety')}; without it: {counts('nosafety')} "
               f"(over {len(tasks)} tasks; safety from the model, progress geometric, no RRT tree / goal head)")
    return "\n".join(out), verdict


def compare_expected(summaries, path):
    exp = {(e["task"], e["mode"]): e for e in json.loads(Path(path).read_text(encoding="utf-8"))["episodes"]}
    print(f"\ncompare with {path} (informational; MuJoCo contact events and borderline 15-deg decisions depend on the build / CPU):")
    for s in summaries:
        e = exp.get((s["task"], s["mode"]))
        if e is None:
            print(f"  {s['task']:18s} {s['mode']:8s}: not in the expected summary")
            continue
        same = s["outcome"] == e["outcome"] and s["steps"] == e["steps"]
        print(f"  {s['task']:18s} {s['mode']:8s}: outcome {s['outcome']} vs {e['outcome']}, steps {s['steps']} vs {e['steps']}, max tilt "
              f"{s['max_true_tilt_deg']:.1f} vs {e['max_true_tilt_deg']:.1f} deg  {'SAME' if same else 'DIFFERENT'}")


# ============================================================================= main
def rel(path):
    """A path as stored in summary.json: relative to the package root when inside it, else its last component (env_demo.py:74-78)."""
    p = Path(path).resolve()
    return p.relative_to(PACKAGE).as_posix() if p.is_relative_to(PACKAGE) else p.name


def run(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.stdout.reconfigure(line_buffering=True)
    import torch
    torch.set_num_threads(int(args.threads))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    import lewm_s1
    import mujoco
    from three_block_scene import ThreeBlockScene
    t0 = time.time()
    names = [t.name for t in TASKS] if args.tasks == "all" else args.tasks.split(",")
    unknown = [n for n in names if n not in {t.name for t in TASKS}]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; known: {[t.name for t in TASKS]}")
    tasks = [(i, t) for i, t in enumerate(TASKS) if t.name in names]
    modes = [False] if args.no_safety else [True, False]
    args.out.mkdir(parents=True, exist_ok=False)
    wm = lewm_s1.load_world_model(args.bundle, args.upstream, device=args.device, head_path=args.head, encoder_builder=args.encoder_builder)
    print(f"loaded {wm!r}\n  bundle sha256 {wm.bundle_sha256}  head sha256 {wm.head_info['sha256']}\n  torch {torch.__version__}  threads "
          f"{torch.get_num_threads()}  device {wm.device}  tf32 off  mujoco {mujoco.__version__}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    cfg = law.LawConfig(normalization={"mean": list(wm.action_normalization["mean"]), "scale": list(wm.action_normalization["scale"])})
    low, high = cfg.env_bounds()
    if not (np.allclose(low, PILOT_ENV_LOW, atol=1e-12) and np.allclose(high, PILOT_ENV_HIGH, atol=1e-12)):
        raise RuntimeError(f"the bundle's action normalisation gives WM-range env bounds {low} {high}, not the pilot's")
    print(f"  action law: {law.SPEED_MPS * 100:g} cm/s, WM-range env bounds of the pilot reproduced from the bundle's normalisation (max |diff| "
          f"{max(np.abs(low - PILOT_ENV_LOW).max(), np.abs(high - PILOT_ENV_HIGH).max()):.1e})")
    pconf = PlannerConfig()
    print(f"  planner: {asdict(pconf)}; arrival = pinch within {100 * ARRIVE_M:g} cm of goal + grip offset for {pconf.arrival_cycles} cycles; "
          f"toppled > {TOPPLE_DEG:g} deg; timeout {args.max_steps} steps; fallback {args.fallback}")
    env = ThreeBlockScene()
    summaries = []
    try:
        for safety in modes:
            for index, task in tasks:
                summaries.append(run_episode(env, wm, task, index, safety, args, cfg, pconf, lewm_s1.letterbox, args.out))
            write_strip("safety" if safety else "nosafety", summaries, args.out / f"strip_{'safety' if safety else 'nosafety'}.png")
    finally:
        env.close()
    cmp_text, verdict = comparison(summaries)
    overall = {"format": "task_demo_v1", "runtime_s": round(time.time() - t0, 1), "arguments": {k: (rel(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "tasks": [asdict(t) for _, t in tasks], "planner": asdict(pconf), "arrive_m": ARRIVE_M, "topple_deg": TOPPLE_DEG,
               "versions": {"torch": torch.__version__, "numpy": np.__version__, "mujoco": mujoco.__version__, "gl": os.environ["MUJOCO_GL"],
                            "threads": torch.get_num_threads()}, "bundle_sha256": wm.bundle_sha256, "head_sha256": wm.head_info["sha256"],
               "episodes": [{k: v for k, v in s.items() if k != "cards"} for s in summaries], "table": table(summaries), "comparison": cmp_text,
               "verdict": verdict}
    (args.out / "summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print("\n" + table(summaries))
    if not args.no_safety:
        print("\n" + cmp_text)
    print(f"\n{verdict}\nwrote {args.out}/summary.json, strip_*.png and one directory per task x mode ({overall['runtime_s']} s)")
    if args.compare:
        compare_expected(summaries, args.compare)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", type=Path, required=True, help="checkout of lucas-maes/le-wm at commit 8edfeb33")
    p.add_argument("--bundle", type=Path, default=PACKAGE / "model_state.pt")
    p.add_argument("--head", type=Path, default=PACKAGE / "heads" / "tilt_head.pt")
    p.add_argument("--encoder-builder", default="hydra", choices=("hydra", "transformers"))
    p.add_argument("--tasks", default="all", help="'all' or a comma-separated list of task names")
    p.add_argument("--no-safety", action="store_true", help="run only the mode without the model's safety check")
    p.add_argument("--out", type=Path, default=Path("task_demo_out"), help="NEW output directory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--gif", action="store_true", help="render every step and write rollout.gif per episode")
    p.add_argument("--max-steps", type=int, default=800, help="timeout (control steps of 0.05 s, the pilot's 800)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", type=Path, help="examples/task_demo_expected_summary.json: print the differences to our run")
    p.add_argument("--fallback", default="hold", choices=("hold", "retreat"),
                   help="what to do when every candidate incl. hold is rejected: hold (the project's rule) or an EXPERIMENTAL lift-and-back-away edge")
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


if __name__ == "__main__":
    sys.exit(main())
