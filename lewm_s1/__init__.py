"""lewm_s1: standalone loader for the LeWM S1 world-model release (bundle + tilt head).

Public API (see ``world_model.py`` for the full docstrings):

* ``load_world_model(bundle_path, upstream_dir, device="cpu", head_path=None)`` -> ``WorldModel``
* ``WorldModel.encode / normalize_actions / action_sequence / rollout / predict_next / tilt_deg / unsafe_prob``
* ``letterbox(image_uint8, size=224)``: raw camera image -> cache-format 224x224x3 frame
* ``UPSTREAM_REPO``, ``UPSTREAM_COMMIT``, ``EXPECTED_UPSTREAM_SHA256``: the upstream code this bundle needs

``examples/train_episode_window.npz`` keys (EXAMPLE_KEYS below) hold one TRAIN window of the training cache:
3 context frames, the 2 recorded commands before the newest frame, 24 future commands (raw and WM units) and the
simulator's true worst-guard tilt of the 24 following frames.
"""
from .world_model import (ACTION_COLUMNS, ACTION_NAMES, EXPECTED_UPSTREAM_SHA256, HISTORY, IMAGE_SIZE, LATENT_DIM,
                          LATENT_INTERFACE, PIXEL_MEAN, PIXEL_STD, TILT_SCALE_DEG, UPSTREAM_COMMIT, UPSTREAM_FILES,
                          UPSTREAM_REPO, TiltHead, WorldModel, action_sequence, denormalize_actions, letterbox,
                          load_world_model, normalize_actions, sha256_file)

__version__ = "1.0"
EXAMPLE_KEYS = {
    "frames": "uint8 [3, 224, 224, 3] cache frames t-2, t-1, t (letterboxed RGB)",
    "frame_indices": "int64 [3] the frame indices t-2, t-1, t inside the episode",
    "raw_actions_hist": "float32 [2, 5] recorded simulator commands a[t-2], a[t-1] (dx, dy, dz, yaw=0, dgripper)",
    "raw_actions_future": "float32 [24, 5] recorded commands a[t], ..., a[t+23]",
    "wm_actions_hist": "float32 [2, 4] the same two commands in WM units (normalize_actions)",
    "wm_actions_future": "float32 [24, 4] the 24 future commands in WM units",
    "true_tilt_deg": "float64 [24] simulator worst-guard tilt (max over the two guards) of frames t+1 .. t+24",
    "true_unsafe": "bool [24] simulator unsafe flag (tilt > 30 deg) of frames t+1 .. t+24",
    "context_tilt_deg": "float64 [3] worst-guard tilt of the context frames t-2 .. t",
    "episode_id, t, horizon, control_dt, cache, episode_sha256, pixels_sha256, action_columns, window_rule, action_units":
        "provenance scalars / strings",
}

__all__ = ["load_world_model", "WorldModel", "TiltHead", "letterbox", "normalize_actions", "denormalize_actions",
           "action_sequence", "sha256_file", "UPSTREAM_REPO", "UPSTREAM_COMMIT", "UPSTREAM_FILES",
           "EXPECTED_UPSTREAM_SHA256", "HISTORY", "LATENT_DIM", "IMAGE_SIZE", "ACTION_COLUMNS", "ACTION_NAMES",
           "PIXEL_MEAN", "PIXEL_STD", "TILT_SCALE_DEG", "LATENT_INTERFACE", "EXAMPLE_KEYS", "__version__"]
