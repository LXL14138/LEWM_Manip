"""Standalone loader for the LeWM "S1" world-model bundle (``model_state.pt``) and its tilt head.

This module imports NOTHING from the training project. It needs only ``torch``, ``numpy``, ``omegaconf`` +
``hydra-core`` (model instantiation), the upstream LeWM code (``jepa.py``, ``module.py``, the model yaml) of
commit ``8edfeb33`` of https://github.com/lucas-maes/le-wm.git and the packages that code imports
(``einops``, ``transformers``, ``stable-pretraining``); ``Pillow`` only for :func:`letterbox`.

Typical use::

    from lewm_s1 import load_world_model
    wm = load_world_model("model_state.pt", "/path/to/le-wm", head_path="heads/tilt_head.pt")
    z_ctx = wm.encode(frames)                      # frames: uint8 [3, 224, 224, 3] RGB, letterboxed (see letterbox)
    a = wm.normalize_actions(raw_env_actions)      # raw: [26, 5] simulator commands in [-1, 1] -> [26, 4] WM units
    z_future = wm.rollout(z_ctx[None], a[None])    # [1, 24, 192] predicted post-projector latents
    tilt = wm.tilt_deg(z_future[0])                # [24] predicted worst-guard tilt in degrees
    p_unsafe = wm.unsafe_prob(z_future[0])         # [24] sigmoid(unsafe logit)

Conventions (all mirrored from the project's evaluation tools so that the numbers agree bit for bit on CPU):

* Images: the cache frames are 224x224x3 uint8 RGB produced by :func:`letterbox` from the simulator's fixed
  front camera (640x480, no overlay, shadows off). ``encode`` turns them into float [0, 1], subtracts the
  ImageNet mean and divides by the ImageNet std per channel (channel-first), then calls
  ``model.encode({'pixels': x})['emb']`` and returns the POST-PROJECTOR embedding (192-d), which is what every
  head of the project reads.
* Actions: the simulator records 5 normalised commands per control step (0.05 s): ``[dx, dy, dz, yaw, dgripper]``
  in [-1, 1]; dx/dy/dz map to +-0.05 m end-effector displacement per step (``action_low/high = +-0.05``), the
  yaw column is always 0 (yaw fixed at 90 deg) and dgripper is the gripper command in [-1, 1]. The world model
  uses columns (0, 1, 2, 4) standardised with the TRAIN-set mean/scale stored in the bundle. Alignment:
  ``state[i] --action[i]--> state[i+1]``.
* Rollout: with history 3, the k-th prediction uses the three newest latents (real context at k=0, then the
  model's own predictions) and the actions ``a[t-2+k], a[t-1+k], a[t+k]``; the action timeline passed to
  ``rollout`` is therefore ``[a[t-2], a[t-1], a[t], ..., a[t+K-1]]`` (2 + K rows).
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["WorldModel", "TiltHead", "load_world_model", "letterbox", "normalize_actions", "denormalize_actions",
           "action_sequence", "sha256_file", "UPSTREAM_REPO", "UPSTREAM_COMMIT", "UPSTREAM_FILES",
           "EXPECTED_UPSTREAM_SHA256"]

UPSTREAM_REPO = "https://github.com/lucas-maes/le-wm.git"
UPSTREAM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
UPSTREAM_FILES = ("jepa.py", "module.py", "config/train/model/lewm.yaml")
# sha256 of the three upstream files at UPSTREAM_COMMIT (the bundle records the same values in "upstream_sha256").
EXPECTED_UPSTREAM_SHA256 = {
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "config/train/model/lewm.yaml": "7be97eaa2c83f809b9ea7a6da5a7d3f3c70502c27537383f54db32335e21e42b",
}
BUNDLE_KEYS = ("model_state_dict", "model_config", "training_config", "action_normalization", "pixel_mean",
               "pixel_std", "resize", "control_dt", "upstream_sha256")

HISTORY = 3
LATENT_DIM = 192
IMAGE_SIZE = 224
ACTION_DIM = 4
ACTION_COLUMNS = (0, 1, 2, 4)                       # of the 5 recorded simulator commands
ACTION_NAMES = ("dx", "dy", "dz", "dgripper")
YAW_COLUMN = 3                                      # always 0 in this scene (yaw fixed at 90 deg)
ACTION_ABS_MAX = 1.0 + 1e-7                         # the simulator's normalised command range
PIXEL_MEAN = (0.485, 0.456, 0.406)                  # ImageNet statistics (also the letterbox fill colour)
PIXEL_STD = (0.229, 0.224, 0.225)
LETTERBOX_FILL = tuple(round(value * 255) for value in PIXEL_MEAN)   # (124, 116, 104)

TILT_HEAD_FORMAT = "lewm_tilt_head_v1"
TILT_SCALE_DEG = 30.0
TILT_OUTPUTS = ("tilt_deg / 30", "unsafe_logit")
UNSAFE_ROW = 1
LATENT_INTERFACE = "model.encode({'pixels': normalized_rgb})['emb']; post-projector"


# ------------------------------------------------------------------------------------------------- helpers
def sha256_file(path):
    """Hex sha256 of a file (1 MiB blocks)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float32(values, name, shape):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def normalize_actions(raw, mean, scale):
    """Raw simulator commands -> the world model's 4 standardised action channels (float32).

    ``raw``: ``[N, 5]`` recorded commands ``[dx, dy, dz, yaw, dgripper]`` (the yaw column must be 0) or ``[N, 4]``
    already reduced to ``[dx, dy, dz, dgripper]``; every value must lie in [-1, 1] (normalised commands, NOT
    metres: dx/dy/dz = +-1 -> +-0.05 m per 0.05 s step). ``mean`` / ``scale``: the bundle's TRAIN statistics.
    Arithmetic is float32 numpy, identical to the training cache and the project's evaluation tools.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] not in (4, 5):
        raise ValueError(f"Expected [steps, 5] raw commands or [steps, 4] (dx, dy, dz, dgripper); got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite action")
    if np.any(np.abs(raw) > ACTION_ABS_MAX):
        raise ValueError("Actions must be the simulator's normalised commands in [-1, 1] (not metres, not already "
                         "standardised WM actions)")
    if raw.shape[1] == 5:
        if np.any(raw[:, YAW_COLUMN] != 0):
            raise ValueError(f"Column {YAW_COLUMN} is the yaw command, fixed to 0 in this scene (yaw 90 deg); "
                             "refusing to drop a non-zero yaw silently")
        raw = raw[:, list(ACTION_COLUMNS)]
    mean = _as_float32(mean, "mean", (ACTION_DIM,))
    scale = _as_float32(scale, "scale", (ACTION_DIM,))
    return ((raw - mean) / scale).astype(np.float32)


def denormalize_actions(wm_actions, mean, scale):
    """Inverse of :func:`normalize_actions` -> ``[N, 4]`` raw ``[dx, dy, dz, dgripper]`` commands (float32)."""
    wm_actions = np.asarray(wm_actions, dtype=np.float32)
    if wm_actions.ndim != 2 or wm_actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected [steps, {ACTION_DIM}] WM actions")
    mean = _as_float32(mean, "mean", (ACTION_DIM,))
    scale = _as_float32(scale, "scale", (ACTION_DIM,))
    return (wm_actions * scale + mean).astype(np.float32)


def action_sequence(history_actions, future_actions):
    """``[a[t-2], a[t-1]]`` (2 x 4) + ``K`` future WM actions -> the ``[2 + K, 4]`` timeline ``rollout`` expects."""
    history_actions = np.asarray(history_actions, np.float32)
    future_actions = np.asarray(future_actions, np.float32)
    if history_actions.shape != (HISTORY - 1, ACTION_DIM):
        raise ValueError(f"Need exactly the {HISTORY - 1} WM actions recorded before the newest context frame")
    if future_actions.ndim != 2 or future_actions.shape[1] != ACTION_DIM or not len(future_actions):
        raise ValueError(f"future_actions must be [K >= 1, {ACTION_DIM}]")
    return np.concatenate([history_actions, future_actions])


def letterbox(image_uint8, size=IMAGE_SIZE):
    """Resize a raw camera image to the cache format: ``HxWx3`` uint8 RGB -> ``size x size x 3`` uint8.

    Port of the project's cache conversion (bundle ``resize`` = "LANCZOS letterbox; no crop; RGB ImageNet-mean
    padding"): the whole image is kept, scaled by ``size / max(W, H)`` with Pillow's LANCZOS filter, and pasted
    centred on a canvas filled with the ImageNet mean colour (124, 116, 104). For the training camera (640x480)
    the content occupies 224x168 pixels at rows 28..195, columns 0..223, with mean-coloured bands above and below.
    Feed the result to :meth:`WorldModel.encode`. The camera must be the simulator's fixed front view (position
    [0.70125, 0, 0.324], fovy 48 deg, no overlay, shadows off); Pillow's LANCZOS kernel can differ by a few LSBs
    between Pillow versions (the project used Pillow 12.3.0).
    """
    from PIL import Image

    image_uint8 = np.asarray(image_uint8)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3 or image_uint8.dtype != np.uint8:
        raise ValueError("Expected an HxWx3 uint8 RGB image")
    if size < 1:
        raise ValueError("size must be positive")
    image = Image.fromarray(image_uint8).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), LETTERBOX_FILL)
    canvas.paste(image, ((size - shape[0]) // 2, (size - shape[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


# ------------------------------------------------------------------------------------------------- tilt head
class TiltHead(nn.Module):
    """``192 -> hidden -> hidden -> n_outputs`` MLP with input standardisation buffers; row 0 = tilt_deg / 30,
    row 1 = unsafe logit. ``forward(z) = net((z - mean) / scale)`` exactly as the project's ``TiltHead``."""

    def __init__(self, latent_dim=LATENT_DIM, hidden=256, n_outputs=2):
        super().__init__()
        self.latent_dim, self.hidden, self.n_outputs = int(latent_dim), int(hidden), int(n_outputs)
        self.register_buffer("mean", torch.zeros(self.latent_dim))
        self.register_buffer("scale", torch.ones(self.latent_dim))
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.n_outputs))

    def forward(self, latent):
        if latent.shape[-1] != self.latent_dim:
            raise ValueError("Wrong latent dimension/interface")
        return self.net((latent.float() - self.mean) / self.scale)


def _load_head(path, bundle_sha256, upstream_sha256, device):
    """Load a ``lewm_tilt_head_v1`` file and verify that it was trained on THIS bundle's latents."""
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != TILT_HEAD_FORMAT:
        raise ValueError(f"{path}: not a {TILT_HEAD_FORMAT} file")
    if int(payload.get("latent_dim", 0)) != LATENT_DIM or int(payload.get("n_outputs", 2)) < 2:
        raise ValueError(f"{path}: unexpected head geometry")
    outputs = tuple(payload.get("outputs") or TILT_OUTPUTS)
    if outputs[:2] != TILT_OUTPUTS or int(payload.get("unsafe_row", UNSAFE_ROW)) != UNSAFE_ROW \
            or float(payload.get("tilt_scale_deg", 0)) != TILT_SCALE_DEG:
        raise ValueError(f"{path}: head rows are not (tilt_deg / 30, unsafe_logit)")
    binding = payload.get("binding") or {}
    if binding.get("wm_checkpoint_sha256") != bundle_sha256:
        raise RuntimeError("This tilt head belongs to a different world model: it is bound to bundle sha256 "
                           f"{binding.get('wm_checkpoint_sha256')} but model_state.pt has sha256 {bundle_sha256}")
    if binding.get("upstream_sha256") != upstream_sha256:
        raise RuntimeError("This tilt head was trained with different upstream LeWM code than the loaded bundle")
    if binding.get("latent_interface") != LATENT_INTERFACE:
        raise RuntimeError(f"This tilt head reads a different latent interface: {binding.get('latent_interface')!r}")
    head = TiltHead(payload["latent_dim"], payload["hidden"], int(payload.get("n_outputs", 2)))
    head.load_state_dict(payload["state_dict"], strict=True)
    head.requires_grad_(False).eval().to(device)
    info = {"path": str(path), "sha256": sha256_file(path), "outputs": list(outputs), "binding": dict(binding),
            "label_rule": payload.get("label_rule"), "provenance": dict(payload.get("provenance") or {}),
            "score_is_calibrated_probability": bool(payload.get("score_is_calibrated_probability", False))}
    return head, info


# ------------------------------------------------------------------------------------------------- loading
def _check_upstream(upstream_dir, expected):
    """Fingerprint the three upstream files; raise a clear error naming the expected commit on any difference."""
    upstream_dir = Path(upstream_dir).expanduser().resolve()
    for relative in UPSTREAM_FILES:
        if not (upstream_dir / relative).is_file():
            raise FileNotFoundError(
                f"Missing upstream LeWM file {upstream_dir / relative}. --upstream must point at a checkout of "
                f"{UPSTREAM_REPO} at commit {UPSTREAM_COMMIT}: git clone {UPSTREAM_REPO} && "
                f"git -C le-wm checkout {UPSTREAM_COMMIT}")
    fingerprints = {relative: sha256_file(upstream_dir / relative) for relative in UPSTREAM_FILES}
    if expected is not None and fingerprints != dict(expected):
        differing = [f"{rel} (expected sha256 {expected.get(rel)}, found {fingerprints[rel]})"
                     for rel in UPSTREAM_FILES if fingerprints[rel] != expected.get(rel)]
        raise RuntimeError(
            "Upstream LeWM code differs from the code this bundle was trained with. Expected git commit "
            f"{UPSTREAM_COMMIT} of {UPSTREAM_REPO}; mismatching files: {'; '.join(differing)}. "
            f"Fix: git clone {UPSTREAM_REPO} && git -C le-wm checkout {UPSTREAM_COMMIT}")
    return upstream_dir, fingerprints


def _import_upstream(upstream_dir):
    """Import ``module`` and ``jepa`` from the upstream checkout (same rules as the project's loader): an already
    imported module of the same name is accepted only if it comes from this very checkout."""
    upstream_dir = Path(upstream_dir).resolve()
    for name in ("jepa", "module"):
        if name in sys.modules:
            existing = Path(getattr(sys.modules[name], "__file__", "") or "").resolve()
            if existing != upstream_dir / f"{name}.py":
                raise RuntimeError(f"Module name collision: {name!r} is already imported from {existing}; "
                                   f"lewm_s1 needs {upstream_dir / (name + '.py')}")
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    module = importlib.import_module("module")
    jepa = importlib.import_module("jepa")
    if Path(module.__file__).resolve() != upstream_dir / "module.py" or Path(jepa.__file__).resolve() != upstream_dir / "jepa.py":
        raise RuntimeError("Imported a different LeWM checkout than the one fingerprinted")
    return module, jepa


def _check_architecture(upstream_dir, bundle_model_config, cfg):
    """Re-resolve the upstream yaml exactly like the training code did and require equality with the bundle."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({"img_size": cfg["image_size"], "embed_dim": cfg["embed_dim"],
                               "history_size": cfg["history_size"]})
    config.model = OmegaConf.load(Path(upstream_dir) / "config/train/model/lewm.yaml")
    config.model.action_encoder.input_dim = cfg["action_dim"]
    if config.model.encoder.get("pretrained", True):
        raise ValueError("The upstream model config requests pretrained weights; this bundle was trained from scratch")
    resolved = OmegaConf.to_container(config.model, resolve=True)
    if resolved != bundle_model_config:
        raise RuntimeError("Upstream config/train/model/lewm.yaml resolves to a different architecture than the "
                           "bundle's model_config")


def _build_transformers_encoder(encoder_config):
    """The encoder ``stable_pretraining.backbone.utils.vit_hf`` builds, constructed directly with ``transformers``
    (same ViTConfig, no stable_pretraining import). Only the 'tiny' / pretrained=False case is supported."""
    from transformers import ViTConfig, ViTModel

    expected = {"_target_": "stable_pretraining.backbone.utils.vit_hf", "size": "tiny", "patch_size": 14,
                "image_size": IMAGE_SIZE, "pretrained": False, "use_mask_token": False}
    if dict(encoder_config) != expected:
        raise RuntimeError(f"encoder_builder='transformers' supports only {expected}, bundle has {dict(encoder_config)}")
    config = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=192 * 4,
                       image_size=IMAGE_SIZE, patch_size=14)
    model = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
    model.config.interpolate_pos_encoding = True
    return model


def load_world_model(bundle_path, upstream_dir, device="cpu", head_path=None, encoder_builder="hydra"):
    """Load ``model_state.pt`` into the upstream JEPA module and wrap it.

    ``bundle_path``: the self-contained bundle (state dict + hydra model config + preprocessing contract).
    ``upstream_dir``: a checkout of https://github.com/lucas-maes/le-wm.git at commit 8edfeb33...; its ``jepa.py``,
    ``module.py`` and ``config/train/model/lewm.yaml`` are fingerprinted against the bundle before anything is
    imported, so a different commit fails loudly.
    ``head_path``: optional ``heads/tilt_head.pt`` (``lewm_tilt_head_v1``), verified to be bound to this bundle.
    ``encoder_builder``: ``"hydra"`` (default) instantiates the whole model tree from the bundle's ``model_config``
    with ``hydra.utils.instantiate`` (this imports ``stable_pretraining``, whose ``vit_hf`` builds the ViT);
    ``"transformers"`` builds the identical ``transformers.ViTModel`` directly and instantiates the rest with
    hydra, avoiding the stable-pretraining dependency. Both paths load the same state dict strictly.
    The model is returned in float32 eval mode with gradients disabled; TF32 matmuls are switched off globally.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if encoder_builder not in ("hydra", "transformers"):
        raise ValueError("encoder_builder must be 'hydra' or 'transformers'")
    device = torch.device(device)
    bundle_path = Path(bundle_path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    missing = [k for k in BUNDLE_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"{bundle_path} lacks bundle keys {missing}")
    cfg = bundle["training_config"]
    expected_cfg = {"history_size": HISTORY, "embed_dim": LATENT_DIM, "image_size": IMAGE_SIZE, "action_dim": ACTION_DIM}
    wrong = {k: cfg.get(k) for k, v in expected_cfg.items() if cfg.get(k) != v}
    if wrong:
        raise ValueError(f"Bundle training_config differs from the S1 contract {expected_cfg}: {wrong}")
    if list(bundle["pixel_mean"]) != list(PIXEL_MEAN) or list(bundle["pixel_std"]) != list(PIXEL_STD):
        raise ValueError("Bundle pixel statistics are not the ImageNet values this loader hard-codes")
    norm = bundle["action_normalization"]
    if list(norm.get("names", [])) != list(ACTION_NAMES) or norm.get("fit_split") != "train":
        raise ValueError("Unexpected action normalisation contract in the bundle")

    upstream_dir, fingerprints = _check_upstream(upstream_dir, bundle["upstream_sha256"])
    module, jepa = _import_upstream(upstream_dir)
    _check_architecture(upstream_dir, bundle["model_config"], cfg)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    model_config = OmegaConf.create(bundle["model_config"])
    if model_config.get("_target_") != "jepa.JEPA":
        raise RuntimeError("Bundle model_config does not target jepa.JEPA")
    if encoder_builder == "hydra":
        model = instantiate(model_config)
    else:
        encoder = _build_transformers_encoder(bundle["model_config"]["encoder"])
        model = instantiate(model_config, encoder=encoder)
    if not isinstance(model, jepa.JEPA):
        raise RuntimeError("Instantiated model is not the upstream JEPA class")
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.float().requires_grad_(False)
    model.to(device).eval()
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Model must be frozen in eval mode")

    bundle_sha256 = sha256_file(bundle_path)
    head = head_info = None
    if head_path is not None:
        head, head_info = _load_head(head_path, bundle_sha256, fingerprints, device)
    return WorldModel(model, bundle, bundle_path, bundle_sha256, upstream_dir, fingerprints, device, head, head_info,
                      encoder_builder)


# ------------------------------------------------------------------------------------------------- wrapper
class WorldModel:
    """Frozen S1 world model + optional tilt head with numpy in / numpy out helpers (see module docstring)."""

    def __init__(self, model, bundle, bundle_path, bundle_sha256, upstream_dir, upstream_sha256, device, head,
                 head_info, encoder_builder):
        self.model = model
        self.device = torch.device(device)
        self.image_size, self.history, self.latent_dim, self.action_dim = IMAGE_SIZE, HISTORY, LATENT_DIM, ACTION_DIM
        self.control_dt = float(bundle["control_dt"])
        self.resize = str(bundle["resize"])
        self.pixel_mean = torch.tensor(list(bundle["pixel_mean"]), dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(list(bundle["pixel_std"]), dtype=torch.float32).view(1, 3, 1, 1)
        self.action_normalization = dict(bundle["action_normalization"])
        self.action_mean = _as_float32(self.action_normalization["mean"], "mean", (ACTION_DIM,))
        self.action_scale = _as_float32(self.action_normalization["scale"], "scale", (ACTION_DIM,))
        self.model_config = bundle["model_config"]
        self.training_config = bundle["training_config"]
        self.training_budget = bundle.get("training_budget")
        self.bundle_path, self.bundle_sha256 = Path(bundle_path), bundle_sha256
        self.upstream_dir, self.upstream_sha256 = Path(upstream_dir), dict(upstream_sha256)
        self.upstream_commit = UPSTREAM_COMMIT
        self.latent_interface = LATENT_INTERFACE
        self.encoder_builder = encoder_builder
        self.head, self.head_info = head, head_info
        self.head_unsafe_threshold = None
        if head_info is not None:
            threshold = head_info["provenance"].get("unsafe_threshold")
            self.head_unsafe_threshold = None if threshold is None else float(threshold)

    # ---- images -> latents
    def encode_torch(self, pixels):
        """Normalised pixels ``[N, 3, H, W]`` or ``[N, T, 3, H, W]`` (float, already (x - mean) / std) -> post-projector
        latents ``[N, 192]`` / ``[N, T, 192]`` (float32, no autocast)."""
        if pixels.ndim == 4:
            return self.encode_torch(pixels[:, None])[:, 0]
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("Expected [N, 3, H, W] or [N, T, 3, H, W] normalised pixels")
        if self.model.training:
            raise RuntimeError("Model left eval mode")
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            emb = self.model.encode({"pixels": pixels.to(self.device)})["emb"].detach().float()
        if not bool(torch.isfinite(emb).all()):
            raise FloatingPointError("Nonfinite latent")
        return emb

    def encode(self, frames, batch_size=None):
        """uint8 RGB frames ``[N, 224, 224, 3]`` (letterboxed like the cache) -> float32 latents ``[N, 192]``.

        Pipeline (bit for bit the project's ``encode_frames``): ``permute`` to channel-first, ``.float() / 255``,
        ``(x - pixel_mean) / pixel_std`` with (1, 3, 1, 1) float32 tensors, encode, take the post-projector
        embedding. ``batch_size=None`` encodes all frames in one batch; batching can change CPU kernels and thus
        the last bits (~1e-6).
        """
        rgb = np.asarray(frames)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 [N, H, W, 3] RGB frames")
        if rgb.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError(f"Frames must be {self.image_size}x{self.image_size} (use lewm_s1.letterbox on raw camera images)")
        if not len(rgb):
            return np.zeros((0, self.latent_dim), np.float32)
        batch_size = len(rgb) if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parts = []
        for start in range(0, len(rgb), batch_size):
            pixels = torch.from_numpy(np.ascontiguousarray(rgb[start:start + batch_size])).permute(0, 3, 1, 2).float() / 255
            pixels = ((pixels - self.pixel_mean) / self.pixel_std).contiguous()[:, None]
            z = self.encode_torch(pixels)[:, 0]
            parts.append(z.cpu().numpy())
        return np.concatenate(parts).astype(np.float32, copy=False)

    # ---- actions
    def normalize_actions(self, raw):
        """Raw simulator commands ``[N, 5]`` (or ``[N, 4]`` = dx, dy, dz, dgripper) in [-1, 1] -> WM actions ``[N, 4]``;
        see :func:`normalize_actions`."""
        return normalize_actions(raw, self.action_mean, self.action_scale)

    def denormalize_actions(self, wm_actions):
        return denormalize_actions(wm_actions, self.action_mean, self.action_scale)

    @staticmethod
    def action_sequence(history_actions, future_actions):
        return action_sequence(history_actions, future_actions)

    # ---- dynamics
    def rollout_torch(self, context, actions):
        """``context`` ``[B, 3, 192]``, ``actions`` ``[B, 2 + K, 4]`` (normalised) -> ``[B, K, 192]`` torch tensors.
        Same sliding context as the project's ``rollout_latents``: at step k the newest three states (real
        context, then predictions) and the actions ``a[t-2+k .. t+k]`` give ``zhat[t+1+k]``."""
        if self.model.training:
            raise RuntimeError("Model left eval mode")
        context = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if context.ndim != 3 or context.shape[1] != self.history or context.shape[2] != self.latent_dim:
            raise ValueError(f"context must be [B, {self.history}, {self.latent_dim}]")
        horizon = actions.shape[1] - (self.history - 1) if actions.ndim == 3 else 0
        if actions.ndim != 3 or horizon < 1 or actions.shape != (len(context), self.history - 1 + horizon, self.action_dim):
            raise ValueError(f"actions must be [B, 2 + K, {self.action_dim}] with K >= 1 (the 2 actions before the newest "
                             "context frame followed by the K future actions)")
        history = self.history
        states, predictions = context, []
        with torch.no_grad():
            for k in range(horizon):
                act_emb = self.model.action_encoder(actions[:, :history + k])[:, -history:]
                prediction = self.model.predict(states[:, -history:], act_emb)[:, -1:]
                if prediction.shape != context[:, -1:].shape:
                    raise RuntimeError("Upstream predicted latent shape changed")
                predictions.append(prediction)
                states = torch.cat((states, prediction), dim=1)
        out = torch.cat(predictions, dim=1).float()
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("Nonfinite rollout")
        return out

    def rollout(self, z_ctx, actions):
        """numpy version of :meth:`rollout_torch`: ``[B, 3, 192]`` x ``[B, 2 + K, 4]`` -> float32 ``[B, K, 192]``."""
        z_ctx, actions = np.asarray(z_ctx, np.float32), np.asarray(actions, np.float32)
        return self.rollout_torch(torch.from_numpy(np.ascontiguousarray(z_ctx)),
                                  torch.from_numpy(np.ascontiguousarray(actions))).cpu().numpy().astype(np.float32, copy=False)

    def predict_next(self, z_hist, a_hist):
        """One step: latents ``[B, 3, 192]`` + the aligned normalised actions ``[B, 3, 4]`` (``a[t-2], a[t-1], a[t]``)
        -> ``zhat[t+1]`` ``[B, 192]``; identical to ``rollout(z_hist, a_hist)[:, 0]``."""
        a_hist = np.asarray(a_hist, np.float32)
        if a_hist.ndim != 3 or a_hist.shape[1:] != (self.history, self.action_dim):
            raise ValueError(f"a_hist must be [B, {self.history}, {self.action_dim}]")
        return self.rollout(z_hist, a_hist)[:, 0]

    # ---- tilt head
    def head_outputs(self, z):
        """Raw head rows ``[N, n_outputs]``: column 0 = tilt_deg / 30, column 1 = unsafe logit."""
        if self.head is None:
            raise RuntimeError("No tilt head loaded (pass head_path= to load_world_model)")
        z = np.asarray(z, np.float32)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected [N, {self.latent_dim}] latents")
        with torch.no_grad():
            out = self.head(torch.from_numpy(np.ascontiguousarray(z)).to(self.device)).float()
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError("Nonfinite tilt-head output")
        return out.cpu().numpy()

    def tilt_deg(self, z):
        """Predicted worst-guard tilt in degrees ``[N]`` = 30 * head(z)[:, 0]."""
        if self.head is None:
            raise RuntimeError("No tilt head loaded (pass head_path= to load_world_model)")
        z = np.asarray(z, np.float32)
        with torch.no_grad():
            tilt = (self.head(torch.from_numpy(np.ascontiguousarray(z)).to(self.device))[..., 0] * TILT_SCALE_DEG).float()
        return tilt.cpu().numpy().astype(np.float32)

    def unsafe_prob(self, z):
        """``sigmoid(unsafe logit)`` ``[N]``. NOT a calibrated probability: the head's provenance records the alarm
        threshold ``head_unsafe_threshold`` (0.9608 for the S1 head = 95 % recall on real dev latents)."""
        out = self.head_outputs(z)
        return torch.sigmoid(torch.from_numpy(out[:, UNSAFE_ROW])).numpy().astype(np.float32)

    def __repr__(self):
        return (f"WorldModel(latent_dim={self.latent_dim}, history={self.history}, image_size={self.image_size}, "
                f"control_dt={self.control_dt}, device={self.device}, upstream_commit={self.upstream_commit[:8]}, "
                f"head={'yes' if self.head is not None else 'no'}, encoder_builder={self.encoder_builder!r})")
