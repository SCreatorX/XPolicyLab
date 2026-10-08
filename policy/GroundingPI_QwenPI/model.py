from __future__ import annotations

"""XPolicyLab adapter for the GroundingPI QwenPI websocket server."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

_POLICY_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = Path("/home/admin1/GroundingPI/vla/starvla")


def _none(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve(value: Any, *bases: Path) -> Path | None:
    if _none(value):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return bases[0] / path


def _image(obs: dict[str, Any], name: str) -> np.ndarray:
    value = obs.get("vision", {}).get(name)
    if isinstance(value, dict):
        value = next((value[key] for key in ("color", "rgb", "colors") if key in value), None)
    if value is None:
        raise KeyError(f"missing camera {name!r}")
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"camera {name!r} must be 3-D, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"camera {name!r} must have three channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * (255.0 if float(np.nanmax(image)) <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image, mode="RGB").resize(size, Image.BILINEAR), dtype=np.uint8)


def _stats_root(raw: dict[str, Any], key: str | None) -> dict[str, Any]:
    if "state" in raw or "action" in raw:
        return raw
    if key is not None:
        value = raw.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"unnorm_key {key!r} is absent from {list(raw)}")
        return value
    if len(raw) != 1:
        raise ValueError(f"unnorm_key is required for statistics {list(raw)}")
    value = next(iter(raw.values()))
    if not isinstance(value, dict):
        raise ValueError("selected normalization statistics is not an object")
    return value


def _block(stats: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if stats is None or not isinstance(stats.get(name), dict):
        raise KeyError(f"normalization statistics do not contain {name!r}")
    return stats[name]


def _broadcast(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    try:
        return np.broadcast_to(array, shape).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(f"{label} shape {array.shape} cannot broadcast to {shape}") from exc


def _transform(values: np.ndarray, stats: dict[str, Any], mode: str, inverse: bool, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if mode == "q99":
        low = _broadcast(stats["q01"], values.shape, f"{label}.q01")
        high = _broadcast(stats["q99"], values.shape, f"{label}.q99")
        if inverse:
            values = np.clip(values, -1.0, 1.0)
            mask = np.asarray(stats.get("mask", True), dtype=bool)
            return np.where(mask, 0.5 * (values + 1.0) * (high - low) + low, values)
        result = np.zeros_like(values)
        valid = high != low
        result[valid] = 2.0 * (values[valid] - low[valid]) / (high[valid] - low[valid]) - 1.0
        return np.clip(result, -1.0, 1.0)
    if mode == "mean_std":
        mean = _broadcast(stats["mean"], values.shape, f"{label}.mean")
        std = _broadcast(stats["std"], values.shape, f"{label}.std")
        return values * std + mean if inverse else np.divide(values - mean, std, out=np.zeros_like(values), where=std != 0)
    if mode == "min_max":
        low = _broadcast(stats["min"], values.shape, f"{label}.min")
        high = _broadcast(stats["max"], values.shape, f"{label}.max")
        return 0.5 * (values + 1.0) * (high - low) + low if inverse else np.divide(values - low, high - low, out=np.zeros_like(values), where=high != low) * 2.0 - 1.0
    if mode == "binary":
        return (values > 0.5).astype(np.float32)
    raise ValueError(f"unsupported {label} normalization mode {mode!r}")


class Model(ModelTemplate):
    """XPL-side QwenPI client; the Qwen3 model remains in the external server."""

    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        self.action_type = str(self.cfg.get("action_type") or "joint")
        self.env_cfg_type = self.cfg.get("env_cfg_type")
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type is required")
        if self.action_type not in {"joint", "ee"}:
            raise ValueError(f"unsupported action_type={self.action_type!r}")
        self.dims = dict(get_robot_action_dim_info(self.env_cfg_type))
        if self.action_type == "ee":
            self.dims["arm_dim"] = [7] * len(self.dims["arm_dim"])
        self.action_dim = sum(self.dims["arm_dim"]) + sum(self.dims["ee_dim"])
        self.cameras = list(self.cfg.get("camera_names") or ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        size = self.cfg.get("image_size", [224, 224])
        self.image_size = (int(size[0]), int(size[1]))
        self.execute_horizon = int(self.cfg.get("execute_horizon", 16))
        if self.execute_horizon < 1:
            raise ValueError("execute_horizon must be positive")
        self.include_state = _bool(self.cfg.get("include_state"), True)
        self.server_returns_normalized = _bool(self.cfg.get("server_returns_normalized"), True)
        # GroundingPI's default StateActionTransform is mean_std; q99/min_max
        # remain configurable for checkpoints trained with those schemes.
        self.state_norm_mode = str(self.cfg.get("state_norm_mode") or "mean_std")
        self.action_norm_mode = str(self.cfg.get("action_norm_mode") or "mean_std")
        self.allow_dummy_policy = _bool(self.cfg.get("allow_dummy_policy"), False)
        self.stats = None
        stats_path = _resolve(self.cfg.get("dataset_statistics_path"), _POLICY_DIR, Path.cwd())
        if stats_path is not None:
            with stats_path.open("r", encoding="utf-8") as handle:
                self.stats = _stats_root(json.load(handle), None if _none(self.cfg.get("unnorm_key")) else self.cfg["unnorm_key"])
        if self.include_state and self.stats is None and not self.allow_dummy_policy:
            raise ValueError("include_state=true requires dataset_statistics_path")
        if self.server_returns_normalized and self.stats is None and not self.allow_dummy_policy:
            raise ValueError("server_returns_normalized=true requires dataset_statistics_path")
        self.client = None if self.allow_dummy_policy else self._connect()
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.pending: dict[int, list[np.ndarray]] = {}
        self.latest = [0]
        print(f"[GroundingPI_QwenPI] action_dim={self.action_dim}, action_type={self.action_type}, include_state={self.include_state}")

    def _connect(self):
        root = (_resolve(self.cfg.get("groundingpi_root"), _POLICY_DIR, Path.cwd()) or _DEFAULT_ROOT).resolve()
        if not (root / "deployment/model_server/tools/websocket_policy_client.py").is_file():
            raise FileNotFoundError(f"GroundingPI websocket client not found below {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        host = str(self.cfg.get("groundingpi_server_host") or "127.0.0.1")
        port = int(self.cfg.get("groundingpi_server_port") or 5694)
        client = WebsocketClientPolicy(host, port)
        print(f"[GroundingPI_QwenPI] connected to ws://{host}:{port}")
        return client

    def _state(self, obs):
        # XPL's public runtime schema calls the single-arm joint field
        # ``arm_joint_state``; the shared pack helper retains the historical
        # ``joint_state`` spelling for that one case.  Alias only at this
        # boundary so the helper remains the source of dimension/order truth.
        if len(self.dims["arm_dim"]) == 1 and self.action_type == "joint":
            state = dict(obs.get("state", {}))
            if "joint_state" not in state and "arm_joint_state" in state:
                state["joint_state"] = state["arm_joint_state"]
            obs = dict(obs)
            obs["state"] = state
        raw = np.asarray(pack_robot_state(obs, self.action_type, self.dims, source_type="obs"), dtype=np.float32).reshape(1, -1)
        return raw if self.stats is None else _transform(raw, _block(self.stats, "state"), self.state_norm_mode, False, "state")

    def _unpack_action(self, action):
        result = unpack_robot_state(action, self.action_type, self.dims, source_type="obs")
        if len(self.dims["arm_dim"]) == 1 and self.action_type == "joint":
            result["arm_joint_state"] = result.pop("joint_state")
        return result

    def _action(self, vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.action_dim:
            raise ValueError(f"server action_dim={vector.shape[0]}, XPL expects {self.action_dim}")
        return vector.astype(np.float32, copy=False)

    def _dummy_chunk(self):
        vector = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_type == "ee":
            offset = 0
            for arm_dim, ee_dim in zip(self.dims["arm_dim"], self.dims["ee_dim"]):
                if arm_dim != 7:
                    raise ValueError(f"dummy ee mode expects 7-D poses, got {arm_dim}")
                vector[offset + 3] = 1.0
                offset += arm_dim + ee_dim
        return [vector.copy() for _ in range(self.execute_horizon)]

    def _infer(self, env_ids):
        if self.allow_dummy_policy:
            for env_id in env_ids:
                self.pending[env_id] = self._dummy_chunk()
            return
        observations = [self.obs_by_env[env_id] for env_id in env_ids]
        payload = {
            "batch_images": [[_resize(_image(obs, camera), self.image_size) for camera in self.cameras] for obs in observations],
            "instructions": [str(obs.get("instruction") or obs.get("instructions") or self.cfg.get("task_name") or "follow the instruction") for obs in observations],
        }
        if self.include_state:
            payload["state"] = np.stack([self._state(obs) for obs in observations], axis=0).astype(np.float32)
        response = self.client.infer(payload)
        if not response.get("ok", False):
            raise RuntimeError(f"GroundingPI inference failed: {response.get('error', response)}")
        data = response.get("data") or {}
        actions = data.get("normalized_actions", data.get("actions"))
        if actions is None:
            raise RuntimeError("GroundingPI response has neither normalized_actions nor actions")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        if actions.ndim != 3 or actions.shape[0] != len(env_ids) or actions.shape[1] == 0:
            raise RuntimeError(f"expected action shape (B={len(env_ids)}, T, D), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("GroundingPI returned non-finite actions")
        if actions.shape[-1] != self.action_dim:
            raise RuntimeError(f"QwenPI action_dim={actions.shape[-1]} does not match XPL dim={self.action_dim}; retrain the head")
        for index, env_id in enumerate(env_ids):
            chunk = actions[index, : self.execute_horizon]
            if self.server_returns_normalized:
                # Keep the time axis here: GroundingPI may store per-position
                # chunk statistics with shape [T, D], not only [D].
                chunk = _transform(
                    chunk, _block(self.stats, "action"), self.action_norm_mode, True, "action"
                )
            self.pending[env_id] = [self._action(action) for action in chunk]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.latest = []
        for index, obs in enumerate(obs_list):
            env_id = int(obs.get("env_idx", index))
            self.obs_by_env[env_id] = obs
            self.latest.append(env_id)

    def get_action(self):
        return self.get_action_batch([self.latest[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        env_idx_list = list(self.latest if env_idx_list is None else env_idx_list)
        if not env_idx_list:
            return []
        missing = [env_id for env_id in env_idx_list if env_id not in self.obs_by_env]
        if missing:
            raise ValueError(f"missing observations for env indices {missing}")
        needs_inference = [env_id for env_id in env_idx_list if not self.pending.get(env_id)]
        if needs_inference:
            self._infer(needs_inference)
        result = []
        for env_id in env_idx_list:
            action = self.pending[env_id].pop(0)
            result.append([self._unpack_action(action)])
        return result

    def reset(self):
        self.obs_by_env.clear()
        self.pending.clear()
        self.latest = [0]

    def __del__(self):
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
