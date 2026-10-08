"""Serializable websocket bridge for GroundingPI QwenPI.

GroundingPI's upstream server calls QwenPI.predict_action directly.  That
method returns a CPU torch.Tensor, while the retained msgpack serializer only
knows NumPy arrays.  This bridge keeps the upstream model untouched and converts
nested tensors before handing the response to the upstream websocket server.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


def _numpy(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, dict):
        return {key: _numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_numpy(item) for item in value)
    return value


class SerializablePolicy:
    def __init__(self, policy):
        self.policy = policy

    def predict_action(self, **payload):
        return _numpy(self.policy.predict_action(**payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundingpi-root", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    args = parser.parse_args()

    root = str(args.groundingpi_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from starVLA.model.framework.base_framework import baseframework

    policy = baseframework.from_pretrained(args.ckpt_path)
    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to(args.device).eval()
    WebsocketPolicyServer(
        policy=SerializablePolicy(policy),
        host=args.host,
        port=args.port,
        metadata={"framework": "GroundingPI_QwenPI", "returns": "normalized_actions"},
    ).serve_forever()


if __name__ == "__main__":
    main()
