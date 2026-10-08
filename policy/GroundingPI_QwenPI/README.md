# GroundingPI_QwenPI

**Contributor:** XPolicyLab integration | **Paper:** GroundingPI / StarVLA | **Original code:** `/home/admin1/GroundingPI/vla/starvla`

This policy adapts GroundingPI's Qwen3-based `QwenPI` VLA to the XPolicyLab policy-server interface. The Qwen3 model and PI action head stay in the GroundingPI process. XPolicyLab sends decoded RGB camera arrays, language instructions, and (when enabled) normalized robot state over the upstream websocket protocol, then converts the returned action chunk into XPL action dictionaries.

The adapter follows the shared conventions in the [XPolicyLab README](../../README.md). The extra `groundingpi_server.py` bridge is needed because the upstream QwenPI implementation returns a `torch.Tensor`, while its websocket msgpack serializer supports NumPy arrays.

## Installation

Install the XPL-side adapter in the environment that runs the policy server:

```bash
cd /home/admin1/XPolicyLab-groundingpi/policy/GroundingPI_QwenPI
bash install.sh
```

GroundingPI itself needs its own environment with the Qwen3/VLM, PyTorch and any CUDA extensions required by `/home/admin1/GroundingPI/vla/starvla`. `install.sh` intentionally does not replace that environment.

## Data processing and training

This integration is evaluation-only. `process_data.sh` and `train.sh` exit with a clear message because dataset conversion and QwenPI training remain GroundingPI-native. The checkpoint must have an action head whose output dimension equals the selected XPL `env_cfg_type` and `action_type`; the adapter fails fast on a mismatch.

GroundingPI statistics are required when the server returns normalized actions or when state is sent. Set `GROUNDINGPI_DATASET_STATS_PATH` to the matching `dataset_statistics.json` and set `GROUNDINGPI_UNNORM_KEY` when that file contains multiple dataset entries. The adapter currently supports `mean_std`, `q99`, `min_max`, and `binary` state/action transforms, including per-position chunk statistics. Complex modality-specific relative-pose or rotation unapply rules still need to be added if the selected checkpoint uses them.

## Evaluation

Use the standard XPL arguments:

```bash
cd /home/admin1/XPolicyLab-groundingpi/policy/GroundingPI_QwenPI
bash eval.sh <bench> <task> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_python> <eval_env>
```

For a wiring-only run, the policy can use the built-in zero-action path (identity quaternion for EE poses). This does not exercise Qwen3 inference:

```bash
EVAL_ENV_TYPE=debug \
GROUNDINGPI_ALLOW_DUMMY_POLICY=true \
GROUNDINGPI_INCLUDE_STATE=false \
bash eval.sh RoboDojo stack_bowls debug arx_x5 joint 0 0 0 <policy_env> <eval_env>
```

To have the XPL policy-server script start the GroundingPI bridge, provide a real checkpoint and matching statistics:

```bash
GROUNDINGPI_SERVER_AUTOSTART=1 \
GROUNDINGPI_CHECKPOINT_PATH=/path/to/qwenpi_checkpoint.pt \
GROUNDINGPI_DATASET_STATS_PATH=/path/to/dataset_statistics.json \
GROUNDINGPI_UNNORM_KEY=<dataset_key_if_needed> \
bash eval.sh RoboDojo stack_bowls qwenpi arx_x5 joint 0 0 0 <policy_env> <eval_env>
```

Alternatively, start `groundingpi_server.py` separately and set `GROUNDINGPI_SERVER_HOST` and `GROUNDINGPI_SERVER_PORT`; leave `GROUNDINGPI_SERVER_AUTOSTART=0`.

## Files

- `model.py`: XPL `ModelTemplate` adapter, image/state packing, action denormalization and action-chunk caching.
- `groundingpi_server.py`: serializable websocket bridge around GroundingPI's `baseframework` checkpoint loader.
- `setup_eval_policy_server.sh`: optional bridge startup followed by the normal XPL policy server.
- `deploy.yml`: default camera names, image size, server endpoint and normalization settings.

Real checkpoint inference still needs to be validated in the target parent workspace, where the RoboDojo `env_cfg` package and the QwenPI checkpoint are available.

The policy interpreter (argument 9) may be a conda name, virtual environment directory, or executable path. `GROUNDINGPI_PYTHON` selects the separate model-server interpreter when autostarting. Startup settings can also be set in `deploy.yml`; explicit environment variables take precedence. Checkpoints resolve through the shared XPL helper: explicit checkpoint path, path-valued `ckpt_name`, then the standard `checkpoints/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` layout. For a separate bridge process, run `python groundingpi_server.py --groundingpi-root /path/to/GroundingPI/vla/starvla --ckpt-path /path/to/checkpoint --port 5694 --use-bf16`.

`joint` and `ee` output dictionaries support the single/dual-arm dimensions registered by the parent workspace. EE poses use `[x, y, z, qw, qx, qy, qz]`. The training state/action order must match XPL packing: arm, gripper, then the second arm and gripper for dual-arm robots. Dimension agreement alone does not guarantee compatible action semantics. Camera order, units, absolute/delta convention, and normalization must also match training.
