#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'MSG'
GroundingPI_QwenPI is evaluation-only in XPolicyLab. Dataset conversion remains in
GroundingPI's upstream training pipeline; no XPL process_data wrapper is provided.
MSG
exit 2
