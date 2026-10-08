#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'MSG'
GroundingPI_QwenPI is evaluation-only in XPolicyLab. Train the QwenPI/PI head
with GroundingPI's upstream scripts, then pass its checkpoint to this adapter.
MSG
exit 2
