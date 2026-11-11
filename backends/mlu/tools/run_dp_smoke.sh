#!/bin/bash
# Multi-card DP smoke for the GPT-2 example on MLU (single process, N threads).
#
# Usage: run_dp_smoke.sh <nthread_per_process> <total_batch_size>
# Environment:
#   MLU_VISIBLE_DEVICES  cards to use (default: first N cards)
#   GPT2_LLMC_FILE       grouped-layout checkpoint (default: baseline asset path)
#   GPT2_INPUT_BIN       training data .bin        (default: baseline asset path)
#
# Step-1 train loss reference (GPT-2 124M FP32, tiny_shakespeare, llmc-official
# data): ~5.25. A world_size-N DP run must report the same value; if it scales
# with N, the kAvg path of the collective is broken (see CnclImpl::ScaleAvg).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-dp"

NTHREAD="${1:?usage: run_dp_smoke.sh <nthread_per_process> <total_batch_size>}"
TOTAL_BATCH="${2:?usage: run_dp_smoke.sh <nthread_per_process> <total_batch_size>}"

ASSETS="${GPT2_ASSETS_DIR:-/mnt/afs/user_home/wangyichen/infinitrain-baseline}"
GPT2_LLMC_FILE="${GPT2_LLMC_FILE:-$ASSETS/models/gpt2-staging/gpt2_124M_grouped.bin}"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-$ASSETS/data/llmc-official/gpt2/tiny_shakespeare_train.bin}"

if [ -z "${MLU_VISIBLE_DEVICES:-}" ]; then
  MLU_VISIBLE_DEVICES="$(seq -s, 0 $((NTHREAD - 1)))"
fi
export MLU_VISIBLE_DEVICES
export GLOG_minloglevel=2

cd "$BUILD_DIR"
exec ./gpt2 --device privateuse1 \
  --llmc_filepath "$GPT2_LLMC_FILE" \
  --input_bin "$GPT2_INPUT_BIN" \
  --nthread_per_process "$NTHREAD" \
  --total_batch_size "$TOTAL_BATCH" \
  --num_iteration 1
