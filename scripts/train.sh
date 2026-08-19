#!/usr/bin/env bash
# Plug-and-play training wrapper around the research train.py.
#
# Prereqs: run from the research repo root (where train.py lives), with the
# training deps installed:  pip install torch numpy tqdm tensorboard ipython
#
# Usage:
#   ./train.sh <train_config.json> [output_dir] [restore_checkpoint]
#
# Example (after prepare_data.py wrote dataset/ + dataset/train_config.json):
#   ./train.sh dataset/train_config.json runs
set -euo pipefail

CONFIG="${1:?path to train_config.json (emitted by prepare_data.py) required}"
OUTPUT_DIR="${2:-runs}"
RESTORE="${3:-}"

if [[ ! -f train.py ]]; then
  echo "error: train.py not found in $(pwd). Run this from the research repo root." >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "error: config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "> training with config=$CONFIG output=$OUTPUT_DIR ${RESTORE:+restore=$RESTORE}"

if [[ -n "$RESTORE" ]]; then
  python train.py --config_path "$CONFIG" --output_path "$OUTPUT_DIR" --restore_path "$RESTORE"
else
  python train.py --config_path "$CONFIG" --output_path "$OUTPUT_DIR"
fi
