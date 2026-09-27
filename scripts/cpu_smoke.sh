#!/bin/zsh
# CPU smoke test: ~300 pretraining steps of MiniMind (64M) on a slice of pretrain_t2t_mini.
# Criterion: the loss decreases. Never run while an AgentOS task is running (16 GB Mac).
set -euo pipefail

LAB=${0:A:h:h}
STEPS=${STEPS:-300}
BATCH=${BATCH:-8}
SEQ=${SEQ:-128}
DEVICE=${DEVICE:-cpu}
ENTRY=${ENTRY:-train_pretrain.py}  # scripts/mps_patch.py for float16 autocast on MPS
LOG=${LOG:-$LAB/results/${DEVICE}-smoke.log}
SRC=$LAB/data/pretrain_t2t_mini.head32mb.jsonl
SLICE=$LAB/data/pretrain_smoke_$((STEPS * BATCH)).jsonl

if command -v agentos >/dev/null && agentos task list | grep -qw running; then
  echo "an AgentOS task is running: retry later" >&2
  exit 1
fi

# One epoch over STEPS*BATCH samples == STEPS optimizer steps (accumulation 1): no max-steps flag upstream.
head -n $((STEPS * BATCH)) "$SRC" > "$SLICE"

cd "$LAB/upstream/trainer"  # upstream resolves ../model (tokenizer) from the cwd
nice -n 10 "$LAB/.venv/bin/python" -u "$ENTRY" \
  --device "$DEVICE" --epochs 1 --batch_size "$BATCH" --max_seq_len "$SEQ" \
  --accumulation_steps 1 --log_interval 10 --save_interval 100000 --num_workers 0 \
  --data_path "$SLICE" --save_dir "$LAB/out" --save_weight "smoke_${DEVICE}" "$@" 2>&1 \
  | perl -MTime::HiRes=time -ne 'BEGIN { $| = 1 } printf "%.3f %s", time, $_' | tee "$LOG"
