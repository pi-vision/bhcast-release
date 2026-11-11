#!/usr/bin/env bash
# Run the registered measurements on one local GPU, with a wall-time cap.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg WANDB_MODE=disabled
python=${BHCAST_AUDIT_PYTHON:-python}
test "$(git -C source/release rev-parse HEAD)" = 7cb2f93423f49fc4edd05cad97f9e643c7202e53
git -C source/release diff --exit-code HEAD
checkpoint=${BHCAST_CHECKPOINT:-source/release/checkpoints/model.pt}
printf '%s  %s\n' 96449f1d1a6acc38b59e329196f2541447b67f40052de404a11322c9dd5bc46e "$checkpoint" | sha256sum --check
mkdir -p results
exec timeout --kill-after=30s 1h "$python" -u "$@"
