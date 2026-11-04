#!/usr/bin/env bash
# Launch every config in <repo>/configs/sweet26/predictor/automl/parallel/
# as its own background process
# (one (featurizer, model) pair per process). Does NOT touch the runs that
# are already going.
#
#   bash launch_parallel.sh            # all 20
#   bash launch_parallel.sh sim02      # only the ones matching "sim02"
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$(command -v python)}"   # run with the deepmol env active
FILTER="${1:-}"
CONFIG_DIR="$(cd ../configs/predictor/automl/parallel && pwd)"
mkdir -p logs_parallel

for cfg in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$cfg" .yaml)
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
    if pgrep -f -- "--config $cfg" > /dev/null; then
        echo "skip $name (already running)"
        continue
    fi
    nohup "$PYTHON" automl_pipeline.py --config "$cfg" \
        > "logs_parallel/$name.log" 2>&1 &
    echo "started $name (pid $!)"
    # output_dir = checkpoints/<YYYYmmdd_HHMMSS>_<study_name>: spacing the
    # launches avoids a burst of 20 identical standardize/featurize jobs.
    sleep 5
done
