#!/usr/bin/env bash
# Run every sanity check, then build + execute the report notebook.
#   bash run_all.sh            (N_TRIALS=100 by default for the valid/test swap)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$(command -v python)}"   # run with the deepmol env active
N_TRIALS="${N_TRIALS:-100}"
mkdir -p logs

$PY 00_featurize.py > logs/00_featurize.log 2>&1
$PY 01_stereochemistry.py > logs/01.log 2>&1 &
$PY 02_split_and_standardize.py > logs/02.log 2>&1 &
$PY 03_mislabel_check.py > logs/03.log 2>&1 &
$PY 04_search_inspection.py > logs/04.log 2>&1 &
$PY 05_tsne.py > logs/05.log 2>&1 &
wait
$PY 06_rf_repeat.py > logs/06.log 2>&1

# 4 searches in parallel, 16 cores each
for th in 0.2 0.3; do
  for sel in valid test; do
    SANITY_N_JOBS=16 $PY 07_swap_valid_test.py --threshold $th --select-on $sel \
      --n-trials "$N_TRIALS" > "logs/07_th${th}_${sel}.log" 2>&1 &
  done
done
wait

$PY build_notebook.py
