"""Featurize (atom_pair) every split of both thresholds once; the other scripts read the cache."""
from common import RUNS, SPLITS, featurize_split

for threshold in RUNS:
    for split in SPLITS:
        X, y, _ = featurize_split(threshold, split)
        print(f"th={threshold} {split:5s} X={X.shape} sweet={y.mean():.4f}")
