"""
t-SNE of the data actually used by the atom_pair + RF runs, thresholds 0.2 and 0.3.

Features = the model's own input (atom_pair, 2048 bits), PCA -> 50 dims, then
openTSNE (FFT, all ~90k molecules, no subsampling). One embedding per
threshold, plotted three ways:
  - colored by split
  - one panel per split over the full cloud in gray (where does valid/test sit?)
  - sweet vs non-sweet per split

Outputs: outputs/05_tsne_th{0.2,0.3}_*.png, cache/tsne_th*.npz
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE
from sklearn.decomposition import PCA

from common import CACHE_DIR, N_JOBS, OUT_DIR, RUNS, SPLIT_COLORS, SPLITS, featurize_split

for threshold in RUNS:
    parts = [featurize_split(threshold, s) for s in SPLITS]
    X = np.vstack([p[0] for p in parts]).astype(np.float32)
    y = np.concatenate([p[1] for p in parts])
    split = np.concatenate([[s] * len(p[0]) for s, p in zip(SPLITS, parts)])

    cache = f"{CACHE_DIR}/tsne_th{threshold}.npz"
    if os.path.exists(cache):
        emb = np.load(cache)["emb"]
    else:
        print(f"th={threshold}: PCA({X.shape}) -> 50, then t-SNE...")
        Z = PCA(n_components=50, random_state=42).fit_transform(X)
        emb = np.asarray(TSNE(n_jobs=N_JOBS, random_state=42, verbose=True).fit(Z))
        np.savez_compressed(cache, emb=emb, y=y, split=split)

    # colored by split (train first so valid/test are on top)
    fig, ax = plt.subplots(figsize=(8, 7))
    for s, size, alpha in [("train", 1, 0.25), ("valid", 2, 0.5), ("test", 2, 0.5)]:
        m = split == s
        ax.scatter(*emb[m].T, s=size, alpha=alpha, c=SPLIT_COLORS[s], label=f"{s} (n={m.sum()})", rasterized=True)
    ax.legend(markerscale=6)
    ax.set_title(f"t-SNE atom_pair — similarity split th={threshold}")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_tsne_th{threshold}_splits.png", dpi=130)
    plt.close(fig)

    # one panel per split / per split x label
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, s in enumerate(SPLITS):
        m = split == s
        ax = axes[0, col]
        ax.scatter(*emb.T, s=1, c="#dddddd", rasterized=True)
        ax.scatter(*emb[m].T, s=1, alpha=0.4, c=SPLIT_COLORS[s], rasterized=True)
        ax.set_title(f"{s} (n={m.sum()})")
        ax = axes[1, col]
        ax.scatter(*emb.T, s=1, c="#eeeeee", rasterized=True)
        ax.scatter(*emb[m & (y == 0)].T, s=1, alpha=0.3, c="#7f7f7f", label="non-sweet", rasterized=True)
        ax.scatter(*emb[m & (y == 1)].T, s=1.5, alpha=0.6, c="#2ca02c", label="sweet", rasterized=True)
        ax.set_title(f"{s} — sweet (n={(m & (y == 1)).sum()}) vs non-sweet")
        ax.legend(markerscale=8, loc="lower right")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"t-SNE atom_pair — th={threshold}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_tsne_th{threshold}_panels.png", dpi=110)
    plt.close(fig)
    print(f"th={threshold}: saved")
