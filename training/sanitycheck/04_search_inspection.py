"""
How was the hyperparameter search done? (grid? random? Bayesian?)

From automl_pipeline.py + search_space.py:
  - optuna PipelineOptimization, sampler = TPESampler(seed=42)  -> Bayesian (TPE),
    NOT grid. Optuna's TPE default: first 10 trials random, then TPE.
  - objective = ObjectiveTrainEval: fit on train, score F1 (threshold 0.5) on VALID.
    Best trial = argmax valid F1. Test is never seen during the search.
  - 100 trials per (featurizer, model) pair; RF space (search_space.build_steps):
      n_estimators     {100, 200, 300}
      max_depth        {5, 10, 15}
      min_samples_leaf 1..50
      min_samples_split 2..20 step 2
      class_weight     {None, balanced}
    => 3*3*50*10*2 = 9000 combinations, 100 sampled.
    Featurizer is fixed (atom_pair, default 2048 bits) — no featurizer params tuned.

This script reads the trials CSVs and reports: state counts, duplicated
configs, where the best lands in the space (edges?), and valid F1 per param.

Outputs: outputs/04_*.csv, outputs/04_search_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_DIR, PAIR_ID, RUNS

SPACE = {
    "n_estimators": [100, 300], "max_depth": [5, 15], "min_samples_leaf": [1, 50],
    "min_samples_split": [2, 20],
}
PARAMS = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "class_weight"]

summary_rows, trials_all = [], []
for threshold, run_dir in RUNS.items():
    with open(f"{run_dir}/config.json") as f:
        cfg = json.load(f)
    t = pd.read_csv(f"{run_dir}/trials_{PAIR_ID}.csv")
    t = t.rename(columns={f"params_model__{p}": p for p in PARAMS})
    t["class_weight"] = t["class_weight"].fillna("None")
    t["threshold"] = threshold
    t["duration_min"] = pd.to_timedelta(t["duration"]).dt.total_seconds() / 60
    trials_all.append(t)

    best = t.loc[t["value"].idxmax()]
    summary_rows.append({
        "threshold": threshold,
        "sampler": f"TPESampler(seed={cfg['automl']['seed']})",
        "metric (on valid)": cfg["automl"]["metric"],
        "n_trials": len(t),
        "states": t["state"].value_counts().to_dict(),
        "unique_configs": int(t[PARAMS].drop_duplicates().shape[0]),
        "best_trial": int(best["number"]),
        "best_valid_f1": round(best["value"], 4),
        "best_params": {p: best[p] for p in PARAMS},
        "best_at_space_edge": [p for p, (lo, hi) in SPACE.items() if best[p] in (lo, hi)],
        "trials_within_0.01_of_best": int((t["value"] >= best["value"] - 0.01).sum()),
        "median_trial_minutes": round(t["duration_min"].median(), 1),
        "valid_f1_class_weight_None(median)": round(t[t.class_weight == "None"]["value"].median(), 4),
        "valid_f1_class_weight_balanced(median)": round(t[t.class_weight == "balanced"]["value"].median(), 4),
    })

trials = pd.concat(trials_all, ignore_index=True)
summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{OUT_DIR}/04_search_summary.csv", index=False)
trials.to_csv(f"{OUT_DIR}/04_trials.csv", index=False)
top = trials.sort_values("value", ascending=False).groupby("threshold").head(10)
top[["threshold", "number", "value"] + PARAMS].to_csv(f"{OUT_DIR}/04_top10_trials.csv", index=False)
print(summary.T.to_string())

# value vs trial number (+ running best)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (threshold, t) in zip(axes, trials.groupby("threshold")):
    ax.scatter(t["number"], t["value"], s=12, c=(t["class_weight"] == "balanced"), cmap="coolwarm")
    ax.plot(t["number"], t["value"].cummax(), color="k", lw=1, label="running best")
    ax.axvline(9.5, ls=":", color="gray")
    ax.text(10, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.05, " random | TPE", fontsize=8, color="gray")
    ax.set_title(f"threshold {threshold} — valid F1 per trial (red = balanced)")
    ax.set_xlabel("trial")
axes[0].set_ylabel("valid F1")
axes[0].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_search_history.png", dpi=130)

# valid F1 vs each hyperparameter
fig, axes = plt.subplots(2, len(PARAMS), figsize=(18, 6), sharey="row")
for row, (threshold, t) in enumerate(trials.groupby("threshold")):
    for col, p in enumerate(PARAMS):
        ax = axes[row, col]
        ax.scatter(t[p].astype(str) if p == "class_weight" else t[p], t["value"], s=10, alpha=0.7)
        ax.set_title(f"th {threshold} — {p}", fontsize=9)
    axes[row, 0].set_ylabel("valid F1")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_search_params.png", dpi=130)
