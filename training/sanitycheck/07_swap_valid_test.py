"""
Swap validation and test: run the same RF search as the AutoML run
(same space as search_space.build_steps, TPESampler(seed=42), F1 @ 0.5)
but SELECT on test instead of valid, then report the held-out valid.
Also runs the original direction (select on valid) with the identical code
so the two are directly comparable.

Every trial is scored on BOTH valid and test (stored per trial), so we can
also see whether valid and test agree on which configs are good.

  python 07_swap_valid_test.py --threshold 0.3 --select-on test --n-trials 100

Outputs: outputs/07_trials_th{t}_select_{set}.csv, outputs/07_best_th{t}_select_{set}.csv
"""
import argparse

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from common import N_JOBS, OUT_DIR, SPLITS, all_metrics, featurize_split

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, required=True)
parser.add_argument("--select-on", choices=["valid", "test"], required=True)
parser.add_argument("--n-trials", type=int, default=100)
parser.add_argument("--n-jobs", type=int, default=N_JOBS)
args = parser.parse_args()

data = {s: featurize_split(args.threshold, s) for s in SPLITS}
holdout = "valid" if args.select_on == "test" else "test"


def build_rf(trial):
    # identical to search_space.build_steps (RandomForestClassifier branch)
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300, step=100),
        max_depth=trial.suggest_int("max_depth", 5, 15, step=5),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, step=1),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20, step=2),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        n_jobs=args.n_jobs,
        random_state=42,
    )


def objective(trial):
    rf = build_rf(trial)
    rf.fit(data["train"][0], data["train"][1])
    scores = {s: f1_score(data[s][1], rf.predict(data[s][0])) for s in ("valid", "test")}
    trial.set_user_attr("f1_valid", scores["valid"])
    trial.set_user_attr("f1_test", scores["test"])
    return scores[args.select_on]


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

tag = f"th{args.threshold}_select_{args.select_on}"
trials = study.trials_dataframe()
trials.to_csv(f"{OUT_DIR}/07_trials_{tag}.csv", index=False)

best = study.best_trial
rf = build_rf(optuna.trial.FixedTrial(best.params))
rf.fit(data["train"][0], data["train"][1])
rows = []
for split in SPLITS:
    X, y, _ = data[split]
    for metric, value in all_metrics(y, rf.predict_proba(X)[:, 1]).items():
        rows.append({"threshold": args.threshold, "selected_on": args.select_on, "split": split,
                     "role": "train" if split == "train" else ("selection" if split == args.select_on else "held-out"),
                     "metric": metric, "value": round(value, 4),
                     "best_trial": best.number, "best_params": str(best.params)})
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/07_best_{tag}.csv", index=False)
print(f"[{tag}] best trial {best.number} {best.params}")
print(pd.DataFrame(rows).query("metric == 'f1_score'")[["split", "role", "value"]].to_string(index=False))
