"""
Simple, transparent re-run of the chosen model: atom_pair + RandomForest with
the AutoML best params, trained on the same train split, 3 times with
different RF seeds (0, 1, 2) + the original seed 42. No deepmol pipeline,
plain sklearn on the cached atom_pair features.

If the reported numbers are real, every run should land close to them and
test > valid should persist across seeds (i.e. it is the data, not luck).

Outputs: outputs/06_rf_repeat.csv
"""
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import N_JOBS, OUT_DIR, RUNS, SPLITS, all_metrics, featurize_split, reported_metrics, rf_params_from_best

SEEDS = [0, 1, 2, 42]
rows = []
for threshold in RUNS:
    data = {s: featurize_split(threshold, s) for s in SPLITS}
    params = rf_params_from_best(threshold)
    reported = reported_metrics(threshold)
    print(f"th={threshold} params={params}")

    for split in SPLITS:
        for metric, value in reported[split].items():
            rows.append({"threshold": threshold, "run": "reported (AutoML)", "split": split,
                         "metric": metric, "value": value})

    for seed in SEEDS:
        start = time.time()
        rf = RandomForestClassifier(**params, n_jobs=N_JOBS, random_state=seed)
        rf.fit(data["train"][0], data["train"][1])
        fit_s = time.time() - start
        for split in SPLITS:
            X, y, _ = data[split]
            for metric, value in all_metrics(y, rf.predict_proba(X)[:, 1]).items():
                rows.append({"threshold": threshold, "run": f"seed {seed}", "split": split,
                             "metric": metric, "value": round(value, 4)})
        print(f"  seed {seed}: fit {fit_s:.0f}s")

results = pd.DataFrame(rows)
results.to_csv(f"{OUT_DIR}/06_rf_repeat.csv", index=False)
print(results[results.metric.isin(["f1_score", "precision_score", "recall_score"])]
      .pivot_table(index=["threshold", "metric", "run"], columns="split", values="value")[SPLITS]
      .to_string())
