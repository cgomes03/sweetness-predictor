"""
Are valid/test mislabeled or swapped somewhere? (would explain test >> valid)

  1. Labels in each split CSV == labels in the raw dataset (by inchikey)?
  2. Reload the saved best_pipeline and re-score it on the valid and test
     CSVs. If the reported "valid"/"test" columns were swapped or computed on
     the wrong set, the numbers won't reproduce.
  3. Code path: deepmol SimilaritySplitter.split returns
     (train_inds, valid_inds, test_inds) and the 2nd _split_fingerprints call
     is (test_size, valid_size) -> (test_inds, valid_inds): consistent, no swap
     (documented in the notebook, not executable here).
  4. Composition by data source (DeepSweet / GoldTaste / FooDB) and the
     reported model's performance per source — a different source mix in
     valid vs test is an innocent explanation for the gap.

Outputs: outputs/03_*.csv
"""
import numpy as np
import pandas as pd
from deepmol.pipeline import Pipeline

from common import (OUT_DIR, PAIR_ID, RUNS, SPLITS, all_metrics, featurize_split,
                    load_raw, load_split_csv, reported_metrics)

raw = load_raw()
raw_label = dict(zip(raw["inchikey"], raw["sweet"]))
raw_source = dict(zip(raw["inchikey"], raw["source"]))

label_rows, repro_rows, source_rows = [], [], []

for threshold, run_dir in RUNS.items():
    pipeline = Pipeline.load(f"{run_dir}/best_pipeline_{PAIR_ID}")
    rf = pipeline.steps[-1][1].model
    reported = reported_metrics(threshold)

    for split in SPLITS:
        df = load_split_csv(threshold, split)
        X, y, ids = featurize_split(threshold, split)

        # 1. labels vs raw
        raw_y = df["ids"].map(raw_label)
        label_rows.append({
            "threshold": threshold, "split": split, "n": len(df),
            "missing_in_raw": int(raw_y.isna().sum()),
            "label_mismatch_vs_raw": int((raw_y.notna() & (raw_y != df["sweet"])).sum()),
            "features_y_matches_csv": bool(np.array_equal(y, df["sweet"].astype(int).to_numpy())),
        })

        # 2. reproduce reported metrics with the saved model
        proba = rf.predict_proba(X)[:, 1]
        recomputed = all_metrics(y, proba)
        for metric, value in recomputed.items():
            repro_rows.append({
                "threshold": threshold, "split": split, "metric": metric,
                "reported": reported[split][metric], "recomputed": round(value, 4),
                "abs_diff": round(abs(value - reported[split][metric]), 4),
            })

        # 4. per-source composition and performance
        sources = df["ids"].map(raw_source).fillna("Unknown").to_numpy()
        preds = (proba >= 0.5).astype(int)
        for source in sorted(set(sources)):
            mask = sources == source
            ys, ps = y[mask], preds[mask]
            source_rows.append({
                "threshold": threshold, "split": split, "source": source,
                "n": int(mask.sum()), "pct_of_split": round(100 * mask.mean(), 1),
                "sweet_frac": round(ys.mean(), 3),
                "TP": int(((ps == 1) & (ys == 1)).sum()), "FP": int(((ps == 1) & (ys == 0)).sum()),
                "FN": int(((ps == 0) & (ys == 1)).sum()), "TN": int(((ps == 0) & (ys == 0)).sum()),
            })

labels = pd.DataFrame(label_rows)
repro = pd.DataFrame(repro_rows)
by_source = pd.DataFrame(source_rows)
by_source["precision"] = (by_source.TP / (by_source.TP + by_source.FP)).round(3)
by_source["recall"] = (by_source.TP / (by_source.TP + by_source.FN)).round(3)
by_source["f1"] = (2 * by_source.TP / (2 * by_source.TP + by_source.FP + by_source.FN)).round(3)

labels.to_csv(f"{OUT_DIR}/03_labels_vs_raw.csv", index=False)
repro.to_csv(f"{OUT_DIR}/03_reproduce_reported.csv", index=False)
by_source.to_csv(f"{OUT_DIR}/03_by_source.csv", index=False)

print(labels.to_string(index=False))
print(repro[repro.metric.isin(["f1_score", "precision_score", "recall_score"])].to_string(index=False))
print("max |reported - recomputed|:", repro["abs_diff"].max())
print(by_source.to_string(index=False))
