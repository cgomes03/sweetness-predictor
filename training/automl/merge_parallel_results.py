"""
Merge the original similarity02/03 AutoML runs with the per-pair parallel runs
(<repo>/configs/sweet26/predictor/automl/parallel/) into one table per threshold, checking that everything
being merged is actually comparable.

Source of each (featurizer, model) pair:
  - morgan_RF: original run dir (the only pair it finished before the switch)
  - everything else: the parallel run dir for that pair. morgan_SVC is taken
    from the parallel run too, even if the original finishes it, so that
    every SVC in the table uses the same svc_fast setup.

Checks (the script stops on any FAIL):
  1. config: identical to the original except cpu / study_name / candidates
     / svc_fast, and every SVC pair has svc_fast=True
  2. split: same split_fingerprint.json for the same featurizer everywhere
  3. trials: 100 COMPLETE trials in each pair's Optuna DB (reported)

Usage (from the automl/ dir):
  python merge_parallel_results.py
Outputs go to merged_similarity/.
"""
import glob
import json
import os
import sqlite3
import sys

import pandas as pd

FEATURIZERS = ["morgan", "maccs", "2d_descriptors", "atom_pair"]
MODELS = ["RandomForestClassifier", "SVC", "LogisticRegression"]
SHORT = {"RandomForestClassifier": "RF", "SVC": "SVC", "LogisticRegression": "LR"}

RUNS = {
    0.2: {
        "original": "checkpoints/20261001_142333_sweet_automl_similarity02_study",
        "parallel_glob": "checkpoints/*_sweet_automl_sim02_par_{feat}_{model}",
    },
    0.3: {
        "original": "checkpoints/20261001_142446_sweet_automl_similarity03_study",
        "parallel_glob": "checkpoints/*_sweet_automl_sim03_par_{feat}_{model}",
    },
}

# Keys allowed to differ between the original and the parallel configs.
IGNORED_KEYS = {"cpu"}
IGNORED_AUTOML_KEYS = {
    "study_name", "featurizer_candidates", "model_candidates", "svc_fast",
}
# Pairs taken from the original (sequential) run; all others from parallel.
ORIGINAL_PAIRS = {("morgan", "RandomForestClassifier")}

OUT_DIR = "merged_similarity"
problems = []


def comparable_config(path):
    with open(f"{path}/config.json") as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if k not in IGNORED_KEYS}
    config["automl"] = {
        k: v for k, v in config["automl"].items() if k not in IGNORED_AUTOML_KEYS
    }
    return config


def pair_done(run_dir, feat, model):
    pair_id = f"{feat}_{model}"
    summary_path = f"{run_dir}/summary_by_model_featurizer.json"
    if not os.path.exists(f"{run_dir}/best_pipeline_{pair_id}/pipeline.json"):
        return False
    if not os.path.exists(summary_path):
        return False
    with open(summary_path) as f:
        return model in json.load(f).get(feat, {})


def trial_counts(run_dir, feat, model):
    with open(f"{run_dir}/config.json") as f:
        study_name = json.load(f)["automl"]["study_name"]
    db = f"{run_dir}/{study_name}_{feat}_{model}.db"
    if not os.path.exists(db):
        return {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    counts = dict(conn.execute("select state, count(*) from trials group by state"))
    # Trials hit by trial_timeout are stored as COMPLETE with value -inf.
    counts["TIMED_OUT"] = conn.execute(
        "select count(*) from trials t left join trial_values v "
        "on v.trial_id = t.trial_id where t.state = 'COMPLETE' "
        "and (v.value is null or v.value < -1e300)"
    ).fetchone()[0]
    conn.close()
    return counts


def split_fingerprint(run_dir, feat):
    path = f"{run_dir}/split_{feat}/split_fingerprint.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_source(threshold, feat, model):
    if (feat, model) in ORIGINAL_PAIRS:
        original = RUNS[threshold]["original"]
        return original if pair_done(original, feat, model) else None
    pattern = RUNS[threshold]["parallel_glob"].format(feat=feat, model=SHORT[model])
    done = [d for d in sorted(glob.glob(pattern)) if pair_done(d, feat, model)]
    if len(done) > 1:
        problems.append(
            f"[{threshold}] {feat}_{model}: {len(done)} finished parallel dirs "
            f"{done} — delete/rename the ones you don't want"
        )
    return done[-1] if done else None


def merge_threshold(threshold):
    original = RUNS[threshold]["original"]
    reference_config = comparable_config(original)
    split_by_feat = {}
    records, sources = [], []

    for feat in FEATURIZERS:
        for model in MODELS:
            source = find_source(threshold, feat, model)
            if source is None:
                problems.append(f"[{threshold}] {feat}_{model}: not finished anywhere")
                continue

            # 1. config
            if comparable_config(source) != reference_config:
                problems.append(
                    f"[{threshold}] {feat}_{model}: config of {source} differs "
                    f"from {original}"
                )
            if model == "SVC":
                with open(f"{source}/config.json") as f:
                    if not json.load(f)["automl"].get("svc_fast", False):
                        problems.append(
                            f"[{threshold}] {feat}_SVC: {source} was not run "
                            f"with svc_fast=True"
                        )

            # 2. split
            fingerprint = split_fingerprint(source, feat)
            if feat in split_by_feat and fingerprint != split_by_feat[feat][0]:
                problems.append(
                    f"[{threshold}] split_{feat} differs between "
                    f"{split_by_feat[feat][1]} and {source}"
                )
            split_by_feat.setdefault(feat, (fingerprint, source))

            # 3. trials
            counts = trial_counts(source, feat, model)

            with open(f"{source}/summary_by_model_featurizer.json") as f:
                result = json.load(f)[feat][model]
            for metric in result["test"]:
                records.append({
                    "threshold": threshold,
                    "featurizer": feat,
                    "model": model,
                    "metric": metric,
                    "train": result["train"][metric],
                    "valid": result["valid"][metric],
                    "test": result["test"][metric],
                })
            sources.append({
                "threshold": threshold,
                "featurizer": feat,
                "model": model,
                "source_dir": source,
                "best_pipeline": f"{source}/best_pipeline_{feat}_{model}",
                "best_valid": result["best_valid"],
                "best_params": json.dumps(result["best_params"]),
                "n_complete": counts.get("COMPLETE", 0),
                "n_timed_out": counts.get("TIMED_OUT", 0),
                "n_fail": counts.get("FAIL", 0),
                "n_pruned": counts.get("PRUNED", 0),
                "split_train_hash": (fingerprint or {}).get("train"),
            })
    return records, sources


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    all_records, all_sources = [], []
    for threshold in RUNS:
        records, sources = merge_threshold(threshold)
        all_records += records
        all_sources += sources

    sources_df = pd.DataFrame(all_sources)
    print(sources_df[[
        "threshold", "featurizer", "model", "n_complete", "n_timed_out", "n_fail",
        "split_train_hash", "source_dir",
    ]].to_string(index=False))

    if problems:
        print("\nFAIL — nothing written:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = pd.DataFrame(all_records)
    results.to_csv(f"{OUT_DIR}/results_long.csv", index=False)
    sources_df.to_csv(f"{OUT_DIR}/sources.csv", index=False)
    pivot = results[results["metric"] == "f1_score"].pivot_table(
        index=["threshold", "model"], columns="featurizer", values="test"
    )
    pivot.to_csv(f"{OUT_DIR}/test_f1_by_threshold_model_featurizer.csv")
    print("\nTest F1:\n", pivot.to_string())
    print(f"\nOK — written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
