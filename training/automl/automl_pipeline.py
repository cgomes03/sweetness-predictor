
import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import yaml
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.pipeline_optimization.objective_wrapper import ObjectiveTrainEval
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "manual",
        "scripts",
    )
)

from sweet_model_pipeline import load_dataset, split_dataset, standardize
from dataset_filters import undersample_majority, exclude_np_pathways
from search_space import (
    CachedTwoDimensionalDescriptors,
    build_default_featurizer,
    build_steps,
)


FINAL_METRICS = [
    Metric(f1_score),
    Metric(precision_score),
    Metric(recall_score),
    Metric(matthews_corrcoef),
    Metric(roc_auc_score),
    Metric(average_precision_score),
    Metric(balanced_accuracy_score),
    Metric(accuracy_score),
]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Existing checkpoints/<RUN_ID>_<study> dir to continue: pairs "
             "that already have best_pipeline_<pair>/ are skipped.",
    )
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    # Same rule as manual/scripts/sweet_model_pipeline.py: a relative
    # dataset_path is relative to the config file, not to the cwd, so the
    # configs can live in <repo>/configs/sweet26/predictor/.
    dataset_path = config.get("dataset_path")
    if dataset_path and not os.path.isabs(dataset_path):
        config_dir = os.path.dirname(os.path.abspath(args.config))
        config["dataset_path"] = os.path.normpath(os.path.join(config_dir, dataset_path))
    return config, args.resume


def _write_results(output_dir, summary, long_records):
    """Persist results after every (featurizer, model) pair so a crash
    mid-run keeps everything that already finished."""
    pd.DataFrame(long_records).to_csv(
        f"{output_dir}/results_by_model_featurizer.csv",
        index=False,
    )
    with open(
        f"{output_dir}/summary_by_model_featurizer.json",
        "w",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)


def _clear_stale_pair(output_dir, pair_id, study_name):
    """Remove leftovers of a pair that was interrupted mid-optimization
    (only called in --resume mode, for pairs without best_pipeline_<pair>/)."""
    stale = [
        study_name,
        f"{output_dir}/{study_name}.db",
        f"{output_dir}/tmp_{pair_id}",
        f"{output_dir}/models_{pair_id}",
    ]
    for path in stale:
        if os.path.isdir(path):
            print(f"  [resume] removing stale {path}/")
            shutil.rmtree(path)
        elif os.path.exists(path):
            print(f"  [resume] removing stale {path}")
            os.remove(path)


def run_automl_pipeline(config, resume_dir=None):
    n_jobs = int(config.get("cpu", 4))
    cache_dir = "checkpoints/cache"

    if resume_dir:
        output_dir = resume_dir.rstrip("/")
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"--resume dir not found: {output_dir}")
        print(f"Resuming run in {output_dir}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # study_name is included so two configs launched in the same second (or
        # even the same run_id from a clock with coarser resolution) never share
        # an output_dir — they used to, and it silently corrupted each other's
        # shared-name files (best_metrics_morgan_RandomForestClassifier.csv,
        # tmp_morgan_.../, models_morgan_.../, etc.) when run in parallel.
        study_name_prefix = config.get("automl", {}).get("study_name", "sweet")
        # output_root groups the runs of one batch under a single folder
        # (e.g. checkpoints/20261029_sim02_clean/); defaults to checkpoints/.
        output_root = config.get("output_root", "checkpoints").rstrip("/")
        output_dir = f"{output_root}/{run_id}_{study_name_prefix}"

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w") as config_file:
        json.dump(config, config_file, indent=2)

    # --- Data loading and standardization ---
    dataframe = load_dataset(
        config["dataset_path"],
        config["smiles_field"],
        config["id_field"],
        [config["label_field"]],
    )

    dataset = standardize(
        dataframe,
        config["dataset_path"],
        cache_dir=cache_dir,
        standardizer_name=config.get("standardizer", "chembl"),
    )

    # --- Whole-dataset preprocessing (before featurization/split) ---
    preprocessing_config = config.get("preprocessing", {})

    exclude_pathways = preprocessing_config.get("exclude_np_pathways")
    if exclude_pathways:
        dataset = exclude_np_pathways(
            dataset, exclude_pathways, config["dataset_path"], cache_dir=cache_dir,
        )

    undersample_config = preprocessing_config.get("undersample_majority")
    if undersample_config:
        dataset = undersample_majority(
            dataset,
            majority_label=undersample_config.get("label", 0),
            max_count=undersample_config["max_count"],
            seed=undersample_config.get("seed", 42),
        )

    # --- AutoML configuration ---
    automl_config = config["automl"]

    metric_name = automl_config.get("metric", "f1_score")
    metric_function = globals().get(metric_name)

    if metric_function is None:
        raise ValueError(f"Unknown metric: {metric_name}")

    featurizer_candidates = automl_config.get(
        "featurizer_candidates",
        ["morgan", "maccs", "atom_pair"],
    )
    model_candidates = automl_config.get(
        "model_candidates",
        ["RandomForestClassifier", "SVC", "LogisticRegression"],
    )

    n_trials = automl_config.get("n_trials", 60)
    seed = int(automl_config.get("seed", 42))


    classification_threshold = float(
        automl_config.get("classification_threshold", 0.5)
    )

    # --- Run one optimization study per (featurizer, model) pair ---
    # The split/featurizer-survivor mask only depends on the featurizer, so
    # it is computed once per featurizer and reused across all models to
    # avoid redundant work.
    summary = {}
    long_records = []

    if resume_dir:
        summary_path = f"{output_dir}/summary_by_model_featurizer.json"
        results_path = f"{output_dir}/results_by_model_featurizer.csv"
        if os.path.exists(summary_path):
            with open(summary_path) as summary_file:
                summary = json.load(summary_file)
        if os.path.exists(results_path):
            long_records = pd.read_csv(results_path).to_dict("records")

    def pair_done(featurizer_name, model_name):
        pair_id = f"{featurizer_name}_{model_name}"
        return (
            os.path.exists(f"{output_dir}/best_pipeline_{pair_id}/pipeline.json")
            and model_name in summary.get(featurizer_name, {})
        )

    for featurizer_name in featurizer_candidates:
        if all(pair_done(featurizer_name, m) for m in model_candidates):
            print(f"  [resume] {featurizer_name}: all models done, skipping")
            continue

        print(f"{'=' * 60}")
        print(f"  Featurizer: {featurizer_name}")
        print(f"{'=' * 60}")

        # Featurize before splitting — same order as the manual pipeline
        # (sweet_model_pipeline.run_pipeline: standardize -> featurize ->
        # split), instead of splitting the raw standardized dataset. Only
        # the survivor mask is kept here; features are recomputed per trial
        # in build_steps() below, so the split itself stays a lightweight
        # (ids/smiles/label) dataset, matching the previous CSV output.
        prefilter_dataset = deepcopy(dataset)
        build_default_featurizer(featurizer_name, n_jobs).featurize(
            prefilter_dataset, inplace=True
        )
        # featurize(inplace=True) already drops molecules that fail to
        # featurize, so prefilter_dataset's own indices are just 0..N-1 in
        # its own (already-shrunk) index space — not indices into the
        # original `dataset`. Map survivors back to `dataset`'s indices by
        # id instead of reusing prefilter_dataset's positions directly.
        surviving_ids = set(prefilter_dataset.ids)
        valid_indices = np.array(
            [i for i, id_ in enumerate(dataset.ids) if id_ in surviving_ids]
        )

        dropped = len(dataset) - len(valid_indices)
        if dropped:
            print(
                f"  Dropped {dropped} molecules that failed "
                f"'{featurizer_name}' featurization"
            )

        split_input_dataset = deepcopy(dataset).select_to_split(valid_indices)

        split_output_dir = f"{output_dir}/split_{featurizer_name}"
        os.makedirs(split_output_dir, exist_ok=True)

        train_dataset, valid_dataset, test_dataset = split_dataset(
            split_input_dataset,
            config["split"],
            split_output_dir,
            save_npy=False,
            dataset_path=config["dataset_path"],
            cache_dir=cache_dir,
        )

        if featurizer_name == "2d_descriptors":
            # Warm the descriptor cache for these exact train/valid/test
            # subsets so the trials that follow don't recompute it 100x.
            for dataset_split in (train_dataset, valid_dataset, test_dataset):
                CachedTwoDimensionalDescriptors(n_jobs=n_jobs).featurize(
                    deepcopy(dataset_split), inplace=True
                )

        summary.setdefault(featurizer_name, {})

        # --- Evaluate with optimised threshold ---
        def evaluate_with_threshold(pipeline, dataset, threshold):
            probas = pipeline.predict_proba(dataset)
            if probas.ndim == 2:
                probas = probas[:, 1]
            y_true = dataset.y
            min_len = min(len(y_true), len(probas))
            y_true = y_true[:min_len]
            probas = probas[:min_len]
            preds = (probas >= threshold).astype(int)
            return {
                "f1_score": round(f1_score(y_true, preds), 4),
                "precision_score": round(precision_score(y_true, preds), 4),
                "recall_score": round(recall_score(y_true, preds), 4),
                "matthews_corrcoef": round(matthews_corrcoef(y_true, preds), 4),
                "roc_auc_score": round(roc_auc_score(y_true, probas), 4),
                "average_precision_score": round(
                    average_precision_score(y_true, probas), 4
                ),
                "balanced_accuracy_score": round(
                    balanced_accuracy_score(y_true, preds), 4
                ),
                "accuracy_score": round(accuracy_score(y_true, preds), 4),
            }

        for model_name in model_candidates:
            pair_id = f"{featurizer_name}_{model_name}"

            if pair_done(featurizer_name, model_name):
                print(f"  [resume] {pair_id}: already done, skipping")
                continue

            print(f"\n{'-' * 60}")
            print(
                f"  Featurizer: {featurizer_name}  |  "
                f"Model: {model_name}  ({n_trials} trials)"
            )
            print(f"{'-' * 60}")

            pair_config = dict(
                automl_config,
                featurizer_candidates=[featurizer_name],
                model_candidates=[model_name],
            )

            steps = build_steps(
                pair_config,
                n_jobs,
                f"{output_dir}/tmp_{pair_id}",
            )

            study_name = (
                f"{automl_config.get('study_name', 'sweet')}"
                f"_{pair_id}"
            )

            if resume_dir:
                _clear_stale_pair(output_dir, pair_id, study_name)

            # Guard against leftover artifacts from interrupted runs.
            if os.path.exists(study_name):
                raise RuntimeError(
                    f"'{study_name}/' already exists in cwd — "
                    f"delete or rename it before running to avoid "
                    f"mixing old and new artifacts."
                )

            pipeline_optimizer = PipelineOptimization(
                direction="maximize",
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=seed),
                storage=f"sqlite:///{output_dir}/{study_name}.db",
            )

            pipeline_optimizer.optimize(
                train_dataset=train_dataset,
                test_dataset=valid_dataset,
                objective_steps=steps,
                objective=ObjectiveTrainEval,
                metric=Metric(metric_function),
                n_trials=n_trials,
                save_top_n=automl_config.get("save_top_n", 3),
                trial_timeout=automl_config.get(
                    "trial_timeout",
                    1800,
                ),
            )

            # Save optimization trial history.
            pipeline_optimizer.trials_dataframe().to_csv(
                f"{output_dir}/trials_{pair_id}.csv",
                index=False,
            )

            best_pipeline = pipeline_optimizer.best_pipeline
            best_valid = pipeline_optimizer.best_value
            best_params = pipeline_optimizer.best_params

            if os.path.exists(study_name):
                shutil.move(
                    study_name,
                    f"{output_dir}/models_{pair_id}",
                )

            # Save the trained best pipeline right away (featurizer + fitted
            # model), independent of the top-N trial folders, so it can be
            # reloaded with Pipeline.load() without re-running the search.
            best_pipeline.path = os.path.abspath(
                f"{output_dir}/best_pipeline_{pair_id}"
            )
            best_pipeline.save()
            trials = pipeline_optimizer.trials_dataframe()
            best_trial = int(
                trials.sort_values("value", ascending=False).iloc[0]["number"]
            )
            with open(f"{output_dir}/best_pipeline_{pair_id}/best_trial.json", "w") as f:
                json.dump(
                    {"trial": best_trial, "valid_value": best_valid,
                     "metric": metric_name, "params": best_params},
                    f, indent=2, default=str,
                )
            print(f"  Saved best pipeline (trial {best_trial}) -> "
                  f"{output_dir}/best_pipeline_{pair_id}")

            print(f"  Classification threshold: {classification_threshold}")

            train_scores = evaluate_with_threshold(
                best_pipeline, train_dataset, classification_threshold,
            )
            valid_scores = evaluate_with_threshold(
                best_pipeline, valid_dataset, classification_threshold,
            )
            test_scores = evaluate_with_threshold(
                best_pipeline, test_dataset, classification_threshold,
            )

            # Save per (featurizer, model) metrics in the same format as
            # the manual pipeline.
            metrics_dataframe = pd.DataFrame(
                [
                    {
                        "Metric": metric_name,
                        "Train": round(train_scores[metric_name], 4),
                        "Valid": round(valid_scores[metric_name], 4),
                        "Test": round(test_scores[metric_name], 4),
                    }
                    for metric_name in train_scores
                ]
            )

            metrics_dataframe.to_csv(
                f"{output_dir}/best_metrics_{pair_id}.csv",
                index=False,
            )

            print(f"\n[{featurizer_name} | {model_name}] Best pipeline metrics:")
            print(metrics_dataframe.to_string(index=False))

            summary[featurizer_name][model_name] = {
                "best_valid": best_valid,
                "best_params": best_params,
                "train": train_scores,
                "valid": valid_scores,
                "test": test_scores,
            }

            for score_name in train_scores:
                long_records.append(
                    {
                        "featurizer": featurizer_name,
                        "model": model_name,
                        "metric": score_name,
                        "train": train_scores[score_name],
                        "valid": valid_scores[score_name],
                        "test": test_scores[score_name],
                    }
                )

            print(
                f"\n[{featurizer_name} | {model_name}] "
                f"valid={best_valid:.4f}"
                f"  test_f1={test_scores['f1_score']:.4f}\n"
            )

            _write_results(output_dir, summary, long_records)

    # --- Consolidated tidy table: one row per (featurizer, model, metric) ---
    # Lets you groupby("model") to compare models, or filter to one model
    # and groupby("featurizer") to find its best featurizer.
    _write_results(output_dir, summary, long_records)
    results_long = pd.DataFrame(long_records)

    # --- Convenience view: test-set primary metric, models x featurizers ---
    primary_metric_table = results_long[
        results_long["metric"] == metric_name
    ].pivot(index="model", columns="featurizer", values="test")

    primary_metric_table.to_csv(
        f"{output_dir}/test_{metric_name}_by_model_and_featurizer.csv"
    )

    print("\n" + "=" * 60)
    print(f"  Test '{metric_name}' — model x featurizer")
    print("=" * 60)
    print(primary_metric_table.to_string())

    return output_dir


if __name__ == "__main__":
    config, resume_dir = load_config()
    run_automl_pipeline(config, resume_dir)
