"""
Shared paths/helpers for the sanity checks of the AutoML atom_pair + RF model.

Everything reads the *exact* artifacts the AutoML runs produced (split CSVs,
best_pipeline, trials, Optuna DB) from ../automl/checkpoints, so the checks
look at what was actually trained/evaluated, not at a re-implementation.
"""
import json
import os

import numpy as np
import pandas as pd
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

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOML_DIR = os.path.join(HERE, "..", "automl")
OUT_DIR = os.path.join(HERE, "outputs")
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

RAW_DATASET = os.path.normpath(os.path.join(
    HERE, "..", "..",
    "DeepSweet/resources/data/merged_foodb_gold_sweetnessv3_filtered.csv",
))
STANDARDIZED_CACHE = os.path.join(
    AUTOML_DIR,
    "checkpoints/cache/merged_foodb_gold_sweetnessv3_filtered_standardized_chembl.csv",
)

# Runs of the chosen model (atom_pair + RandomForest) for each similarity
# threshold — same dirs merged_similarity/sources.csv points to.
RUNS = {
    0.2: os.path.join(AUTOML_DIR, "checkpoints/20261005_152420_sweet_automl_sim02_par_atom_pair_RF"),
    0.3: os.path.join(AUTOML_DIR, "checkpoints/20261005_152521_sweet_automl_sim03_par_atom_pair_RF"),
}
PAIR_ID = "atom_pair_RandomForestClassifier"
SPLITS = ["train", "valid", "test"]
SPLIT_COLORS = {"train": "#4C72B0", "valid": "#DD8452", "test": "#C44E52"}

N_JOBS = int(os.environ.get("SANITY_N_JOBS", 32))


def load_split_csv(threshold, split):
    """The train/valid/test CSV the AutoML run wrote (ids = inchikey, smiles = standardized)."""
    path = f"{RUNS[threshold]}/split_atom_pair/{split}_dataset.csv"
    return pd.read_csv(path, index_col=0)


def load_raw():
    return pd.read_csv(RAW_DATASET)


def best_trial(threshold):
    with open(f"{RUNS[threshold]}/best_pipeline_{PAIR_ID}/best_trial.json") as f:
        return json.load(f)


def reported_metrics(threshold):
    with open(f"{RUNS[threshold]}/summary_by_model_featurizer.json") as f:
        return json.load(f)["atom_pair"]["RandomForestClassifier"]


def rf_params_from_best(threshold):
    params = best_trial(threshold)["params"]
    return {
        k.replace("model__", ""): v for k, v in params.items() if k.startswith("model__")
    }


def featurize_split(threshold, split):
    """
    Atom-pair features for one split, built with the same deepmol featurizer
    and defaults as search_space.build_steps() (nBits=2048, includeChirality=False).
    Cached as npz so the other scripts don't recompute ~90k fingerprints.
    """
    cache = f"{CACHE_DIR}/atompair_th{threshold}_{split}.npz"
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        return data["X"], data["y"], data["ids"]

    from deepmol.compound_featurization import AtomPairFingerprint
    from deepmol.loaders import CSVLoader

    csv = f"{RUNS[threshold]}/split_atom_pair/{split}_dataset.csv"
    dataset = CSVLoader(
        dataset_path=csv, smiles_field="smiles", id_field="ids",
        labels_fields=["sweet"], mode="classification",
    ).create_dataset()
    n_before = len(dataset)
    AtomPairFingerprint(n_jobs=N_JOBS).featurize(dataset, inplace=True)
    if len(dataset) != n_before:
        print(f"  WARNING: {n_before - len(dataset)} molecules dropped by featurizer")
    X = np.asarray(dataset.X, dtype=np.uint8)
    y = np.asarray(dataset.y).astype(int).ravel()
    ids = np.asarray(dataset.ids).astype(str)
    np.savez_compressed(cache, X=X, y=y, ids=ids)
    return X, y, ids


def all_metrics(y_true, proba, threshold=0.5):
    """Same metrics/threshold logic as automl_pipeline.evaluate_with_threshold."""
    preds = (proba >= threshold).astype(int)
    return {
        "f1_score": f1_score(y_true, preds),
        "precision_score": precision_score(y_true, preds),
        "recall_score": recall_score(y_true, preds),
        "matthews_corrcoef": matthews_corrcoef(y_true, preds),
        "roc_auc_score": roc_auc_score(y_true, proba),
        "average_precision_score": average_precision_score(y_true, proba),
        "balanced_accuracy_score": balanced_accuracy_score(y_true, preds),
        "accuracy_score": accuracy_score(y_true, preds),
    }
