"""
How is the dataset standardized and split, and is the split sound?

Pipeline as run by automl_pipeline.py:
  raw CSV -> ChEMBLStandardizer (standardize_mol + get_parent_mol, cached CSV)
  -> _deduplicate_structures (same std SMILES: keep 1; conflicting labels: drop all)
  -> drop molecules that fail atom_pair featurization
  -> SimilaritySplitter(homogenous_threshold=0.2|0.3) 70/15/15, per class
  -> split cache json (ids) -> split_atom_pair/*.csv

Checks:
  a. funnel of molecule counts per step + what standardization changed
  b. split sizes, class balance, overlaps between splits, duplicated ids
  c. deepmol SimilaritySplitter._split_fingerprints seeds group 0 with the
     LOCAL index 0 (`indices_in_group = ([0], [])`) instead of indexes[0]:
     dataset position 0 is duplicated and the first molecule of some
     class/sub-split is silently lost. Verified here on the real split.
  d. split cache json == the split CSVs that were trained on
  e. nearest-neighbour Tanimoto to train (Morgan r=2, 1024 bits — the
     fingerprint SimilaritySplitter uses) for valid vs test

Outputs: outputs/02_*.csv, outputs/02_nn_similarity.png
"""
import glob
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from common import (AUTOML_DIR, N_JOBS, OUT_DIR, RUNS, SPLIT_COLORS, SPLITS,
                    STANDARDIZED_CACHE, load_raw, load_split_csv)

RDLogger.DisableLog("rdApp.*")

raw = load_raw()
std = pd.read_csv(STANDARDIZED_CACHE, index_col=0)

# --- a. standardization: what changed ---------------------------------------
m = raw[["inchikey", "canon_smiles", "sweet"]].merge(
    std.rename(columns={"ids": "inchikey", "smiles": "std_smiles"}), on="inchikey", how="left",
    suffixes=("", "_std"),
)
changed = m["std_smiles"].notna() & (m["canon_smiles"] != m["std_smiles"])
std_changes = pd.DataFrame([
    {"what": "raw rows", "n": len(raw)},
    {"what": "rows lost in standardization (mol=None)", "n": int(m["std_smiles"].isna().sum())},
    {"what": "SMILES changed by standardizer", "n": int(changed.sum())},
    {"what": "multi-fragment in raw ('.')", "n": int(raw["canon_smiles"].str.contains(r"\.").sum())},
    {"what": "multi-fragment after standardization", "n": int(std["smiles"].astype(str).str.contains(r"\.").sum())},
    {"what": "label changed by standardization", "n": int((m["std_smiles"].notna() & (m["sweet"] != m["sweet_std"])).sum())},
])
std_changes.to_csv(f"{OUT_DIR}/02_standardization_changes.csv", index=False)
m[changed][["inchikey", "canon_smiles", "std_smiles", "sweet"]].head(30).to_csv(
    f"{OUT_DIR}/02_standardization_examples.csv", index=False)
print(std_changes.to_string(index=False))


def has_carbon(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and any(a.GetSymbol() == "C" for a in mol.GetAtoms())


# --- reproduce _deduplicate_structures on the cache (same order) ------------
groups = {}
for i, smi in enumerate(std["smiles"]):
    groups.setdefault(smi, []).append(i)
keep, n_dup, n_conf = [], 0, 0
labels = std["sweet"].to_numpy()
for idxs in groups.values():
    if len(idxs) == 1:
        keep.append(idxs[0])
    elif len(set(labels[idxs])) > 1:
        n_conf += len(idxs)
    else:
        keep.append(idxs[0])
        n_dup += len(idxs) - 1
dedup = std.iloc[sorted(keep)].reset_index(drop=True)

funnel, split_rows, bug_rows, cache_rows = [], [], [], []
nn_records = []

for threshold, run_dir in RUNS.items():
    parts = {s: load_split_csv(threshold, s) for s in SPLITS}
    all_ids = pd.concat([parts[s]["ids"] for s in SPLITS])

    # split input = dedup order minus molecules the featurizer dropped. We
    # don't know those directly, so: in-dedup ids not present in any split
    # are either featurizer drops or lost to the splitter bug.
    absent = dedup[~dedup["ids"].isin(set(all_ids))]
    funnel.append({
        "threshold": threshold, "raw": len(raw), "standardized": len(std),
        "dedup_collapsed": n_dup, "dedup_conflicting_dropped": n_conf,
        "after_dedup": len(dedup), "absent_from_all_splits": len(absent),
        "rows_in_splits": len(all_ids), "unique_ids_in_splits": all_ids.nunique(),
    })

    # --- b. sizes, balance, overlaps
    for s in SPLITS:
        df = parts[s]
        split_rows.append({
            "threshold": threshold, "split": s, "n": len(df),
            "frac": round(len(df) / len(all_ids), 4),
            "n_sweet": int(df["sweet"].sum()), "sweet_frac": round(df["sweet"].mean(), 4),
            "duplicated_ids": int(df["ids"].duplicated().sum()),
            "no_carbon": int(sum(not has_carbon(x) for x in df["smiles"])),
        })
    overlaps = {
        f"{a}&{b}": len(set(parts[a]["ids"]) & set(parts[b]["ids"]))
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    }

    # --- c. index-0 bug
    dup_ids = all_ids[all_ids.duplicated(keep=False)].unique()
    for dup_id in dup_ids:
        bug_rows.append({
            "threshold": threshold, "kind": "duplicated", "id": dup_id,
            "position_in_dedup_order": int(dedup.index[dedup["ids"] == dup_id][0]),
            "label": dedup.loc[dedup["ids"] == dup_id, "sweet"].iloc[0],
            "counts": {s: int((parts[s]["ids"] == dup_id).sum()) for s in SPLITS},
        })
    for _, row in absent.iterrows():
        pos = int(row.name)
        first_of_class = pos == int(dedup.index[dedup["sweet"] == row["sweet"]][0])
        bug_rows.append({
            "threshold": threshold, "kind": "absent", "id": row["ids"],
            "position_in_dedup_order": pos, "label": row["sweet"],
            "counts": f"first of its class in dedup order: {first_of_class}",
        })
    bug_rows.append({"threshold": threshold, "kind": "overlaps", "id": "", "counts": overlaps})

    # --- d. cache json vs CSVs
    for path in glob.glob(f"{AUTOML_DIR}/checkpoints/cache/merged_foodb_gold_sweetnessv3_filtered_split_similarity_*_ids.json"):
        with open(path) as f:
            cached = json.load(f)
        match = {s: list(map(str, cached[s])) == list(parts[s]["ids"].astype(str)) for s in SPLITS}
        cache_rows.append({"threshold": threshold, "cache_file": path.split("/")[-1], **match})
    with open(f"{run_dir}/config.json") as f:
        split_cfg = json.load(f)["split"]
    cache_rows.append({
        "threshold": threshold, "cache_file": "expected hash from config",
        "train": hashlib.sha256(json.dumps(split_cfg, sort_keys=True).encode()).hexdigest()[:12],
    })

    # --- e. nearest-neighbour Tanimoto to train (splitter's own fingerprint)
    def fps(smiles):
        return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, 1024) for x in smiles]

    train_fps = fps(parts["train"]["smiles"])

    def nn_chunk(chunk):
        return [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in chunk]

    for s in ["valid", "test"]:
        q = fps(parts[s]["smiles"])
        # one chunk per worker: train_fps is shipped once per task
        step = int(np.ceil(len(q) / N_JOBS))
        chunks = [q[i:i + step] for i in range(0, len(q), step)]
        sims = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(nn_chunk)(c) for c in chunks))
        nn_records.append(pd.DataFrame({
            "threshold": threshold, "split": s, "ids": parts[s]["ids"].values,
            "sweet": parts[s]["sweet"].values, "nn_sim_train": sims,
        }))

pd.DataFrame(funnel).to_csv(f"{OUT_DIR}/02_funnel.csv", index=False)
pd.DataFrame(split_rows).to_csv(f"{OUT_DIR}/02_split_composition.csv", index=False)
pd.DataFrame(bug_rows).to_csv(f"{OUT_DIR}/02_splitter_index0_bug.csv", index=False)
pd.DataFrame(cache_rows).to_csv(f"{OUT_DIR}/02_split_cache_vs_csv.csv", index=False)

nn = pd.concat(nn_records, ignore_index=True)
nn.to_csv(f"{OUT_DIR}/02_nn_similarity.csv", index=False)
nn_summary = nn.groupby(["threshold", "split", "sweet"])["nn_sim_train"].describe(
    percentiles=[0.1, 0.25, 0.5]).round(3)
nn_summary["frac_identical_fp(=1.0)"] = nn.groupby(["threshold", "split", "sweet"])["nn_sim_train"].apply(
    lambda s: round((s >= 0.999).mean(), 3))
nn_summary.to_csv(f"{OUT_DIR}/02_nn_similarity_summary.csv")

print(pd.DataFrame(funnel).to_string(index=False))
print(pd.DataFrame(split_rows).to_string(index=False))
print(pd.DataFrame(bug_rows).to_string(index=False))
print(pd.DataFrame(cache_rows).to_string(index=False))
print(nn_summary.to_string())

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for row, threshold in enumerate(RUNS):
    for col, (label, name) in enumerate([(0, "non-sweet"), (1, "sweet")]):
        ax = axes[row, col]
        for s in ["valid", "test"]:
            v = nn[(nn.threshold == threshold) & (nn.split == s) & (nn.sweet == label)]["nn_sim_train"]
            ax.hist(v, bins=50, range=(0, 1), histtype="step", lw=1.6, color=SPLIT_COLORS[s],
                    density=True, label=f"{s} (median {v.median():.2f}, n={len(v)})")
        ax.set_title(f"threshold {threshold} — {name}")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("max Tanimoto to train (Morgan r2 1024)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_nn_similarity.png", dpi=130)
