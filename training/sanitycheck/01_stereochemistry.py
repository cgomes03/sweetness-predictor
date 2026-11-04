"""
Does stereochemistry survive the pipeline, and does it reach the model?

  1. raw CSV -> ChEMBL-standardized cache: is the stereo (@, /, \\) kept in the SMILES?
  2. atom_pair featurizer: deepmol's AtomPairFingerprint defaults to
     includeChirality=False, so stereoisomers get the SAME feature vector.
     How many stereoisomer groups are there, how many have conflicting
     labels, and how many cross the train/valid/test boundary?
  3. Identical feature vectors across splits (exact duplicates the model
     cannot tell apart) — overall and with conflicting labels.

Outputs: outputs/01_*.csv
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from common import OUT_DIR, RUNS, SPLITS, featurize_split, load_raw, load_split_csv, STANDARDIZED_CACHE

RDLogger.DisableLog("rdApp.*")


def has_stereo(smiles):
    s = pd.Series(smiles).astype(str)
    return s.str.contains("@") | s.str.contains(r"[/\\]")


def flat_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


# --- 1. raw vs standardized -------------------------------------------------
raw = load_raw()
std = pd.read_csv(STANDARDIZED_CACHE, index_col=0)
merged = raw[["inchikey", "canon_smiles", "sweet"]].merge(
    std.rename(columns={"ids": "inchikey", "smiles": "std_smiles", "sweet": "std_sweet"}),
    on="inchikey", how="inner",
)
merged["raw_stereo"] = has_stereo(merged["canon_smiles"]).values
merged["std_stereo"] = has_stereo(merged["std_smiles"]).values
lost = merged[merged["raw_stereo"] & ~merged["std_stereo"]]

std_summary = pd.DataFrame([
    {"step": "raw CSV", "n": len(raw), "with_stereo": int(has_stereo(raw["canon_smiles"]).sum())},
    {"step": "ChEMBL standardized", "n": len(std), "with_stereo": int(has_stereo(std["smiles"]).sum())},
    {"step": "stereo lost in standardization", "n": len(merged), "with_stereo": len(lost)},
])
std_summary["pct"] = (100 * std_summary["with_stereo"] / std_summary["n"]).round(2)
std_summary.to_csv(f"{OUT_DIR}/01_stereo_standardization.csv", index=False)
lost[["inchikey", "canon_smiles", "std_smiles", "sweet"]].to_csv(
    f"{OUT_DIR}/01_stereo_lost_examples.csv", index=False)
print(std_summary.to_string(index=False))

# --- 2/3. per threshold: stereoisomers + identical feature vectors -----------
group_rows, dup_rows, example_rows = [], [], []

for threshold in RUNS:
    frames = []
    for split in SPLITS:
        df = load_split_csv(threshold, split)
        X, y, ids = featurize_split(threshold, split)
        assert list(df["ids"].astype(str)) == list(ids), "split CSV / features misaligned"
        df = df.assign(split=split, fp_hash=[hash(row.tobytes()) for row in X])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["has_stereo"] = has_stereo(data["smiles"]).values
    data["flat"] = [flat_smiles(s) for s in data["smiles"]]

    # Stereoisomer groups: same structure once stereo is removed, >1 member.
    groups = data.groupby("flat").agg(
        n=("ids", "size"),
        n_labels=("sweet", "nunique"),
        n_splits=("split", "nunique"),
        n_fp=("fp_hash", "nunique"),
    )
    stereo_groups = groups[groups["n"] > 1]
    group_rows.append({
        "threshold": threshold,
        "molecules": len(data),
        "molecules_with_stereo_in_smiles": int(data["has_stereo"].sum()),
        "stereoisomer_groups": len(stereo_groups),
        "molecules_in_stereoisomer_groups": int(stereo_groups["n"].sum()),
        "groups_conflicting_label": int((stereo_groups["n_labels"] > 1).sum()),
        "groups_across_splits": int((stereo_groups["n_splits"] > 1).sum()),
        "groups_same_atompair_fp": int((stereo_groups["n_fp"] == 1).sum()),
    })

    # Identical atom-pair vectors: valid/test molecules whose exact feature
    # vector also exists in train.
    train = data[data["split"] == "train"]
    train_labels = train.groupby("fp_hash")["sweet"].agg(lambda s: set(s))
    for split in ["valid", "test"]:
        part = data[data["split"] == split]
        in_train = part["fp_hash"].isin(train_labels.index)
        conflict = [
            in_tr and (row_y not in train_labels[h])
            for h, row_y, in_tr in zip(part["fp_hash"], part["sweet"], in_train)
        ]
        dup_rows.append({
            "threshold": threshold,
            "split": split,
            "n": len(part),
            "same_fp_in_train": int(in_train.sum()),
            "same_fp_in_train_pct": round(100 * in_train.mean(), 2),
            "same_fp_label_conflict": int(np.sum(conflict)),
            "sweet_with_same_fp_in_train": int((in_train & (part["sweet"] == 1)).sum()),
        })

    # A few concrete conflicting stereoisomer groups for the notebook.
    conflicting = stereo_groups[stereo_groups["n_labels"] > 1].index[:10]
    ex = data[data["flat"].isin(conflicting)][["flat", "ids", "smiles", "sweet", "split"]]
    example_rows.append(ex.assign(threshold=threshold).sort_values(["flat", "sweet"]))

pd.DataFrame(group_rows).to_csv(f"{OUT_DIR}/01_stereoisomer_groups.csv", index=False)
pd.DataFrame(dup_rows).to_csv(f"{OUT_DIR}/01_identical_fp_vs_train.csv", index=False)
pd.concat(example_rows).to_csv(f"{OUT_DIR}/01_conflicting_stereoisomers_examples.csv", index=False)
print(pd.DataFrame(group_rows).to_string(index=False))
print(pd.DataFrame(dup_rows).to_string(index=False))

# --- demo: two enantiomers, chirality off vs on ------------------------------
demo = []
for name, a, b in [
    ("D-/L-glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "OC[C@@H]1OC(O)[C@@H](O)[C@H](O)[C@H]1O"),
    ("D-/L-phenylalanine", "N[C@H](Cc1ccccc1)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O"),
]:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    for chir in (False, True):
        fa = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(ma, nBits=2048, includeChirality=chir)
        fb = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mb, nBits=2048, includeChirality=chir)
        demo.append({"pair": name, "includeChirality": chir, "identical_fp": fa == fb})
pd.DataFrame(demo).to_csv(f"{OUT_DIR}/01_chirality_demo.csv", index=False)
print(pd.DataFrame(demo).to_string(index=False))
