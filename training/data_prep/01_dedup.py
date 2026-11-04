
import os

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = "data"
os.makedirs(OUT, exist_ok=True)

CSV = "data/merged_foodb_gold_sweetnessv3.csv"  # raw merged source, not shipped in this repo
COL_SMILES = "smiles"
COL_LABEL = "sweet"

resumo = []


def registar(passo, d):
    """Guarda quantos registos (doces / nao doces) ha em d neste passo."""
    doces = int((d[COL_LABEL] == 1).sum())
    resumo.append({"passo": passo, "registos": len(d),
                   "doces": doces, "nao_doces": len(d) - doces})


# --- 1. canonicalizar sem estereoquimica -------------------------------------

def canon_flat(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


df = pd.read_csv(CSV)
df[COL_LABEL] = df[COL_LABEL].astype(int)
registar("dataset original", df)

df["smiles_original"] = df[COL_SMILES]
df[COL_SMILES] = df[COL_SMILES].apply(canon_flat)
registar("removidos: SMILES invalidos", df[df[COL_SMILES].isna()])
df = df.dropna(subset=[COL_SMILES]).reset_index(drop=True)
registar("depois de canonicalizar", df)


# --- 2. conflitos de label ---------------------------------------------------

n_labels = df.groupby(COL_SMILES)[COL_LABEL].transform("nunique")
conflitos = df[n_labels > 1]
registar("removidos: labels em conflito", conflitos)
conflitos.to_csv(f"{OUT}/conflitos_labels.csv", index=False)
n_mol_conflito = conflitos[COL_SMILES].nunique()


# --- 3. um registo por molecula, conflituosos fora ---------------------------

sem_conflitos = df[n_labels == 1]
limpo = (sem_conflitos
           .drop_duplicates(COL_SMILES, keep="first")
           .reset_index(drop=True))
registar("removidos: duplicados",
         sem_conflitos[sem_conflitos.duplicated(COL_SMILES, keep="first")])
registar("dataset limpo", limpo)

limpo.to_csv(f"{OUT}/sweet26_clean.csv", index=False)


# --- 4. resumo para os metodos -----------------------------------------------

resumo = pd.DataFrame(resumo)
resumo.to_csv(f"{OUT}/resumo_limpeza.csv", index=False)

print(resumo.to_string(index=False))
print(f"\nmoleculas distintas com labels em conflito: {n_mol_conflito}")
print(f"-> {OUT}/sweet26_clean.csv")
print(f"-> {OUT}/resumo_limpeza.csv")
