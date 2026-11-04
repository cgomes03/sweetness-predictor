"""
Sweetness predictor: P(sweet) for SMILES, from one trained DeepMol model.

Default model: morgan + SVC (radius 1, 2048 bits, RBF, C 9.36), similarity
split threshold 0.2, the best valid-F1 pair of the 2026-10-29 AutoML search on
the CLEANED dataset (valid F1 0.8149, test F1 0.8288).

Why not the earlier atom_pair + RandomForest (test F1 0.9779)
-------------------------------------------------------------
That 0.9779 was leakage, not performance. The dataset it was trained on held
duplicate stereoisomers and salts of the same molecule, and the similarity
splitter put copies of one compound on both sides of the split. Cleaning the
dataset -- canonical SMILES without stereochemistry, conflicting labels
dropped, one record per molecule (data/sweet26_clean.csv, built by
training/data_prep/01_dedup.py) -- takes the honest test F1 of the best model
down to 0.83.

One scoring path, two environments
----------------------------------
There is a single implementation of "standardize, featurize, predict"
(`_Scorer` below), so every caller gets the same numbers.

deepmol brings TensorFlow and a pinned RDKit, so it usually lives in its own
conda env. The env boundary is handled inside this file, and callers do not
need to know it is there:

    in the deepmol env   SweetnessPredictor ─────────────> _Scorer (in-process)
    anywhere else        SweetnessPredictor ─pipe─> this file --serve ─> _Scorer

The subprocess is persistent: importing deepmol costs ~12 s, paid once per run
rather than once per call.

Usage
-----
    from sweetness_predictor import SweetnessPredictor
    p = SweetnessPredictor()
    p.calculate_scores(["OC[C@H]1O...", "CCO"])   # -> array([0.98, 0.01])

Self-test:

    python sweetness_predictor.py            # a few known molecules
    python sweetness_predictor.py --test-set # metrics on the saved splits

Layout this file expects
------------------------
    sweetness_predictor.py
    model/config.json                     the training run's config
    model/best_pipeline_morgan_SVC/       featurizer.pkl, pipeline.json, model/
    data/splits/{train,valid,test}_dataset.csv

Set SWEETNESS_MODEL_DIR to point `model/` somewhere else, which is what a
checkout keeping the 233 MB model.pkl outside git needs.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import SanitizeMol

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Default model
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

# The model lives next to this file. SWEETNESS_MODEL_DIR overrides it, for a
# checkout that keeps the 233 MB model.pkl outside the repository.
RUN_DIR = Path(os.environ.get("SWEETNESS_MODEL_DIR", _HERE / "model"))
MODEL_SUBDIR = "best_pipeline_morgan_SVC"
MODEL_PATH = RUN_DIR / MODEL_SUBDIR

_SPLITS_DIR = _HERE / "data" / "splits"
TRAIN_SET_PATH = _SPLITS_DIR / "train_dataset.csv"
VALID_SET_PATH = _SPLITS_DIR / "valid_dataset.csv"
TEST_SET_PATH = _SPLITS_DIR / "test_dataset.csv"

# Sweetness is maximized, so anything unscorable gets the floor.
WORST_FITNESS = 0.0

# Conda env holding deepmol. Resolved under the running user's home rather
# than hardcoded to one path, so this works for whoever runs it.
DEEPMOL_ENV = "deepmol"

# Root of the trained-model tree, for resolving a bare run id.
CHECKPOINTS_ROOT = RUN_DIR.parent
_PIPELINES = ()


def deepmol_available() -> bool:
    """Whether this interpreter can import deepmol, without paying the import."""
    import importlib.util

    try:
        return importlib.util.find_spec("deepmol") is not None
    except (ImportError, ValueError):
        return False


def find_deepmol_python() -> str:
    """Path to the deepmol env's interpreter."""
    home = Path.home()
    for base in ("miniconda3", "anaconda3", "conda", "miniforge3", "mambaforge"):
        candidate = home / base / "envs" / DEEPMOL_ENV / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find the '{DEEPMOL_ENV}' conda env's python. Pass "
        "python_executable=... to SweetnessPredictor if it lives elsewhere."
    )


def resolve_checkpoint(checkpoint) -> Path:
    """Accepts an absolute path, one relative to CHECKPOINTS_ROOT, or a run id."""
    path = Path(checkpoint)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    direct = CHECKPOINTS_ROOT / path
    if direct.exists():
        return direct

    for pipeline in _PIPELINES:
        candidate = CHECKPOINTS_ROOT / pipeline / "checkpoints" / path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint}' not found under {CHECKPOINTS_ROOT} "
        f"(searched directly and in {'/'.join(_PIPELINES)}/checkpoints/)"
    )


# --------------------------------------------------------------------------
# SMILES helpers -- shared by the scorer and both backends
# --------------------------------------------------------------------------
def as_smiles(smiles_or_mol) -> str:
    """Accepts a SMILES string, an RDKit Mol or None, as the EA passes all three."""
    if smiles_or_mol is None:
        return ""
    if isinstance(smiles_or_mol, str):
        return smiles_or_mol
    try:
        return Chem.MolToSmiles(smiles_or_mol)
    except Exception:
        return ""


def is_scorable(smi) -> bool:
    """
    Whether a SMILES is worth sending to the model.

    An empty SMILES is *valid* to RDKit -- it yields an empty molecule whose
    all-zero fingerprint the model scores around 0.42. Left in, every molecule
    the generator failed to produce would collect that instead of
    worst_fitness, and an EA would be free to drift towards candidates that do
    not parse.
    """
    if not smi or not str(smi).strip():
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return False
    return mol is not None and mol.GetNumAtoms() > 0


def build_featurizer(name, params, n_jobs):
    """
    For the older `manual` runs, which save a bare model plus a featurizer
    name. Kept in step with featurize() in new_analysis/manual/scripts/
    sweet_model_pipeline.py -- these names are what the run configs record.
    """
    from deepmol.compound_featurization import (
        AtomPairFingerprint,
        MACCSkeysFingerprint,
        MorganFingerprint,
        TwoDimensionDescriptors,
    )

    if name == "2d_descriptors":
        return TwoDimensionDescriptors(n_jobs=n_jobs, **params)
    if name in ("morgan", "morgan_fingerprints"):
        return MorganFingerprint(n_jobs=n_jobs, **params)
    if name == "maccs":
        return MACCSkeysFingerprint(n_jobs=n_jobs, **params)
    if name == "atom_pair":
        return AtomPairFingerprint(n_jobs=n_jobs, **params)
    raise ValueError(f"Unknown featurizer: {name}")


# --------------------------------------------------------------------------
# The scorer -- the only place a molecule is turned into a probability
# --------------------------------------------------------------------------
class _Scorer:
    """
    Loads the model and scores batches. Needs deepmol, so it only ever runs in
    the deepmol env: directly when that is where we already are, otherwise
    inside the worker this file spawns.

    Two saved layouts, told apart by a pipeline.json in the model folder:

      manual/checkpoints/<RUN_ID>/
        config.json     names the featurizer
        rf_model/       a DeepMol SklearnModel -- featurize here, then predict

      automl/checkpoints/<RUN_ID>/best_pipeline_<feat>_<model>/
        pipeline.json   marks it as a Pipeline
        config.json     lists the Pipeline's steps
        featurizer.pkl  the featurizer Optuna chose for this trial
        model/

    The automl layout is the safer of the two: Optuna picks a featurizer per
    trial, so no single name could be recorded at the run level, and each saved
    pipeline instead carries its own. Nothing here chooses a featurizer for it,
    so nothing here can choose wrong.
    """

    def __init__(self, model_path=MODEL_PATH, n_jobs=4):
        self.model_path = Path(model_path)
        self.n_jobs = n_jobs

        if not self.model_path.exists():
            raise FileNotFoundError(f"No model at {self.model_path}")

        self.mode = "pipeline" if (self.model_path / "pipeline.json").exists() else "sklearn"
        run_config = self._read_run_config()

        if self.mode == "pipeline":
            self.featurizer_name = "(bundled in pipeline)"
            # Whether the Pipeline carries a standardizer step decides who
            # standardizes. Read from its own config rather than assumed: this
            # checkpoint's pipeline is featurizer + model only, and assuming
            # otherwise silently scores unstandardized molecules with a model
            # trained on standardized ones.
            self.pipeline_standardizes = self._pipeline_has_standardizer()
            self.standardizer_name = (
                "(bundled in pipeline)" if self.pipeline_standardizes
                else run_config.get("standardizer", "chembl")
            )
        else:
            # Read from the run's own config, and refuse to guess when it is
            # not recorded. A model trained on 2d_descriptors but scored with
            # Morgan fingerprints does not error -- it returns confident
            # nonsense, and an EA would optimize against it for hours none the
            # wiser. morgan and atom_pair are both 2048-wide, so swapping those
            # two does not even trip sklearn's feature-count check.
            if "featurizer" not in run_config:
                hint = ""
                if "automl" in run_config:
                    candidates = run_config["automl"].get("featurizer_candidates", [])
                    hint = (
                        f" This is an automl run (candidates: {', '.join(candidates)}); "
                        "Optuna picks one per trial, so the run config names none. "
                        "Point model_path at one of its pipeline folders instead -- "
                        "best_pipeline_<feat>_<model>/ or models_<feat>_<model>/trial_N/."
                    )
                raise ValueError(
                    f"{self.model_path} holds no pipeline.json and its run config "
                    "records no 'featurizer', so there is no way to know how this "
                    "model was featurized. Refusing to guess: the wrong featurizer "
                    "does not error, it returns plausible-looking scores that mean "
                    "nothing." + hint
                )
            self.featurizer_name = run_config["featurizer"]
            self.featurizer_params = run_config.get("featurizer_params") or {}
            self.standardizer_name = run_config.get("standardizer", "chembl")
            self.pipeline_standardizes = False

        self._model = None
        self._featurizer = None

    # -- config ----------------------------------------------------------

    def _read_run_config(self):
        """
        The run's config.json -- one level up from a pipeline folder, or in the
        model folder's own directory for the manual layout.
        """
        for candidate in (self.model_path.parent / "config.json",
                          self.model_path / "config.json"):
            if candidate.exists():
                with open(candidate) as f:
                    config = json.load(f)
                # A Pipeline's own config.json is a step list keyed "0", "1";
                # that is not the run config.
                if not all(key.isdigit() for key in config):
                    return config
        if self.mode == "pipeline":
            # A Pipeline bundles its featurizer, so a missing run config only
            # costs us the standardizer name, which defaults to chembl.
            return {}
        raise FileNotFoundError(
            f"No run config.json near {self.model_path} -- the featurizer used "
            "at training time cannot be recovered, and guessing it would "
            "silently produce confident nonsense."
        )

    def _pipeline_has_standardizer(self) -> bool:
        """
        Whether the saved Pipeline includes a standardizer step.

        Its config.json is a step list: {"0": {"name": "featurizer", ...},
        "1": {"name": "model", ...}}. No deepmol needed to read it, so the
        answer is available before the model is loaded.
        """
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            return False
        try:
            with open(config_path) as f:
                steps = json.load(f)
        except Exception:
            return False
        return any(
            "standardiz" in str(step.get("name", "")).lower()
            for step in steps.values()
            if isinstance(step, dict)
        )

    # -- lazy loading ----------------------------------------------------

    @property
    def model(self):
        """Loaded on first use and reused -- the expensive part."""
        if self._model is None:
            if self.mode == "pipeline":
                from deepmol.pipeline import Pipeline
                self._model = Pipeline.load(str(self.model_path))
            else:
                from deepmol.models import SklearnModel
                self._model = SklearnModel.load(str(self.model_path))
        return self._model

    @property
    def featurizer(self):
        if self._featurizer is None and self.mode == "sklearn":
            self._featurizer = build_featurizer(
                self.featurizer_name, self.featurizer_params, self.n_jobs
            )
        return self._featurizer

    # -- scoring ---------------------------------------------------------

    @staticmethod
    def _sanitize(dataset):
        """
        Re-sanitizes molecules after standardization, which is what makes them
        featurizable at all: the standardizer leaves mols whose ring info is
        not initialized, and both Morgan fingerprints and the 2D descriptors
        then fail on every single molecule with "RingInfo not initialized" --
        scoring a whole population as worst_fitness without raising anything.

        Mirrors _sanitize_mols() in new_analysis/manual/scripts/
        sweet_model_pipeline.py, so scoring matches training.
        """
        fixed = []
        for mol in dataset.mols:
            if mol is None:
                fixed.append(None)
                continue
            try:
                SanitizeMol(mol)
                fixed.append(mol)
            except Exception:
                fixed.append(None)
        dataset._mols = np.array(fixed, dtype=object)
        return dataset

    def score(self, smiles_list) -> np.ndarray:
        """
        One raw probability per input, in the order given. NaN for molecules
        that could not be scored -- the caller decides what that is worth.

        Standardization and featurization both drop molecules they cannot
        handle, and generated molecules fail far more often than curated ones,
        so results are mapped back by id. Zipping the shrunken output onto the
        input would shift every score after the first failure onto the wrong
        molecule -- silently, and in an EA's favour.
        """
        from deepmol.datasets import SmilesDataset

        scores = np.full(len(smiles_list), np.nan, dtype=float)
        if len(smiles_list) == 0:
            return scores

        keep = [i for i, smi in enumerate(smiles_list) if is_scorable(smi)]
        if not keep:
            return scores
        to_score = [smiles_list[i] for i in keep]

        # Each molecule carries its position in to_score as its id, so the
        # survivors can be mapped back after rows are dropped.
        try:
            dataset = SmilesDataset(
                smiles=list(to_score), ids=[str(i) for i in range(len(to_score))]
            )
        except Exception as e:
            print(f"could not build dataset: {e}", file=sys.stderr)
            return scores

        if not self.pipeline_standardizes:
            if self.standardizer_name == "chembl":
                from deepmol.standardizer import ChEMBLStandardizer
                ChEMBLStandardizer().standardize(dataset, inplace=True)
            elif self.standardizer_name not in (None, "", "none"):
                raise ValueError(f"Unknown standardizer: {self.standardizer_name}")
            # Between standardizing and featurizing -- see _sanitize.
            dataset = self._sanitize(dataset)

        if self.mode == "pipeline":
            # return_invalid=True keeps one row per molecule (NaN where
            # featurization failed), so ids stay aligned.
            proba = np.asarray(self.model.predict_proba(dataset, return_invalid=True))
        else:
            self.featurizer.featurize(dataset, inplace=True)
            if len(dataset.smiles) == 0:
                return scores
            proba = np.asarray(self.model.predict_proba(dataset))

        if proba.size == 0:
            return scores

        # Binary classifiers return either (N, 2) or a flat (N,).
        if proba.ndim == 2 and proba.shape[1] >= 2:
            proba = proba[:, 1]
        proba = proba.ravel()

        keep_arr = np.asarray(keep)
        survivors = np.array([int(i) for i in dataset.ids])
        n = min(len(survivors), len(proba))
        if n:
            scores[keep_arr[survivors[:n]]] = proba[:n]
        return scores

    def describe(self) -> dict:
        return {
            "model": str(self.model_path),
            "mode": self.mode,
            "featurizer": self.featurizer_name,
            "standardizer": self.standardizer_name,
        }


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class _InProcessBackend:
    """Used when this interpreter already has deepmol."""

    def __init__(self, model_path, n_jobs=4):
        self.scorer = _Scorer(model_path, n_jobs=n_jobs)

    def score(self, smiles_list) -> Optional[np.ndarray]:
        return self.scorer.score(smiles_list)

    def describe(self) -> dict:
        return dict(self.scorer.describe(), backend="in-process")

    def close(self):
        pass


class _WorkerBackend:
    """
    Used when this interpreter has no deepmol: spawns `this file --serve` under
    the deepmol env's python and talks JSON-lines over a pipe.

    The process is kept alive between calls. The 12 s of imports is then paid
    once per run rather than once per generation, which is the whole reason
    this is a persistent worker and not reg_api's one-shot script.

    Protocol, one JSON object per line each way:

        out  {"smiles": [...]}        in   {"scores": [float|null, ...]}
        out  {"cmd": "quit"}          (exits)
        on startup, once loaded:      {"ready": true, ...}
        on failure:                   {"error": "..."}
    """

    def __init__(self, model_path, n_jobs=4, python_executable=None,
                 startup_timeout=600, score_timeout=1800, log_path=None):
        # Set before anything that can raise: close() runs from __del__ and
        # atexit on half-built objects too, and a failed __init__ would
        # otherwise bury the real error under an AttributeError.
        self._proc = None
        self._log = None
        self._restarts = 0

        self.model_path = Path(model_path)
        self.n_jobs = n_jobs
        self.startup_timeout = startup_timeout
        self.score_timeout = score_timeout
        self._ready = {}

        self.python_executable = python_executable or find_deepmol_python()
        if not os.path.exists(self.python_executable):
            raise FileNotFoundError(
                f"No interpreter at {self.python_executable}. Expected the "
                f"'{DEEPMOL_ENV}' conda env; pass python_executable=... if it "
                "lives elsewhere."
            )

        if log_path is None:
            handle, log_path = tempfile.mkstemp(prefix="sweetness_worker_", suffix=".log")
            os.close(handle)
        self.log_path = log_path

        # So a crashing run does not leave the worker behind holding its model.
        atexit.register(self.close)

    # -- lifecycle -------------------------------------------------------

    def _log_tail(self, n_lines=15):
        try:
            with open(self.log_path) as f:
                return "".join(f.readlines()[-n_lines:]).strip()
        except Exception:
            return "(no log available)"

    def _start(self):
        command = [
            self.python_executable, os.path.abspath(__file__), "--serve",
            "--model-path", str(self.model_path),
            "--n-jobs", str(self.n_jobs),
        ]
        print(
            f"Starting sweetness worker ({DEEPMOL_ENV} env) for "
            f"{self.model_path.name} -- loading deepmol, ~12 s"
        )

        # stderr goes to a file, not a pipe. deepmol logs and progress bars
        # freely, and a pipe nobody drains fills its buffer and blocks the
        # worker mid-write, deadlocking the run. A file also survives the
        # worker, so a startup failure can still be explained.
        self._log = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1,
        )

        message = self._read_message(self.startup_timeout)
        if message is None:
            self.close()
            raise RuntimeError(
                f"Sweetness worker did not become ready within "
                f"{self.startup_timeout}s.\nWorker log ({self.log_path}):\n"
                f"{self._log_tail()}"
            )
        if "error" in message:
            self.close()
            raise RuntimeError(
                f"Sweetness worker failed to start: {message['error']}\n"
                f"Worker log ({self.log_path}):\n{self._log_tail()}"
            )

        self._ready = message
        print(
            f"Sweetness worker ready: {self.model_path.name} [{message.get('mode')}] "
            f"(featurizer={message.get('featurizer')}, "
            f"standardizer={message.get('standardizer')})"
        )

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _read_message(self, timeout):
        """
        Reads one JSON line from the worker, or None on timeout or EOF.

        select() rather than a bare readline(): a worker that dies mid-batch,
        or wedges, would otherwise block the caller forever with no diagnostic.
        """
        if self._proc is None or self._proc.stdout is None:
            return None

        stream = self._proc.stdout
        ready, _, _ = select.select([stream], [], [], timeout)
        if not ready:
            return None

        line = stream.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            # The worker redirects fd 1 to stderr before importing anything,
            # so this should not happen; if it does, the log says what leaked.
            print(f"Sweetness: unparseable worker output: {line[:200]!r}", file=sys.stderr)
            return None

    def close(self):
        """Shuts the worker down. Safe to call repeatedly."""
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    # Ask it to quit, then escalate: terminate, then kill.
                    if proc.stdin is not None:
                        try:
                            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                            proc.stdin.flush()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
            except Exception:
                pass
            self._proc = None

        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None

    def __del__(self):
        self.close()

    # -- scoring ---------------------------------------------------------

    def score(self, smiles_list) -> Optional[np.ndarray]:
        """Scores a batch. None means the batch could not be scored at all."""
        # One retry, on a fresh worker: a worker killed by the OS (OOM is the
        # realistic one) would otherwise fail every remaining generation.
        for attempt in (0, 1):
            try:
                if not self._alive():
                    self._start()
                scores = self._request(smiles_list)
            except Exception as e:
                print(f"Sweetness: scoring failed - {e}", file=sys.stderr)
                self.close()
                if attempt == 0:
                    continue
                return None

            if scores is not None:
                break

            # No scores and no exception means the worker went away mid-batch.
            self.close()
            if attempt == 0:
                self._restarts += 1
                print(
                    f"Sweetness: worker died mid-batch, restarting "
                    f"(restart #{self._restarts}). Log: {self.log_path}",
                    file=sys.stderr,
                )
        else:
            return None

        if scores is None:
            return None

        # null is the worker's NaN -- JSON has no NaN.
        out = np.array([np.nan if s is None else float(s) for s in scores], dtype=float)
        if len(out) != len(smiles_list):
            # The worker aligns its output by id, so this should not happen.
            # Returning a misaligned array would hand the caller other
            # molecules' scores, so refuse it rather than pass it on.
            print(
                f"Sweetness: got {len(out)} scores for {len(smiles_list)} "
                "molecules, discarding batch",
                file=sys.stderr,
            )
            return None
        return out

    def _request(self, smiles_list):
        """Sends one batch and waits for its scores. None if the worker went away."""
        if self._proc is None or self._proc.stdin is None:
            return None
        try:
            self._proc.stdin.write(json.dumps({"smiles": list(smiles_list)}) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None

        message = self._read_message(self.score_timeout)
        if message is None:
            return None
        if "error" in message:
            print(f"Sweetness: worker error - {message['error']}", file=sys.stderr)
            return None
        return message.get("scores")

    def describe(self) -> dict:
        return {
            "model": str(self.model_path),
            "mode": self._ready.get("mode", "(not started)"),
            "featurizer": self._ready.get("featurizer", "(not started)"),
            "standardizer": self._ready.get("standardizer", "(not started)"),
            "backend": f"worker ({self.python_executable})",
        }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
class SweetnessPredictor:
    """
    P(sweet) for SMILES or RDKit Mols.

    The same scoring interface the EA's engines expose (calculate_scores /
    calculate_score / method_str, worst_fitness 0.0), so a case study or a
    training metric can call it directly.

    Which backend runs is decided once, here, by whether this interpreter has
    deepmol -- so the same two lines work in both envs.
    """

    def __init__(self, model_path=MODEL_PATH, threshold=None,
                 worst_fitness=WORST_FITNESS, n_jobs=4, backend=None,
                 python_executable=None, **worker_kwargs):
        """
        model_path     A pipeline/model folder, a run id, or a path relative to
                       CHECKPOINTS_ROOT. Defaults to the morgan + SVC winner.
        threshold      If set, scores below it collapse to worst_fitness. None
                       returns the raw probability, which is what an EA wants:
                       a continuous gradient to climb rather than a cliff. Each
                       run's config.json records the threshold its reported
                       metrics used; that is for reading metrics, not a default
                       to quietly apply to an objective.
        worst_fitness  Score for molecules that cannot be scored.
        backend        "in-process", "worker", or None to decide automatically.
        """
        self.model_path = resolve_checkpoint(model_path)
        self.threshold = threshold
        self.worst_fitness = worst_fitness

        if backend is None:
            backend = "in-process" if deepmol_available() else "worker"
        self.backend_name = backend

        if backend == "in-process":
            self._backend = _InProcessBackend(self.model_path, n_jobs=n_jobs)
        elif backend == "worker":
            self._backend = _WorkerBackend(
                self.model_path, n_jobs=n_jobs,
                python_executable=python_executable, **worker_kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    # -- scoring ---------------------------------------------------------

    def calculate_scores(self, smiles_list) -> np.ndarray:
        """
        One score per input, in input order. Molecules that cannot be scored
        get worst_fitness.

        Batched because callers score a whole population or sample at a time:
        featurizing in one pass costs ~8 ms per molecule against ~65 ms
        looping one at a time.
        """
        if smiles_list is None or len(smiles_list) == 0:
            return np.array([])

        smiles = [as_smiles(s) for s in smiles_list]
        results = np.full(len(smiles), self.worst_fitness, dtype=float)

        raw = self._backend.score(smiles)
        if raw is None:
            return results

        scored = ~np.isnan(raw)
        results[scored] = raw[scored]

        if self.threshold is not None:
            results[results < self.threshold] = self.worst_fitness
        return results

    def calculate_score(self, smiles_or_mol) -> float:
        """Single-molecule score."""
        scores = self.calculate_scores([smiles_or_mol])
        return float(scores[0]) if len(scores) else self.worst_fitness

    def __call__(self, smiles_list) -> np.ndarray:
        return self.calculate_scores(smiles_list)

    # -- housekeeping ----------------------------------------------------

    def describe(self) -> dict:
        return self._backend.describe()

    def close(self):
        """Releases the worker, if there is one. Safe to call repeatedly."""
        self._backend.close()

    def method_str(self):
        return f"Sweetness_{self.model_path.name}"


_default_predictor = None


def predict_sweetness(smiles: List[str], worst_fitness: float = WORST_FITNESS) -> np.ndarray:
    """Function form, sharing one loaded model across calls."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = SweetnessPredictor()
    _default_predictor.worst_fitness = worst_fitness
    return _default_predictor.calculate_scores(smiles)


# --------------------------------------------------------------------------
# Worker mode -- `python sweetness_predictor.py --serve`, in the deepmol env
# --------------------------------------------------------------------------
def serve(model_path, n_jobs=4) -> int:
    """
    Reads {"smiles": [...]} lines and answers {"scores": [...]} lines, holding
    the loaded model between batches.

    Scores come back aligned with the SMILES that were sent: one per input, in
    order, null for anything that could not be scored.
    """
    # Claim a private channel for the protocol BEFORE importing anything heavy.
    #
    # deepmol, TensorFlow and RDKit all write to stdout as a side effect of
    # being imported ("No normalization for SPS. Feature removed!", cuDNN
    # registration errors, deepmol's own logger). Any one of those lines lands
    # in the middle of the JSON stream and the parent cannot parse it.
    #
    # So: duplicate the real stdout to a private fd, then point fd 1 at stderr.
    # Everything that thinks it is printing to stdout from here on goes to
    # stderr instead, where the parent collects it as a log, and fd 1's
    # duplicate carries only protocol messages.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol = os.fdopen(protocol_fd, "w")

    def send(obj):
        """Writes one protocol message. Flushed so the parent never waits."""
        protocol.write(json.dumps(obj) + "\n")
        protocol.flush()

    try:
        scorer = _Scorer(model_path, n_jobs=n_jobs)
        scorer.model  # pay the load now, so "ready" means ready
    except Exception as e:
        # Reported over the protocol so the parent can raise something useful
        # instead of just timing out waiting for "ready".
        send({"error": f"failed to load model: {type(e).__name__}: {e}"})
        return 1

    send(dict(scorer.describe(), ready=True))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as e:
            send({"error": f"unparseable request: {e}"})
            continue

        if request.get("cmd") == "quit":
            break

        try:
            scores = scorer.score(request.get("smiles") or [])
            # JSON has no NaN, so unscorable comes back as null.
            send({"scores": [None if np.isnan(s) else float(s) for s in scores]})
        except Exception as e:
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            send({"error": f"{type(e).__name__}: {e}"})

    return 0


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
KNOWN_MOLECULES = {
    "sucrose": "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "aspartame": "COC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CC(=O)O",
    "saccharin": "O=C1NS(=O)(=O)c2ccccc21",
    "caffeine (bitter)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ethanol": "CCO",
    "invalid SMILES": "gsserdhserds",
}


def test_single_molecules(predictor=None):
    predictor = predictor or SweetnessPredictor()
    print(f"\nBackend: {predictor.describe()}")
    preds = predictor.calculate_scores(list(KNOWN_MOLECULES.values()))
    print("\n--- Single molecules ---")
    for name, p in zip(KNOWN_MOLECULES, preds):
        print(f"  {name:20s} P(sweet) = {p:.3f}")
    assert len(preds) == len(KNOWN_MOLECULES)
    assert preds[-1] == 0.0, "an unparseable SMILES must score worst_fitness"
    return preds


def test_test_dataset(threshold: float = 0.5, predictor=None):
    """
    Metrics on the run's own splits. Reported by the AutoML run for this
    model: valid F1 0.8149, test F1 0.8288.
    """
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        f1_score, matthews_corrcoef, precision_score, recall_score,
        roc_auc_score,
    )

    predictor = predictor or SweetnessPredictor(worst_fitness=np.nan)
    predictor.worst_fitness = np.nan  # unscorable molecules are excluded, not scored 0
    print(f"\nBackend: {predictor.describe()}")

    for label, path in (("Train", TRAIN_SET_PATH),
                        ("Valid", VALID_SET_PATH),
                        ("Test", TEST_SET_PATH)):
        split = pd.read_csv(path, index_col=0)
        probas = predictor.calculate_scores(split["smiles"].tolist())

        ok = ~np.isnan(probas)
        y_true = split["sweet"].to_numpy()[ok].astype(int)
        probas = probas[ok]
        preds = (probas >= threshold).astype(int)

        print(f"\n--- {label} set, threshold {threshold} "
              f"({ok.sum()} / {len(split)} molecules predicted) ---")
        for name, value in {
            "f1_score": f1_score(y_true, preds),
            "precision_score": precision_score(y_true, preds),
            "recall_score": recall_score(y_true, preds),
            "matthews_corrcoef": matthews_corrcoef(y_true, preds),
            "roc_auc_score": roc_auc_score(y_true, probas),
            "average_precision_score": average_precision_score(y_true, probas),
            "balanced_accuracy_score": balanced_accuracy_score(y_true, preds),
            "accuracy_score": accuracy_score(y_true, preds),
        }.items():
            print(f"  {name:25s} {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--serve", action="store_true",
                        help="run as a scoring worker over stdin/stdout")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--test-set", action="store_true",
                        help="metrics on the run's train/valid/test splits")
    parser.add_argument("--backend", choices=["in-process", "worker"], default=None,
                        help="force a backend instead of detecting deepmol")
    args = parser.parse_args()

    if args.serve:
        return serve(args.model_path, n_jobs=args.n_jobs)

    predictor = SweetnessPredictor(
        model_path=args.model_path, n_jobs=args.n_jobs, backend=args.backend
    )
    try:
        if args.test_set:
            test_test_dataset(predictor=predictor)
        else:
            test_single_molecules(predictor=predictor)
    finally:
        predictor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
