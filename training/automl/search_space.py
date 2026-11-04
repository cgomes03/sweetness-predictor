import hashlib
from copy import deepcopy

from deepmol.compound_featurization import (
    AtomPairFingerprint,
    MACCSkeysFingerprint,
    MorganFingerprint,
    TwoDimensionDescriptors,
)
from deepmol.models import SklearnModel
from deepmol.scalers.sklearn_scalers import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost e opcional: so e preciso para model_candidates XGBClassifier
    XGBClassifier = None
import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(pipeline, dataset, metric_fn=f1_score,
                        low=0.3, high=0.7, step=0.01):
    """
    Sweep classification thresholds on a validation set and return
    the threshold that maximises the given metric.
    """
    probas = pipeline.predict_proba(dataset)
    # predict_proba pode devolver (n, 2) ou (n,); queremos P(sweet)
    if probas.ndim == 2:
        probas = probas[:, 1]

    y_true = dataset.y
    # Alinhar comprimentos caso o pipeline tenha descartado moléculas
    min_len = min(len(y_true), len(probas))
    y_true = y_true[:min_len]
    probas = probas[:min_len]

    best_t, best_score = 0.5, 0.0
    for t in np.arange(low, high + step, step):
        preds = (probas >= t).astype(int)
        score = metric_fn(y_true, preds)
        if score > best_score:
            best_score = score
            best_t = t

    return round(best_t, 2), round(best_score, 4)

class CachedTwoDimensionalDescriptors(TwoDimensionDescriptors):
    """
    Cache the fully featurized dataset, including features, IDs, labels,
    and molecules, rather than caching only the feature matrix.

    The original featurization process may remove molecules that fail during
    featurization. Caching only the feature matrix could therefore lead to
    misalignment between the features and the corresponding labels.
    """

    _cache = {}

    def featurize(self, dataset, inplace=True):
        dataset_key = hashlib.md5(
            ",".join(map(str, dataset.ids)).encode()
        ).hexdigest()

        if dataset_key in self._cache:
            cached_dataset = deepcopy(self._cache[dataset_key])

            if inplace:
                dataset.__dict__.update(cached_dataset.__dict__)
                return dataset

            return cached_dataset

        result = super().featurize(dataset, inplace=inplace)
        out = dataset if inplace else result
        self._cache[dataset_key] = deepcopy(out)

        return out


def build_default_featurizer(featurizer_name, n_jobs):
    """
    Build a featurizer with default hyperparameters (no Optuna trial
    involved). Used to prefilter molecules that fail featurization
    before splitting, mirroring the featurizer choices in build_steps().
    """
    if featurizer_name == "morgan":
        return MorganFingerprint(radius=2, size=2048)

    elif featurizer_name == "maccs":
        return MACCSkeysFingerprint()

    elif featurizer_name == "2d_descriptors":
        return CachedTwoDimensionalDescriptors(n_jobs=n_jobs)

    elif featurizer_name == "atom_pair":
        return AtomPairFingerprint()

    else:
        raise ValueError(f"Unknown featurizer: {featurizer_name}")


def build_steps(automl_config, n_jobs, model_dir):
    featurizer_candidates = automl_config.get(
        "featurizer_candidates",
        ["morgan", "maccs", "2d_descriptors"],
    )
    model_candidates = automl_config.get(
        "model_candidates",
        ["RandomForestClassifier", "SVC", "LogisticRegression"],
    )

    def steps(trial):
        featurizer_name = trial.suggest_categorical(
            "featurizer",
            featurizer_candidates,
        )

        if featurizer_name == "morgan":
            radius = trial.suggest_int("morgan_radius", 1, 3)
            n_bits = trial.suggest_categorical(
                "morgan_bits",
                [1024, 2048],
            )
            featurizer = MorganFingerprint(
                radius=radius,
                size=n_bits,
            )

        elif featurizer_name == "maccs":
            featurizer = MACCSkeysFingerprint()

        elif featurizer_name == "2d_descriptors":
            featurizer = CachedTwoDimensionalDescriptors(
                n_jobs=n_jobs
            )

        elif featurizer_name == "atom_pair":
            featurizer = AtomPairFingerprint()

        else:
            raise ValueError(
                f"Unknown featurizer: {featurizer_name}"
            )

        model_name = trial.suggest_categorical(
            "model",
            model_candidates,
        )

        if model_name == "RandomForestClassifier":
            n_estimators = trial.suggest_int("model__n_estimators", 100, 300, step=100)
            max_depth = trial.suggest_int("model__max_depth", 5, 15, step=5)
            min_samples_leaf = trial.suggest_int("model__min_samples_leaf", 1, 50, step=1)
            min_samples_split = trial.suggest_int("model__min_samples_split", 2, 20, step=2)
            class_weight = trial.suggest_categorical("model__class_weight", [None, "balanced"])
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                class_weight=class_weight,
                n_jobs=n_jobs,
                random_state=42,
            )

        elif model_name == "SVC":
            kernel = trial.suggest_categorical(
                "model__kernel",
                ["linear", "rbf"],
            )
            C = trial.suggest_float(
                "model__C",
                0.1,
                10.0,
                log=True,
            )

            if automl_config.get("svc_fast", False):
                # Same Platt (sigmoid) calibration over 5 folds as
                # probability=True, but the 5 fold SVCs are fitted in
                # parallel and there is no extra full-data refit (the
                # calibrated fold models are averaged instead). libsvm is
                # single-threaded, so this is the only way SVC uses >1 core.
                classifier = CalibratedClassifierCV(
                    SVC(
                        kernel=kernel,
                        C=C,
                        cache_size=2000,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=5,
                    n_jobs=5,
                    ensemble=True,
                )
            else:
                classifier = SVC(
                    kernel=kernel,
                    C=C,
                    probability=True,
                    random_state=42,
                )

        elif model_name == "LogisticRegression":
            C = trial.suggest_float(
                "model__C",
                0.01,
                10.0,
                log=True,
            )

            classifier = LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=42,
            )

        elif model_name == "XGBClassifier":
            if XGBClassifier is None:
                raise ImportError(
                    "model_candidates inclui XGBClassifier mas o xgboost nao "
                    "esta instalado neste env (pip install xgboost)."
                )
            # scale_pos_weight faz o papel do class_weight='balanced' do RF:
            # o dataset limpo tem ~4.9 nao doces por cada doce, por isso o
            # intervalo 1-8 cobre "sem correcao" e "corrigido em excesso".
            classifier = XGBClassifier(
                n_estimators=trial.suggest_int(
                    "model__n_estimators", 100, 500, step=100),
                max_depth=trial.suggest_int("model__max_depth", 3, 12),
                learning_rate=trial.suggest_float(
                    "model__learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("model__subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float(
                    "model__colsample_bytree", 0.3, 1.0),
                min_child_weight=trial.suggest_int(
                    "model__min_child_weight", 1, 10),
                reg_lambda=trial.suggest_float(
                    "model__reg_lambda", 0.1, 10.0, log=True),
                scale_pos_weight=trial.suggest_float(
                    "model__scale_pos_weight", 1.0, 8.0),
                # hist: muito mais rapido do que 'exact' em fingerprints
                # esparsas, e o resultado e equivalente
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=n_jobs,
                random_state=42,
            )

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

        sklearn_model = SklearnModel(
            model=classifier,
            model_dir=model_dir,
        )

        # 2D descriptors are dense and have highly heterogeneous scales,
        # whereas molecular fingerprints are binary and do not require scaling.
        if featurizer_name == "2d_descriptors":
            return [
                ("featurizer", featurizer),
                ("scaler", StandardScaler()),
                ("model", sklearn_model),
            ]

        return [
            ("featurizer", featurizer),
            ("model", sklearn_model),
        ]

    return steps