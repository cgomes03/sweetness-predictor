# Sweetness predictor

A binary classifier that estimates the probability that a molecule tastes
sweet, given its SMILES string.

This repository accompanies the MSc dissertation *Multiobjective Optimisation
of Chemical Compounds with Generative Artificial Intelligence with Applications
in the Food Industry*, carried out at the University of Minho. Within that work
the predictor serves two roles: as an objective function scoring candidate
molecules during evolutionary multiobjective optimisation, and as an evaluation
metric monitored while a generative model is fine-tuned towards sweet chemical
space.

The code was developed as a component of
[molevogen](https://github.com/BioSystemsUM/molevogen), the framework for
molecular generation and evolutionary optimisation in which the dissertation
work was conducted. It is extracted and published here so that the predictor
can be inspected, reused and cited independently of that framework.

```python
from sweetness_predictor import SweetnessPredictor

p = SweetnessPredictor()
p.calculate_scores(["OC[C@H]1O[C@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"])
# array([0.797])
```

The model is a support vector classifier with a radial basis function kernel
applied to Morgan fingerprints. It was selected by an automated search over
featuriser and model families, conducted with [DeepMol](https://github.com/BioSystemsUM/DeepMol)
and Optuna.

## Model

| Property              | Value                                   |
| --------------------- | --------------------------------------- |
| Featuriser            | Morgan fingerprint, radius 1, 2048 bits |
| Classifier            | Support vector classifier, RBF kernel, C = 9.365 |
| Standardisation       | ChEMBL standardiser                     |
| Partition fingerprint | Atom-pair, used only to group molecules for the split |
| Selection metric      | F1 score on the validation set          |
| Search                | Optuna, 60 trials per featuriser/model pair; best was trial 22 |

Two distinct fingerprints are involved and should not be conflated. The model
featurises molecules with Morgan fingerprints. The similarity-based partition
of the dataset was computed with atom-pair fingerprints, which affects only how
molecules were allocated to the training, validation and test sets.

## Performance

Decision threshold 0.5. Validation and test sets were held out under a
similarity-based partition, described under [Data](#data).

| Metric    | Train  | Validation | Test   |
| --------- | ------ | ---------- | ------ |
| F1        | 0.9806 | 0.8149     | 0.8288 |
| Precision | 0.9733 | 0.9262     | 0.9282 |
| Recall    | 0.9880 | 0.7275     | 0.7487 |
| MCC       | 0.9770 | 0.7927     | 0.8069 |


## Installation

The predictor depends on DeepMol, which in turn requires TensorFlow and a
pinned version of RDKit:

```bash
conda env create -f environment.yml   # creates an environment named "deepmol"
conda activate deepmol
```

`environment.yml` is a complete export of a working environment and therefore
pins exact builds and targets Linux. The versions that govern reproducibility
are Python 3.11, `deepmol==1.2.1`, `rdkit==2025.9.6`, `scikit-learn==1.5.2`,
`numpy==1.26.4` and `tensorflow==2.15.0`.

The scikit-learn version must be respected. The classifier is distributed as a
pickled estimator, and a mismatched version may either refuse to deserialise it
or load it and produce incorrect scores without raising an error.

Calling code need not run inside that environment. If DeepMol cannot be
imported, `SweetnessPredictor` launches a persistent worker process in the
`deepmol` environment and communicates with it over a pipe, so a caller in any
other environment obtains identical scores. Importing DeepMol costs
approximately twelve seconds, incurred once per run rather than once per call.

## Obtaining the model weights

`model/best_pipeline_morgan_SVC/model/model.pkl` is 233 MB, since a support
vector classifier with an RBF kernel retains its support vectors. This exceeds
GitHub's 100 MB limit for files in a repository, so the file is distributed as a
release asset rather than tracked in version control.

Download it into place:

```bash
gh release download v1.0 --pattern model.pkl \
    --dir model/best_pipeline_morgan_SVC/model/
```

Alternatively, set the `SWEETNESS_MODEL_DIR` environment variable to the
directory holding `best_pipeline_morgan_SVC/`, wherever that is kept.

## Repository layout

```
sweetness_predictor.py          standardisation, featurisation and prediction
environment.yml                 the deepmol conda environment
model/
  config.json                   configuration of the training run
  best_metrics_morgan_SVC.csv   the performance table above
  best_pipeline_morgan_SVC/     featuriser, pipeline definition and classifier
data/
  sweet26_clean.csv             the curated dataset, 65,255 molecules
  splits/                       training, validation and test partitions
training/
  data_prep/01_dedup.py         constructs the curated dataset from merged sources
  automl/                       search pipeline, search space and launchers
  configs/                      one configuration per featuriser/model pair
  sanitycheck/                  leakage and partition integrity checks
```

## Data

`data/sweet26_clean.csv` contains 65,255 unique molecules, of which 11,121
(17.0 per cent) are labelled sweet. It was assembled from three sources and
deduplicated:

| Source                 | Molecules |
| ---------------------- | --------- |
| Sweetness dataset v3.1 | 58,052    |
| FooDB                  | 52,748    |
| Gold taste dataset     | 12,111    |

The saved partitions contain 63,953 of these molecules at 15.8 per cent sweet;
ChEMBL standardisation discards compounds it cannot process.

| Partition  | Molecules | Sweet | Proportion sweet |
| ---------- | --------- | ----- | ---------------- |
| Training   | 44,767    | 7,092 | 15.8 per cent    |
| Validation | 9,592     | 1,519 | 15.8 per cent    |
| Test       | 9,594     | 1,520 | 15.8 per cent    |

The partition is similarity-based, not random. Molecules are grouped by
structural similarity, using atom-pair fingerprints with a homogeneity
threshold of 0.2, before being divided in proportions of 70, 15 and 15 per cent,
so that structurally near-identical compounds cannot be separated across
partitions. A random partition of this dataset yields considerably higher
scores that do not reflect performance on novel chemistry, which is the regime
in which a generative model operates.


## Reproducing the model selection

With the `deepmol` environment active, from the repository root:

```bash
python training/automl/automl_pipeline.py \
    --config training/configs/predictor/automl/parallel/sim02_morgan_SVC.yaml
```

`training/automl/launch_parallel.sh` runs every featuriser and model
combination, and `merge_parallel_results.py` consolidates the outcomes into a
single table. The search is computationally demanding: the original run used
50 CPU cores with a timeout of 30 minutes per trial, and a fixed random seed of
42.

## Self-test

```bash
python sweetness_predictor.py            # scores a set of reference molecules
python sweetness_predictor.py --test-set # recomputes the performance table
```

The first command is expected to produce:

```
sucrose              P(sweet) = 0.797
aspartame            P(sweet) = 0.808
saccharin            P(sweet) = 0.036
caffeine (bitter)    P(sweet) = 0.037
ethanol              P(sweet) = 0.518
invalid SMILES       P(sweet) = 0.000
```

