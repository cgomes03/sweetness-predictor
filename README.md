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
p.calculate_scores(
    ["OC[C@H]1O[C@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"]
)
# array([0.797])
```

The model is a support vector classifier with a radial basis function kernel
applied to Morgan fingerprints. It was selected by an automated search over
featuriser and model families, conducted with
[DeepMol](https://github.com/BioSystemsUM/DeepMol) and Optuna.

## Model

| Property              | Value                                   |
| --------------------- | --------------------------------------- |
| Featuriser            | Morgan fingerprint, radius 1, 2048 bits |
| Classifier            | Support vector classifier, RBF kernel, C = 9.365 |
| Standardisation       | ChEMBL standardiser                     |
| Partition fingerprint | Atom-pair, used only to group molecules for the split |
| Selection metric      | F1 score on the validation set          |
| Search                | Optuna, 60 trials per featuriser/model pair; best was trial 22 |

Stereochemistry is not encoded: fingerprints are computed with chirality
disabled, so stereoisomers receive identical scores. This is deliberate, and
follows from the generative model the predictor is paired with, whose SMILES
vocabulary contains no stereochemical tokens.

## Data

`data/sweet26_clean.csv` holds the curated dataset, assembled from three
sources:

| Source | Role |
| ------ | ---- |
| [ChemTastesDB](https://doi.org/10.1016/j.fochms.2022.100090) | Curated tastants with experimentally confirmed labels; the dominant source of sweet compounds |
| [FooDB](https://foodb.ca/) | Food compounds with flavour annotations; a compound is labelled sweet if a flavour descriptor contains "sweet" |
| [DeepSweet](https://github.com/BioSystemsUM/DeepSweet) compilation | Earlier compilation over several taste and flavour databases, retained for comparability |

Molecular identity was resolved by full InChIKey computed with RDKit.
Compounds carrying conflicting labels across sources were removed, as were
compounds labelled non-sweet only by the FooDB extraction.

**Label-quality caveat.** Most negative labels originate from FooDB, directly
or through the inherited compilation, and reflect the absence of a sweet
descriptor rather than experimental evidence of a non-sweet taste. Some
compounds labelled non-sweet may be sweet but unannotated, so a fraction of
the apparent false positives are not necessarily errors. Precision should be
read with this in mind.

### Splits

`data/splits/` holds the training, validation, and test partitions. They were
produced by DeepMol's `SimilaritySplitter` over atom-pair fingerprints, which
groups structurally similar molecules and keeps them on the same side of the
split, so that the held-out sets sit in regions of chemical space the model
has not seen. A random split would place near-identical molecules in both
training and test sets and give an optimistic estimate of performance on
unfamiliar chemistry, which is the regime this predictor operates in when
scoring generated molecules.

The splitter's `homogenous_threshold` is compared against Tanimoto
*similarity* and the procedure is a greedy heuristic: it gives no guarantee
on the maximum similarity between any training and test pair. The partition
distributed here uses a threshold of 0.2.

## Performance

Decision threshold 0.5, on the splits described above.

| Metric    | Train  | Validation | Test   |
| --------- | ------ | ---------- | ------ |
| F1        | 0.9806 | 0.8149     | 0.8288 |
| Precision | 0.9733 | 0.9262     | 0.9282 |
| Recall    | 0.9880 | 0.7275     | 0.7487 |
| MCC       | 0.9770 | 0.7927     | 0.8069 |

Only the test column is an unbiased estimate. The validation set was the
selection signal driving the Optuna search, so validation metrics are
optimistic by construction.

## Scope

The model is a fingerprint-based QSAR classifier. It reads substructure
presence, not physicochemical properties, and says nothing about toxicity,
safety, solubility, metabolic fate or sweetness intensity relative to sucrose.
It estimates one thing: the probability that a structure belongs to the sweet
class as labelled in the dataset above. It is a research tool for ranking and
filtering candidate structures, not a basis for any claim about a compound's
fitness for consumption.

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



## Repository layout

```
sweetness_predictor.py          standardisation, featurisation and prediction
environment.yml                 the deepmol conda environment
model/
  config.json                   configuration of the training run
  best_metrics_morgan_SVC.csv   the performance table above
  best_pipeline_morgan_SVC/     featuriser, pipeline definition and classifier
data/
  sweet26_clean.csv             the curated dataset
  splits/                       training, validation and test partitions
training/
  data_prep/01_dedup.py         constructs the curated dataset from merged sources
  automl/                       search pipeline, search space and launchers
  configs/                      one configuration per featuriser/model pair
```



