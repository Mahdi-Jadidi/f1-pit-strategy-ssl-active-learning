# F1 Pit Strategy via Semi-Supervised and Active Learning

An end-to-end pit-stop strategy research pipeline built on Ergast-style race data. It creates lap and championship-context features, assigns auditable strategy labels, benchmarks three supervised models, runs iterative pseudo-labeling at multiple thresholds, compares three active-learning query policies, and exports plateau and comparative analyses.

## Pipeline

```text
raw Ergast tables -> cleaning -> lap/pit features -> strategy labels
                  -> stratified pools -> model selection
                  -> pseudo-labeling + active learning -> comparison
```

The implementation is split across `data.py`, `features.py`, `labeling.py`, `models.py`, `semi_supervised.py`, `active_learning.py`, and `pipeline.py`.

## Run

```bash
pip install -e .
f1-pit-strategy run --data-dir . --output-dir outputs
```

For a prepared run, `data-dir` needs `labeled.csv` and `unlabeled.csv`. Add `--oracle-unlabeled` when the unlabeled pool contains a hidden `strategy` column and active-learning simulation should be included. Raw feature preparation is available through `f1-pit-strategy prepare --data-dir /path/to/ergast`.

## Outputs

`baseline_metrics.csv`, `pseudo_labeling_history.csv`, `active_learning_history.csv`, `comparative_summary.csv`, `plateau_summary.csv`, and serialized best estimators.

## Topics

`formula-1` `tabular-ml` `semi-supervised-learning` `active-learning` `scikit-learn`
