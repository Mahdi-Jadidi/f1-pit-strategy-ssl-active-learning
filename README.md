# F1 Pit Strategy via Semi-Supervised and Active Learning

A research-oriented machine-learning pipeline for inferring Formula 1 pit-stop strategy from race, lap-time, pit-stop, and championship-context data. The central constraint is realistic: labelled strategy decisions are limited, while historical telemetry-like records are plentiful.

## Problem

Can a strategy classifier become useful when only a small fraction of the available race records are labelled? This repository evaluates three ways to use the unlabelled pool: supervised learning, confidence-filtered pseudo-labelling, and active learning with an oracle simulation.

## What was built

- A reproducible feature pipeline from Ergast-style race tables, including lap context, pit timing, driver and constructor standings, and race state.
- Auditable strategy labels rather than an opaque target construction.
- Baselines across logistic regression, random forest, and gradient boosting.
- Iterative pseudo-labelling with multiple confidence thresholds and plateau detection.
- Three query policies for active learning: uncertainty, margin, and entropy.

## Research contribution

The project does more than train a classifier: it measures how performance changes as labels are acquired or inferred. Its comparative outputs make it possible to see when pseudo-labels stop helping, which acquisition policy extracts the most value from a labelling budget, and whether a gain is stable across iterations.

## Outputs

Each experiment writes `baseline_metrics.csv`, pseudo-labelling and active-learning histories, a comparative summary, plateau analysis, and serialized estimators. This makes the reported conclusion traceable back to every round of the experiment.

## Reproduce

```bash
pip install -e .
f1-pit-strategy run --data-dir . --output-dir outputs
```

Prepared runs require `labeled.csv` and `unlabeled.csv`; use `prepare` to construct them from raw Ergast-style files. GitHub Actions runs package quality checks and tests.
