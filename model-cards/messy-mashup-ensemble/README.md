---
tags:
- audio
- music-genre-classification
- experimental
---

# Messy Mashup Ensemble Artifacts

This repository stores artifacts from Vikas's Messy Mashup audio project. At revision `78b927e52f6a66b1ed4a84c67f1933fd394dbffa`, the public file inventory contains `ensemble_config.json` and one AST checkpoint under `ast_seed42/`.

## Current package status

The repository name describes an ensemble experiment, but the inspected tree contains only one member checkpoint. A complete multi-model ensemble has not been verified. There is no root `config.json` or `preprocessor_config.json`, so this is not a self-contained Transformers pipeline at the repository root.

Use the matching training or inference implementation from the [project repository](https://github.com/23f3001800/deep-audio-classifier) to establish preprocessing, class order and aggregation rules before loading these files. A generic `pipeline(model="Vikas25S/messy-mashup-ensemble")` example would be misleading for the current layout.

## Evaluation and intended use

These artifacts are suitable for inspecting the experiment and reproducing it once the loader and missing metadata are supplied. No independently reproduced ensemble score is available. The AST-v2 project's historical validation score must not be attributed to this repository.

Before presenting an ensemble result, record each member's revision, preprocessing, label mapping, blend weights and evaluation split. Compare the ensemble with its strongest single member on the same held-out songs. Include model size and inference time so any accuracy gain can be weighed against cost.

## Limitations and license

Training provenance, a runnable ensemble loader and a checkpoint-specific evaluation receipt are incomplete. Weight and dataset redistribution terms are not established by the inspected files. This card does not assign a license or claim that this package is production-ready.
