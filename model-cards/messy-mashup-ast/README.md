---
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-spectrogram-transformer
- music-genre-classification
- audio
---

# Messy Mashup AST

This is an Audio Spectrogram Transformer checkpoint shared by Vikas for the Messy Mashup music-classification project. It predicts one of ten output classes from audio features.

## Checkpoint details

- Inspected revision: `2e13c40cad5c0d8e9322ddf9b4a25879fa42d68a`.
- Architecture: `ASTForAudioClassification`, 12 transformer layers, hidden size 768 and 12 attention heads.
- Input feature shape: up to 1,024 frames with 128 mel bins; patch size 16, frequency/time stride 10.
- Saved configuration records Transformers 4.57.1 and float32 weights.
- Related source: [deep-audio-classifier](https://github.com/23f3001800/deep-audio-classifier).

Output order in the saved configuration: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

## Loading gap

This repository contains weights and a model configuration but no `preprocessor_config.json`. The matching feature extractor must be recovered from the original training run. Copying another checkpoint's preprocessing settings without checking provenance could silently change predictions. A standalone inference example is therefore not claimed yet.

## Training and evaluation status

The checkpoint is associated with the Messy Mashup project, but its exact training run, data split, seed and validation results are not established by the inspected Hub files. The AST-v2 project's historical F1 score is not a verified result for this revision. No independent genre-accuracy result is claimed.

## Intended use and limitations

Use this checkpoint for music-classification experiments and educational comparisons after checking its packaging requirements. It is a closed-set classifier: it cannot establish that a clip belongs to an unsupported genre, identify an artist or determine music ownership. Short clips, unfamiliar subgenres and audio unlike the training mashups may produce unreliable predictions.

A credible new accuracy report needs labelled audio, a song-level split that excludes training sources, exact checkpoint and dataset hashes, per-class support and a confusion matrix. Synthetic silence, tone and noise can test numerical behaviour, but they cannot measure genre accuracy.

## License and provenance

The inspected Hub package does not establish a checkpoint-specific license. The source repository's code-license statement should not be treated as permission to redistribute all weights or training recordings. Confirm the base-model and training-data terms before redistribution. This card does not assign new licensing rights.
