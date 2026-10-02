---
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-spectrogram-transformer
- music-genre-classification
- audio
base_model: MIT/ast-finetuned-audioset-10-10-0.4593
---

# Messy Mashup AST V2

This is an Audio Spectrogram Transformer checkpoint shared by Vikas for the Messy Mashup music-classification project. It predicts one of ten output classes from audio features.

## Checkpoint details

- Inspected revision: `eeef47a362c32313b6c4df23e126675e16c0eb9c`.
- Architecture: `ASTForAudioClassification`, 12 transformer layers, hidden size 768 and 12 attention heads.
- Input feature shape: up to 1,024 frames with 128 mel bins; patch size 16, frequency/time stride 10.
- Saved configuration records Transformers 4.57.1 and float32 weights.
- Related source: [deep-audio-classifier](https://github.com/23f3001800/deep-audio-classifier).

Output order in the saved configuration: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

The repository includes an AST feature extractor configured for 16 kHz audio, normalization mean -4.2677393 and standard deviation 4.5689974.

## Run inference

Install `torch`, `transformers==4.57.1`, `librosa` and `soundfile`. The example loads a fixed checkpoint revision and follows the project's ten-second input convention.

```python
import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

model_id = "Vikas25S/messy-mashup-ast-v2"
revision = "eeef47a362c32313b6c4df23e126675e16c0eb9c"
extractor = ASTFeatureExtractor.from_pretrained(model_id, revision=revision)
model = ASTForAudioClassification.from_pretrained(model_id, revision=revision).eval()
audio, _ = librosa.load("clip.wav", sr=16000, mono=True)
audio = np.pad(audio[:160000], (0, max(0, 160000 - len(audio))))
inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    probabilities = model(**inputs).logits.softmax(dim=-1)[0]
index = int(probabilities.argmax())
print(model.config.id2label[index], float(probabilities[index]))
```

The softmax score is a relative class score, not a calibrated probability that the prediction is correct. This closed-set model still selects a class for silence and out-of-domain audio.
## Fresh loading and inference check

On 2 October 2026, a clean GitHub Actions CPU run loaded this pinned revision and ran six forward passes: two each on ten seconds of silence, a 440 Hz tone and seeded noise. All three inputs produced finite `(1, 10)` logits and repeatable outputs. The loader also verified the ten genre labels and the 16 kHz feature extractor.

The loaded model contains 86,196,490 parameters. Weights SHA-256: `dc38dd0ebf3428bc9bbec6c86b57b8ca6ae19f8d59fe62d5bfb118cc35855d17`.

[Run and logs](https://github.com/23f3001800/deep-audio-classifier/actions/runs/37021499380). This is a real checkpoint usability check on synthetic signals; it does not measure genre accuracy or establish robustness on real music.

## Training and reported validation

The [project README](https://github.com/23f3001800/deep-audio-classifier/blob/bad6bf8a2986b1db4ea943ab7f74dfcfe7631887/README.md) describes fine-tuning from `MIT/ast-finetuned-audioset-10-10-0.4593` on stems from 1,000 songs across ten genres, with ESC-50 noise and stem-mixing augmentation. It describes a song-level validation holdout and two training phases: a frozen backbone followed by full fine-tuning with layer-wise learning-rate decay.

| Result | Value | Evidence status |
|---|---:|---|
| Validation macro F1 | 0.8871 | Historical project report |
| Validation accuracy | 0.8905 | Historical project report |
| Independent test macro F1 | Not available | No labelled independent test run verified |

These numbers have not been independently reproduced for the pinned Hub revision. Different notebook versions use different split descriptions; an exact split manifest and checkpoint-to-run mapping are still needed. Subsequent fixes to the extracted training loop do not retroactively change this checkpoint or improve its reported score.

## Intended use and limitations

Use this checkpoint for music-classification experiments and educational comparisons after checking its packaging requirements. It is a closed-set classifier: it cannot establish that a clip belongs to an unsupported genre, identify an artist or determine music ownership. Short clips, unfamiliar subgenres and audio unlike the training mashups may produce unreliable predictions.

A credible new accuracy report needs labelled audio, a song-level split that excludes training sources, exact checkpoint and dataset hashes, per-class support and a confusion matrix. Synthetic silence, tone and noise can test numerical behaviour, but they cannot measure genre accuracy.

## License and provenance

The inspected Hub package does not establish a checkpoint-specific license. The source repository's code-license statement should not be treated as permission to redistribute all weights or training recordings. Confirm the base-model and training-data terms before redistribution. This card does not assign new licensing rights.
