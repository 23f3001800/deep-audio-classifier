---
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-spectrogram-transformer
- music-genre-classification
- audio
---

# Messy Mashup AST — Legacy Checkpoint

This is an Audio Spectrogram Transformer checkpoint shared by Vikas for the Messy Mashup music-classification project. It predicts one of ten output classes from audio features.

## Checkpoint details

- Inspected revision: `2ce148cd5715e079dc89bb23356012187117dedb`.
- Architecture: `ASTForAudioClassification`, 12 transformer layers, hidden size 768 and 12 attention heads.
- Input feature shape: up to 1,024 frames with 128 mel bins; patch size 16, frequency/time stride 10.
- Saved configuration records Transformers 4.57.1 and float32 weights.
- Related source: [deep-audio-classifier](https://github.com/23f3001800/deep-audio-classifier).

Output order in the saved configuration: `LABEL_0`, `LABEL_1`, `LABEL_2`, `LABEL_3`, `LABEL_4`, `LABEL_5`, `LABEL_6`, `LABEL_7`, `LABEL_8`, `LABEL_9`.

The checkpoint exports generic class names. The class-to-genre mapping for this exact revision has not been verified, so this card does not substitute the mapping from a different model. Resolve it from the original training run before interpreting an output as a genre.

The repository includes an AST feature extractor configured for 16 kHz audio, normalization mean -4.2677393 and standard deviation 4.5689974.

## Run inference

Install `torch`, `transformers==4.57.1`, `librosa` and `soundfile`. The example loads a fixed checkpoint revision and follows the project's ten-second input convention.

```python
import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

model_id = "Vikas25S/ast-messy-mashup-classifier"
revision = "2ce148cd5715e079dc89bb23356012187117dedb"
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
## Training and evaluation status

The checkpoint is associated with the Messy Mashup project, but its exact training run, data split, seed and validation results are not established by the inspected Hub files. The AST-v2 project's historical F1 score is not a verified result for this revision. No independent genre-accuracy result is claimed.

## Intended use and limitations

Use this checkpoint for music-classification experiments and educational comparisons after checking its packaging requirements. It is a closed-set classifier: it cannot establish that a clip belongs to an unsupported genre, identify an artist or determine music ownership. Short clips, unfamiliar subgenres and audio unlike the training mashups may produce unreliable predictions.

A credible new accuracy report needs labelled audio, a song-level split that excludes training sources, exact checkpoint and dataset hashes, per-class support and a confusion matrix. Synthetic silence, tone and noise can test numerical behaviour, but they cannot measure genre accuracy.

## License and provenance

The inspected Hub package does not establish a checkpoint-specific license. The source repository's code-license statement should not be treated as permission to redistribute all weights or training recordings. Confirm the base-model and training-data terms before redistribution. This card does not assign new licensing rights.
