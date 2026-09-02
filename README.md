# 🎵 Deep Audio Classifier

Fine-tuned [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) for 10-class music genre classification on noisy stem mashups — trained as part of the **Messy Mashup** project, where test audio consists of stems mixed from different songs with added environmental noise (ESC-50).

**Author:** Vikas · Roll No. 23f3001800
**Live Demo:** [audioclassificationproj.streamlit.app](https://audioclassificationproj.streamlit.app/)

[![Model](https://img.shields.io/badge/model-AST--v2-blue)](https://huggingface.co/Vikas25S/messy-mashup-ast-v2)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## Overview

| Property | Value |
|---|---|
| Base model | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Task | 10-class music genre classification |
| Input | FBANK features — 10s audio at 16kHz |
| Output | Probabilities over 10 genres |
| Parameters | 86.2M total, 10-class head replaced |
| Val Macro F1 | **0.8871** |
| Val Accuracy | **0.8905** |

**Genres:** `blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`

---

## Repository Structure

```
deep-audio-classifier/
├── app.py              # Streamlit app — the live demo entry point
├── notebooks/          # Exploratory analysis & training notebooks
├── reports/            # Evaluation reports, plots, write-ups
├── scripts/            # Standalone scripts (training, preprocessing, etc.)
├── src/                # Core library code (model, data, inference utils)
└── requirements.txt    # Python dependencies
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/23f3001800/deep-audio-classifier.git
cd deep-audio-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app locally

```bash
streamlit run app.py
```

This launches the same interface as the [live demo](https://audioclassificationproj.streamlit.app/) — upload an audio clip and get genre predictions with per-class confidence scores.

---

## Using the Model Directly

The fine-tuned model is hosted on Hugging Face at [`Vikas25S/messy-mashup-ast-v2`](https://huggingface.co/Vikas25S/messy-mashup-ast-v2) and can be used independently of this app:

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch
import librosa
import numpy as np

# Load model and feature extractor
model_id = "Vikas25S/messy-mashup-ast-v2"
fe    = ASTFeatureExtractor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id)
model.eval()

# Load and preprocess audio
y, _ = librosa.load("your_audio.wav", sr=16000, mono=True)

# Crop or pad to 10 seconds
CLIP_SAMPLES = 160000
if len(y) >= CLIP_SAMPLES:
    y = y[:CLIP_SAMPLES]
else:
    y = np.pad(y, (0, CLIP_SAMPLES - len(y)))

# Extract features
inputs = fe(y, sampling_rate=16000, return_tensors="pt", padding="max_length")

# Predict
with torch.no_grad():
    logits = model(input_values=inputs["input_values"]).logits

probs  = torch.softmax(logits, dim=-1)[0]
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

top_idx   = probs.argmax().item()
top_genre = genres[top_idx]
top_conf  = probs[top_idx].item()

print(f"Predicted genre : {top_genre}")
print(f"Confidence      : {top_conf:.2%}")

for genre, prob in zip(genres, probs.tolist()):
    bar = "█" * int(prob * 30)
    print(f"  {genre:12s} {bar:<30s} {prob:.3f}")
```

---

## Training Details

### Dataset

- **Training data:** 1,000 songs × 10 genres × 4 stems (drums, vocals, bass, other)
- **Test data:** 3,020 noisy mashups — stems from different songs + ESC-50 environmental noise
- **Validation split:** Last 10 songs per genre held out (100 songs total, deterministic)

### Key Design Decisions

**Stem mixing from different songs**
Each training sample mixes drums, vocals, bass, and other stems from 4 different songs of the same genre. This forces the model to learn genre-level features rather than song-specific patterns — exactly how the test mashups were constructed.

**ESC-50 noise curriculum**
Environmental noise (ESC-50) was injected during training in 3 stages:

| Stage | Epochs | Pool | Noise Probability |
|---|---|---|---|
| 1 | 1–7 | ESC-10 (easy) | 50% |
| 2 | 7–14 | ESC-50 non-musical | 75% |
| 3 | 14+ | Full ESC-50 | 90% |

**2-phase fine-tuning**
- Phase 1 (5 epochs): backbone frozen, only the 10-class head trains at `lr=3e-4`
- Phase 2 (15 epochs): full model fine-tuned with Layer-wise LR Decay (LLRD, decay=0.85)

**Class-weighted loss**
Weights tuned to address hard genres (rock, blues, reggae consistently confused):

```
rock=1.40  blues=1.30  reggae=1.20  country=1.10
metal=0.95  jazz=0.50  classical=0.50
```

**Augmentations applied during training**
- Time stretch ±15% (P=0.50)
- Pitch shift ±2 semitones (P=0.40)
- Random gain per stem 0.6–1.4× (P=0.50)
- SpecAugment: frequency masking ±24 bins, time masking ±96 frames (P=0.70)

### Training Config

```python
SAMPLE_RATE     = 16000
CLIP_DURATION   = 10          # seconds
BATCH_SIZE      = 8           # × accum_steps=4 → effective=32
PHASE1_LR       = 3e-4
PHASE2_LR       = 1e-4
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.1
GRAD_CLIP       = 1.0
EARLY_STOPPING  = 5           # patience
```

---

## Per-Class Results

| Genre | Validation F1 | Notes |
|---|---|---|
| classical | 1.000 | Acoustically unique — orchestral, no drums |
| jazz | 1.000 | Piano + brass + distinct harmony |
| hiphop | 0.990 | 808 bass, rap vocals, clear signature |
| pop | 0.980 | Polished production, distinct |
| disco | 0.959 | Strong 4-on-floor kick |
| metal | 0.865 | Distorted guitar, some overlap with rock |
| country | 0.845 | Acoustic guitar, some overlap with blues |
| reggae | 0.759 | Offbeat skank, some overlap with hiphop |
| blues | 0.766 | 12-bar structure, confused with rock |
| rock | 0.706 | Guitar + drums + vocals, hardest genre |

---

## Limitations

- Trained on 30-second stems at 16kHz. Very short clips (< 5s) may produce unreliable results.
- Performance degrades on genres with high acoustic overlap (rock/blues/reggae trio).
- Optimised for noisy mashup conditions — clean single-instrument audio may behave differently from the training distribution.
- Training data is limited to 100 songs per genre — uncommon subgenres within each category may not be well represented.

## Intended Use

- Music genre classification research
- Audio tagging pipelines
- Educational demonstrations of audio transformer fine-tuning

**Not intended for:** commercial music rights identification, artist attribution, or any application requiring high precision on ambiguous borderline genres.

---

## Citation

```bibtex
@misc{messy-mashup-ast-v2,
  author    = {Vikas25S},
  title     = {Messy Mashup AST V2 — Music Genre Classifier},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/Vikas25S/messy-mashup-ast-v2}
}
```

Base model citation:

```bibtex
@article{gong2021ast,
  title   = {AST: Audio Spectrogram Transformer},
  author  = {Gong, Yuan and Chung, Yu-An and Glass, James},
  journal = {Interspeech 2021},
  year    = {2021}
}
```

---

## License

Released under the [MIT License](LICENSE).

## Author

**Vikas** — Roll No. 23f3001800 — Jan 2026 Deep Learning Project
