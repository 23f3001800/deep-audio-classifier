"""Build evidence-backed README files for the five existing public Hub repos."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "model-cards"
inventory = json.loads((ROOT / "inventory.json").read_text())
SOURCE = "https://github.com/23f3001800/deep-audio-classifier"

def inference(model_id, revision):
    return f'''## Run inference

Install `torch`, `transformers==4.57.1`, `librosa` and `soundfile`. The example loads a fixed checkpoint revision and follows the project's ten-second input convention.

```python
import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

model_id = "{model_id}"
revision = "{revision}"
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
'''

for item in inventory["models"]:
    model_id, revision = item["id"], item["sha"]
    name = model_id.split("/")[1]
    if name == "llama-3.2-email-formatter":
        card = f'''---
library_name: peft
pipeline_tag: text-generation
base_model: unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
language:
- en
tags:
- llama
- lora
- email-formatting
---

# Llama 3.2 Email Formatter

This repository contains a LoRA adapter for an email-formatting experiment by Vikas. It contains adapter weights and a tokenizer, rather than a standalone copy of the base language model.

## Checkpoint details

- Inspected revision: `{revision}`.
- Base checkpoint recorded in `adapter_config.json`: `unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit`.
- Adapter: LoRA, rank 16, alpha 16, dropout 0.05.
- Target modules: query, key, value and output projections, plus gate, up and down projections.
- Task: causal language modelling. The saved adapter config records PEFT 0.18.1.

## Intended use

Explore formatting an English email draft while preserving the supplied facts. Review every output before using it: a language model can invent a recipient, date, commitment or other detail that was absent from the input. This checkpoint does not send email.

## Loading and reproducibility

Load this repository as a PEFT adapter on its recorded base checkpoint. The base is a bitsandbytes 4-bit model, so the runtime must support that quantization setup. Use the repository's tokenizer and chat template. A verified end-to-end inference example is still pending; successful adapter upload alone does not establish that generation works in a fresh environment.

The training dataset, split, prompt format, seed, optimizer settings and training duration are not documented in the inspected repository. Do not infer those settings from the adapter configuration. No held-out formatting, factual-preservation or latency results have been published with this checkpoint.

## Evaluation needed

Compare the base model and this adapter on the same unseen email drafts. Measure format adherence and factual preservation separately, and record examples of added or removed facts. Include ambiguous requests and drafts containing instruction-like text. Report the checkpoint revisions, test prompts and decoding settings alongside results.

## License information

The previous generated card labelled this repository Apache-2.0. That label alone does not establish redistribution rights for the base model. The [upstream base card](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit) identifies Llama 3.2 terms. The adapter author's intended license and the training-data terms still need to be recorded; this card does not grant new rights or relicense the checkpoint.
'''
    elif name == "messy-mashup-ensemble":
        card = f'''---
tags:
- audio
- music-genre-classification
- experimental
---

# Messy Mashup Ensemble Artifacts

This repository stores artifacts from Vikas's Messy Mashup audio project. At revision `{revision}`, the public file inventory contains `ensemble_config.json` and one AST checkpoint under `ast_seed42/`.

## Current package status

The repository name describes an ensemble experiment, but the inspected tree contains only one member checkpoint. A complete multi-model ensemble has not been verified. There is no root `config.json` or `preprocessor_config.json`, so this is not a self-contained Transformers pipeline at the repository root.

Use the matching training or inference implementation from the [project repository]({SOURCE}) to establish preprocessing, class order and aggregation rules before loading these files. A generic `pipeline(model="{model_id}")` example would be misleading for the current layout.

## Evaluation and intended use

These artifacts are suitable for inspecting the experiment and reproducing it once the loader and missing metadata are supplied. No independently reproduced ensemble score is available. The AST-v2 project's historical validation score must not be attributed to this repository.

Before presenting an ensemble result, record each member's revision, preprocessing, label mapping, blend weights and evaluation split. Compare the ensemble with its strongest single member on the same held-out songs. Include model size and inference time so any accuracy gain can be weighed against cost.

## Limitations and license

Training provenance, a runnable ensemble loader and a checkpoint-specific evaluation receipt are incomplete. Weight and dataset redistribution terms are not established by the inspected files. This card does not assign a license or claim that this package is production-ready.
'''
    else:
        config = json.loads(item["config.json"])
        labels = [config["id2label"][str(i)] for i in range(10)]
        v2 = name == "messy-mashup-ast-v2"
        generic = name == "ast-messy-mashup-classifier"
        title = {"messy-mashup-ast-v2": "Messy Mashup AST V2", "messy-mashup-ast": "Messy Mashup AST", "ast-messy-mashup-classifier": "Messy Mashup AST — Legacy Checkpoint"}[name]
        card = f'''---
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-spectrogram-transformer
- music-genre-classification
- audio
'''
        if v2:
            card += "base_model: MIT/ast-finetuned-audioset-10-10-0.4593\n"
        card += f'''---

# {title}

This is an Audio Spectrogram Transformer checkpoint shared by Vikas for the Messy Mashup music-classification project. It predicts one of ten output classes from audio features.

## Checkpoint details

- Inspected revision: `{revision}`.
- Architecture: `ASTForAudioClassification`, 12 transformer layers, hidden size 768 and 12 attention heads.
- Input feature shape: up to 1,024 frames with 128 mel bins; patch size 16, frequency/time stride 10.
- Saved configuration records Transformers 4.57.1 and float32 weights.
- Related source: [deep-audio-classifier]({SOURCE}).

Output order in the saved configuration: {", ".join(f"`{label}`" for label in labels)}.

'''
        if generic:
            card += "The checkpoint exports generic class names. The class-to-genre mapping for this exact revision has not been verified, so this card does not substitute the mapping from a different model. Resolve it from the original training run before interpreting an output as a genre.\n\n"
        if item["preprocessor_config.json"]:
            card += "The repository includes an AST feature extractor configured for 16 kHz audio, normalization mean -4.2677393 and standard deviation 4.5689974.\n\n"
            card += inference(model_id, revision)
        else:
            card += "## Loading gap\n\nThis repository contains weights and a model configuration but no `preprocessor_config.json`. The matching feature extractor must be recovered from the original training run. Copying another checkpoint's preprocessing settings without checking provenance could silently change predictions. A standalone inference example is therefore not claimed yet.\n\n"
        if v2:
            card += '''## Fresh loading and inference check

On 2 October 2026, a clean GitHub Actions CPU run loaded this pinned revision and ran six forward passes: two each on ten seconds of silence, a 440 Hz tone and seeded noise. All three inputs produced finite `(1, 10)` logits and repeatable outputs. The loader also verified the ten genre labels and the 16 kHz feature extractor.

The loaded model contains 86,196,490 parameters. Weights SHA-256: `dc38dd0ebf3428bc9bbec6c86b57b8ca6ae19f8d59fe62d5bfb118cc35855d17`.

[Run and logs](https://github.com/23f3001800/deep-audio-classifier/actions/runs/37021499380). This is a real checkpoint usability check on synthetic signals; it does not measure genre accuracy or establish robustness on real music.

'''
            card += f'''## Training and reported validation

The [project README]({SOURCE}/blob/bad6bf8a2986b1db4ea943ab7f74dfcfe7631887/README.md) describes fine-tuning from `MIT/ast-finetuned-audioset-10-10-0.4593` on stems from 1,000 songs across ten genres, with ESC-50 noise and stem-mixing augmentation. It describes a song-level validation holdout and two training phases: a frozen backbone followed by full fine-tuning with layer-wise learning-rate decay.

| Result | Value | Evidence status |
|---|---:|---|
| Validation macro F1 | 0.8871 | Historical project report |
| Validation accuracy | 0.8905 | Historical project report |
| Independent test macro F1 | Not available | No labelled independent test run verified |

These numbers have not been independently reproduced for the pinned Hub revision. Different notebook versions use different split descriptions; an exact split manifest and checkpoint-to-run mapping are still needed. Subsequent fixes to the extracted training loop do not retroactively change this checkpoint or improve its reported score.

'''
        else:
            card += "## Training and evaluation status\n\nThe checkpoint is associated with the Messy Mashup project, but its exact training run, data split, seed and validation results are not established by the inspected Hub files. The AST-v2 project's historical F1 score is not a verified result for this revision. No independent genre-accuracy result is claimed.\n\n"
        card += '''## Intended use and limitations

Use this checkpoint for music-classification experiments and educational comparisons after checking its packaging requirements. It is a closed-set classifier: it cannot establish that a clip belongs to an unsupported genre, identify an artist or determine music ownership. Short clips, unfamiliar subgenres and audio unlike the training mashups may produce unreliable predictions.

A credible new accuracy report needs labelled audio, a song-level split that excludes training sources, exact checkpoint and dataset hashes, per-class support and a confusion matrix. Synthetic silence, tone and noise can test numerical behaviour, but they cannot measure genre accuracy.

## License and provenance

The inspected Hub package does not establish a checkpoint-specific license. The source repository's code-license statement should not be treated as permission to redistribute all weights or training recordings. Confirm the base-model and training-data terms before redistribution. This card does not assign new licensing rights.
'''
    destination = ROOT / name / "README.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(card)

manifest = {"models": [{"repo_id": x["id"], "parent_commit": x["sha"],
                         "path": x["id"].split("/")[1] + "/README.md"}
                        for x in inventory["models"]]}
(ROOT / "publish-manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Prepared {len(manifest['models'])} model cards")
