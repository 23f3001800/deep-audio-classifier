# dl-genai-project-26-t1

## 🎵 Messy Mashup: Audio Spectrogram Transformer (AST)

# Model Description

This model is a fine-tuned Audio Spectrogram Transformer (AST) designed for robust music genre classification in highly degraded, noisy, and chaotic audio environments. It was specifically developed for the Messy Mashup Challenge, which requires the model to classify genres from dynamically time-stretched tracks mixed with random environmental noise.

Base Model: MIT/ast-finetuned-audioset-10-10-0.4593

Task: 10-Class Music Genre Classification

Input: Raw 1D Audio Waveforms (converted to 2D Mel-spectrograms under the hood)

Output: Predicted Genre Label (e.g., Jazz, Metal, Pop)

## 🏗️ Architecture & Training Pipeline

Standard models fail on this dataset because they overfit to pristine studio stems. This model overcomes that domain shift through a custom-built Combinatorial Domain Randomization Engine.

During training, the model never sees the same track twice. The PyTorch DataLoader operates as a dynamic audio synthesizer:

Combinatorial Selection: Randomly selects 4 distinct songs from a target genre, extracting isolated stems (drums, bass, vocals, other).

Algorithmic Time-Stretching: Utilizes a Phase Vocoder to rhythmically align the BPM of all stems to the drum track without destroying pitch or harmonic integrity.

Stochastic Noise Injection: Randomly samples environmental audio from the ESC-50 Dataset and overlays it at varying Signal-to-Noise Ratios (SNRs) to simulate the hostile test environment.

Global Self-Attention: The AST processes the resulting Mel-spectrogram, using attention mechanisms to mathematically isolate the harmonic structure of the genre while ignoring the injected ESC-50 noise patches.

## 📊 Classes
The model is trained to recognize the following 10 genres:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## 💻 How to Use
You can easily load this model for inference using the Hugging Face transformers library:

Python
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 1. Load Model and Extractor
model_id = "your-username/ast-messy-mashup-classifier" # Replace with your repo
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)

# 2. Load your noisy audio file
audio_path = "path/to/your/mashup.wav"
y, _ = librosa.load(audio_path, sr=22050, duration=5.0)

# 3. Extract features and predict
inputs = extractor(y, sampling_rate=22050, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class = torch.argmax(logits, dim=-1).item()
print(f"Predicted Genre: {model.config.id2label[predicted_class]}")
⚠️ Limitations
The model expects a sample rate of 22050 Hz and a duration of 5.0 seconds. Audio outside these bounds must be resampled or padded/truncated prior to inference.

Extreme stretching artifacts (e.g., forcing a 60 BPM track to 180 BPM) during mashup generation may occasionally degrade confidence scores
