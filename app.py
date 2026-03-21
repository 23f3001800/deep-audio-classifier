# app.py — HuggingFace Spaces Streamlit deployment
import streamlit as st
import torch
import librosa
import numpy as np
import tempfile
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# ── Config ────────────────────────────────────────────────────
REPO_ID     = "Vikas25S/messy-mashup-ast-v2"
GENRES      = ["blues","classical","country","disco","hiphop",
               "jazz","metal","pop","reggae","rock"]
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000 * 10

# ── Load model (cached so it only loads once) ─────────────────
@st.cache_resource
def load_model():
    fe    = ASTFeatureExtractor.from_pretrained(REPO_ID)
    model = ASTForAudioClassification.from_pretrained(REPO_ID)
    model.eval()
    return fe, model

# ── Inference function ────────────────────────────────────────
def predict(audio_path, fe, model):
    # Load and resample to 16kHz
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Crop/pad to 10 seconds
    if len(y) >= CLIP_SAMPLES:
        y = y[:CLIP_SAMPLES]
    else:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))

    # Extract features
    inputs = fe(
        y,
        sampling_rate  = SAMPLE_RATE,
        return_tensors = "pt",
        padding        = "max_length"
    )

    # Predict
    with torch.no_grad():
        logits = model(input_values=inputs["input_values"].squeeze(0).unsqueeze(0)).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()

    return {genre: float(probs[i]) for i, genre in enumerate(GENRES)}

# ── Streamlit UI ──────────────────────────────────────────────
st.title("🎵 Music Genre Classifier")
st.write("Upload a WAV audio file to classify its genre.")

fe, model = load_model()
st.success("✅ Model loaded")

uploaded = st.file_uploader("Upload audio (WAV)", type=["wav"])

if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.audio(uploaded)

    with st.spinner("Classifying..."):
        results = predict(tmp_path, fe, model)

    # Sort by confidence
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # Display results
    st.subheader("Predictions:")
    for genre, prob in results.items():
        st.progress(prob, text=f"{genre:<12} {prob:.2%}")

    top_genre = list(results.keys())[0]
    st.success(f"🎯 Predicted Genre: **{top_genre.upper()}**")


