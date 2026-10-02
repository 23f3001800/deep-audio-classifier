"""Real checkpoint inference on synthetic inputs; this is not an accuracy test."""
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import ASTFeatureExtractor, ASTForAudioClassification


def main():
    torch.set_num_threads(2)
    repo = "Vikas25S/messy-mashup-ast-v2"
    revision = "eeef47a362c32313b6c4df23e126675e16c0eb9c"
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]))
    model = ASTForAudioClassification.from_pretrained(snapshot, local_files_only=True).eval()
    extractor = ASTFeatureExtractor.from_pretrained(snapshot, local_files_only=True)
    expected = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    assert [model.config.id2label[i] for i in range(10)] == expected
    assert extractor.sampling_rate == 16000
    rng = np.random.default_rng(42)
    waves = {"silence": np.zeros(160000, dtype=np.float32),
             "tone_440hz": (0.1 * np.sin(2 * np.pi * 440 * np.arange(160000) / 16000)).astype(np.float32),
             "noise_seed42": rng.normal(0, 0.01, 160000).astype(np.float32)}
    cases = []
    for name, wave in waves.items():
        inputs = extractor(wave, sampling_rate=16000, return_tensors="pt")
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            repeated = model(**inputs).logits
        assert logits.shape == (1, 10)
        assert torch.isfinite(logits).all()
        assert torch.allclose(logits, repeated, atol=1e-6, rtol=1e-5)
        cases.append({"input": name, "input_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                      "shape": list(logits.shape), "finite": True, "repeatable": True,
                      "two_forward_passes_ms": 1000 * (time.perf_counter() - start)})
    receipt = {"model": repo, "revision": revision, "run_id": os.getenv("GITHUB_RUN_ID"),
               "finished_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "torch": torch.__version__, "device": "cpu", "threads": 2,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "weights_sha256": hashlib.sha256((snapshot / "model.safetensors").read_bytes()).hexdigest(),
               "cases": cases, "accuracy_evaluated": False,
               "scope": "Synthetic silence, tone and noise test loading, shape, numerical stability and repeatability only. No genre ground truth."}
    out = Path("hub-smoke-results")
    out.mkdir(exist_ok=True)
    (out / "audio-smoke.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
