"""Opt-in parity check against the official local Community-1 pipeline."""

import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "community-1"
ONNX_DIR = ROOT / "models" / "onnx-community-1"
SAMPLE = ROOT / "third_party" / "cpp-annote" / "audio" / "conversation.wav"


def _turns(annotation):
    return [
        (round(float(segment.start), 6), round(float(segment.end), 6), str(label))
        for segment, _, label in annotation.itertracks(yield_label=True)
    ]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_COMMUNITY1_PARITY") != "1"
    or not MODEL_DIR.is_dir()
    or not ONNX_DIR.is_dir()
    or not SAMPLE.is_file(),
    reason="set RUN_COMMUNITY1_PARITY=1 after downloading/exporting Community-1 artifacts",
)
def test_onnx_pipeline_matches_community1_reference_on_shared_waveform():
    """Compare both final outputs, avoiding decoder/resampler differences."""

    torch = pytest.importorskip("torch")
    reference_api = pytest.importorskip("pyannote.audio")

    from onnx_pyannote import ONNXSpeakerDiarization
    from utils.audio import decode_audio

    waveform = decode_audio(str(SAMPLE))[: 20 * 16_000]
    file = {
        "uri": "same-waveform",
        "waveform": torch.from_numpy(waveform)[None],
        "sample_rate": 16_000,
    }
    reference = reference_api.Pipeline.from_pretrained(MODEL_DIR)(file)
    actual = ONNXSpeakerDiarization(ONNX_DIR)(waveform)

    assert _turns(actual.speaker_diarization) == _turns(reference.speaker_diarization)
    assert _turns(actual.exclusive_speaker_diarization) == _turns(
        reference.exclusive_speaker_diarization
    )
    assert actual.speaker_embeddings.shape == reference.speaker_embeddings.shape
    np.testing.assert_allclose(
        actual.speaker_embeddings, reference.speaker_embeddings, rtol=1e-4, atol=1e-4
    )
