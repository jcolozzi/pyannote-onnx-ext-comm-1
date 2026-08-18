# Pyannote ONNX Extended Community-1

This repository provides an ONNX Runtime backend for the gated
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline. It preserves the reference pipeline's Community-1 behavior instead
of replacing its post-processing with generic clustering.

It includes:

- the official 10-second segmentation schedule with a 1-second hop;
- pyannote's hard powerset-to-multilabel conversion;
- exact 16 kHz Kaldi FBank preprocessing and local-speaker masking;
- the shipped PLDA transform, AHC initialization, and VBx clustering;
- regular overlap-preserving diarization; and
- separately reconstructed exclusive diarization, using a speaker-count cap of
  one exactly as Community-1 does.

## Setup and export

Accept the Community-1 model terms on Hugging Face, authenticate, then
download and convert the checkpoint. PLDA/VBx is not an ONNX graph: its
supplied `.npz` files are copied alongside the two neural models.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
hf auth login
hf download pyannote/speaker-diarization-community-1 --local-dir models/community-1
python export_onnx.py
```

The result is an artifact directory like this:

```text
models/onnx-community-1/
  segmentation.onnx
  embedding.onnx
  plda/plda.npz
  plda/xvec_transform.npz
  community-1.json
```

The neural inference is ONNX Runtime. The exact FBank frontend uses
`torchaudio` because the Kaldi feature extraction operation cannot be exported
to ONNX without changing the model interface. This is intentional: changing
it to a generic Mel frontend breaks Community-1 parity.

## Use from Python

```python
from onnx_pyannote import ONNXSpeakerDiarization

pipeline = ONNXSpeakerDiarization(
    "models/onnx-community-1",
    providers=["CPUExecutionProvider"],
)
output = pipeline("meeting.wav")

for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.3f} {turn.end:.3f} {speaker}")

# This is a separate Community-1 reconstruction, not a turn-level tie-breaker.
exclusive = output.exclusive_speaker_diarization
```

The default return value is `DiarizationOutput`, which contains both results
and speaker embeddings. Existing callers that require one `Annotation` can
request it explicitly:

```python
regular = pipeline("meeting.wav", return_exclusive=False)
exclusive = pipeline("meeting.wav", return_exclusive=True)
```

`num_speakers`, `min_speakers`, and `max_speakers` are supported. As in the
official pipeline, forcing a speaker count uses K-Means after the VBx estimate.

## Parity verification

The normal unit tests require no gated model. The opt-in integration test uses
the downloaded Community-1 source pipeline and feeds it the exact same decoded
waveform as the ONNX backend, so decoder/resampler variations do not mask a
pipeline mismatch:

```powershell
$env:RUN_COMMUNITY1_PARITY = "1"
pytest -q -m integration
```

On the bundled 73.15-second `cpp-annote` conversation sample, the ONNX backend
produced exactly the same 16 regular turns and 16 exclusive turns as
`pyannote.audio` Community-1. It ran in 34.448 seconds on this CPU compared
with 76.235 seconds for the reference pipeline. The sample has no detected
overlap, so the integration test also exercises the separate exclusive path on
inputs where it differs when overlap is present.

## Testing

```powershell
pytest -q
```

The exporter validates ONNX Runtime model outputs against the source PyTorch
checkpoint unless `--skip-validation` is passed.
