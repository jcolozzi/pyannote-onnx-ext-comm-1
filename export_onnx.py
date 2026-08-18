"""Export the local pyannote Community-1 neural checkpoints to ONNX.

Community-1 has three runtime components:

* the 10-second segmentation network;
* the masked speaker-embedding network; and
* PLDA/VBx clustering data (``plda/*.npz``).

Only the first two are neural networks and are therefore converted to ONNX.
The PLDA files are copied unchanged and ``community-1.json`` records the
parameters needed by an ONNX Runtime implementation of the whole pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from pyannote.audio import Pipeline


SAMPLE_RATE = 16_000
CHUNK_DURATION = 10.0
NUM_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
NUM_FRAMES = 589


class FbankMaskedEmbedding(torch.nn.Module):
    """Export the embedding network after Community-1's Kaldi FBank frontend.

    ``torchaudio.compliance.kaldi.fbank`` is not traceable to ONNX.  Keeping
    it outside the graph is intentional and also matches the public WeSpeaker
    ONNX model convention: callers provide mean-centered 80-bin FBank input.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, fbank: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(fbank, weights=mask)[1]


class SegmentationToMultilabel(torch.nn.Module):
    """Match pyannote's hard powerset-to-multilabel conversion exactly."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mapping",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        powerset_scores = self.model(waveform)
        powerset = torch.nn.functional.one_hot(
            torch.argmax(powerset_scores, dim=-1),
            num_classes=self.mapping.shape[0],
        ).to(dtype=self.mapping.dtype)
        return torch.matmul(powerset, self.mapping)


def _export(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    destination: Path,
    input_names: list[str],
    output_name: str,
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    """Export a Community-1 model and validate its ONNX protobuf."""

    model.eval()
    with torch.inference_mode():
        export_args = dict(
            input_names=input_names,
            output_names=[output_name],
            opset_version=opset,
            dynamo=False,
        )
        if dynamic_axes is not None:
            export_args["dynamic_axes"] = dynamic_axes

        torch.onnx.export(
            model,
            inputs if len(inputs) > 1 else inputs[0],
            str(destination),
            **export_args,
        )

    onnx.checker.check_model(str(destination))


def _validate_onnx(
    segmentation: torch.nn.Module,
    embedding_source: torch.nn.Module,
    output_dir: Path,
) -> None:
    """Check ONNX Runtime outputs against the PyTorch source checkpoints."""

    torch.manual_seed(0)
    waveform = torch.randn(1, 1, NUM_SAMPLES)
    mask = torch.ones(1, NUM_FRAMES)

    with torch.inference_mode():
        reference_segmentation = segmentation(waveform).cpu().numpy()
        reference_embedding = embedding_source(waveform, weights=mask).cpu().numpy()
        fbank = embedding_source.compute_fbank(waveform).cpu().numpy()

    segmentation_session = ort.InferenceSession(
        str(output_dir / "segmentation.onnx"),
        providers=["CPUExecutionProvider"],
    )
    embedding_session = ort.InferenceSession(
        str(output_dir / "embedding.onnx"),
        providers=["CPUExecutionProvider"],
    )
    actual_segmentation = segmentation_session.run(
        ["segmentation"], {"waveforms": waveform.numpy()}
    )[0]
    actual_embedding = embedding_session.run(
        ["embedding"], {"fbank": fbank, "mask": mask.numpy()}
    )[0]

    np.testing.assert_allclose(
        actual_segmentation, reference_segmentation, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        actual_embedding, reference_embedding, rtol=1e-4, atol=1e-4
    )


def export_onnx(model_dir: Path, output_dir: Path, opset: int, validate: bool) -> None:
    """Export Community-1 from a local Hugging Face snapshot."""

    if not (model_dir / "config.yaml").is_file():
        raise FileNotFoundError(
            f"{model_dir} is not a local Community-1 model directory; config.yaml is missing."
        )

    pipeline = Pipeline.from_pretrained(model_dir)
    segmentation = SegmentationToMultilabel(pipeline._segmentation.model.eval())
    embedding_source = pipeline._embedding.model_.eval()
    embedding = FbankMaskedEmbedding(embedding_source.resnet.eval())

    output_dir.mkdir(parents=True, exist_ok=True)
    _export(
        segmentation,
        (torch.zeros(1, 1, NUM_SAMPLES),),
        output_dir / "segmentation.onnx",
        ["waveforms"],
        "segmentation",
        opset,
        dynamic_axes={
            "waveforms": {0: "batch"},
            "segmentation": {0: "batch"},
        },
    )
    _export(
        embedding,
        (torch.zeros(1, 998, 80), torch.ones(1, NUM_FRAMES)),
        output_dir / "embedding.onnx",
        ["fbank", "mask"],
        "embedding",
        opset,
        dynamic_axes={
            "fbank": {0: "batch", 1: "num_fbank_frames"},
            "mask": {0: "batch", 1: "num_mask_frames"},
            "embedding": {0: "batch"},
        },
    )

    plda_dir = output_dir / "plda"
    plda_dir.mkdir(exist_ok=True)
    for name in ("plda.npz", "xvec_transform.npz"):
        shutil.copy2(model_dir / "plda" / name, plda_dir / name)

    metadata = {
        "model": "pyannote/speaker-diarization-community-1",
        "sample_rate": SAMPLE_RATE,
        "chunk_duration": CHUNK_DURATION,
        "num_samples": NUM_SAMPLES,
        "num_frames": NUM_FRAMES,
        "frame_step": 0.016875,
        "frame_duration": 0.0619375,
        "segmentation": {
            "format": "hard multi-label speaker activity",
            "output_speakers": 3,
            "powerset_mapping": [
                [],
                [0],
                [1],
                [2],
                [0, 1],
                [0, 2],
                [1, 2],
            ],
        },
        "embedding": {
            "dimension": 256,
            "input": "mean-centered 80-bin Kaldi FBank",
            "fbank_num_frames": 998,
            "fbank": {
                "frame_length_ms": 25,
                "frame_shift_ms": 10,
                "window": "hamming",
                "dither": 0.0,
                "snip_edges": True,
            },
            "mask_input": "589-frame speaker activity mask",
        },
        "clustering": {
            "algorithm": "VBx",
            "threshold": 0.6,
            "Fa": 0.07,
            "Fb": 0.8,
            "plda_lda_dimension": 128,
        },
        "cpp_annote_compatible": True,
    }
    (output_dir / "community-1.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if validate:
        _validate_onnx(segmentation, embedding_source, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/community-1"),
        help="local Hugging Face Community-1 snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/onnx-community-1"),
        help="directory for ONNX models and PLDA artifacts",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        opset=args.opset,
        validate=not args.skip_validation,
    )
    print(f"Exported Community-1 ONNX artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
