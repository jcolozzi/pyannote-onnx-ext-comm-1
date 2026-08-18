"""Small ONNX-export helpers that are safe to import in the test suite."""

from __future__ import annotations

import torch


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
