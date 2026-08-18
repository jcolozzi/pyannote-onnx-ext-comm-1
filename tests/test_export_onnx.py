import torch

from onnx_export import SegmentationToMultilabel


class _FixedPowersetModel(torch.nn.Module):
    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.scores


def test_segmentation_export_uses_pyannote_hard_powerset_conversion():
    # The winning powerset classes are: silence, speaker 1, and speakers 2+3.
    scores = torch.tensor(
        [[[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
          [-2.0, -1.0, -3.0, -4.0, -5.0, -6.0, -7.0],
          [-7.0, -6.0, -1.0, -5.0, -4.0, -3.0, -0.5]]]
    )
    model = SegmentationToMultilabel(_FixedPowersetModel(scores))

    activity = model(torch.zeros(1, 1, 160_000))

    assert torch.equal(
        activity,
        torch.tensor([[[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 1.0]]]),
    )
