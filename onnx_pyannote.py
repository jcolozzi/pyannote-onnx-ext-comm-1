"""ONNX Runtime implementation of pyannote Community-1 diarization.

The exported models contain the neural networks only. Community-1 also
requires the supplied PLDA parameters, VBx clustering, the exact Kaldi FBank
frontend, and reconstruction from the local-speaker segmentation tracks. This
module implements those non-neural stages so its regular and exclusive output
follow the public ``pyannote/speaker-diarization-community-1`` pipeline.

The VBx and PLDA helpers are a small adaptation of
``pyannote.audio.utils.vbx`` (Apache-2.0) and Community-1 pipeline logic
(MIT). See ``THIRD_PARTY_NOTICES.md`` for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import onnxruntime as ort
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans
import torch
import torchaudio

from utils.audio import decode_audio


SAMPLE_RATE = 16_000
CHUNK_DURATION = 10.0
CHUNK_STEP = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
NUM_LOCAL_SPEAKERS = 3
NUM_SEGMENTATION_FRAMES = 589
FRAME_DURATION = 0.0619375
FRAME_STEP = 0.016875
MIN_EMBEDDING_SAMPLES = 400


@dataclass(frozen=True)
class DiarizationOutput:
    """Community-1's regular and transcription-friendly diarization results."""

    speaker_diarization: Annotation
    exclusive_speaker_diarization: Annotation
    speaker_embeddings: np.ndarray

    def to_dict(self) -> dict[str, list[dict[str, float | str]]]:
        """Return a portable, JSON-friendly representation of both outputs."""

        def turns(annotation: Annotation) -> list[dict[str, float | str]]:
            return [
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "speaker": str(speaker),
                }
                for segment, _, speaker in annotation.itertracks(yield_label=True)
            ]

        return {
            "diarization": turns(self.speaker_diarization),
            "exclusive_diarization": turns(self.exclusive_speaker_diarization),
        }


class _PLDA:
    """Community-1's x-vector transform and PLDA projection."""

    def __init__(self, transform_path: Path, plda_path: Path, lda_dimension: int = 128):
        transform = np.load(transform_path)
        mean1, mean2, lda = transform["mean1"], transform["mean2"], transform["lda"]

        plda = np.load(plda_path)
        mu, transform_matrix, psi = plda["mu"], plda["tr"], plda["psi"]

        within = np.linalg.inv(transform_matrix.T.dot(transform_matrix))
        between = np.linalg.inv((transform_matrix.T / psi).dot(transform_matrix))
        eigenvalues, whitening = eigh(between, within)

        self._mean1 = mean1
        self._mean2 = mean2
        self._lda = lda
        self._mu = mu
        self._transform = whitening.T[::-1]
        self._psi = eigenvalues[::-1]
        self.lda_dimension = lda_dimension

    @staticmethod
    def _l2_norm(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return values / np.linalg.norm(values)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    @property
    def phi(self) -> np.ndarray:
        return self._psi[: self.lda_dimension]

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = np.sqrt(self._lda.shape[1]) * self._l2_norm(
            self._lda.T.dot(
                np.sqrt(self._lda.shape[0])
                * self._l2_norm(embeddings - self._mean1).T
            ).T
            - self._mean2
        )
        return (transformed - self._mu).dot(self._transform.T)[:, : self.lda_dimension]


def _vbx(
    features: np.ndarray,
    phi: np.ndarray,
    *,
    fa: float,
    fb: float,
    priors: np.ndarray,
    responsibilities: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the GMM form of VBx used by Community-1."""

    dimension = features.shape[1]
    gamma = responsibilities
    pi = priors
    constant = -0.5 * (
        np.sum(features**2, axis=1, keepdims=True) + dimension * np.log(2 * np.pi)
    )
    rho = features * np.sqrt(phi)
    previous_elbo: float | None = None

    for _ in range(max_iterations):
        inverse_precision = 1.0 / (
            1 + fa / fb * gamma.sum(axis=0, keepdims=True).T * phi
        )
        alpha = fa / fb * inverse_precision * gamma.T.dot(rho)
        log_probability = fa * (
            rho.dot(alpha.T)
            - 0.5 * (inverse_precision + alpha**2).dot(phi)
            + constant
        )

        log_pi = np.log(pi + 1e-8)
        log_marginal = logsumexp(log_probability + log_pi, axis=-1)
        gamma = np.exp(log_probability + log_pi - log_marginal[:, None])
        pi = gamma.sum(axis=0)
        pi = pi / pi.sum()
        elbo = np.sum(log_marginal) + fb * 0.5 * np.sum(
            np.log(inverse_precision) - inverse_precision - alpha**2 + 1
        )

        if previous_elbo is not None and elbo - previous_elbo < 1e-4:
            break
        previous_elbo = elbo

    return gamma, pi


def _cluster_vbx(
    ahc_clusters: np.ndarray,
    features: np.ndarray,
    phi: np.ndarray,
    *,
    fa: float,
    fb: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.zeros((len(ahc_clusters), int(ahc_clusters.max()) + 1))
    initial[np.arange(len(ahc_clusters)), ahc_clusters.astype(int)] = 1.0
    initial = softmax(initial * 7.0, axis=1)
    return _vbx(
        features,
        phi,
        fa=fa,
        fb=fb,
        priors=np.ones(initial.shape[1]) / initial.shape[1],
        responsibilities=initial,
        max_iterations=20,
    )


class ONNXSpeakerDiarization:
    """Faithful ONNX Runtime backend for Community-1.

    ``model_dir`` must contain the files generated by ``export_onnx.py``. The
    returned :class:`DiarizationOutput` mirrors current pyannote.audio output:
    regular diarization preserves overlap, while exclusive diarization is
    reconstructed independently with the instantaneous speaker count capped at
    one.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        model_name: str | None = None,
        segmentation_path: str | Path | None = None,
        embedding_path: str | Path | None = None,
        plda_dir: str | Path | None = None,
        providers: list[str] | None = None,
        segmentation_batch_size: int = 32,
        embedding_batch_size: int = 32,
        return_exclusive: bool | None = None,
    ):
        """Load exported Community-1 artifacts.

        ``return_exclusive`` is a compatibility switch. Leave it as ``None``
        (the default) to return both reference outputs. ``True`` or ``False``
        returns only the requested annotation, as older releases did.
        """

        if model_name not in (None, "speaker-diarization-community-1"):
            raise ValueError(
                "This backend implements only speaker-diarization-community-1; "
                "use an exported Community-1 model directory."
            )
        if model_dir is None:
            model_dir = Path("models/onnx-community-1")
        model_dir = Path(model_dir)
        segmentation_path = Path(segmentation_path or model_dir / "segmentation.onnx")
        embedding_path = Path(embedding_path or model_dir / "embedding.onnx")
        plda_dir = Path(plda_dir or model_dir / "plda")

        missing = [
            str(path)
            for path in (
                segmentation_path,
                embedding_path,
                plda_dir / "plda.npz",
                plda_dir / "xvec_transform.npz",
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Community-1 ONNX artifacts are incomplete. Run `python export_onnx.py` "
                f"first. Missing: {', '.join(missing)}"
            )

        self.segmentation_session = ort.InferenceSession(
            str(segmentation_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.embedding_session = ort.InferenceSession(
            str(embedding_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._validate_model_interfaces()
        self.segmentation_input_name = self.segmentation_session.get_inputs()[0].name
        self.segmentation_output_name = self.segmentation_session.get_outputs()[0].name
        self.embedding_input_names = [item.name for item in self.embedding_session.get_inputs()]
        self.embedding_output_name = self.embedding_session.get_outputs()[0].name

        self.plda = _PLDA(
            plda_dir / "xvec_transform.npz", plda_dir / "plda.npz", lda_dimension=128
        )
        self.segmentation_batch_size = self._supported_segmentation_batch_size(
            segmentation_batch_size
        )
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.return_exclusive = return_exclusive
        self._chunks = SlidingWindow(start=0.0, duration=CHUNK_DURATION, step=CHUNK_STEP)
        self._frames = SlidingWindow(
            start=0.0, duration=FRAME_DURATION, step=FRAME_STEP
        )

    def _validate_model_interfaces(self) -> None:
        segmentation_input = self.segmentation_session.get_inputs()[0]
        segmentation_output = self.segmentation_session.get_outputs()[0]
        embedding_inputs = self.embedding_session.get_inputs()
        embedding_output = self.embedding_session.get_outputs()[0]

        if len(self.segmentation_session.get_inputs()) != 1 or len(
            self.segmentation_session.get_outputs()
        ) != 1:
            raise ValueError("segmentation.onnx must have one waveform input and one output")
        if len(embedding_inputs) != 2 or len(self.embedding_session.get_outputs()) != 1:
            raise ValueError("embedding.onnx must have FBank and mask inputs and one output")
        if segmentation_input.shape[-1] != CHUNK_SAMPLES:
            raise ValueError("segmentation.onnx is not a 10-second Community-1 model")
        if segmentation_output.shape[-2:] != [NUM_SEGMENTATION_FRAMES, NUM_LOCAL_SPEAKERS]:
            raise ValueError("segmentation.onnx must output (batch, 589, 3) activity")
        if embedding_output.shape[-1] != 256:
            raise ValueError("embedding.onnx must output 256-dimensional embeddings")

    def _supported_segmentation_batch_size(self, requested: int) -> int:
        batch_dimension = self.segmentation_session.get_inputs()[0].shape[0]
        if isinstance(batch_dimension, int) and batch_dimension == 1:
            if requested != 1:
                warnings.warn(
                    "segmentation.onnx has a fixed batch dimension; processing one chunk at a time. "
                    "Re-export with the current export_onnx.py for batched inference.",
                    stacklevel=2,
                )
            return 1
        return max(1, requested)

    @staticmethod
    def _audio_array(audio: str | Path | np.ndarray | dict) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            waveform = decode_audio(str(audio), SAMPLE_RATE)
        elif isinstance(audio, dict):
            if "waveform" not in audio:
                raise ValueError("audio dictionaries must include a 'waveform' entry")
            sample_rate = audio.get("sample_rate", SAMPLE_RATE)
            waveform = np.asarray(audio["waveform"], dtype=np.float32)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    torch.as_tensor(waveform), sample_rate, SAMPLE_RATE
                ).numpy()
        else:
            waveform = np.asarray(audio, dtype=np.float32)

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1:
            raise ValueError("audio must be a mono waveform or a path to an audio file")
        return np.ascontiguousarray(waveform, dtype=np.float32)

    @staticmethod
    def _audio_uri(audio: str | Path | np.ndarray | dict) -> str | None:
        """Mirror pyannote's annotation URI when the caller supplies one."""

        if isinstance(audio, (str, Path)):
            return str(audio)
        if isinstance(audio, dict):
            return audio.get("uri")
        return None

    @staticmethod
    def _chunk_starts(num_samples: int) -> Iterable[int]:
        num_complete = (
            (num_samples - CHUNK_SAMPLES) // SAMPLE_RATE + 1
            if num_samples >= CHUNK_SAMPLES
            else 0
        )
        for index in range(num_complete):
            yield index * SAMPLE_RATE
        if num_samples < CHUNK_SAMPLES or (num_samples - CHUNK_SAMPLES) % SAMPLE_RATE:
            yield num_complete * SAMPLE_RATE

    @staticmethod
    def _make_chunk(waveform: np.ndarray, start: int) -> np.ndarray:
        chunk = waveform[start : start + CHUNK_SAMPLES]
        if len(chunk) == CHUNK_SAMPLES:
            return chunk
        return np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

    def run_segmentation(self, waveform: np.ndarray) -> SlidingWindowFeature:
        """Apply hard Community-1 powerset conversion on 10s/1s chunks."""

        starts = list(self._chunk_starts(len(waveform)))
        outputs: list[np.ndarray] = []
        for offset in range(0, len(starts), self.segmentation_batch_size):
            batch = np.stack(
                [
                    self._make_chunk(waveform, start)
                    for start in starts[offset : offset + self.segmentation_batch_size]
                ]
            )[:, None, :].astype(np.float32, copy=False)
            output = self.segmentation_session.run(
                [self.segmentation_output_name], {self.segmentation_input_name: batch}
            )[0]
            outputs.append(output.astype(np.float32, copy=False))

        return SlidingWindowFeature(np.concatenate(outputs, axis=0), self._chunks)

    @staticmethod
    def _compute_fbank(waveforms: np.ndarray) -> np.ndarray:
        """Match WeSpeakerResNet34's torchaudio Kaldi FBank frontend exactly."""

        tensor = torch.from_numpy(waveforms) * (1 << 15)
        features = torch.vmap(
            lambda waveform: torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                round_to_power_of_two=True,
                snip_edges=True,
                dither=0.0,
                sample_frequency=SAMPLE_RATE,
                window_type="hamming",
                use_energy=False,
            )
        )(tensor)
        return (features - torch.mean(features, dim=1, keepdim=True)).numpy()

    def get_embeddings(
        self, waveform: np.ndarray, segmentations: SlidingWindowFeature
    ) -> np.ndarray:
        """Extract one overlap-masked embedding per local speaker and chunk."""

        data = np.nan_to_num(segmentations.data, nan=0.0).astype(np.float32)
        num_chunks, num_frames, num_speakers = data.shape
        min_clean_frames = ceil(
            num_frames * MIN_EMBEDDING_SAMPLES / (CHUNK_DURATION * SAMPLE_RATE)
        )
        clean = data * (np.sum(data, axis=2, keepdims=True) < 2)
        masks = np.where(
            np.sum(clean, axis=1, keepdims=True) > min_clean_frames,
            clean,
            data,
        )

        starts = list(self._chunk_starts(len(waveform)))
        chunks = np.stack([self._make_chunk(waveform, start) for start in starts])[:, None, :]
        flat_waveforms = np.repeat(chunks, num_speakers, axis=0)
        flat_masks = np.transpose(masks, (0, 2, 1)).reshape(-1, num_frames)

        batches: list[np.ndarray] = []
        for offset in range(0, len(flat_masks), self.embedding_batch_size):
            waveforms_batch = np.ascontiguousarray(
                flat_waveforms[offset : offset + self.embedding_batch_size], dtype=np.float32
            )
            masks_batch = np.ascontiguousarray(
                flat_masks[offset : offset + self.embedding_batch_size], dtype=np.float32
            )
            fbank = self._compute_fbank(waveforms_batch)
            inputs = {
                self.embedding_input_names[0]: fbank,
                self.embedding_input_names[1]: masks_batch,
            }
            batches.append(self.embedding_session.run([self.embedding_output_name], inputs)[0])

        return np.concatenate(batches, axis=0).reshape(num_chunks, num_speakers, -1)

    @staticmethod
    def _aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow,
        *,
        skip_average: bool,
        missing: float,
    ) -> SlidingWindowFeature:
        """Port of pyannote.audio.core.Inference.aggregate for Community-1."""

        num_chunks, num_chunk_frames, num_classes = scores.data.shape
        chunks = scores.sliding_window
        frames = SlidingWindow(start=chunks.start, duration=frames.duration, step=frames.step)
        num_frames = (
            frames.closest_frame(
                chunks.start
                + chunks.duration
                + (num_chunks - 1) * chunks.step
                + 0.5 * frames.duration
            )
            + 1
        )
        aggregated = np.zeros((num_frames, num_classes), dtype=np.float32)
        overlap_count = np.zeros_like(aggregated)
        aggregate_mask = np.zeros_like(aggregated)

        for chunk_index, (_, score) in enumerate(scores):
            score = score.copy()
            mask = 1 - np.isnan(score)
            np.nan_to_num(score, copy=False, nan=0.0)
            start = frames.closest_frame(chunks[chunk_index].start + 0.5 * frames.duration)
            target = slice(start, start + num_chunk_frames)
            aggregated[target] += score * mask
            overlap_count[target] += mask
            aggregate_mask[target] = np.maximum(aggregate_mask[target], mask)

        result = aggregated if skip_average else aggregated / np.maximum(overlap_count, 1e-12)
        result[aggregate_mask == 0.0] = missing
        return SlidingWindowFeature(result, frames)

    def speaker_count(self, segmentations: SlidingWindowFeature) -> SlidingWindowFeature:
        counts = SlidingWindowFeature(
            np.sum(segmentations.data, axis=-1, keepdims=True), segmentations.sliding_window
        )
        result = self._aggregate(counts, self._frames, skip_average=False, missing=0.0)
        result.data = np.rint(result.data).astype(np.uint8)
        return result

    @staticmethod
    def _set_num_speakers(
        num_speakers: int | None, min_speakers: int | None, max_speakers: int | None
    ) -> tuple[int | None, int, float]:
        min_speakers = num_speakers or min_speakers or 1
        max_speakers = num_speakers or max_speakers or np.inf
        if min_speakers > max_speakers:
            raise ValueError("min_speakers must be smaller than or equal to max_speakers")
        return (
            min_speakers if min_speakers == max_speakers else num_speakers,
            min_speakers,
            max_speakers,
        )

    @staticmethod
    def _filter_embeddings(
        embeddings: np.ndarray, segmentations: SlidingWindowFeature
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        single = np.sum(segmentations.data, axis=2, keepdims=True) == 1
        clean_frames = np.sum(segmentations.data * single, axis=1)
        active = clean_frames >= 0.2 * segmentations.data.shape[1]
        valid = ~np.any(np.isnan(embeddings), axis=2)
        chunk_indices, speaker_indices = np.where(active * valid)
        return embeddings[chunk_indices, speaker_indices], chunk_indices, speaker_indices

    @staticmethod
    def _constrained_argmax(scores: np.ndarray) -> np.ndarray:
        scores = np.nan_to_num(scores, nan=np.nanmin(scores))
        num_chunks, num_speakers, _ = scores.shape
        clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        for chunk, cost in enumerate(scores):
            speakers, assignments = linear_sum_assignment(cost, maximize=True)
            clusters[chunk, speakers] = assignments
        return clusters

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature,
        *,
        num_speakers: int | None,
        min_speakers: int,
        max_speakers: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform Community-1 AHC initialisation, PLDA, and VBx clustering."""

        train, _, _ = self._filter_embeddings(embeddings, segmentations)
        num_chunks, num_local_speakers, dimension = embeddings.shape
        if len(train) < 2:
            clusters = np.zeros((num_chunks, num_local_speakers), dtype=np.int8)
            return clusters, np.mean(train, axis=0, keepdims=True)

        normalized = train / np.linalg.norm(train, axis=1, keepdims=True)
        dendrogram = linkage(normalized, method="centroid", metric="euclidean")
        ahc_clusters = fcluster(dendrogram, 0.6, criterion="distance") - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)
        responsibilities, priors = _cluster_vbx(
            ahc_clusters, self.plda(train), self.plda.phi, fa=0.07, fb=0.8
        )
        weights = responsibilities[:, priors > 1e-7]
        centroids = weights.T @ train.reshape(-1, dimension) / weights.sum(
            axis=0, keepdims=True
        ).T

        constrained = True
        automatic_count = centroids.shape[0]
        forced_count = num_speakers
        if automatic_count < min_speakers:
            forced_count = min_speakers
        elif automatic_count > max_speakers:
            forced_count = int(max_speakers)
        if forced_count and forced_count != automatic_count:
            constrained = False
            labels = KMeans(
                n_clusters=forced_count, n_init=3, random_state=42, copy_x=False
            ).fit_predict(normalized)
            centroids = np.vstack(
                [np.mean(train[labels == cluster], axis=0) for cluster in range(forced_count)]
            )

        distances = cdist(
            embeddings.reshape(-1, dimension), centroids, metric="cosine"
        ).reshape(num_chunks, num_local_speakers, -1)
        scores = 2 - distances
        if constrained:
            scores[segmentations.data.sum(axis=1) == 0] = scores.min() - 1
            clusters = self._constrained_argmax(scores)
        else:
            clusters = np.argmax(scores, axis=2)
        return clusters.astype(np.int8, copy=False), centroids

    def to_diarization(
        self, segmentations: SlidingWindowFeature, count: SlidingWindowFeature
    ) -> SlidingWindowFeature:
        activations = self._aggregate(
            segmentations, count.sliding_window, skip_average=True, missing=0.0
        )
        max_speakers_per_frame = int(np.max(count.data))
        if activations.data.shape[1] < max_speakers_per_frame:
            activations.data = np.pad(
                activations.data,
                ((0, 0), (0, max_speakers_per_frame - activations.data.shape[1])),
            )

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)
        speakers = np.argsort(-activations.data, axis=-1)
        binary = np.zeros_like(activations.data)
        for frame_index, frame_count in enumerate(count.data[:, 0]):
            binary[frame_index, speakers[frame_index, : int(frame_count)]] = 1.0
        return SlidingWindowFeature(binary, activations.sliding_window)

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        num_chunks, num_frames, _ = segmentations.data.shape
        num_clusters = int(np.max(hard_clusters)) + 1
        clustered = np.full((num_chunks, num_frames, num_clusters), np.nan)
        for chunk_index, (cluster, (_, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            for cluster_index in np.unique(cluster):
                if cluster_index != -2:
                    clustered[chunk_index, :, cluster_index] = np.max(
                        segmentation[:, cluster == cluster_index], axis=1
                    )
        return self.to_diarization(
            SlidingWindowFeature(clustered, segmentations.sliding_window), count
        )

    @staticmethod
    def _to_annotation(diarization: SlidingWindowFeature) -> Annotation:
        """Convert reconstructed binary frames using pyannote's Binarize semantics."""

        annotation = Annotation()
        timestamps = [
            diarization.sliding_window[index].middle for index in range(len(diarization.data))
        ]
        for speaker, scores in enumerate(diarization.data.T):
            active = scores[0] > 0.5
            start = timestamps[0]
            for timestamp, score in zip(timestamps[1:], scores[1:]):
                if active and score < 0.5:
                    annotation[Segment(start, timestamp), str(speaker)] = speaker
                    start = timestamp
                    active = False
                elif not active and score > 0.5:
                    start = timestamp
                    active = True
            if active:
                annotation[Segment(start, timestamps[-1]), str(speaker)] = speaker
        return annotation

    @staticmethod
    def build_exclusive_annotation(annotation: Annotation) -> Annotation:
        """Deprecated compatibility helper; pipeline results use reconstruction instead.

        A bare annotation lacks the frame activations and speaker counts needed
        for Community-1's reference exclusive diarization. This helper only
        keeps historical deterministic tie-breaking for callers migrating to
        :attr:`DiarizationOutput.exclusive_speaker_diarization`.
        """

        events = [
            (float(segment.start), float(segment.end), str(speaker))
            for segment, _, speaker in annotation.itertracks(yield_label=True)
            if segment.end > segment.start
        ]
        if not events:
            return Annotation()
        result = Annotation()
        boundaries = sorted({time for start, end, _ in events for time in (start, end)})
        order = list(dict.fromkeys(speaker for _, _, speaker in events))
        for start, end in zip(boundaries, boundaries[1:]):
            active = [speaker for left, right, speaker in events if left < end and right > start]
            if active:
                if len(active) == 1:
                    selected = active[0]
                else:
                    center = (start + end) / 2.0

                    def score(candidate: str) -> tuple[float, float]:
                        coverage = sum(
                            min(end, right) - max(start, left)
                            for left, right, speaker in events
                            if speaker == candidate and left < end and right > start
                        )
                        distance = min(
                            abs(center - (left + right) / 2.0)
                            for left, right, speaker in events
                            if speaker == candidate
                        )
                        return coverage, -distance

                    selected = max(
                        active, key=lambda candidate: (*score(candidate), -order.index(candidate))
                    )
                result[Segment(start, end)] = selected
        return result.support(collar=0.0)

    def __call__(
        self,
        audio: str | Path | np.ndarray | dict,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        return_exclusive: bool | None = None,
    ) -> DiarizationOutput | Annotation:
        waveform = self._audio_array(audio)
        uri = self._audio_uri(audio)
        num_speakers, min_speakers, max_speakers = self._set_num_speakers(
            num_speakers, min_speakers, max_speakers
        )
        segmentations = self.run_segmentation(waveform)
        count = self.speaker_count(segmentations)

        if np.max(count.data) == 0:
            output = DiarizationOutput(
                Annotation(uri=uri), Annotation(uri=uri), np.zeros((0, 256))
            )
        else:
            embeddings = self.get_embeddings(waveform, segmentations)
            hard_clusters, centroids = self.cluster_embeddings(
                embeddings,
                segmentations,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            count.data = np.minimum(count.data, max_speakers).astype(np.int8)
            hard_clusters[np.sum(segmentations.data, axis=1) == 0] = -2

            diarization = self._to_annotation(
                self.reconstruct(segmentations, hard_clusters, count)
            )
            diarization.uri = uri
            count.data = np.minimum(count.data, 1).astype(np.int8)
            exclusive = self._to_annotation(
                self.reconstruct(segmentations, hard_clusters, count)
            )
            exclusive.uri = uri
            labels = diarization.labels()
            if len(labels) > len(centroids):
                centroids = np.pad(centroids, ((0, len(labels) - len(centroids)), (0, 0)))
            mapping = {label: f"SPEAKER_{int(label):02d}" for label in labels}
            diarization = diarization.rename_labels(mapping=mapping)
            exclusive = exclusive.rename_labels(mapping=mapping)
            ordered_centroids = centroids[[int(label) for label in labels]]
            output = DiarizationOutput(diarization, exclusive, ordered_centroids)

        selection = self.return_exclusive if return_exclusive is None else return_exclusive
        if selection is True:
            return output.exclusive_speaker_diarization
        if selection is False:
            return output.speaker_diarization
        return output
