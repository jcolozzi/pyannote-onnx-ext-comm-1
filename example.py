"""Run the exported Community-1 ONNX pipeline on an audio file."""

from pathlib import Path
import time

from onnx_pyannote import ONNXSpeakerDiarization


def main() -> None:
    audio_path = Path("example.wav")
    if not audio_path.is_file():
        print(f"{audio_path} not found. Provide an audio file before running this example.")
        return

    pipeline = ONNXSpeakerDiarization(
        "models/onnx-community-1", providers=["CPUExecutionProvider"]
    )
    started = time.perf_counter()
    output = pipeline(audio_path)
    print(f"Diarization completed in {time.perf_counter() - started:.2f} seconds.")

    print("Regular diarization:")
    for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True):
        print(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}")

    print("Exclusive diarization:")
    for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(
        yield_label=True
    ):
        print(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}")


if __name__ == "__main__":
    main()
