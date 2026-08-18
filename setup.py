from setuptools import setup, find_packages

setup(
    name="pyannote-onnx-extended",
    version="0.1.0",
    description="Community-1 speaker diarization with ONNX Runtime neural inference",
    author="User",
    packages=find_packages(),
    py_modules=["onnx_pyannote"],
    install_requires=[
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "av>=11",
        "scikit-learn",
        "huggingface_hub",
        "pyannote.core",
        "scipy",
        "torch",
        "torchaudio",
    ],
    python_requires=">=3.8",
)
