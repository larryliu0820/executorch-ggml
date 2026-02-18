"""executorch-ggml: ExecuTorch backend delegating to ggml."""

from setuptools import setup, find_packages

setup(
    name="executorch-ggml",
    version="0.1.0",
    description="ExecuTorch backend delegating to ggml",
    author="Larry Liu",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.10",
    install_requires=[
        "flatbuffers>=24.3.25",
        "torch>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
