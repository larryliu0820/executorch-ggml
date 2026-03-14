"""
Setup configuration for executorch-ggml.

Metadata and dependencies are defined in pyproject.toml (PEP 621).
This file only handles the custom CMake build logic for the native extension.

The native extension is always built when dependencies are available.
Dependencies are discovered automatically from third-party/ submodules, or
can be specified explicitly via environment variables:
  - LLAMA_CPP_DIR=/path/to/llama.cpp
  - EXECUTORCH_DIR=/path/to/executorch (repo root containing executorch/)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self) -> None:
        for ext in self.extensions:
            self.build_extension(ext)

    def _discover_dep(self, env_key: str, candidates: List[pathlib.Path]) -> str | None:
        v = os.environ.get(env_key)
        if v:
            return v
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def build_extension(self, ext: Extension) -> None:
        import subprocess

        root = pathlib.Path(__file__).resolve().parent

        # Prefer deps vendored under this repo's third-party/ if present.
        llama_dir = self._discover_dep(
            "LLAMA_CPP_DIR",
            [root / "third-party" / "llama.cpp", root / "third-party" / "llama"],
        )
        et_dir = self._discover_dep(
            "EXECUTORCH_DIR",
            [root / "third-party" / "executorch"],
        )

        if not llama_dir or not et_dir:
            print(
                "[executorch-ggml] Skipping native extension build. "
                "Could not locate dependencies. Set LLAMA_CPP_DIR and EXECUTORCH_DIR, "
                "or vendor them under third-party/{llama.cpp,executorch}."
            )
            return

        build_temp = root / "build"
        build_temp.mkdir(parents=True, exist_ok=True)

        # Auto-discover python interpreter: use the one running setup.py
        python_exe = sys.executable

        cmake_args: List[str] = [
            "-G",
            "Ninja",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DLLAMA_CPP_DIR={llama_dir}",
            f"-DEXECUTORCH_DIR={et_dir}",
        ]

        env = os.environ.copy()

        subprocess.check_call(["cmake", "-S", str(root), "-B", str(build_temp), *cmake_args], env=env)
        subprocess.check_call(
            ["cmake", "--build", str(build_temp), "--target", "executorch_ggml_backend_py",
             "--parallel", str(os.cpu_count() or 4)],
            env=env,
        )


# Minimal setup - metadata comes from pyproject.toml
setup(
    ext_modules=[CMakeExtension("executorch_ggml._ggml_backend")],
    cmdclass={"build_ext": CMakeBuild},
)
