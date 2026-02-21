"""
Setup configuration for executorch-ggml.

Metadata and dependencies are defined in pyproject.toml (PEP 621).
This file only handles the custom CMake build logic for the native extension.

To build the native extension during pip install, set:
  - EXECUTORCH_GGML_BUILD_NATIVE=1
  - LLAMA_CPP_DIR=/path/to/llama.cpp
  - EXECUTORCH_DIR=/path/to/executorch (repo root containing executorch/)

Example:
  LLAMA_CPP_DIR=... EXECUTORCH_DIR=... EXECUTORCH_GGML_BUILD_NATIVE=1 pip install .
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self) -> None:
        # IMPORTANT: pip build isolation environments typically won't have
        # pybind11/executorch available. We only build the native extension
        # when explicitly requested.
        if os.environ.get("EXECUTORCH_GGML_BUILD_NATIVE", "0") not in {"1", "true", "True", "yes"}:
            print(
                "[executorch-ggml] Skipping native extension build. "
                "Set EXECUTORCH_GGML_BUILD_NATIVE=1 to build _ggml_backend."
            )
            return

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

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Auto-discover python interpreter: use the one running setup.py
        python_exe = sys.executable

        cmake_args: List[str] = [
            "-G",
            "Ninja",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DLLAMA_CPP_DIR={llama_dir}",
            f"-DEXECUTORCH_DIR={et_dir}",
            "-DEXECUTORCH_GGML_BUILD_PYTHON_EXTENSION=ON",
        ]

        env = os.environ.copy()

        # Ensure we use a real flatc binary (not the ExecuTorch python wrapper).
        def pick_flatc() -> str | None:
            candidates = []
            w = shutil.which("flatc")
            if w:
                candidates.append(w)
            # common locations (brew/system)
            candidates += ["/opt/homebrew/bin/flatc", "/usr/local/bin/flatc", "/usr/bin/flatc"]

            for c in candidates:
                if not c:
                    continue
                p = pathlib.Path(c)
                if not p.exists():
                    continue
                # avoid venv/python wrapper scripts
                if str(p).startswith(str(pathlib.Path(sys.executable).parent)):
                    continue
                return str(p)
            return None

        flatc = pick_flatc()
        if flatc:
            env["FLATC"] = flatc
        else:
            print("[executorch-ggml] Warning: could not locate system flatc; CMake may fail")

        subprocess.check_call(["cmake", "-S", str(root), "-B", str(build_temp), *cmake_args], env=env)
        subprocess.check_call(
            ["cmake", "--build", str(build_temp), "--target", "executorch_ggml_backend_py"],
            env=env,
        )


# Minimal setup - metadata comes from pyproject.toml
setup(
    ext_modules=[CMakeExtension("executorch_ggml._ggml_backend")],
    cmdclass={"build_ext": CMakeBuild},
)
