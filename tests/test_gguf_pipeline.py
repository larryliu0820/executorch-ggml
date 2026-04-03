"""End-to-end test suite for GGUF-to-PTE pipeline and GGUFModule.

Downloads Qwen3-0.6B-Q8_0.gguf from HuggingFace on first run (cached
by huggingface_hub for subsequent runs).
"""

import os
import tempfile

import pytest
import torch

# Test dependencies
pytest.importorskip("transformers")
pytest.importorskip("optimum")

from executorch_ggml.gguf_analyzer import GGUFAnalyzer
from executorch_ggml.weight_mapping import WeightNameMapper, MultiArchWeightMapper
from executorch_ggml.export_gguf import export_gguf_to_pte, GGUFExportConfig
from executorch_ggml.gguf_module import GGUFModule


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gguf_path():
    """Download Qwen3-0.6B-Q8_0.gguf (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
    )


@pytest.fixture
def output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="session")
def pte_path(gguf_path, tmp_path_factory):
    """Export a weight-less PTE once for the session."""
    out = str(tmp_path_factory.mktemp("pte") / "qwen3.pte")
    config = GGUFExportConfig(
        max_seq_len=128,
        preserve_dynamic_shapes=True,
        enable_quantization=True,
    )
    export_gguf_to_pte(gguf_path, out, config)
    return out


# ---------------------------------------------------------------------------
# GGUFAnalyzer
# ---------------------------------------------------------------------------

class TestGGUFAnalyzer:

    def test_initialization(self, gguf_path):
        analyzer = GGUFAnalyzer(gguf_path)
        assert analyzer.gguf_path == gguf_path
        assert analyzer.reader is not None

    def test_architecture_detection(self, gguf_path):
        arch = GGUFAnalyzer(gguf_path).get_model_architecture()
        assert arch == "qwen3"

    def test_model_config(self, gguf_path):
        config = GGUFAnalyzer(gguf_path).get_model_config()
        assert config["embedding_length"] == 1024
        assert config["block_count"] == 28
        assert config["vocab_size"] == 151936

    def test_tensor_names(self, gguf_path):
        names = GGUFAnalyzer(gguf_path).get_tensor_names()
        assert len(names) == 310  # Qwen3-0.6B has 310 tensors
        assert "token_embd.weight" in names

    def test_file_info(self, gguf_path):
        info = GGUFAnalyzer(gguf_path).get_file_info()
        assert info["architecture"] == "qwen3"
        assert info["tensor_count"] == 310
        assert info["file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# WeightNameMapper
# ---------------------------------------------------------------------------

class TestWeightNameMapper:

    def test_initialization(self):
        mapper = WeightNameMapper("qwen3", 28)
        assert mapper.arch == "qwen3"
        assert mapper.n_blocks == 28

    def test_pytorch_to_gguf(self):
        mapper = WeightNameMapper("qwen3", 28)
        # Spot-check a few known mappings
        assert isinstance(mapper.pytorch_to_gguf("tok_embeddings.weight"), str)

    def test_gguf_to_pytorch(self):
        mapper = WeightNameMapper("qwen3", 28)
        assert isinstance(mapper.gguf_to_pytorch("token_embd.weight"), str)

    def test_weight_parameter_detection(self):
        mapper = WeightNameMapper("qwen3", 28)
        assert mapper.is_weight_parameter("layers.0.attention.wq.weight")
        assert not mapper.is_weight_parameter("cache_position")

    def test_multi_arch_mapper(self):
        mm = MultiArchWeightMapper()
        assert mm.get_mapper("qwen3", 28).arch == "qwen3"
        assert mm.get_mapper("llama", 32).arch == "llama"

    def test_validation_with_real_gguf(self, gguf_path):
        analyzer = GGUFAnalyzer(gguf_path)
        gguf_names = analyzer.get_tensor_names()
        config = analyzer.get_model_config()
        mapper = WeightNameMapper("qwen3", config["block_count"])

        validation = mapper.validate_mapping(
            ["tok_embeddings.weight", "norm.weight"],
            gguf_names,
        )
        assert isinstance(validation, dict)
        assert "successful_mappings" in validation


# ---------------------------------------------------------------------------
# Export Pipeline
# ---------------------------------------------------------------------------

class TestGGUFExportPipeline:

    def test_export_config_defaults(self):
        config = GGUFExportConfig()
        assert config.max_seq_len == 128
        assert config.enable_quantization is False
        assert config.skip_weight_data is True

    def test_export_produces_small_pte(self, gguf_path, output_dir):
        """Weight-less PTE should be much smaller than the GGUF."""
        out = os.path.join(output_dir, "test.pte")
        config = GGUFExportConfig(
            max_seq_len=128,
            preserve_dynamic_shapes=True,
            enable_quantization=True,
        )
        export_gguf_to_pte(gguf_path, out, config)

        pte_size = os.path.getsize(out)
        gguf_size = os.path.getsize(gguf_path)
        assert 0 < pte_size < gguf_size
        # Weight-less PTE for Qwen3-0.6B should be under 1 MB
        assert pte_size < 1_000_000


# ---------------------------------------------------------------------------
# GGUFModule
# ---------------------------------------------------------------------------

class TestGGUFModule:

    def test_initialization(self, pte_path, gguf_path):
        module = GGUFModule(pte_path, gguf_path)
        assert module.pte_path == pte_path
        assert module.gguf_path == gguf_path

    def test_model_info(self, pte_path, gguf_path):
        module = GGUFModule(pte_path, gguf_path)
        info = module.get_model_info()
        assert info["architecture"] == "qwen3"
        assert info["pte_size_mb"] < 1.0
        assert info["gguf_size_mb"] > 100

    def test_tensor_listing(self, pte_path, gguf_path):
        module = GGUFModule(pte_path, gguf_path)
        names = module.list_gguf_tensors()
        assert len(names) == 310

    def test_weight_loading(self, pte_path, gguf_path):
        module = GGUFModule(pte_path, gguf_path)
        t = module.load_weight_tensor("token_embd.weight")
        assert isinstance(t, torch.Tensor)
        assert t.numel() > 0

    def test_forward_callable(self, pte_path, gguf_path):
        module = GGUFModule(pte_path, gguf_path)
        assert callable(module.forward)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_imports():
    """All GGUF modules are importable."""
    from executorch_ggml.gguf_analyzer import GGUFAnalyzer  # noqa: F401
    from executorch_ggml.weight_mapping import WeightNameMapper  # noqa: F401
    from executorch_ggml.export_gguf import export_gguf_to_pte  # noqa: F401
    from executorch_ggml.gguf_module import GGUFModule  # noqa: F401
