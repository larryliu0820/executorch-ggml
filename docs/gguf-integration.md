# GGUF Integration for ExecuTorch GGML Backend

This document describes the GGUF-to-PTE export pipeline and GGUFModule runtime class that enable memory-efficient inference by separating model graph structure from weight data.

## Overview

The GGUF integration provides:

1. **GGUF Analysis**: Extract model metadata and architecture from GGUF files
2. **Weight-less PTE Export**: Generate PTE files without embedded weights but with external weight references
3. **GGUFModule Runtime**: Load PTE graph structure + GGUF weights for inference
4. **Weight Name Mapping**: Bidirectional conversion between PyTorch FQNs and GGUF tensor names

## Quick Start

### Export GGUF to PTE

```python
from executorch_ggml import export_gguf_to_pte, GGUFExportConfig

# Export Qwen3 GGUF to weight-less PTE
config = GGUFExportConfig(max_seq_len=128, enable_quantization=False)
pte_path = export_gguf_to_pte(
    gguf_path="qwen3-0.6b-q8_0.gguf",
    output_pte_path="qwen3_weightless.pte",
    export_config=config
)
```

### Runtime Inference with GGUFModule

```python
from executorch_ggml import GGUFModule

# Load PTE graph + GGUF weights
module = GGUFModule("qwen3_weightless.pte", "qwen3-0.6b-q8_0.gguf")

# Run inference
method = module.load_method("forward")
outputs = method.forward((input_ids, cache_position))
```

## Architecture

### Export Pipeline

1. **GGUF Analysis** (`GGUFAnalyzer`)
   - Extract model architecture (qwen3, llama, etc.)
   - Parse configuration (layers, heads, vocab size, etc.)
   - Enumerate tensor names

2. **Weight Name Mapping** (`WeightNameMapper`)
   - Convert PyTorch FQNs ↔ GGUF tensor names
   - Architecture-specific mapping rules
   - Weight parameter detection

3. **PTE Export with External References**
   - Uses existing `ExecutorchBackendConfig(external_constants=...)`
   - GGUF tensor names become external weight reference tags
   - No core ExecuTorch modifications required

### Runtime Pipeline

1. **PTE Loading**
   - Load graph structure using existing `_load_for_executorch()`
   - External weight references preserved

2. **GGUF Weight Loading** (`GGUFModule`)
   - Lazy weight loading from GGUF file
   - Memory-efficient tensor caching
   - Dynamic weight resolution (future)

## Supported Architectures

Currently supported:
- **Qwen3**: Qwen3-0.6B and compatible models
- **Llama**: Llama-2/3 and compatible models (basic support)

Extensible architecture system allows adding new model types by updating weight mapping rules.

## API Reference

### GGUFAnalyzer

```python
from executorch_ggml import GGUFAnalyzer

analyzer = GGUFAnalyzer("model.gguf")

# Model information
arch = analyzer.get_model_architecture()  # "qwen3"
config = analyzer.get_model_config()      # {embedding_length: 896, ...}
tensors = analyzer.get_tensor_names()     # ["token_embd.weight", ...]

# File information
info = analyzer.get_file_info()
analyzer.print_summary()
```

### WeightNameMapper

```python
from executorch_ggml import WeightNameMapper

mapper = WeightNameMapper("qwen3", n_blocks=32)

# Bidirectional mapping
gguf_name = mapper.pytorch_to_gguf("layers.0.attention.wq.weight")
# -> "blk.0.attn_q.weight"

pytorch_name = mapper.gguf_to_pytorch("blk.0.attn_q.weight")
# -> "layers.0.attention.wq.weight"

# Weight detection
is_weight = mapper.is_weight_parameter("layers.0.attention.wq.weight")  # True
```

### GGUFExportConfig

```python
from executorch_ggml import GGUFExportConfig

config = GGUFExportConfig(
    max_seq_len=128,              # Model sequence length
    enable_quantization=False,    # Enable GGML quantization
    preserve_dynamic_shapes=False,
    use_custom_kv_cache=False,
    use_custom_sdpa=False,
)
```

### export_gguf_to_pte

```python
from executorch_ggml import export_gguf_to_pte

pte_path = export_gguf_to_pte(
    gguf_path="model.gguf",           # Input GGUF file
    output_pte_path="model.pte",      # Output PTE file
    export_config=config,             # Export configuration
)
```

### GGUFModule

```python
from executorch_ggml import GGUFModule

module = GGUFModule("model.pte", "model.gguf")

# Model information
module.print_info()
info = module.get_model_info()

# Weight access
tensor_names = module.list_gguf_tensors()
weight_tensor = module.load_weight_tensor("token_embd.weight")

# Method execution
method = module.load_method("forward")
outputs = method.forward((input_ids, cache_position))

# Cache management
module.preload_weights()      # Load all weights
module.clear_weight_cache()   # Free memory
```

## Example Usage

### Complete Pipeline Demo

```python
#!/usr/bin/env python3
"""Complete GGUF-to-PTE pipeline example"""

import torch
from executorch_ggml import (
    GGUFAnalyzer, export_gguf_to_pte, GGUFExportConfig, GGUFModule
)

# Step 1: Analyze GGUF file
print("Analyzing GGUF file...")
analyzer = GGUFAnalyzer("qwen3-0.6b-q8_0.gguf")
analyzer.print_summary()

# Step 2: Export to weight-less PTE
print("Exporting to PTE...")
config = GGUFExportConfig(max_seq_len=128)
pte_path = export_gguf_to_pte(
    "qwen3-0.6b-q8_0.gguf",
    "qwen3_weightless.pte",
    config
)

# Step 3: Runtime inference
print("Loading for inference...")
module = GGUFModule(pte_path, "qwen3-0.6b-q8_0.gguf")

# Create sample inputs
batch_size, seq_len = 1, 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
cache_position = torch.arange(seq_len, dtype=torch.long)

# Run inference
method = module.load_method("forward")
outputs = method.forward((input_ids, cache_position))
print(f"Inference successful: {len(outputs)} output tensors")
```

### Command Line Demo

```bash
# Run complete pipeline demo
python examples/gguf_pipeline_demo.py qwen3-0.6b-q8_0.gguf

# Analysis only
python examples/gguf_pipeline_demo.py qwen3-0.6b-q8_0.gguf --analyze-only

# Custom configuration
python examples/gguf_pipeline_demo.py qwen3-0.6b-q8_0.gguf \
    --max-seq-len 256 \
    --output custom_model.pte
```

## Testing

### Run Test Suite

```bash
# All tests
pytest tests/test_gguf_pipeline.py -v

# Skip slow tests
pytest tests/test_gguf_pipeline.py -v -m "not slow"

# Specific test class
pytest tests/test_gguf_pipeline.py::TestGGUFAnalyzer -v
```

### Test Requirements

- GGUF test files (place in test directory or specify path)
- ExecuTorch runtime dependencies
- Model dependencies (transformers, optimum)

## Implementation Details

### Zero-Modification Approach

The implementation leverages existing ExecuTorch infrastructure without modifying core files:

- **External Constants**: Uses existing `external_constants_pass.py` with custom tag generator
- **Serialization**: Uses existing PTE serialization with `TensorDataLocation.EXTERNAL`
- **Runtime Loading**: Uses existing `_load_for_executorch()` API

### Weight Reference Flow

1. **Export**: PyTorch FQNs → GGUF tensor names → External reference tags
2. **Serialization**: Tags stored as `fully_qualified_name` in PTE metadata
3. **Runtime**: External references → GGUF tensor lookup → PyTorch tensors

### Memory Efficiency

- **Lazy Loading**: Weights loaded on-demand from GGUF file
- **Caching**: LRU-style tensor cache for frequently accessed weights
- **Shared Storage**: Multiple methods can share same GGUF weight data

## Limitations and Future Work

### Current Limitations

- **Architecture Support**: Limited to Qwen3 and basic Llama support
- **Dynamic Weight Resolution**: Not yet implemented in C++ backend
- **Full Integration**: Runtime weight resolution needs C++ backend integration
- **Performance**: Python-level weight loading may be slower than native

### Future Enhancements

1. **Extended Architecture Support**: Mistral, Falcon, GPT variants
2. **C++ Weight Resolution**: Native backend integration for better performance
3. **Dynamic Shape Support**: Better dynamic sequence length handling
4. **Streaming Inference**: Efficient KV-cache management for long sequences
5. **Quantization Integration**: Seamless GGUF quantization with ExecuTorch quantization

## Troubleshooting

### Common Issues

**Import Errors**
```
ImportError: GGUF functionality not available
```
- Install llama.cpp dependencies: `pip install -r requirements-gguf.txt`
- Ensure gguf-py is available in Python path

**Export Failures**
```
ValueError: Unsupported architecture: xyz
```
- Currently only qwen3 and llama are supported
- Add architecture support in `WeightNameMapper`

**Runtime Errors**
```
ImportError: ExecuTorch runtime not available
```
- Build ExecuTorch with GGML backend: `cmake -DGGML_CUDA=ON ...`
- Install runtime dependencies: `pip install executorch[runtime]`

**Memory Issues**
```
OutOfMemoryError during export
```
- Reduce `max_seq_len` in export config
- Use quantization: `enable_quantization=True`
- Ensure sufficient RAM for model + export overhead

### Debug Tips

1. **Enable Verbose Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Validate Mappings**
   ```python
   mapper = WeightNameMapper("qwen3", 32)
   validation = mapper.validate_mapping(pytorch_names, gguf_names)
   mapper.print_mapping_summary(validation)
   ```

3. **Check File Integrity**
   ```python
   analyzer = GGUFAnalyzer("model.gguf")
   info = analyzer.get_file_info()
   print(f"GGUF version: {info['gguf_version']}")
   ```

## Contributing

To add support for new architectures:

1. **Add Mapping Rules** in `weight_mapping.py`:
   ```python
   _ARCH_MAPPINGS = {
       "new_arch": [
           ("gguf_pattern", "pytorch_pattern"),
           # ... more mappings
       ]
   }
   ```

2. **Add Config Parser** in `gguf_analyzer.py`:
   ```python
   def _get_new_arch_config(self) -> Dict[str, Any]:
       # Extract architecture-specific config
   ```

3. **Add Model Creator** in `export_gguf.py`:
   ```python
   def _create_new_arch_model_from_config(config_dict, max_seq_len):
       # Create PyTorch model from config
   ```

4. **Add Tests** in `test_gguf_pipeline.py`

5. **Update Documentation**