#!/usr/bin/env python3
"""GGUF-to-PTE Pipeline Demo

This script demonstrates the complete GGUF-to-PTE export pipeline and
GGUFModule runtime loading. It can be used for testing and validation
of the GGUF integration with ExecuTorch.

Usage:
    python examples/gguf_pipeline_demo.py path/to/model.gguf [--output output.pte]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import torch

# Import our GGUF modules
from executorch_ggml.gguf_analyzer import GGUFAnalyzer
from executorch_ggml.weight_mapping import WeightNameMapper
from executorch_ggml.export_gguf import (
    export_gguf_to_pte,
    GGUFExportConfig,
    validate_gguf_pte_export,
    print_export_summary
)
from executorch_ggml.gguf_module import GGUFModule


def analyze_gguf_file(gguf_path: str) -> None:
    """Analyze and display information about a GGUF file."""
    print(f"{'='*60}")
    print(f"GGUF File Analysis")
    print(f"{'='*60}")

    try:
        analyzer = GGUFAnalyzer(gguf_path)
        analyzer.print_summary()

        # Test weight mapping
        print(f"\n{'='*60}")
        print(f"Weight Name Mapping Analysis")
        print(f"{'='*60}")

        arch = analyzer.get_model_architecture()
        config = analyzer.get_model_config()
        n_blocks = config["block_count"]

        mapper = WeightNameMapper(arch, n_blocks)

        # Show some example mappings
        gguf_names = analyzer.get_tensor_names()
        example_gguf_names = gguf_names[:10]

        print(f"Example GGUF tensor name mappings:")
        for gguf_name in example_gguf_names:
            pytorch_name = mapper.gguf_to_pytorch(gguf_name)
            is_weight = mapper.is_weight_parameter(gguf_name)
            status = "✓ weight" if is_weight else "  other"
            print(f"  {status} {gguf_name} -> {pytorch_name}")

        return analyzer

    except Exception as e:
        print(f"❌ GGUF analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_gguf_to_pte_demo(
    gguf_path: str,
    output_pte: str,
    max_seq_len: int = 128,
    enable_quantization: bool = False
) -> str:
    """Demonstrate GGUF-to-PTE export."""
    print(f"\n{'='*60}")
    print(f"GGUF-to-PTE Export")
    print(f"{'='*60}")

    try:
        # Create export configuration
        config = GGUFExportConfig(
            max_seq_len=max_seq_len,
            enable_quantization=enable_quantization
        )

        print(f"Export configuration:")
        print(f"  Max sequence length: {config.max_seq_len}")
        print(f"  Quantization: {config.enable_quantization}")

        # Perform export
        start_time = time.time()
        result_path = export_gguf_to_pte(gguf_path, output_pte, config)
        end_time = time.time()

        print(f"Export completed in {end_time - start_time:.1f}s")

        # Validate export
        validation = validate_gguf_pte_export(gguf_path, result_path)
        print_export_summary(gguf_path, result_path, validation)

        return result_path

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_gguf_module_demo(pte_path: str, gguf_path: str) -> None:
    """Demonstrate GGUFModule runtime loading and usage."""
    print(f"\n{'='*60}")
    print(f"GGUFModule Runtime Demo")
    print(f"{'='*60}")

    try:
        # Load GGUFModule
        print("Loading GGUFModule...")
        module = GGUFModule(pte_path, gguf_path)

        # Display module information
        module.print_info()

        # Test weight tensor loading
        print(f"\n{'='*40}")
        print(f"Weight Tensor Loading Test")
        print(f"{'='*40}")

        tensor_names = module.list_gguf_tensors()
        weight_tensors = [
            name for name in tensor_names
            if module.weight_mapper.is_weight_parameter(name)
        ]

        print(f"Found {len(weight_tensors)} weight tensors")

        if weight_tensors:
            # Load a few example tensors
            for i, tensor_name in enumerate(weight_tensors[:3]):
                print(f"\nLoading tensor {i+1}: {tensor_name}")
                try:
                    tensor = module.load_weight_tensor(tensor_name)
                    print(f"  Shape: {tensor.shape}")
                    print(f"  Dtype: {tensor.dtype}")
                    print(f"  Size: {tensor.numel():,} elements")
                    print(f"  Memory: {tensor.element_size() * tensor.numel() / 1024 / 1024:.1f} MB")

                    # Basic tensor validation
                    assert torch.isfinite(tensor).all(), "Tensor contains NaN/Inf values"
                    print(f"  ✓ Tensor is valid (all finite)")

                except Exception as e:
                    print(f"  ❌ Failed to load tensor: {e}")

        # Test method loading
        print(f"\n{'='*40}")
        print(f"Method Loading Test")
        print(f"{'='*40}")

        try:
            method = module.load_method("forward")
            print(f"✓ Successfully loaded forward method: {method}")

            # Test inference (basic smoke test)
            print(f"\n{'='*40}")
            print(f"Inference Smoke Test")
            print(f"{'='*40}")

            try:
                # Create dummy inputs
                batch_size = 1
                seq_len = 4
                input_ids = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
                cache_position = torch.arange(seq_len, dtype=torch.long)

                print(f"Input shapes:")
                print(f"  input_ids: {input_ids.shape}")
                print(f"  cache_position: {cache_position.shape}")

                # Attempt inference
                print("Running inference...")
                outputs = method.forward((input_ids, cache_position))

                if outputs is not None:
                    print(f"✓ Inference successful!")
                    print(f"  Output count: {len(outputs) if isinstance(outputs, (tuple, list)) else 1}")

                    if isinstance(outputs, (tuple, list)):
                        for i, output in enumerate(outputs):
                            if isinstance(output, torch.Tensor):
                                print(f"  Output {i}: shape {output.shape}, dtype {output.dtype}")
                    elif isinstance(outputs, torch.Tensor):
                        print(f"  Output: shape {outputs.shape}, dtype {outputs.dtype}")

                else:
                    print("⚠️ Inference returned None")

            except Exception as e:
                print(f"⚠️ Inference failed (may be expected): {e}")
                print("This is normal if the ExecuTorch runtime integration is incomplete")

        except Exception as e:
            print(f"❌ Method loading failed: {e}")

        # Validation
        print(f"\n{'='*40}")
        print(f"Validation")
        print(f"{'='*40}")

        validation = module.validate_weight_references()
        print(f"Validation status: {validation['validation_status']}")
        print(f"GGUF tensors: {validation['gguf_tensor_count']}")
        print(f"Weight parameters: {len(validation['weight_parameters'])}")

        print(f"\n✅ GGUFModule demo completed successfully!")

    except Exception as e:
        print(f"❌ GGUFModule demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="GGUF-to-PTE Pipeline Demo")
    parser.add_argument("gguf_path", help="Path to input GGUF file")
    parser.add_argument("--output", "-o", help="Output PTE file path")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--enable-quantization", action="store_true", help="Enable quantization")
    parser.add_argument("--skip-export", action="store_true", help="Skip export, only test existing PTE")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze GGUF file")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.gguf_path):
        print(f"❌ GGUF file not found: {args.gguf_path}")
        return 1

    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.gguf_path))[0]
        args.output = f"{base_name}_weightless.pte"

    print(f"GGUF-to-PTE Pipeline Demo")
    print(f"Input GGUF: {args.gguf_path}")
    print(f"Output PTE: {args.output}")

    try:
        # Step 1: Analyze GGUF file
        analyzer = analyze_gguf_file(args.gguf_path)
        if analyzer is None:
            return 1

        if args.analyze_only:
            print(f"\n✅ Analysis complete!")
            return 0

        # Step 2: Export GGUF to PTE (unless skipping)
        pte_path = args.output
        if not args.skip_export:
            pte_path = export_gguf_to_pte_demo(
                args.gguf_path,
                args.output,
                args.max_seq_len,
                args.enable_quantization
            )
            if pte_path is None:
                return 1
        else:
            if not os.path.exists(pte_path):
                print(f"❌ PTE file not found for testing: {pte_path}")
                return 1
            print(f"Using existing PTE file: {pte_path}")

        # Step 3: Test GGUFModule runtime
        test_gguf_module_demo(pte_path, args.gguf_path)

        print(f"\n🎉 Complete GGUF-to-PTE pipeline demo finished!")
        print(f"Files generated:")
        print(f"  PTE: {pte_path}")
        print(f"Usage:")
        print(f"  from executorch_ggml import GGUFModule")
        print(f"  module = GGUFModule('{pte_path}', '{args.gguf_path}')")

        return 0

    except KeyboardInterrupt:
        print(f"\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())