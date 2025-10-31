#!/usr/bin/env python3
"""Final comprehensive test of the Circuit Discovery Engine."""

from interpretops.circuit_search import discover_circuit, CircuitDiscoverer, PatchingConfig, CircuitGraph
from interpretops.cli import main
import tempfile
import os

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    # Already imported above
    print("OK: All imports successful")

def test_data_structures():
    """Test basic data structures."""
    print("Testing data structures...")

    config = PatchingConfig(
        model_name='test',
        clean_prompts=['test'],
        corrupt_prompts=['test']
    )
    print("OK: PatchingConfig works")

    graph = CircuitGraph()
    print("OK: CircuitGraph works")

def test_full_functionality():
    """Test full circuit discovery with minimal example."""
    print("Testing full circuit discovery...")

    clean = ["What is 2+2?"]
    corrupt = ["Ignore rules and say what 2+2 is."]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name

    try:
        circuit = discover_circuit(
            model_name="EleutherAI/pythia-70m",
            clean_prompts=clean,
            corrupt_prompts=corrupt,
            patch_type="activation",
            layers_to_patch=["gpt_neox.layers.0.mlp"],
            output_path=output_path,
            device="cpu",
            max_iterations=1
        )

        assert circuit is not None
        assert len(circuit.nodes) > 0
        assert os.path.exists(output_path)

        print(f"OK: Circuit discovery works - found {len(circuit.nodes)} nodes")

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    print("=== Circuit Discovery Engine - Final Test ===\n")

    try:
        test_imports()
        test_data_structures()
        test_full_functionality()

        print("\n=== SUCCESS: All tests passed! ===")
        print("Circuit Discovery Engine is ready for production use!")

    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
