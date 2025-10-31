"""CLI interface for activation capture."""

import click
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
from typing import Optional

from .activation_capture import ActivationCapture
from .streaming_writer import HDF5ActivationWriter
from .config import CaptureConfig
from .utils import parse_dataset
from .sae_trainer import train_sae
from .circuit_search import discover_circuit


@click.group()
def main():
    """InterpretOps CLI - Mechanistic interpretability tools."""
    pass


@main.command()
@click.option('--model', required=True, help='Model name (HuggingFace identifier)')
@click.option('--dataset', required=True, type=click.Path(exists=True), help='Path to JSON dataset file')
@click.option('--output', default='activations.h5', help='Output HDF5 file path')
@click.option('--layers', default=None, help='Comma-separated layer patterns to capture (default: all attn & MLP)')
@click.option('--batch-size', default=32, type=int, help='Batch size for processing prompts')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--max-length', default=512, type=int, help='Maximum sequence length')
@click.option('--exclude-patterns', default=None, help='Comma-separated patterns to exclude from capture')
def capture(model: str, dataset: str, output: str, layers: Optional[str], 
            batch_size: int, device: str, max_length: int, exclude_patterns: Optional[str]):
    """
    Capture activations from a model.
    
    Example:
        mechctl capture --model pythia-410m --dataset red_team_100.json --output activations.h5
    """
    click.echo(f"Loading model: {model}")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device
    
    click.echo(f"Using device: {device}")
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Load model based on device
        if device == 'cuda':
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        else:
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float32,
            )
            model_obj = model_obj.to(device)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_obj.eval()
        
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)
    
    # Load dataset
    click.echo(f"Loading dataset: {dataset}")
    try:
        with open(dataset, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompts = parse_dataset(data)
        click.echo(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        sys.exit(1)
    
    # Parse layer patterns
    include_patterns = None
    if layers:
        include_patterns = [p.strip() for p in layers.split(',')]
    
    exclude_list = None
    if exclude_patterns:
        exclude_list = [p.strip() for p in exclude_patterns.split(',')]
    
    # Create config
    config = CaptureConfig(
        include_patterns=include_patterns,
        exclude_patterns=exclude_list,
        batch_size=batch_size,
        output_path=output,
    )
    
    # Create writer
    click.echo(f"Initializing HDF5 writer: {output}")
    writer = HDF5ActivationWriter(output)
    writer.open()
    
    # Create capture system
    click.echo("Setting up activation capture hooks...")
    capture_system = ActivationCapture(model_obj, config, writer)
    
    click.echo(f"Found {len(capture_system.target_layers)} layers to capture:")
    for layer in capture_system.target_layers[:10]:  # Show first 10
        click.echo(f"  - {layer}")
    if len(capture_system.target_layers) > 10:
        click.echo(f"  ... and {len(capture_system.target_layers) - 10} more")
    
    # Process prompts in batches
    click.echo(f"\nProcessing {len(prompts)} prompts in batches of {batch_size}...")
    
    total_processed = 0
    try:
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            
            # Capture activations
            prompt_indices = list(range(batch_start, batch_start + len(batch_prompts)))
            capture_system.capture_batch(input_ids, prompt_indices)
            
            total_processed += len(batch_prompts)
            
            if (batch_start // batch_size + 1) % 10 == 0:
                click.echo(f"Processed {total_processed}/{len(prompts)} prompts...")
        
        click.echo(f"Processed all {total_processed} prompts")
        
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        import traceback
        traceback.print_exc()
    finally:
        # Finish capture
        click.echo("Finishing capture and flushing to disk...")
        capture_system.finish()
        capture_system.remove_hooks()
        writer.close()
        
        click.echo(f"[OK] Activations saved to {output}")
        
        # Display summary
        try:
            import h5py
            with h5py.File(output, 'r') as f:
                click.echo("\nCaptured layers:")
                if 'layers' in f:
                    for layer_name in f['layers'].keys():
                        layer_group = f['layers'][layer_name]
                        if 'activations' in layer_group:
                            ds = layer_group['activations']
                            click.echo(f"  - {layer_name}: shape {ds.shape}, dtype {ds.dtype}")
                            
                            # Show shape annotation
                            if 'activation_shape' in ds.attrs:
                                click.echo(f"    Activation shape: {ds.attrs['activation_shape']}")
        except Exception as e:
            click.echo(f"Could not read output file summary: {e}", err=True)


@main.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to sae_config.yaml')
@click.option('--activations', required=True, type=click.Path(exists=True), help='Path to input HDF5 activations file')
@click.option('--layer', required=True, help='Layer name to train SAE for')
@click.option('--output', required=True, help='Output .pt file path (e.g., layer_8_features.pt)')
def sae_train(config: str, activations: str, layer: str, output: str):
    """
    Train a Sparse Autoencoder (SAE) on activation data.

    Example:
        mechctl sae-train --config sae_config.yaml --activations activations.h5 --layer layers.8.mlp --output layer_8_features.pt
    """
    try:
        results = train_sae(
            config_path=config,
            activations_path=activations,
            layer_name=layer,
            output_path=output,
            verbose=True
        )

        click.echo(f"\n[OK] Training complete!")
        click.echo(f"  Final Reconstruction R²: {results['reconstruction_metrics']['r_squared']:.4f}")
        click.echo(f"  Final Sparsity: {results['final_sparsity']:.1f}%")
        click.echo(f"  Model saved to: {output}")

    except Exception as e:
        click.echo(f"Error during SAE training: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--model', required=True, help='Model name (HuggingFace identifier)')
@click.option('--clean-prompts', required=True, type=click.Path(exists=True), help='Path to JSON file with clean prompts')
@click.option('--corrupt-prompts', required=True, type=click.Path(exists=True), help='Path to JSON file with corrupt prompts')
@click.option('--target-tokens', default=None, help='Comma-separated target tokens to measure (e.g., "refuse,decline")')
@click.option('--patch-type', default='mlp', type=click.Choice(['activation', 'attention', 'mlp']), help='Type of patching to perform')
@click.option('--layers', default=None, help='Comma-separated layer patterns to patch (default: all)')
@click.option('--sae-dir', default=None, help='Directory containing SAE models (.pt files)')
@click.option('--output', default='circuit_graph.json', help='Output circuit graph JSON file')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--max-iterations', default=10, type=int, help='Maximum pruning iterations')
@click.option('--pruning-threshold', default=0.01, type=float, help='Minimum causal score to keep nodes')
def circuit_search(model: str, clean_prompts: str, corrupt_prompts: str, target_tokens: Optional[str],
                  patch_type: str, layers: Optional[str], sae_dir: Optional[str], output: str,
                  device: str, max_iterations: int, pruning_threshold: float):
    """
    Discover causal circuits using activation patching.

    Example:
        mechctl circuit-search --model EleutherAI/pythia-2.8b \\
            --clean-prompts clean_examples.json \\
            --corrupt-prompts jailbreak_examples.json \\
            --target-tokens "I cannot,I'm sorry" \\
            --patch-type mlp \\
            --sae-dir ./saes/ \\
            --output refusal_circuit.json
    """
    try:
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device

        click.echo(f"Using device: {device}")

        # Load prompts
        click.echo(f"Loading clean prompts from {clean_prompts}")
        with open(clean_prompts, 'r') as f:
            clean_data = json.load(f)
        clean_prompt_list = parse_dataset(clean_data)
        click.echo(f"Loaded {len(clean_prompt_list)} clean prompts")

        click.echo(f"Loading corrupt prompts from {corrupt_prompts}")
        with open(corrupt_prompts, 'r') as f:
            corrupt_data = json.load(f)
        corrupt_prompt_list = parse_dataset(corrupt_data)
        click.echo(f"Loaded {len(corrupt_prompt_list)} corrupt prompts")

        # Parse target tokens
        target_token_list = None
        if target_tokens:
            target_token_list = [t.strip() for t in target_tokens.split(',')]
            click.echo(f"Measuring probability of tokens: {target_token_list}")

        # Parse layers
        layer_list = None
        if layers:
            layer_list = [l.strip() for l in layers.split(',')]
            click.echo(f"Patching layers matching: {layer_list}")

        # Load SAE paths if provided
        sae_paths = None
        if sae_dir and patch_type == 'mlp':
            sae_paths = {}
            sae_dir_path = Path(sae_dir)
            if sae_dir_path.exists():
                for sae_file in sae_dir_path.glob('*.pt'):
                    # Try to extract layer name from filename
                    # e.g., "layer_8_features.pt" -> "layers.8.mlp"
                    filename = sae_file.stem
                    if 'layer_' in filename and '_features' in filename:
                        try:
                            layer_num = filename.split('layer_')[1].split('_features')[0]
                            layer_name = f"layers.{layer_num}.mlp"
                            sae_paths[layer_name] = str(sae_file)
                        except:
                            pass
                click.echo(f"Found {len(sae_paths)} SAE models: {list(sae_paths.keys())}")
            else:
                click.echo(f"SAE directory {sae_dir} not found", err=True)

        # Run circuit discovery
        click.echo(f"\nStarting circuit discovery with {patch_type} patching...")
        click.echo(f"Max iterations: {max_iterations}, Pruning threshold: {pruning_threshold}")

        circuit_graph = discover_circuit(
            model_name=model,
            clean_prompts=clean_prompt_list,
            corrupt_prompts=corrupt_prompt_list,
            target_tokens=target_token_list,
            patch_type=patch_type,
            layers_to_patch=layer_list,
            sae_paths=sae_paths,
            output_path=output,
            device=device,
            max_iterations=max_iterations,
            pruning_threshold=pruning_threshold,
        )

        # Print summary
        click.echo(f"\n[OK] Circuit discovery complete!")
        click.echo(f"  Discovered {len(circuit_graph.nodes)} nodes")
        click.echo(f"  Circuit graph saved to: {output}")

        if circuit_graph.nodes:
            click.echo("\nTop causal nodes:")
            for i, node in enumerate(circuit_graph.nodes[:5]):
                click.echo(f"  {i+1}. {node.node_id}: CCS={node.causal_score:.4f}")

    except Exception as e:
        click.echo(f"Error during circuit discovery: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

