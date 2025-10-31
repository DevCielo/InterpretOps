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


if __name__ == '__main__':
    main()

