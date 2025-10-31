# Quick Start Guide

## Installation

After running `pip install -e .`, you should have the `mechctl` command available.

## Verify Installation

Check that the CLI is installed:

```bash
mechctl --help
```

You should see the help output with the `capture` command.

## Test with a Small Model

First, let's test with a small model to make sure everything works:

**Note:** If you encounter CUDA compatibility errors (e.g., "no kernel image is available"), use CPU mode with `--device cpu`:

```bash
mechctl capture --model gpt2 --dataset example_dataset.json --output test_activations.h5 --batch-size 2 --device cpu
```

Or for CUDA (if compatible):

```bash
mechctl capture --model gpt2 --dataset example_dataset.json --output test_activations.h5 --batch-size 2
```

This will:
1. Load GPT-2 (small model for quick testing)
2. Process the 10 example prompts
3. Save activations to `test_activations.h5`

## Full Test with Pythia-410m

For the full test as specified in the requirements, you'll need:

1. **Create a dataset with 1000 prompts** (or use an existing one)

2. **Run the capture command:**
```bash
# For GPU (if compatible):
mechctl capture --model EleutherAI/pythia-410m --dataset red_team_100.json --output activations.h5 --batch-size 32

# For CPU (if GPU issues):
mechctl capture --model EleutherAI/pythia-410m --dataset red_team_100.json --output activations.h5 --batch-size 8 --device cpu
```

3. **Verify the output:**
   - Check that `activations.h5` was created
   - The CLI will show a summary of captured layers and shapes
   - You can inspect the file with Python:

```python
import h5py

with h5py.File('activations.h5', 'r') as f:
    print("Layers captured:")
    for layer_name in f['layers'].keys():
        ds = f['layers'][layer_name]['activations']
        print(f"  {layer_name}: shape {ds.shape}, dtype {ds.dtype}")
        if 'activation_shape' in ds.attrs:
            print(f"    Activation shape: {ds.attrs['activation_shape']}")
    
    print("\nMetadata:")
    for key in f['metadata'].attrs.keys():
        print(f"  {key}: {f['metadata'].attrs[key]}")
```

## Monitor Memory Usage

On Linux/Mac, you can monitor memory during capture:

```bash
# In another terminal
watch -n 1 free -h
# or
top -p $(pgrep -f mechctl)
```

On Windows, use Task Manager or PowerShell:
```powershell
Get-Process python | Select-Object ProcessName, WorkingSet
```

## Troubleshooting

- **CUDA compatibility errors**: If you see "no kernel image is available for execution on the device" or "CUDA capability sm_XXX is not compatible", your PyTorch installation doesn't support your GPU. 
  - **Quick fix**: Use CPU mode: `--device cpu` (slower but will work)
  - **Permanent fix**: Install a PyTorch build that supports your GPU. Check [PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command for your CUDA version
  - **Example for newer GPUs**: You may need PyTorch nightly build: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124`

- **Model download issues**: The first run will download the model from HuggingFace (models are cached in `~/.cache/huggingface/`)

- **Out of memory**: 
  - Reduce `--batch-size` (try `--batch-size 1` or `--batch-size 4`)
  - Use CPU with `--device cpu`
  - Reduce `--max-length` (default is 512)

- **Missing dataset**: Make sure your JSON file exists and is valid JSON. You can test with the included `example_dataset.json`

- **HDF5 errors**: Ensure you have write permissions in the output directory

- **Slow performance on CPU**: This is expected. CPU processing is much slower than GPU. Consider reducing batch size or max-length for faster testing

