# InterpretOps

Interpretation operations toolkit for mechanistic interpretability.

## Installation

```bash
pip install -e .
```

## Usage

### Activation Capture

Capture activations from a model:

```bash
mechctl capture --model pythia-410m --dataset red_team_100.json --output activations.h5
```

### Options

- `--model`: Model name (HuggingFace model identifier)
- `--dataset`: Path to JSON dataset file
- `--output`: Output HDF5 file path
- `--layers`: Layer selection patterns (comma-separated, default: all attention & MLP layers)
- `--batch-size`: Batch size for processing (default: 32)

