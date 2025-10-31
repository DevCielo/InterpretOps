"""Streaming HDF5 writer for activations."""

import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
from collections import defaultdict


class HDF5ActivationWriter:
    """
    Streaming writer for activations to HDF5 format.
    
    Handles batching, shape annotations, and memory-efficient writing.
    """
    
    def __init__(self, output_path: str, chunk_size: Optional[Tuple[int, ...]] = None):
        """
        Initialize HDF5 writer.
        
        Args:
            output_path: Path to output HDF5 file
            chunk_size: Optional chunk size for HDF5 datasets (auto-determined if None)
        """
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.file: Optional[h5py.File] = None
        self.datasets: Dict[str, h5py.Dataset] = {}
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.dtypes: Dict[str, np.dtype] = {}
        self.counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open HDF5 file for writing."""
        if self.file is None:
            self.file = h5py.File(self.output_path, 'w')
            self._initialized = True
    
    def close(self):
        """Close HDF5 file and flush all buffers."""
        if self.file is not None:
            # Flush all buffers before closing
            with self.lock:
                for layer_name in list(self.buffers.keys()):
                    if self.buffers[layer_name]:
                        self._flush_layer(layer_name)
            
            # Close the file
            self.file.close()
            self.file = None
    
    def add_activation(self, layer_name: str, activation: np.ndarray, prompt_idx: int):
        """
        Add an activation to the buffer for a layer.
        
        Args:
            layer_name: Name of the layer (will be sanitized for HDF5)
            activation: Activation tensor as numpy array
            prompt_idx: Index of the prompt this activation corresponds to
        """
        if not self._initialized:
            self.open()
        
        # Sanitize layer name
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        
        with self.lock:
            # Convert to numpy if needed
            if not isinstance(activation, np.ndarray):
                activation = activation.detach().cpu().numpy() if hasattr(activation, 'detach') else np.array(activation)
            
            # Store shape and dtype on first activation
            if safe_name not in self.shapes:
                self.shapes[safe_name] = activation.shape
                self.dtypes[safe_name] = activation.dtype
            
            # Add to buffer
            self.buffers[safe_name].append(activation)
            self.counts[safe_name] += 1
    
    def flush(self, force_all: bool = False):
        """
        Flush all buffered activations to disk.
        
        Args:
            force_all: If True, flush all layers regardless of buffer size
        """
        with self.lock:
            for layer_name in list(self.buffers.keys()):
                if force_all or len(self.buffers[layer_name]) > 0:
                    self._flush_layer(layer_name)
    
    def _flush_layer(self, layer_name: str):
        """Flush activations for a specific layer."""
        if not self.buffers[layer_name]:
            return
        
        # Handle variable-length sequences by finding max shape
        activations_list = self.buffers[layer_name]
        
        # Check if all have same shape
        shapes = [arr.shape for arr in activations_list]
        if len(set(shapes)) == 1:
            # All same shape - can stack directly
            activations = np.stack(activations_list, axis=0)
        else:
            # Different shapes - need to pad to max shape
            # Find max shape for each dimension
            max_shape = tuple(max(dim_sizes) for dim_sizes in zip(*shapes))
            
            # Pad all activations to max shape
            padded_activations = []
            for arr in activations_list:
                if arr.shape != max_shape:
                    # Create padded array with zeros
                    padded = np.zeros(max_shape, dtype=arr.dtype)
                    # Use slices to fill in the actual data
                    slices = tuple(slice(0, s) for s in arr.shape)
                    padded[slices] = arr
                    padded_activations.append(padded)
                else:
                    padded_activations.append(arr)
            
            activations = np.stack(padded_activations, axis=0)
        
        # Get or create dataset
        # activations has shape (N, ...) where N is number of prompts
        # For dataset creation, we use a single sample (remove first dimension)
        if layer_name not in self.datasets:
            sample_activation = activations[0] if activations.shape[0] > 0 else activations.squeeze(0)
            self._create_dataset(layer_name, sample_activation)
        
        # Append to dataset
        dataset = self.datasets[layer_name]
        current_size = dataset.shape[0]
        new_size = current_size + activations.shape[0]
        
        # Check if shape matches (except first dimension)
        # Also check rank (number of dimensions)
        if len(dataset.shape) != len(activations.shape):
            # Different ranks - can't easily combine, skip this batch or handle separately
            # For now, we'll try to reshape to match
            raise ValueError(
                f"Rank mismatch for layer {layer_name}: dataset has {len(dataset.shape)} dims, "
                f"activations have {len(activations.shape)} dims. Dataset shape: {dataset.shape}, "
                f"Activations shape: {activations.shape}"
            )
        
        if len(dataset.shape) > 1 and dataset.shape[1:] != activations.shape[1:]:
            # Dataset already exists with different shape - need to resize
            # Use the maximum shape to accommodate both (but same rank)
            if len(dataset.shape) != len(activations.shape):
                raise ValueError(f"Rank mismatch: dataset {len(dataset.shape)}D vs activations {len(activations.shape)}D")
            
            max_shape = tuple(max(d, a) for d, a in zip(dataset.shape[1:], activations.shape[1:]))
            
            # Resize dataset to max shape (maintain rank)
            temp_shape = (new_size,) + max_shape
            dataset.resize(temp_shape)
            
            # Pad new activations to match dataset shape
            if max_shape != activations.shape[1:]:
                padded_new = np.zeros((activations.shape[0],) + max_shape, dtype=activations.dtype)
                slices = tuple(slice(0, s) for s in activations.shape[1:])
                padded_new[:, slices] = activations
                activations = padded_new
        else:
            # Shapes match (or first time) - just resize
            dataset.resize((new_size,) + activations.shape[1:])
        
        # Write new data
        dataset[current_size:new_size] = activations
        
        # Clear buffer
        self.buffers[layer_name] = []
        
        # Flush to disk
        self.file.flush()
    
    def _create_dataset(self, layer_name: str, sample_activation: np.ndarray):
        """Create HDF5 dataset for a layer."""
        # Create group structure: /layers/{layer_name}/
        group_path = f"layers/{layer_name}"
        if group_path not in self.file:
            group = self.file.create_group(group_path)
        else:
            group = self.file[group_path]
        
        # Determine shape: [max_samples, *activation_shape]
        # Start with resizable first dimension
        shape = (0,) + sample_activation.shape
        maxshape = (None,) + sample_activation.shape
        
        # Determine chunk size
        if self.chunk_size:
            chunks = self.chunk_size
        else:
            # Auto-chunk: use first dimension as batch, rest as sample shape
            chunks = (min(32, shape[1] if len(shape) > 1 else 32),) + sample_activation.shape
        
        # Create dataset
        dataset = group.create_dataset(
            'activations',
            shape=shape,
            maxshape=maxshape,
            dtype=sample_activation.dtype,
            chunks=chunks,
            compression='gzip',
            compression_opts=4
        )
        
        # Store shape and dtype metadata
        dataset.attrs['activation_shape'] = sample_activation.shape
        dataset.attrs['dtype'] = str(sample_activation.dtype)
        dataset.attrs['layer_name'] = layer_name
        
        self.datasets[layer_name] = dataset
    
    def add_metadata(self, metadata: Dict[str, any]):
        """Add metadata to the HDF5 file."""
        if not self._initialized:
            self.open()
        
        if 'metadata' not in self.file:
            meta_group = self.file.create_group('metadata')
        else:
            meta_group = self.file['metadata']
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float)) for v in value):
                meta_group.attrs[key] = value
            else:
                # Store as string representation
                meta_group.attrs[key] = str(value)

