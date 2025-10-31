"""Tests for activation capture functionality."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import tempfile

from interpretops.activation_capture import ActivationCapture
from interpretops.streaming_writer import HDF5ActivationWriter
from interpretops.config import CaptureConfig


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        mlp_out = self.mlp(x)
        x = x + mlp_out
        return x


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.transformer = nn.ModuleList([
            SimpleTransformerBlock() for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(128)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.transformer:
            x = layer(x)
        x = self.layer_norm(x)
        return x


def test_layer_detection():
    """Test that attention and MLP layers are detected."""
    from interpretops.utils import detect_attention_layers, detect_mlp_layers
    
    model = SimpleModel(num_layers=2)
    
    attn_layers = detect_attention_layers(model)
    mlp_layers = detect_mlp_layers(model)
    
    assert len(attn_layers) > 0
    assert len(mlp_layers) > 0
    assert any('attn' in layer.lower() for layer in attn_layers)
    assert any('mlp' in layer.lower() for layer in mlp_layers)


def test_activation_capture():
    """Test basic activation capture."""
    model = SimpleModel(num_layers=2)
    model.eval()
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        output_path = f.name
    
    try:
        config = CaptureConfig(
            batch_size=2,
            output_path=output_path
        )
        
        writer = HDF5ActivationWriter(output_path)
        writer.open()
        
        capture = ActivationCapture(model, config, writer)
        
        # Create some dummy inputs
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Capture activations
        capture.capture_batch(input_ids, prompt_indices=[0, 1])
        
        # Finish and close
        capture.finish()
        capture.remove_hooks()
        writer.close()
        
        # Verify output file
        assert os.path.exists(output_path)
        
        with h5py.File(output_path, 'r') as f:
            # Check metadata
            assert 'metadata' in f
            assert 'target_layers' in f['metadata'].attrs
            
            # Check layers
            if 'layers' in f:
                assert len(f['layers']) > 0
                
                # Verify shape annotations
                for layer_name in f['layers'].keys():
                    layer_group = f['layers'][layer_name]
                    if 'activations' in layer_group:
                        ds = layer_group['activations']
                        assert 'activation_shape' in ds.attrs
                        assert ds.shape[0] > 0  # Should have captured activations
    
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def test_hdf5_writer():
    """Test HDF5 writer functionality."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        output_path = f.name
    
    try:
        writer = HDF5ActivationWriter(output_path)
        writer.open()
        
        # Add some test activations
        for i in range(5):
            activation = np.random.randn(10, 128).astype(np.float32)
            writer.add_activation('test_layer', activation, i)
        
        writer.flush()
        writer.add_metadata({'test_key': 'test_value', 'num_items': 5})
        writer.close()
        
        # Verify
        with h5py.File(output_path, 'r') as f:
            assert 'layers' in f
            assert 'test_layer' in f['layers']
            assert 'activations' in f['layers']['test_layer']
            
            ds = f['layers']['test_layer']['activations']
            assert ds.shape[0] == 5
            assert 'activation_shape' in ds.attrs
            
            # Check metadata
            assert 'metadata' in f
            assert f['metadata'].attrs['test_key'] == 'test_value'
    
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    test_layer_detection()
    test_hdf5_writer()
    test_activation_capture()
    print("All tests passed!")

