"""Core activation capture system with PyTorch hooks."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import threading

from .streaming_writer import HDF5ActivationWriter
from .config import CaptureConfig
from .utils import detect_attention_layers, detect_mlp_layers, sanitize_layer_name


class ActivationCapture:
    """
    Manages PyTorch hooks to capture activations from model layers.
    
    Supports streaming to HDF5 with memory-efficient batching.
    """
    
    def __init__(self, model: nn.Module, config: CaptureConfig, writer: HDF5ActivationWriter):
        """
        Initialize activation capture system.
        
        Args:
            model: PyTorch model to capture activations from
            config: Capture configuration
            writer: HDF5 writer instance for streaming activations
        """
        self.model = model
        self.config = config
        self.writer = writer
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.current_prompt_idx = 0
        self.batch_counter = 0
        self.lock = threading.Lock()
        
        # Detect layers to capture
        self.target_layers = self._detect_target_layers()
        
        # Register hooks
        self._register_hooks()
    
    def _detect_target_layers(self) -> List[str]:
        """Detect which layers should be captured based on config."""
        all_layers = []
        
        # Detect attention and MLP layers
        attention_layers = detect_attention_layers(self.model)
        mlp_layers = detect_mlp_layers(self.model)
        
        # Combine and filter by config
        candidate_layers = list(set(attention_layers + mlp_layers))
        
        for layer_name in candidate_layers:
            if self.config.should_capture_layer(layer_name):
                all_layers.append(layer_name)
        
        return all_layers
    
    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for layer_name in self.target_layers:
            # Get the module
            module = self._get_module_by_name(layer_name)
            if module is None:
                continue
            
            # Create hook closure with proper binding using default argument
            def make_hook(name=layer_name):  # Default arg captures the value
                def hook_fn(module, input, output):
                    # Extract the actual activation (output might be tuple)
                    if isinstance(output, tuple):
                        # For attention, often output is (attn_output, weights)
                        activation = output[0]
                    else:
                        activation = output
                    
                    # Capture activation
                    with self.lock:
                        self.writer.add_activation(name, activation, self.current_prompt_idx)
                
                return hook_fn
            
            # Register hook (default arg ensures proper closure)
            hook = module.register_forward_hook(make_hook())
            self.hooks.append(hook)
    
    def _get_module_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get module by its full name."""
        parts = layer_name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module
    
    def capture_batch(self, inputs: torch.Tensor, prompt_indices: Optional[List[int]] = None):
        """
        Capture activations for a batch of inputs.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, ...)
            prompt_indices: Optional list of prompt indices for tracking
        """
        batch_size = inputs.shape[0]
        
        if prompt_indices is None:
            prompt_indices = list(range(self.current_prompt_idx, self.current_prompt_idx + batch_size))
        
        with self.lock:
            # Update prompt indices for hooks
            old_indices = self.current_prompt_idx
        
        try:
            # Process batch
            for i, prompt_idx in enumerate(prompt_indices):
                with self.lock:
                    self.current_prompt_idx = prompt_idx
                
                # Forward pass for single item (to capture per-prompt activations)
                single_input = inputs[i:i+1]
                with torch.no_grad():
                    _ = self.model(single_input)
        
        finally:
            with self.lock:
                self.current_prompt_idx = old_indices
                self.batch_counter += 1
                
                # Flush if needed
                if self.batch_counter % self.config.flush_interval == 0:
                    self.writer.flush()
    
    def finish(self):
        """Finish capture and flush all remaining activations."""
        with self.lock:
            self.writer.flush(force_all=True)
            
            # Add metadata
            metadata = {
                'model_name': type(self.model).__name__,
                'num_prompts': self.current_prompt_idx + 1,
                'target_layers': self.target_layers,
                'batch_size': self.config.batch_size,
            }
            self.writer.add_metadata(metadata)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

