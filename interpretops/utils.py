"""Utility functions for activation capture."""

import re
from typing import List, Dict, Any, Optional
import torch.nn as nn


def detect_attention_layers(model: nn.Module, layer_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Detect attention layers in a Transformer model.
    
    Args:
        model: PyTorch model to analyze
        layer_patterns: Optional list of patterns to match (e.g., ['attn', 'attention'])
                       If None, uses default patterns
    
    Returns:
        List of full layer names (module paths) that match attention patterns
    """
    if layer_patterns is None:
        layer_patterns = ['attn', 'attention', 'self_attn']
    
    attention_layers = []
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        name_lower = name.lower()
        
        # Check if module name or type matches patterns
        if any(pattern.lower() in name_lower for pattern in layer_patterns):
            # Verify it's actually an attention-like module
            if 'attention' in module_type or 'attn' in module_type or hasattr(module, 'q_proj'):
                attention_layers.append(name)
        elif 'attention' in module_type or 'attn' in module_type:
            attention_layers.append(name)
    
    return attention_layers


def detect_mlp_layers(model: nn.Module, layer_patterns: Optional[List[str]] = None) -> List[str]:
    """
    Detect MLP/FFN layers in a Transformer model.
    
    Args:
        model: PyTorch model to analyze
        layer_patterns: Optional list of patterns to match (e.g., ['mlp', 'ffn', 'feed_forward'])
                       If None, uses default patterns
    
    Returns:
        List of full layer names (module paths) that match MLP patterns
    """
    if layer_patterns is None:
        layer_patterns = ['mlp', 'ffn', 'feed_forward', 'feedforward']
    
    mlp_layers = []
    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        name_lower = name.lower()
        
        # Check if module name or type matches patterns
        if any(pattern.lower() in name_lower for pattern in layer_patterns):
            # Verify it's actually an MLP-like module
            if 'mlp' in module_type or 'ffn' in module_type or hasattr(module, 'fc1') or hasattr(module, 'gate_proj'):
                mlp_layers.append(name)
        elif 'mlp' in module_type or 'ffn' in module_type or 'feedforward' in module_type:
            mlp_layers.append(name)
    
    return mlp_layers


def parse_dataset(json_data: Any) -> List[str]:
    """
    Parse a JSON dataset into a list of prompt strings.
    
    Supports multiple formats:
    - List of strings: ["prompt1", "prompt2", ...]
    - List of dicts with 'prompt' key: [{"prompt": "..."}, ...]
    - List of dicts with 'text' key: [{"text": "..."}, ...]
    
    Args:
        json_data: JSON data (list or dict)
    
    Returns:
        List of prompt strings
    """
    if isinstance(json_data, list):
        prompts = []
        for item in json_data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                # Try common keys
                prompt = item.get('prompt') or item.get('text') or item.get('input') or item.get('query')
                if prompt:
                    prompts.append(prompt)
                else:
                    raise ValueError(f"Could not find prompt field in item: {item.keys()}")
            else:
                raise ValueError(f"Unexpected item type in dataset: {type(item)}")
        return prompts
    elif isinstance(json_data, dict):
        # Single item or nested structure
        if 'prompt' in json_data:
            return [json_data['prompt']]
        elif 'text' in json_data:
            return [json_data['text']]
        else:
            raise ValueError(f"Could not find prompt/text field in dict: {json_data.keys()}")
    else:
        raise ValueError(f"Dataset must be a list or dict, got {type(json_data)}")


def validate_shape(shape: tuple, expected_dims: Optional[int] = None) -> bool:
    """
    Validate that a shape has the expected number of dimensions.
    
    Args:
        shape: Shape tuple to validate
        expected_dims: Expected number of dimensions (None to skip check)
    
    Returns:
        True if valid, False otherwise
    """
    if expected_dims is not None:
        return len(shape) == expected_dims
    return True


def sanitize_layer_name(layer_name: str) -> str:
    """
    Sanitize a layer name for use in HDF5 group names.
    
    Args:
        layer_name: Original layer name (may contain dots, etc.)
    
    Returns:
        Sanitized layer name safe for HDF5
    """
    # Replace dots with underscores for HDF5 compatibility
    return layer_name.replace('.', '_').replace('/', '_')

