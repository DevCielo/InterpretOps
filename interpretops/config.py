"""Configuration system for activation capture."""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class CaptureConfig:
    """Configuration for activation capture."""
    
    # Layer selection
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    target_layers: Optional[List[str]] = None  # Specific layer names (overrides patterns)
    
    # Batching parameters
    batch_size: int = 32
    flush_interval: int = 100  # Flush after N batches
    
    # Output settings
    output_path: str = "activations.h5"
    chunk_size: tuple = None  # HDF5 chunking, will be auto-determined if None
    
    # Memory management
    max_buffer_size: int = 1000  # Maximum activations to hold before forcing flush
    
    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.include_patterns is None:
            self.include_patterns = ['attn', 'mlp', 'attention', 'ffn']
        if self.exclude_patterns is None:
            self.exclude_patterns = []
    
    def should_capture_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be captured based on config.
        
        Args:
            layer_name: Full module path name
        
        Returns:
            True if layer should be captured
        """
        # If specific layers are specified, use those
        if self.target_layers is not None:
            return layer_name in self.target_layers
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if re.search(pattern, layer_name, re.IGNORECASE):
                return False
        
        # Check include patterns
        if self.include_patterns:
            for pattern in self.include_patterns:
                if re.search(pattern, layer_name, re.IGNORECASE):
                    return True
            return False
        
        return True
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'CaptureConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

