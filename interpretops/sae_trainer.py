"""Sparse Autoencoder (SAE) training system for activation features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class SAEConfig:
    """Configuration for SAE training."""
    
    hidden_dim: int = 8192  # SAE dictionary size
    sparsity_coef: float = 0.001  # L1 regularization coefficient
    learning_rate: float = 0.0001
    batch_size: int = 4096
    max_activations: Optional[int] = None  # None = use all available
    epochs: int = 10
    device: str = "cuda"  # "cuda" or "cpu"
    optimizer: str = "adam"  # "adam" or "sgd"
    weight_decay: float = 0.0
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SAEConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'hidden_dim': self.hidden_dim,
            'sparsity_coef': self.sparsity_coef,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_activations': self.max_activations,
            'epochs': self.epochs,
            'device': self.device,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
        }


class SAE(nn.Module):
    """
    Sparse Autoencoder for learning feature dictionaries from activations.
    
    Architecture:
    - Encoder: Linear(input_dim → hidden_dim) + ReLU
    - Decoder: Linear(hidden_dim → input_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize SAE.
        
        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of feature dictionary (number of features)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: maps activations to sparse feature space
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: reconstructs activations from features
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize decoder bias to mean of data (common practice)
        # Will be set during training
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAE.
        
        Args:
            x: Input activations of shape (batch_size, input_dim)
        
        Returns:
            Tuple of (reconstructed activations, feature activations)
            - x_recon: Reconstructed activations (batch_size, input_dim)
            - features: Sparse feature activations (batch_size, hidden_dim)
        """
        # Encode: activation → sparse features
        features = F.relu(self.encoder(x))
        
        # Decode: sparse features → reconstruction
        x_recon = self.decoder(features)
        
        return x_recon, features
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        return F.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to activations."""
        return self.decoder(features)


def load_activations_from_hdf5(
    hdf5_path: str,
    layer_name: str,
    max_samples: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, int]:
    """
    Load activations from HDF5 file for a specific layer.
    
    Handles shape flattening: [N, seq_len, hidden_dim] → [N*seq_len, hidden_dim]
    
    Args:
        hdf5_path: Path to HDF5 activation file
        layer_name: Name of layer to load (will be sanitized)
        max_samples: Maximum number of activation vectors to load (None = all)
        device: Device to load tensors on
    
    Returns:
        Tuple of (activations_tensor, original_hidden_dim)
        - activations: Tensor of shape (total_samples, hidden_dim)
        - hidden_dim: The hidden dimension size
    """
    # Sanitize layer name (match how it's stored in HDF5)
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError(f"No layers found in HDF5 file {hdf5_path}")
        
        available = list(f['layers'].keys())
        layer_path = f"layers/{safe_layer_name}/activations"
        
        if layer_path not in f:
            # Try to find a close match
            # Check if layer name contains a number that might match
            layer_num_match = re.search(r'(\d+)', layer_name)
            
            suggestions = []
            if layer_num_match:
                layer_num = layer_num_match.group(1)
                # Look for layers containing this exact number (not substring)
                # For "8", match "_8_" or "_8_mlp" but not "_18_"
                layer_lower = layer_name.lower()
                for avail in available:
                    avail_lower = avail.lower()
                    # Check for exact number match - number surrounded by underscores or at boundaries
                    # This avoids matching "8" inside "18"
                    num_with_underscores = f'_{layer_num}_'
                    num_at_start = f'{layer_num}_'
                    num_at_end = f'_{layer_num}'
                    
                    has_exact_match = (
                        num_with_underscores in avail_lower or
                        avail_lower.startswith(num_at_start) or
                        avail_lower.endswith(num_at_end)
                    )
                    
                    if has_exact_match:
                        # Check if both have same type (mlp or attention)
                        has_mlp_match = 'mlp' in layer_lower and 'mlp' in avail_lower
                        has_attn_match = ('attn' in layer_lower or 'attention' in layer_lower) and 'attention' in avail_lower
                        if has_mlp_match or has_attn_match:
                            suggestions.append(avail)
                            if len(suggestions) >= 3:
                                break
            
            # Build error message
            error_msg = f"Layer '{layer_name}' (sanitized: '{safe_layer_name}') not found in {hdf5_path}\n"
            
            if suggestions:
                error_msg += f"\nDid you mean one of these?\n"
                for sug in suggestions:
                    error_msg += f"  - {sug}\n"
            
            error_msg += f"\nAll available layers ({len(available)} total):\n"
            # Show first 10 and last 5 if there are many
            if len(available) <= 15:
                for avail in available:
                    error_msg += f"  - {avail}\n"
            else:
                for avail in available[:10]:
                    error_msg += f"  - {avail}\n"
                error_msg += f"  ... ({len(available) - 15} more) ...\n"
                for avail in available[-5:]:
                    error_msg += f"  - {avail}\n"
            
            raise ValueError(error_msg)
        
        dataset = f[layer_path]
        data = dataset[:]  # Load entire dataset
        
        # Get shape info
        original_shape = data.shape
        # Shape is [N, *activation_shape] where activation_shape might be [seq_len, hidden_dim] or [hidden_dim]
        
        # Flatten all but the last dimension
        if len(original_shape) == 2:
            # Already 2D: [N, hidden_dim]
            activations_2d = data
            hidden_dim = original_shape[1]
        elif len(original_shape) == 3:
            # 3D: [N, seq_len, hidden_dim] → [N*seq_len, hidden_dim]
            N, seq_len, hidden_dim = original_shape
            activations_2d = data.reshape(N * seq_len, hidden_dim)
        else:
            # Higher dimensional: flatten all but last
            hidden_dim = original_shape[-1]
            total_samples = np.prod(original_shape[:-1])
            activations_2d = data.reshape(total_samples, hidden_dim)
        
        # Limit to max_samples if specified
        if max_samples is not None and len(activations_2d) > max_samples:
            # Randomly sample
            indices = np.random.choice(len(activations_2d), max_samples, replace=False)
            activations_2d = activations_2d[indices]
        
        # Convert to torch tensor
        activations_tensor = torch.from_numpy(activations_2d).float().to(device)
        
        return activations_tensor, hidden_dim


def compute_sparsity(features: torch.Tensor) -> float:
    """
    Compute sparsity percentage (percentage of features that are zero).
    
    Args:
        features: Feature activations of shape (batch_size, hidden_dim)
    
    Returns:
        Sparsity percentage (0-100)
    """
    total_features = features.numel()
    zero_features = (features == 0).sum().item()
    return 100.0 * zero_features / total_features


def compute_feature_stats(
    model: SAE,
    activations: torch.Tensor,
    batch_size: int = 4096,
    device: str = "cuda"
) -> Dict:
    """
    Compute feature statistics over a dataset.
    
    Args:
        model: Trained SAE model
        activations: All activations to compute stats over (N, hidden_dim)
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Dictionary with feature statistics:
        - dead_features: List of feature indices never activated
        - usage_frequency: Per-feature activation frequency (percentage)
        - avg_l1_norm: Average L1 norm of features
    """
    model.eval()
    hidden_dim = model.hidden_dim
    
    # Track feature usage across all activations
    feature_usage = torch.zeros(hidden_dim, device=device)
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(device)
            
            # Encode to get features
            features = model.encode(batch)  # (batch_size, hidden_dim)
            
            # Count how many samples activate each feature
            activated = (features > 0).float()  # (batch_size, hidden_dim)
            feature_usage += activated.sum(dim=0)  # Sum over batch
            
            total_samples += len(batch)
    
    # Compute statistics
    usage_frequency = (feature_usage / total_samples * 100).cpu().numpy()  # Percentage
    dead_features = torch.where(feature_usage == 0)[0].cpu().tolist()
    
    # Compute average L1 norm
    model.train()
    total_l1 = 0.0
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(device)
            features = model.encode(batch)
            total_l1 += features.abs().sum().item()
    
    avg_l1_norm = total_l1 / len(activations)
    
    return {
        'dead_features': dead_features,
        'usage_frequency': usage_frequency.tolist(),
        'num_dead_features': len(dead_features),
        'avg_l1_norm': avg_l1_norm,
    }


class SAETrainer:
    """
    Trainer for Sparse Autoencoder.
    
    Manages training loop, metrics tracking, and model saving.
    """
    
    def __init__(self, config: SAEConfig):
        """
        Initialize trainer.
        
        Args:
            config: SAE training configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Will be set when training starts
        self.model: Optional[SAE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
    def train(
        self,
        activations: torch.Tensor,
        input_dim: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train SAE on activations.
        
        Args:
            activations: Training activations of shape (N, input_dim)
            input_dim: Dimension of input activations
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with training metrics:
            - reconstruction_loss: List of reconstruction MSE per epoch
            - sparsity_loss: List of L1 sparsity loss per epoch
            - total_loss: List of total loss per epoch
            - sparsity: List of sparsity percentages per epoch
        """
        # Initialize model
        self.model = SAE(input_dim=input_dim, hidden_dim=self.config.hidden_dim)
        self.model = self.model.to(self.device)
        
        # Initialize decoder bias to mean of data (common SAE practice)
        with torch.no_grad():
            data_mean = activations.mean(dim=0).to(self.device)
            self.model.decoder.bias.copy_(data_mean)
        
        # Setup optimizer
        if self.config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Training metrics
        metrics = {
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'total_loss': [],
            'sparsity': [],
        }
        
        # Move activations to device
        activations = activations.to(self.device)
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Shuffle data
            indices = torch.randperm(len(activations))
            activations_shuffled = activations[indices]
            
            epoch_recon_loss = 0.0
            epoch_sparsity_loss = 0.0
            epoch_total_loss = 0.0
            epoch_sparsity = 0.0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(activations_shuffled), self.config.batch_size):
                batch = activations_shuffled[i:i+self.config.batch_size]
                
                # Forward pass
                x_recon, features = self.model(batch)
                
                # Compute losses
                recon_loss = F.mse_loss(x_recon, batch)
                sparsity_loss = features.abs().mean() * self.config.sparsity_coef
                total_loss = recon_loss + sparsity_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Track metrics
                batch_sparsity = compute_sparsity(features)
                
                epoch_recon_loss += recon_loss.item()
                epoch_sparsity_loss += sparsity_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_sparsity += batch_sparsity
                num_batches += 1
            
            # Average metrics over epoch
            metrics['reconstruction_loss'].append(epoch_recon_loss / num_batches)
            metrics['sparsity_loss'].append(epoch_sparsity_loss / num_batches)
            metrics['total_loss'].append(epoch_total_loss / num_batches)
            metrics['sparsity'].append(epoch_sparsity / num_batches)
            
            if verbose and (epoch + 1) % max(1, self.config.epochs // 10) == 0:
                print(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Recon Loss={metrics['reconstruction_loss'][-1]:.6f}, "
                    f"Sparsity Loss={metrics['sparsity_loss'][-1]:.6f}, "
                    f"Sparsity={metrics['sparsity'][-1]:.1f}%"
                )
        
        return metrics
    
    def compute_reconstruction_metrics(
        self,
        activations: torch.Tensor,
        batch_size: int = 4096
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.
        
        Args:
            activations: Test activations (N, input_dim)
            batch_size: Batch size for evaluation
        
        Returns:
            Dictionary with:
            - mse: Mean squared error
            - correlation: Pearson correlation coefficient
            - r_squared: R-squared coefficient of determination
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        
        all_recon = []
        all_original = []
        
        with torch.no_grad():
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i+batch_size].to(self.device)
                
                x_recon, _ = self.model(batch)
                
                all_recon.append(x_recon.cpu())
                all_original.append(batch.cpu())
        
        # Concatenate all batches
        all_recon = torch.cat(all_recon, dim=0)
        all_original = torch.cat(all_original, dim=0)
        
        # Compute MSE
        mse = F.mse_loss(all_recon, all_original).item()
        
        # Compute correlation (Pearson)
        # Flatten for correlation computation
        recon_flat = all_recon.flatten()
        orig_flat = all_original.flatten()
        
        # Compute correlation coefficient
        recon_centered = recon_flat - recon_flat.mean()
        orig_centered = orig_flat - orig_flat.mean()
        
        numerator = (recon_centered * orig_centered).sum()
        recon_std = torch.sqrt((recon_centered ** 2).sum())
        orig_std = torch.sqrt((orig_centered ** 2).sum())
        
        if recon_std > 0 and orig_std > 0:
            correlation = (numerator / (recon_std * orig_std)).item()
        else:
            correlation = 0.0
        
        # Compute R-squared
        ss_res = ((all_original - all_recon) ** 2).sum().item()
        ss_tot = ((all_original - all_original.mean()) ** 2).sum().item()
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        self.model.train()
        
        return {
            'mse': mse,
            'correlation': correlation,
            'r_squared': r_squared,
        }
    
    def save_model(
        self,
        output_path: str,
        activations: Optional[torch.Tensor] = None
    ):
        """
        Save trained SAE model with weights and feature statistics.
        
        Args:
            output_path: Path to save .pt file
            activations: Optional activations for computing feature stats
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare state dict
        state_dict = {
            'encoder.weight': self.model.encoder.weight.cpu(),
            'encoder.bias': self.model.encoder.bias.cpu(),
            'decoder.weight': self.model.decoder.weight.cpu(),
            'decoder.bias': self.model.decoder.bias.cpu(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
        }
        
        # Compute feature stats if activations provided
        feature_stats = {}
        if activations is not None:
            feature_stats = compute_feature_stats(
                self.model,
                activations,
                batch_size=self.config.batch_size,
                device=str(self.device)
            )
        
        # Add config and stats to state dict
        state_dict['feature_stats'] = feature_stats
        state_dict['config'] = self.config.to_dict()
        
        # Save
        torch.save(state_dict, output_path)


def train_sae(
    config_path: str,
    activations_path: str,
    layer_name: str,
    output_path: str,
    verbose: bool = True
) -> Dict:
    """
    Main entry point for training SAE.
    
    Args:
        config_path: Path to sae_config.yaml
        activations_path: Path to input HDF5 activations file
        layer_name: Name of layer to train SAE for
        output_path: Path to save trained model (.pt file)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training results and metrics
    """
    # Load config
    config = SAEConfig.from_yaml(config_path)
    
    if verbose:
        print(f"Loading activations from {activations_path} (layer: {layer_name})")
    
    # Load activations
    activations, input_dim = load_activations_from_hdf5(
        activations_path,
        layer_name,
        max_samples=config.max_activations,
        device=config.device
    )
    
    if verbose:
        print(f"Loaded {len(activations)} activation vectors (dim={input_dim})")
        print(f"Training SAE with hidden_dim={config.hidden_dim}")
    
    # Initialize trainer
    trainer = SAETrainer(config)
    
    # Train
    training_metrics = trainer.train(activations, input_dim, verbose=verbose)
    
    # Compute final reconstruction metrics
    if verbose:
        print("Computing final reconstruction metrics...")
    recon_metrics = trainer.compute_reconstruction_metrics(activations)
    
    if verbose:
        print(f"\nFinal Metrics:")
        print(f"  Reconstruction MSE: {recon_metrics['mse']:.6f}")
        print(f"  Correlation: {recon_metrics['correlation']:.4f}")
        print(f"  R²: {recon_metrics['r_squared']:.4f}")
        print(f"  Final Sparsity: {training_metrics['sparsity'][-1]:.1f}%")
    
    # Compute feature stats and save
    if verbose:
        print("Computing feature statistics...")
    trainer.save_model(output_path, activations=activations)
    
    if verbose:
        print(f"Saved trained SAE to {output_path}")
    
    return {
        'training_metrics': training_metrics,
        'reconstruction_metrics': recon_metrics,
        'final_sparsity': training_metrics['sparsity'][-1],
    }

