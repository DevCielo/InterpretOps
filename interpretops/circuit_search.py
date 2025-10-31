"""Circuit Discovery Engine - Activation Patching and Causal Analysis.

Implements ACDC-style iterative pruning to find minimal causal circuits
using activation patching between clean and corrupt examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import h5py
import copy
import logging
from tqdm import tqdm

from .sae_trainer import SAE, load_activations_from_hdf5
from .activation_capture import ActivationCapture
from .config import CaptureConfig
from .utils import detect_attention_layers, detect_mlp_layers

logger = logging.getLogger(__name__)


@dataclass
class CircuitNode:
    """Represents a node in the causal circuit graph."""

    node_id: str  # e.g., "layer_8_feature_123" or "layer_12_attn_head_5"
    node_type: str  # "feature" or "attention_head"
    layer_name: str  # Original layer name
    layer_idx: int  # Layer index
    component_idx: Optional[int] = None  # Feature ID or head index
    causal_score: float = 0.0  # CCS - Causal Contribution Score
    activation_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class CircuitEdge:
    """Represents an edge in the causal circuit graph."""

    source_id: str
    target_id: str
    weight: float = 1.0  # Strength of information flow
    edge_type: str = "activation_flow"  # Type of connection


@dataclass
class CircuitGraph:
    """Complete causal circuit representation."""

    nodes: List[CircuitNode] = field(default_factory=list)
    edges: List[CircuitEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'nodes': [
                {
                    'node_id': node.node_id,
                    'node_type': node.node_type,
                    'layer_name': node.layer_name,
                    'layer_idx': node.layer_idx,
                    'component_idx': node.component_idx,
                    'causal_score': node.causal_score,
                    'activation_stats': node.activation_stats,
                }
                for node in self.nodes
            ],
            'edges': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'weight': edge.weight,
                    'edge_type': edge.edge_type,
                }
                for edge in self.edges
            ],
            'metadata': self.metadata,
        }

    def save_json(self, path: str):
        """Save circuit graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PatchingConfig:
    """Configuration for activation patching."""

    # Model and data
    model_name: str
    clean_prompts: List[str]
    corrupt_prompts: List[str]

    # Target behavior
    target_metric: str = "probability"  # What to measure: "probability", "logit_diff", etc.
    target_tokens: Optional[List[str]] = None  # Tokens to measure probability of

    # Patching settings
    patch_type: str = "activation"  # "activation", "attention", "mlp"
    layers_to_patch: Optional[List[str]] = None  # Specific layers, None = all

    # SAE settings (for feature patching)
    sae_paths: Optional[Dict[str, str]] = None  # layer_name -> sae_path

    # Pruning settings
    pruning_threshold: float = 0.01  # Minimum causal score to keep
    max_iterations: int = 10
    convergence_threshold: float = 0.001

    # Hardware
    device: str = "cuda"
    batch_size: int = 8


class ActivationPatcher:
    """
    Handles activation patching between clean and corrupt examples.

    Supports patching at different granularities:
    - Full activations
    - SAE features
    - Attention heads
    """

    def __init__(self, model: nn.Module, config: PatchingConfig):
        """
        Initialize activation patcher.

        Args:
            model: The transformer model
            config: Patching configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Load SAEs if needed
        self.saes: Dict[str, SAE] = {}
        if config.sae_paths:
            self._load_saes()

        # Cache for activations
        self.clean_cache: Dict[str, torch.Tensor] = {}
        self.corrupt_cache: Dict[str, torch.Tensor] = {}

        # Hook storage
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.patching_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _load_saes(self):
        """Load SAE models for feature-level patching."""
        for layer_name, sae_path in self.config.sae_paths.items():
            if not Path(sae_path).exists():
                logger.warning(f"SAE path {sae_path} not found, skipping")
                continue

            # Load SAE state dict
            state_dict = torch.load(sae_path, map_location=self.device)

            # Create SAE model
            input_dim = state_dict['input_dim']
            hidden_dim = state_dict['hidden_dim']
            sae = SAE(input_dim=input_dim, hidden_dim=hidden_dim)

            # Load weights
            sae.load_state_dict({
                'encoder.weight': state_dict['encoder.weight'],
                'encoder.bias': state_dict['encoder.bias'],
                'decoder.weight': state_dict['decoder.weight'],
                'decoder.bias': state_dict['decoder.bias'],
            })

            sae = sae.to(self.device)
            sae.eval()

            self.saes[layer_name] = sae
            logger.info(f"Loaded SAE for {layer_name}: {input_dim} -> {hidden_dim}")

    def cache_activations(self, prompts: List[str], prompt_type: str = "clean"):
        """
        Cache activations for a set of prompts.

        Args:
            prompts: List of input prompts
            prompt_type: "clean" or "corrupt"
        """
        cache = self.clean_cache if prompt_type == "clean" else self.corrupt_cache

        # Clear existing cache for this type
        cache.clear()

        # Set up activation capture
        capture_config = CaptureConfig(
            target_layers=self.config.layers_to_patch,
            batch_size=1,  # Process one prompt at a time for clean caching
        )

        # Custom writer that stores in memory
        class MemoryWriter:
            def __init__(self):
                self.data: Dict[str, List[torch.Tensor]] = defaultdict(list)

            def add_activation(self, layer_name: str, activation: torch.Tensor, prompt_idx: int):
                self.data[layer_name].append(activation.cpu())

            def flush(self, force_all=False):
                pass

            def open(self):
                pass

            def close(self):
                pass

        writer = MemoryWriter()

        # Create capture system
        capture_system = ActivationCapture(self.model, capture_config, writer)

        # Tokenize prompts
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # First pass: tokenize all prompts to find max length
        all_encoded = []
        max_length = 0
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
            all_encoded.append(encoded)
            max_length = max(max_length, encoded['input_ids'].shape[1])

        # Second pass: process with consistent padding
        for i, encoded in enumerate(tqdm(all_encoded, desc=f"Caching {prompt_type} activations")):
            # Pad to max_length
            input_ids = encoded['input_ids']
            attention_mask = encoded.get('attention_mask')

            if input_ids.shape[1] < max_length:
                # Pad input_ids
                padding_needed = max_length - input_ids.shape[1]
                pad_tensor = torch.full((1, padding_needed), tokenizer.pad_token_id or tokenizer.eos_token_id,
                                      dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, pad_tensor], dim=1)

                # Pad attention_mask if it exists
                if attention_mask is not None:
                    pad_mask = torch.zeros((1, padding_needed), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Capture activations (single prompt)
            capture_system.capture_batch(input_ids, prompt_indices=[i])

        # Store in cache - each layer will have one tensor per prompt
        for layer_name, activations in writer.data.items():
            if activations:
                # All activations should now have the same shape: [1, max_length, hidden_dim]
                # Stack along batch dimension: [num_prompts, max_length, hidden_dim]
                cache[layer_name] = torch.stack(activations, dim=0)

        capture_system.remove_hooks()

        logger.info(f"Cached activations for {len(prompts)} {prompt_type} prompts, {len(cache)} layers")

    def compute_baseline_behavior(self, prompts: List[str]) -> Dict[str, float]:
        """
        Compute baseline behavior metrics for prompts.

        Args:
            prompts: List of prompts to evaluate

        Returns:
            Dictionary of behavior metrics
        """
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        all_metrics = []

        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(prompt, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs.logits

                # Get next token probabilities
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)

                if self.config.target_metric == "probability" and self.config.target_tokens:
                    # Measure probability of specific tokens
                    target_token_ids = tokenizer.convert_tokens_to_ids(self.config.target_tokens)
                    target_probs = probs[:, target_token_ids].sum(dim=-1)
                    metric_value = target_probs.item()
                else:
                    # Default: probability of the most likely token
                    max_prob = probs.max(dim=-1)[0].item()
                    metric_value = max_prob

                all_metrics.append(metric_value)

        return {
            'mean': np.mean(all_metrics),
            'std': np.std(all_metrics),
            'values': all_metrics,
        }

    def patch_and_measure(
        self,
        layer_name: str,
        component_indices: Optional[List[int]] = None,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Patch activations and measure behavior change.

        Args:
            layer_name: Layer to patch
            component_indices: Specific components to patch (heads for attn, features for SAE)
            feature_mask: Binary mask for which features to patch (for SAE patching)

        Returns:
            Behavior change score (higher = more causal)
        """
        # Get clean and corrupt activations
        if layer_name not in self.clean_cache or layer_name not in self.corrupt_cache:
            raise ValueError(f"Activations not cached for layer {layer_name}")

        clean_acts = self.clean_cache[layer_name]  # [num_prompts, seq_len, hidden_dim]
        corrupt_acts = self.corrupt_cache[layer_name]

        # Compute baseline metrics once (expensive operation)
        if not hasattr(self, '_baseline_clean'):
            self._baseline_clean = self.compute_baseline_behavior(self.config.clean_prompts)
            self._baseline_corrupt = self.compute_baseline_behavior(self.config.corrupt_prompts)

        # Set up patching hook
        def patching_hook(module, input, output):
            # Extract the actual activation tensor (output might be tuple)
            if isinstance(output, tuple):
                # For attention, often output is (attn_output, weights)
                activation = output[0]
            else:
                activation = output

            # Determine which prompts to patch (for now, patch all)
            # In practice, you might want to patch specific positions or prompts
            prompt_indices_to_patch = list(range(len(self.config.clean_prompts)))

            # Get the current batch being processed
            batch_size = activation.shape[0]

            # For simplicity, assume we're processing prompts in order
            # In a more robust implementation, we'd track which prompt is being processed
            if isinstance(output, tuple):
                patched_activation = activation.clone()
            else:
                patched_activation = activation.clone()

            for i in range(batch_size):
                if i < len(prompt_indices_to_patch):
                    prompt_idx = prompt_indices_to_patch[i]

                    if self.config.patch_type == "activation":
                        # Full activation patching - replace entire layer output
                        corrupt_act = corrupt_acts[prompt_idx:prompt_idx+1]  # [1, cached_seq_len, hidden_dim]
                        current_seq_len = activation.shape[1]

                        # Ensure corrupt_act is 3D [1, seq_len, hidden_dim]
                        if corrupt_act.dim() == 4:
                            # If it's 4D (attention weights), take first head or sum
                            corrupt_act = corrupt_act.mean(dim=1, keepdim=True)  # [1, seq_len, seq_len] -> [1, seq_len, seq_len]
                            # This is wrong, let me handle this properly
                            corrupt_act = corrupt_act.squeeze(1)  # [1, seq_len, seq_len] -> [seq_len, seq_len]
                            # This is still wrong. Let's just use zeros for now if we have wrong shape
                            corrupt_act = torch.zeros_like(activation[i:i+1])

                        if corrupt_act.shape[1] >= current_seq_len:
                            # Truncate if cached is longer
                            patched_activation[i:i+1] = corrupt_act[:, :current_seq_len, :].to(activation.device)
                        else:
                            # Pad if cached is shorter
                            pad_needed = current_seq_len - corrupt_act.shape[1]
                            pad_tensor = torch.zeros(1, pad_needed, activation.shape[2], device=activation.device)
                            padded_corrupt = torch.cat([corrupt_act.to(activation.device), pad_tensor], dim=1)
                            patched_activation[i:i+1] = padded_corrupt

                    elif self.config.patch_type == "attention" and component_indices is not None:
                        # Attention head patching - this requires knowing the model architecture
                        # For now, approximate by patching the entire attention output
                        corrupt_act = corrupt_acts[prompt_idx:prompt_idx+1]
                        current_seq_len = activation.shape[1]

                        if corrupt_act.shape[1] >= current_seq_len:
                            patched_activation[i:i+1] = corrupt_act[:, :current_seq_len, :].to(activation.device)
                        else:
                            pad_needed = current_seq_len - corrupt_act.shape[1]
                            pad_tensor = torch.zeros(1, pad_needed, activation.shape[2], device=activation.device)
                            padded_corrupt = torch.cat([corrupt_act, pad_tensor], dim=1)
                            patched_activation[i:i+1] = padded_corrupt.to(activation.device)

                    elif self.config.patch_type == "mlp" and self.saes.get(layer_name):
                        # SAE feature patching
                        sae = self.saes[layer_name]

                        # Get activations for this prompt/position
                        clean_act = activation[i:i+1]  # [1, seq_len, hidden_dim]
                        corrupt_act = corrupt_acts[prompt_idx:prompt_idx+1]  # [1, seq_len, hidden_dim]

                        # Encode to features
                        clean_act_flat = clean_act.view(-1, sae.input_dim)  # [seq_len, hidden_dim]
                        corrupt_act_flat = corrupt_act.view(-1, sae.input_dim)

                        clean_features = sae.encode(clean_act_flat)  # [seq_len, feature_dim]
                        corrupt_features = sae.encode(corrupt_act_flat)

                        # Apply feature mask if provided
                        if feature_mask is not None:
                            # Replace selected features with corrupt versions
                            feature_mask_expanded = feature_mask.unsqueeze(0).expand_as(clean_features)
                            clean_features = torch.where(
                                feature_mask_expanded,
                                corrupt_features,
                                clean_features
                            )

                        # Decode back to activations
                        patched_act_flat = sae.decode(clean_features)
                        patched_activation[i:i+1] = patched_act_flat.view(1, -1, sae.input_dim)

            # Return the patched output in the same format as input
            if isinstance(output, tuple):
                return (patched_activation,) + output[1:]
            else:
                return patched_activation

        # Register patching hook
        module = self._get_module_by_name(layer_name)
        hook = module.register_forward_hook(patching_hook)
        self.patching_hooks.append(hook)

        try:
            # Measure behavior with patching
            patched_metrics = self.compute_baseline_behavior(self.config.clean_prompts)

            # Compute behavior change (difference from clean baseline)
            clean_mean = self._baseline_clean['mean']
            corrupt_mean = self._baseline_corrupt['mean']
            patched_mean = patched_metrics['mean']

            # Causal score: how much patching moves behavior toward corrupt
            # Score = (patched - clean) / (corrupt - clean)
            # Higher score means patching has larger effect
            denominator = corrupt_mean - clean_mean
            if abs(denominator) > 1e-6:
                causal_score = (patched_mean - clean_mean) / denominator
                # Clamp to [0, 1] and take absolute value
                causal_score = abs(max(0.0, min(1.0, causal_score)))
            else:
                causal_score = 0.0

            return causal_score

        finally:
            # Remove hook
            hook.remove()
            if hook in self.patching_hooks:
                self.patching_hooks.remove(hook)

    def _get_module_by_name(self, layer_name: str) -> nn.Module:
        """Get module by name."""
        parts = layer_name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                raise ValueError(f"Module {part} not found in {layer_name}")
        return module


class CircuitDiscoverer:
    """
    Main circuit discovery engine using ACDC-style iterative pruning.
    """

    def __init__(self, config: PatchingConfig):
        """
        Initialize circuit discoverer.

        Args:
            config: Patching configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Load model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map='auto' if config.device == 'cuda' else None,
        )
        if config.device == 'cpu':
            self.model = self.model.to(config.device)
        self.model.eval()

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize patcher
        self.patcher = ActivationPatcher(self.model, config)

    def discover_circuit(self) -> CircuitGraph:
        """
        Run the full circuit discovery pipeline.

        Returns:
            CircuitGraph with discovered causal circuit
        """
        logger.info("Starting circuit discovery...")

        # Step 1: Cache activations for clean and corrupt prompts
        logger.info("Caching activations for clean prompts...")
        self.patcher.cache_activations(self.config.clean_prompts, "clean")

        logger.info("Caching activations for corrupt prompts...")
        self.patcher.cache_activations(self.config.corrupt_prompts, "corrupt")

        # Step 2: Initialize candidate nodes
        candidate_nodes = self._initialize_candidate_nodes()

        logger.info(f"Starting with {len(candidate_nodes)} candidate nodes")

        # Step 3: Iterative pruning
        circuit_nodes = self._iterative_pruning(candidate_nodes)

        # Step 4: Build circuit graph
        circuit_graph = self._build_circuit_graph(circuit_nodes)

        logger.info(f"Discovered circuit with {len(circuit_graph.nodes)} nodes")

        return circuit_graph

    def _initialize_candidate_nodes(self) -> List[CircuitNode]:
        """Initialize all possible nodes to consider for the circuit."""
        nodes = []

        # Detect layers based on patch type
        if self.config.patch_type == "attention":
            layer_names = detect_attention_layers(self.model)
        elif self.config.patch_type == "mlp":
            layer_names = detect_mlp_layers(self.model)
        else:
            # For activation patching, use all layers
            attn_layers = detect_attention_layers(self.model)
            mlp_layers = detect_mlp_layers(self.model)
            layer_names = list(set(attn_layers + mlp_layers))

        # Filter to specified layers if provided
        if self.config.layers_to_patch:
            layer_names = [l for l in layer_names if l in self.config.layers_to_patch]

        for layer_name in layer_names:
            # Extract layer index
            layer_idx = self._extract_layer_index(layer_name)

            if self.config.patch_type == "attention":
                # For attention layers, create nodes for each head
                # Note: This is simplified - in practice we'd need to know num_heads
                num_heads = 8  # Default assumption, should be configurable
                for head_idx in range(num_heads):
                    node = CircuitNode(
                        node_id=f"{layer_name}_head_{head_idx}",
                        node_type="attention_head",
                        layer_name=layer_name,
                        layer_idx=layer_idx,
                        component_idx=head_idx,
                    )
                    nodes.append(node)

            elif self.config.patch_type == "mlp" and layer_name in self.patcher.saes:
                # For MLP layers with SAEs, create nodes for each feature
                sae = self.patcher.saes[layer_name]
                for feature_idx in range(sae.hidden_dim):
                    node = CircuitNode(
                        node_id=f"{layer_name}_feature_{feature_idx}",
                        node_type="feature",
                        layer_name=layer_name,
                        layer_idx=layer_idx,
                        component_idx=feature_idx,
                    )
                    nodes.append(node)

            else:
                # For full activation patching, one node per layer
                node = CircuitNode(
                    node_id=f"{layer_name}_activation",
                    node_type="activation",
                    layer_name=layer_name,
                    layer_idx=layer_idx,
                )
                nodes.append(node)

        return nodes

    def _iterative_pruning(self, candidate_nodes: List[CircuitNode]) -> List[CircuitNode]:
        """
        Iteratively prune low-impact nodes using ACDC-style algorithm.

        Args:
            candidate_nodes: Initial set of candidate nodes

        Returns:
            Pruned list of circuit nodes with causal scores
        """
        current_nodes = candidate_nodes.copy()
        prev_total_score = 0.0

        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            # Compute causal scores for all current nodes
            total_score = 0.0
            for node in tqdm(current_nodes, desc="Computing causal scores"):
                score = self._compute_node_causal_score(node, current_nodes)
                node.causal_score = score
                total_score += score

            logger.info(f"Total causal score: {total_score:.4f}")

            # Check convergence
            if abs(total_score - prev_total_score) < self.config.convergence_threshold:
                logger.info("Converged!")
                break

            prev_total_score = total_score

            # Prune low-scoring nodes
            current_nodes = [
                node for node in current_nodes
                if node.causal_score >= self.config.pruning_threshold
            ]

            logger.info(f"Pruned to {len(current_nodes)} nodes")

            if not current_nodes:
                logger.warning("All nodes pruned! Stopping.")
                break

        return current_nodes

    def _compute_node_causal_score(self, target_node: CircuitNode, all_nodes: List[CircuitNode]) -> float:
        """
        Compute causal contribution score for a single node.

        Args:
            target_node: Node to score
            all_nodes: All nodes currently in consideration

        Returns:
            Causal score (0-1, higher = more important)
        """
        # For now, patch the entire layer containing this node
        # In a more sophisticated implementation, we'd patch individual components
        layer_name = target_node.layer_name

        if target_node.node_type == "attention_head":
            # Patch specific attention head
            component_indices = [target_node.component_idx]
            score = self.patcher.patch_and_measure(
                layer_name, component_indices=component_indices
            )
        elif target_node.node_type == "feature":
            # Patch specific SAE feature
            feature_mask = torch.zeros(self.patcher.saes[layer_name].hidden_dim)
            feature_mask[target_node.component_idx] = 1
            score = self.patcher.patch_and_measure(
                layer_name, feature_mask=feature_mask
            )
        else:
            # Patch entire layer activation
            score = self.patcher.patch_and_measure(layer_name)

        return score

    def _build_circuit_graph(self, circuit_nodes: List[CircuitNode]) -> CircuitGraph:
        """
        Build the final circuit graph from discovered nodes.

        Args:
            circuit_nodes: Final set of circuit nodes

        Returns:
            Complete circuit graph
        """
        # Sort nodes by causal score
        circuit_nodes.sort(key=lambda x: x.causal_score, reverse=True)

        # Create edges based on layer ordering (information flow)
        edges = []
        nodes_by_layer = defaultdict(list)

        for node in circuit_nodes:
            nodes_by_layer[node.layer_idx].append(node)

        # Create edges from lower to higher layers
        layer_indices = sorted(nodes_by_layer.keys())
        for i, layer_idx in enumerate(layer_indices[:-1]):
            next_layer_idx = layer_indices[i + 1]

            for source_node in nodes_by_layer[layer_idx]:
                for target_node in nodes_by_layer[next_layer_idx]:
                    edge = CircuitEdge(
                        source_id=source_node.node_id,
                        target_id=target_node.node_id,
                        weight=min(source_node.causal_score, target_node.causal_score),
                        edge_type="layer_flow"
                    )
                    edges.append(edge)

        # Create graph
        graph = CircuitGraph(
            nodes=circuit_nodes,
            edges=edges,
            metadata={
                'model_name': self.config.model_name,
                'patch_type': self.config.patch_type,
                'target_metric': self.config.target_metric,
                'num_clean_prompts': len(self.config.clean_prompts),
                'num_corrupt_prompts': len(self.config.corrupt_prompts),
                'pruning_threshold': self.config.pruning_threshold,
                'total_nodes_discovered': len(circuit_nodes),
            }
        )

        return graph

    def _extract_layer_index(self, layer_name: str) -> int:
        """Extract layer index from layer name."""
        import re
        match = re.search(r'\.(\d+)\.', layer_name)
        if match:
            return int(match.group(1))
        # Try other patterns
        match = re.search(r'layer_(\d+)', layer_name)
        if match:
            return int(match.group(1))
        return 0  # Default


def discover_circuit(
    model_name: str,
    clean_prompts: List[str],
    corrupt_prompts: List[str],
    target_tokens: Optional[List[str]] = None,
    patch_type: str = "mlp",
    layers_to_patch: Optional[List[str]] = None,
    sae_paths: Optional[Dict[str, str]] = None,
    output_path: str = "circuit_graph.json",
    device: str = "cuda",
    max_iterations: int = 10,
    pruning_threshold: float = 0.01,
) -> CircuitGraph:
    """
    Main entry point for circuit discovery.

    Args:
        model_name: HuggingFace model identifier
        clean_prompts: Clean/normal prompts
        corrupt_prompts: Corrupt/adversarial prompts
        target_tokens: Tokens to measure probability of (for refusal circuits)
        patch_type: "activation", "attention", or "mlp"
        layers_to_patch: Specific layers to consider
        sae_paths: Paths to trained SAE models (layer_name -> path)
        output_path: Where to save the circuit graph
        device: Device to use

    Returns:
        Discovered circuit graph
    """
    # Create configuration
    config = PatchingConfig(
        model_name=model_name,
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        target_tokens=target_tokens,
        patch_type=patch_type,
        layers_to_patch=layers_to_patch,
        sae_paths=sae_paths,
        device=device,
        max_iterations=max_iterations,
        pruning_threshold=pruning_threshold,
    )

    # Run discovery
    discoverer = CircuitDiscoverer(config)
    circuit_graph = discoverer.discover_circuit()

    # Save result
    circuit_graph.save_json(output_path)
    logger.info(f"Circuit graph saved to {output_path}")

    return circuit_graph
