"""
Complete AGT-NO (Adaptive Graph Transformer Neural Operator) architecture.

Integrates encoder-processor-decoder components with physics constraints,
multi-physics coupling, and spectral representations for motor fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
import math

from .constraints import PhysicsConstraintLayer, PDEConstraint
from .multi_physics_coupling import MultiPhysicsCoupling
from .spectral_operator import SpectralOperatorLayer, FourierBasis, create_fourier_spectral_layer


class GraphConvolution(nn.Module):
    """Graph convolution layer for spatial relationships."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with adjacency matrix.
        
        Args:
            input: Input features (batch_size, n_nodes, in_features)
            adj: Adjacency matrix (batch_size, n_nodes, n_nodes)
            
        Returns:
            Output features (batch_size, n_nodes, out_features)
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        output = output + self.bias
        return self.dropout(output)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for neural operators."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-head attention forward pass."""
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output


class AGTNOEncoder(nn.Module):
    """Encoder module with lifting, graph convolution, and attention mechanisms."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Lifting layer: map input to higher dimensional space
        self.lifting = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers // 2)
        ])
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, n_heads, dropout)
            for _ in range(n_layers // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Positional encoding for spatial coordinates
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.1)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoder forward pass.
        
        Args:
            x: Input features (batch_size, n_points, input_dim)
            coords: Spatial coordinates (batch_size, n_points, coord_dim)
            adj_matrix: Adjacency matrix (batch_size, n_points, n_points)
            
        Returns:
            Encoded features (batch_size, n_points, output_dim)
        """
        batch_size, n_points, _ = x.shape
        
        # Lifting to higher dimensional space
        h = self.lifting(x)
        
        # Add positional encoding based on coordinates
        if n_points <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :n_points, :]
            h = h + pos_enc
        
        # Create default adjacency matrix if not provided
        if adj_matrix is None:
            # Simple distance-based adjacency
            adj_matrix = self._create_adjacency_matrix(coords)
        
        # Alternate between graph convolution and attention
        for i, (graph_conv, attention) in enumerate(zip(self.graph_convs, self.attention_layers)):
            # Graph convolution
            h = graph_conv(h, adj_matrix)
            h = F.relu(h)
            
            # Multi-head attention
            h = attention(h, h, h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def _create_adjacency_matrix(self, coords: torch.Tensor, 
                                sigma: float = 1.0) -> torch.Tensor:
        """Create adjacency matrix based on spatial coordinates."""
        batch_size, n_points, coord_dim = coords.shape
        
        # Compute pairwise distances
        coords_expanded = coords.unsqueeze(2)  # (batch, n_points, 1, coord_dim)
        coords_transposed = coords.unsqueeze(1)  # (batch, 1, n_points, coord_dim)
        
        distances = torch.norm(coords_expanded - coords_transposed, dim=-1)
        
        # Gaussian kernel for adjacency
        adj_matrix = torch.exp(-distances**2 / (2 * sigma**2))
        
        # Remove self-loops
        eye = torch.eye(n_points, device=coords.device).unsqueeze(0)
        adj_matrix = adj_matrix * (1 - eye)
        
        return adj_matrix


class AGTNOProcessor(nn.Module):
    """Processor module with spectral layers, nonlinear operations, and physics constraints."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_modes: int = 32, n_layers: int = 4,
                 physics_constraints: Optional[List[PDEConstraint]] = None,
                 coupling_system: Optional[MultiPhysicsCoupling] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Spectral operator layers
        self.spectral_layers = nn.ModuleList([
            create_fourier_spectral_layer(
                n_modes=n_modes, 
                input_dim=hidden_dim, 
                output_dim=hidden_dim,
                n_channels=1
            ) for _ in range(n_layers)
        ])
        
        # Nonlinear activation layers
        self.nonlinear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_layers)
        ])
        
        # Physics constraint layer
        if physics_constraints:
            self.physics_constraint_layer = PhysicsConstraintLayer(physics_constraints)
        else:
            self.physics_constraint_layer = None
        
        # Multi-physics coupling
        self.coupling_system = coupling_system
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                control_params: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Processor forward pass.
        
        Args:
            x: Input features (batch_size, n_points, input_dim)
            coords: Spatial coordinates (batch_size, n_points, coord_dim)
            control_params: Control parameters for operator control
            
        Returns:
            Tuple of (processed_features, losses_dict)
        """
        h = x
        total_losses = {}
        
        # Process through spectral and nonlinear layers
        for i in range(self.n_layers):
            # Spectral operator layer
            spectral_out, spectral_losses = self.spectral_layers[i](coords, control_params)
            
            # Combine with current features
            h = h + spectral_out
            h = self.layer_norms[i](h)
            
            # Nonlinear transformation
            h = h + self.nonlinear_layers[i](h)
            
            # Accumulate losses
            for key, value in spectral_losses.items():
                if key in total_losses:
                    total_losses[key] += value
                else:
                    total_losses[key] = value
        
        # Apply physics constraints
        if self.physics_constraint_layer:
            h, physics_residuals = self.physics_constraint_layer(h, x, coords)
            total_losses.update(physics_residuals)
        
        # Apply multi-physics coupling
        if self.coupling_system:
            # Convert tensor to field dictionary for coupling
            fields = self._tensor_to_fields(h)
            modified_fields, coupling_loss = self.coupling_system(fields, coords)
            h = self._fields_to_tensor(modified_fields, h.shape)
            total_losses['coupling_loss'] = coupling_loss
        
        # Final projection
        output = self.final_proj(h)
        
        return output, total_losses
    
    def _tensor_to_fields(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert tensor to field dictionary for multi-physics coupling."""
        batch_size, n_points, features = tensor.shape
        
        # Simple mapping - in practice would be more sophisticated
        fields = {}
        if features >= 6:
            fields['electric_field'] = tensor[:, :, :3]
            fields['magnetic_field'] = tensor[:, :, 3:6]
        if features >= 7:
            fields['temperature'] = tensor[:, :, 6:7]
        if features >= 10:
            fields['displacement'] = tensor[:, :, 7:10]
        
        return fields
    
    def _fields_to_tensor(self, fields: Dict[str, torch.Tensor], 
                         original_shape: torch.Size) -> torch.Tensor:
        """Convert field dictionary back to tensor."""
        batch_size, n_points, features = original_shape
        
        # Reconstruct tensor from fields
        tensor_parts = []
        
        if 'electric_field' in fields:
            tensor_parts.append(fields['electric_field'])
        if 'magnetic_field' in fields:
            tensor_parts.append(fields['magnetic_field'])
        if 'temperature' in fields:
            tensor_parts.append(fields['temperature'])
        if 'displacement' in fields:
            tensor_parts.append(fields['displacement'])
        
        if tensor_parts:
            reconstructed = torch.cat(tensor_parts, dim=-1)
            # Pad or truncate to match original shape
            if reconstructed.shape[-1] < features:
                padding = torch.zeros(batch_size, n_points, 
                                    features - reconstructed.shape[-1],
                                    device=reconstructed.device)
                reconstructed = torch.cat([reconstructed, padding], dim=-1)
            elif reconstructed.shape[-1] > features:
                reconstructed = reconstructed[:, :, :features]
            return reconstructed
        else:
            return torch.zeros(original_shape, device=fields.get('electric_field', 
                                                               torch.zeros(1)).device)


class AGTNODecoder(nn.Module):
    """Decoder module with projection, conservation laws, and output generation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_classes: int = 10, enforce_conservation: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.enforce_conservation = enforce_conservation
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Classification head for fault detection
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Conservation law enforcement
        if enforce_conservation:
            self.conservation_layer = ConservationLayer(output_dim)
        else:
            self.conservation_layer = None
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Decoder forward pass.
        
        Args:
            x: Input features (batch_size, n_points, input_dim)
            coords: Spatial coordinates (batch_size, n_points, coord_dim)
            
        Returns:
            Tuple of (reconstructed_output, classification_output, losses_dict)
        """
        losses = {}
        
        # Projection to output space
        reconstructed = self.projection(x)
        
        # Apply conservation laws
        if self.conservation_layer:
            reconstructed, conservation_loss = self.conservation_layer(reconstructed, coords)
            losses['conservation_loss'] = conservation_loss
        
        # Classification for fault detection
        # Transpose for AdaptiveAvgPool1d: (batch, features, n_points)
        x_transposed = reconstructed.transpose(1, 2)
        classification = self.classifier(x_transposed)
        
        return reconstructed, classification, losses


class ConservationLayer(nn.Module):
    """Layer to enforce conservation laws in the output."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Learnable conservation weights
        self.conservation_weights = nn.Parameter(torch.ones(feature_dim) * 0.1)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enforce conservation laws.
        
        Args:
            x: Input features (batch_size, n_points, feature_dim)
            coords: Spatial coordinates
            
        Returns:
            Tuple of (conserved_output, conservation_loss)
        """
        # Simple conservation: ensure spatial integral is preserved
        spatial_integral = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, feature_dim)
        
        # Conservation loss: penalize large deviations from mean
        conservation_loss = torch.mean(self.conservation_weights * 
                                     torch.var(x, dim=1))
        
        # Apply soft conservation constraint
        conserved_output = x - 0.1 * (x - spatial_integral)
        
        return conserved_output, conservation_loss


class AGTNO(nn.Module):
    """Complete Adaptive Graph Transformer Neural Operator architecture."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64,
                 n_classes: int = 10, n_modes: int = 32, n_heads: int = 8,
                 n_encoder_layers: int = 4, n_processor_layers: int = 6,
                 physics_constraints: Optional[List[PDEConstraint]] = None,
                 coupling_system: Optional[MultiPhysicsCoupling] = None,
                 edge_optimization: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.edge_optimization = edge_optimization
        
        # Encoder-Processor-Decoder architecture
        self.encoder = AGTNOEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_encoder_layers
        )
        
        self.processor = AGTNOProcessor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_processor_layers,
            physics_constraints=physics_constraints,
            coupling_system=coupling_system
        )
        
        self.decoder = AGTNODecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_classes=n_classes
        )
        
        # Edge deployment optimization
        if edge_optimization:
            self.edge_optimizer = EdgeOptimizer(n_modes, compression_ratio=0.5)
        else:
            self.edge_optimizer = None
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor,
                control_params: Optional[torch.Tensor] = None,
                adj_matrix: Optional[torch.Tensor] = None,
                return_intermediate: bool = False
               ) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
                         Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Complete AGT-NO forward pass.
        
        Args:
            x: Input features (batch_size, n_points, input_dim)
            coords: Spatial coordinates (batch_size, n_points, coord_dim)
            control_params: Control parameters for operator control
            adj_matrix: Adjacency matrix for graph convolution
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Tuple of (reconstructed_output, classification_output, total_losses, [intermediate_outputs])
        """
        total_losses = {}
        intermediate_outputs = {}
        
        # Encoder
        encoded = self.encoder(x, coords, adj_matrix)
        if return_intermediate:
            intermediate_outputs['encoded'] = encoded
        
        # Processor
        processed, processor_losses = self.processor(encoded, coords, control_params)
        total_losses.update(processor_losses)
        if return_intermediate:
            intermediate_outputs['processed'] = processed
        
        # Decoder
        reconstructed, classification, decoder_losses = self.decoder(processed, coords)
        total_losses.update(decoder_losses)
        
        # Edge optimization if enabled
        if self.edge_optimizer and not self.training:
            reconstructed, classification = self.edge_optimizer(reconstructed, classification)
        
        if return_intermediate:
            return reconstructed, classification, total_losses, intermediate_outputs
        else:
            return reconstructed, classification, total_losses
    
    def get_physics_residuals(self, x: torch.Tensor, coords: torch.Tensor
                             ) -> Dict[str, torch.Tensor]:
        """Get physics residuals for analysis."""
        with torch.no_grad():
            _, _, losses = self.forward(x, coords)
            
            physics_residuals = {}
            for key, value in losses.items():
                if 'physics' in key.lower() or 'constraint' in key.lower():
                    physics_residuals[key] = value
            
            return physics_residuals
    
    def compress_for_edge(self, compression_ratio: float = 0.5) -> 'AGTNO':
        """Create compressed version for edge deployment."""
        if self.edge_optimizer:
            return self.edge_optimizer.compress_model(self, compression_ratio)
        else:
            raise ValueError("Edge optimization not enabled")


class EdgeOptimizer(nn.Module):
    """Edge deployment optimization with model compression."""
    
    def __init__(self, n_modes: int, compression_ratio: float = 0.5):
        super().__init__()
        self.n_modes = n_modes
        self.compression_ratio = compression_ratio
        self.compressed_modes = int(n_modes * compression_ratio)
        
        # Learnable compression matrix
        self.compression_matrix = nn.Parameter(
            torch.randn(n_modes, self.compressed_modes) * 0.1
        )
    
    def forward(self, reconstructed: torch.Tensor, classification: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply edge optimization during inference."""
        # Simple optimization: reduce precision for edge deployment
        if not self.training:
            # Quantize to reduce memory and computation
            reconstructed = torch.round(reconstructed * 128) / 128
            classification = torch.round(classification * 128) / 128
        
        return reconstructed, classification
    
    def compress_model(self, model: AGTNO, compression_ratio: float) -> AGTNO:
        """Create compressed version of the model."""
        # This would implement actual model compression techniques
        # For now, return the original model
        return model


def create_motor_agtno(input_dim: int = 12, n_classes: int = 10,
                      physics_constraints: Optional[List[PDEConstraint]] = None,
                      coupling_system: Optional[MultiPhysicsCoupling] = None
                     ) -> AGTNO:
    """Create AGT-NO model specifically configured for motor fault diagnosis.
    
    Args:
        input_dim: Input feature dimension (vibration signals, current, voltage, etc.)
        n_classes: Number of fault classes
        physics_constraints: List of physics constraints to enforce
        coupling_system: Multi-physics coupling system
        
    Returns:
        Configured AGT-NO model
    """
    return AGTNO(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=64,
        n_classes=n_classes,
        n_modes=32,
        n_heads=8,
        n_encoder_layers=4,
        n_processor_layers=6,
        physics_constraints=physics_constraints,
        coupling_system=coupling_system,
        edge_optimization=True
    )