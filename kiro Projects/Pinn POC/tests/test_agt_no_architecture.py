"""
Unit tests for complete AGT-NO architecture.

Tests encoder-processor-decoder integration, physics constraint enforcement,
and edge deployment optimization.
"""

import torch
import pytest
import numpy as np
from src.physics.agt_no_architecture import (
    GraphConvolution, MultiHeadAttention, AGTNOEncoder, AGTNOProcessor,
    AGTNODecoder, ConservationLayer, AGTNO, EdgeOptimizer, create_motor_agtno
)
from src.physics.constraints import MaxwellConstraint, HeatEquationConstraint
from src.physics.multi_physics_coupling import create_motor_coupling_system


class TestGraphConvolution:
    """Test graph convolution layer."""
    
    def test_graph_conv_initialization(self):
        """Test graph convolution initialization."""
        conv = GraphConvolution(in_features=64, out_features=128, dropout=0.1)
        assert conv.in_features == 64
        assert conv.out_features == 128
        assert conv.weight.shape == (64, 128)
        assert conv.bias.shape == (128,)
    
    def test_graph_conv_forward(self):
        """Test graph convolution forward pass."""
        conv = GraphConvolution(in_features=32, out_features=64)
        
        batch_size, n_nodes = 2, 10
        input_features = torch.randn(batch_size, n_nodes, 32)
        adj_matrix = torch.randn(batch_size, n_nodes, n_nodes)
        
        output = conv(input_features, adj_matrix)
        
        assert output.shape == (batch_size, n_nodes, 64)
    
    def test_graph_conv_gradient_flow(self):
        """Test gradient flow through graph convolution."""
        conv = GraphConvolution(in_features=16, out_features=32)
        
        input_features = torch.randn(1, 5, 16, requires_grad=True)
        adj_matrix = torch.randn(1, 5, 5)
        
        output = conv(input_features, adj_matrix)
        loss = torch.sum(output**2)
        loss.backward()
        
        assert input_features.grad is not None
        assert conv.weight.grad is not None


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""
    
    def test_attention_initialization(self):
        """Test multi-head attention initialization."""
        attention = MultiHeadAttention(d_model=256, n_heads=8)
        assert attention.d_model == 256
        assert attention.n_heads == 8
        assert attention.d_k == 32  # 256 / 8
    
    def test_attention_forward(self):
        """Test multi-head attention forward pass."""
        attention = MultiHeadAttention(d_model=128, n_heads=4)
        
        batch_size, seq_len = 2, 20
        query = torch.randn(batch_size, seq_len, 128)
        key = torch.randn(batch_size, seq_len, 128)
        value = torch.randn(batch_size, seq_len, 128)
        
        output = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, 128)
    
    def test_attention_with_mask(self):
        """Test multi-head attention with mask."""
        attention = MultiHeadAttention(d_model=64, n_heads=2)
        
        batch_size, seq_len = 1, 10
        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)
        mask = torch.ones(batch_size, 2, seq_len, seq_len)  # n_heads included
        
        output = attention(query, key, value, mask)
        assert output.shape == (batch_size, seq_len, 64)


class TestAGTNOEncoder:
    """Test AGT-NO encoder module."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = AGTNOEncoder(
            input_dim=12, hidden_dim=128, output_dim=64,
            n_heads=4, n_layers=4
        )
        assert encoder.input_dim == 12
        assert encoder.hidden_dim == 128
        assert encoder.output_dim == 64
        assert len(encoder.graph_convs) == 2  # n_layers // 2
        assert len(encoder.attention_layers) == 2
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = AGTNOEncoder(input_dim=8, hidden_dim=64, output_dim=32)
        
        batch_size, n_points = 2, 15
        x = torch.randn(batch_size, n_points, 8)
        coords = torch.randn(batch_size, n_points, 2)
        
        output = encoder(x, coords)
        
        assert output.shape == (batch_size, n_points, 32)
    
    def test_encoder_with_adjacency(self):
        """Test encoder with custom adjacency matrix."""
        encoder = AGTNOEncoder(input_dim=6, hidden_dim=32, output_dim=16)
        
        batch_size, n_points = 1, 8
        x = torch.randn(batch_size, n_points, 6)
        coords = torch.randn(batch_size, n_points, 2)
        adj_matrix = torch.randn(batch_size, n_points, n_points)
        
        output = encoder(x, coords, adj_matrix)
        assert output.shape == (batch_size, n_points, 16)
    
    def test_adjacency_matrix_creation(self):
        """Test automatic adjacency matrix creation."""
        encoder = AGTNOEncoder(input_dim=4, hidden_dim=32, output_dim=16)
        
        coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        adj_matrix = encoder._create_adjacency_matrix(coords)
        
        assert adj_matrix.shape == (1, 3, 3)
        # Diagonal should be zero (no self-loops)
        assert torch.allclose(torch.diag(adj_matrix[0]), torch.zeros(3))


class TestAGTNOProcessor:
    """Test AGT-NO processor module."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = AGTNOProcessor(
            input_dim=64, hidden_dim=128, output_dim=64,
            n_modes=16, n_layers=4
        )
        assert processor.input_dim == 64
        assert processor.hidden_dim == 128
        assert processor.output_dim == 64
        assert len(processor.spectral_layers) == 4
        assert len(processor.nonlinear_layers) == 4
    
    def test_processor_forward(self):
        """Test processor forward pass."""
        processor = AGTNOProcessor(
            input_dim=32, hidden_dim=64, output_dim=32,
            n_modes=8, n_layers=2
        )
        
        batch_size, n_points = 2, 10
        x = torch.randn(batch_size, n_points, 32)
        coords = torch.randn(batch_size, n_points, 2)
        
        output, losses = processor(x, coords)
        
        assert output.shape == (batch_size, n_points, 32)
        assert isinstance(losses, dict)
    
    def test_processor_with_physics_constraints(self):
        """Test processor with physics constraints."""
        constraints = [MaxwellConstraint(weight=1.0)]
        processor = AGTNOProcessor(
            input_dim=32, hidden_dim=64, output_dim=32,
            physics_constraints=constraints
        )
        
        batch_size, n_points = 1, 8
        x = torch.randn(batch_size, n_points, 32)
        coords = torch.randn(batch_size, n_points, 2)
        
        output, losses = processor(x, coords)
        
        assert output.shape == (batch_size, n_points, 32)
        assert 'total_physics_loss' in losses
    
    def test_processor_with_coupling(self):
        """Test processor with multi-physics coupling."""
        coupling_system = create_motor_coupling_system()
        processor = AGTNOProcessor(
            input_dim=32, hidden_dim=64, output_dim=32,
            coupling_system=coupling_system
        )
        
        batch_size, n_points = 1, 6
        x = torch.randn(batch_size, n_points, 32)
        coords = torch.randn(batch_size, n_points, 2)
        
        output, losses = processor(x, coords)
        
        assert output.shape == (batch_size, n_points, 32)
        assert 'coupling_loss' in losses
    
    def test_tensor_field_conversion(self):
        """Test tensor to field dictionary conversion."""
        processor = AGTNOProcessor(input_dim=16, hidden_dim=32, output_dim=16)
        
        tensor = torch.randn(2, 5, 10)  # 10 features
        fields = processor._tensor_to_fields(tensor)
        
        assert 'electric_field' in fields
        assert 'magnetic_field' in fields
        assert 'temperature' in fields
        assert 'displacement' in fields
        
        # Test conversion back
        reconstructed = processor._fields_to_tensor(fields, tensor.shape)
        assert reconstructed.shape == tensor.shape


class TestAGTNODecoder:
    """Test AGT-NO decoder module."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = AGTNODecoder(
            input_dim=64, hidden_dim=128, output_dim=32,
            n_classes=10, enforce_conservation=True
        )
        assert decoder.input_dim == 64
        assert decoder.output_dim == 32
        assert decoder.n_classes == 10
        assert decoder.conservation_layer is not None
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = AGTNODecoder(input_dim=32, hidden_dim=64, output_dim=16, n_classes=5)
        
        batch_size, n_points = 2, 12
        x = torch.randn(batch_size, n_points, 32)
        coords = torch.randn(batch_size, n_points, 2)
        
        reconstructed, classification, losses = decoder(x, coords)
        
        assert reconstructed.shape == (batch_size, n_points, 16)
        assert classification.shape == (batch_size, 5)
        assert isinstance(losses, dict)
    
    def test_decoder_without_conservation(self):
        """Test decoder without conservation laws."""
        decoder = AGTNODecoder(
            input_dim=16, hidden_dim=32, output_dim=8,
            n_classes=3, enforce_conservation=False
        )
        
        x = torch.randn(1, 8, 16)
        coords = torch.randn(1, 8, 2)
        
        reconstructed, classification, losses = decoder(x, coords)
        
        assert reconstructed.shape == (1, 8, 8)
        assert classification.shape == (1, 3)
        assert 'conservation_loss' not in losses


class TestConservationLayer:
    """Test conservation layer."""
    
    def test_conservation_layer_initialization(self):
        """Test conservation layer initialization."""
        layer = ConservationLayer(feature_dim=32)
        assert layer.feature_dim == 32
        assert layer.conservation_weights.shape == (32,)
    
    def test_conservation_layer_forward(self):
        """Test conservation layer forward pass."""
        layer = ConservationLayer(feature_dim=16)
        
        x = torch.randn(2, 10, 16)
        coords = torch.randn(2, 10, 2)
        
        conserved_output, conservation_loss = layer(x, coords)
        
        assert conserved_output.shape == x.shape
        assert isinstance(conservation_loss, torch.Tensor)
        assert conservation_loss >= 0


class TestCompleteAGTNO:
    """Test complete AGT-NO architecture."""
    
    def test_agtno_initialization(self):
        """Test complete AGT-NO initialization."""
        model = AGTNO(
            input_dim=12, hidden_dim=128, output_dim=32,
            n_classes=8, n_modes=16
        )
        assert model.input_dim == 12
        assert model.hidden_dim == 128
        assert model.output_dim == 32
        assert model.n_classes == 8
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'processor')
        assert hasattr(model, 'decoder')
    
    def test_agtno_forward(self):
        """Test complete AGT-NO forward pass."""
        model = AGTNO(input_dim=8, hidden_dim=64, output_dim=16, n_classes=5)
        
        batch_size, n_points = 2, 15
        x = torch.randn(batch_size, n_points, 8)
        coords = torch.randn(batch_size, n_points, 2)
        
        reconstructed, classification, losses = model(x, coords)
        
        assert reconstructed.shape == (batch_size, n_points, 16)
        assert classification.shape == (batch_size, 5)
        assert isinstance(losses, dict)
    
    def test_agtno_with_intermediate_outputs(self):
        """Test AGT-NO with intermediate output return."""
        model = AGTNO(input_dim=6, hidden_dim=32, output_dim=8, n_classes=3)
        
        x = torch.randn(1, 10, 6)
        coords = torch.randn(1, 10, 2)
        
        reconstructed, classification, losses, intermediate = model(
            x, coords, return_intermediate=True
        )
        
        assert 'encoded' in intermediate
        assert 'processed' in intermediate
        assert intermediate['encoded'].shape == (1, 10, 32)
        assert intermediate['processed'].shape == (1, 10, 32)
    
    def test_agtno_with_physics_constraints(self):
        """Test AGT-NO with physics constraints."""
        constraints = [MaxwellConstraint(weight=1.0), HeatEquationConstraint(weight=0.5)]
        model = AGTNO(
            input_dim=10, hidden_dim=64, output_dim=16,
            physics_constraints=constraints
        )
        
        x = torch.randn(1, 8, 10)
        coords = torch.randn(1, 8, 2)
        
        reconstructed, classification, losses = model(x, coords)
        
        assert 'total_physics_loss' in losses
    
    def test_agtno_with_coupling_system(self):
        """Test AGT-NO with multi-physics coupling."""
        coupling_system = create_motor_coupling_system()
        model = AGTNO(
            input_dim=12, hidden_dim=64, output_dim=16,
            coupling_system=coupling_system
        )
        
        x = torch.randn(1, 6, 12)
        coords = torch.randn(1, 6, 2)
        
        reconstructed, classification, losses = model(x, coords)
        
        assert 'coupling_loss' in losses
    
    def test_physics_residuals_extraction(self):
        """Test physics residuals extraction."""
        constraints = [MaxwellConstraint(weight=1.0)]
        model = AGTNO(input_dim=8, physics_constraints=constraints)
        
        x = torch.randn(1, 5, 8)
        coords = torch.randn(1, 5, 2)
        
        residuals = model.get_physics_residuals(x, coords)
        
        assert isinstance(residuals, dict)
        # Should contain physics-related losses
        physics_keys = [k for k in residuals.keys() if 'physics' in k.lower()]
        assert len(physics_keys) > 0
    
    def test_gradient_flow_through_complete_model(self):
        """Test gradient flow through complete AGT-NO."""
        model = AGTNO(input_dim=6, hidden_dim=32, output_dim=8, n_classes=3)
        
        x = torch.randn(1, 8, 6, requires_grad=True)
        coords = torch.randn(1, 8, 2)
        
        reconstructed, classification, losses = model(x, coords)
        
        # Compute total loss
        total_loss = torch.sum(reconstructed**2) + torch.sum(classification**2)
        for loss_value in losses.values():
            if isinstance(loss_value, torch.Tensor):
                total_loss += loss_value
        
        total_loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestEdgeOptimizer:
    """Test edge deployment optimizer."""
    
    def test_edge_optimizer_initialization(self):
        """Test edge optimizer initialization."""
        optimizer = EdgeOptimizer(n_modes=32, compression_ratio=0.5)
        assert optimizer.n_modes == 32
        assert optimizer.compression_ratio == 0.5
        assert optimizer.compressed_modes == 16
    
    def test_edge_optimizer_forward(self):
        """Test edge optimizer forward pass."""
        optimizer = EdgeOptimizer(n_modes=16, compression_ratio=0.75)
        optimizer.eval()  # Set to evaluation mode
        
        reconstructed = torch.randn(2, 10, 8)
        classification = torch.randn(2, 5)
        
        opt_reconstructed, opt_classification = optimizer(reconstructed, classification)
        
        assert opt_reconstructed.shape == reconstructed.shape
        assert opt_classification.shape == classification.shape
    
    def test_edge_optimization_in_agtno(self):
        """Test edge optimization integration in AGT-NO."""
        model = AGTNO(input_dim=6, edge_optimization=True)
        model.eval()  # Set to evaluation mode for edge optimization
        
        x = torch.randn(1, 8, 6)
        coords = torch.randn(1, 8, 2)
        
        reconstructed, classification, losses = model(x, coords)
        
        # Should work without errors
        assert reconstructed.shape[0] == 1
        assert classification.shape[0] == 1


class TestMotorAGTNOCreation:
    """Test motor-specific AGT-NO creation."""
    
    def test_create_motor_agtno(self):
        """Test creation of motor-specific AGT-NO."""
        model = create_motor_agtno(input_dim=12, n_classes=8)
        
        assert isinstance(model, AGTNO)
        assert model.input_dim == 12
        assert model.n_classes == 8
        assert model.edge_optimization
    
    def test_create_motor_agtno_with_physics(self):
        """Test motor AGT-NO creation with physics constraints."""
        constraints = [MaxwellConstraint(weight=1.0)]
        coupling_system = create_motor_coupling_system()
        
        model = create_motor_agtno(
            input_dim=10, n_classes=6,
            physics_constraints=constraints,
            coupling_system=coupling_system
        )
        
        assert isinstance(model, AGTNO)
        assert model.processor.physics_constraint_layer is not None
        assert model.processor.coupling_system is not None
    
    def test_motor_agtno_inference(self):
        """Test motor AGT-NO inference."""
        model = create_motor_agtno(input_dim=8, n_classes=4)
        model.eval()
        
        # Simulate motor sensor data
        batch_size, n_sensors = 2, 12
        sensor_data = torch.randn(batch_size, n_sensors, 8)
        sensor_coords = torch.randn(batch_size, n_sensors, 2)
        
        with torch.no_grad():
            reconstructed, fault_classification, losses = model(sensor_data, sensor_coords)
        
        assert reconstructed.shape == (batch_size, n_sensors, 64)
        assert fault_classification.shape == (batch_size, 4)
        
        # Check fault probabilities sum to reasonable values
        fault_probs = torch.softmax(fault_classification, dim=1)
        assert torch.allclose(torch.sum(fault_probs, dim=1), torch.ones(batch_size))


class TestAGTNOIntegration:
    """Test integration between all AGT-NO components."""
    
    def test_end_to_end_training_step(self):
        """Test end-to-end training step."""
        model = AGTNO(input_dim=8, hidden_dim=32, output_dim=16, n_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training data
        x = torch.randn(2, 10, 8)
        coords = torch.randn(2, 10, 2)
        target_reconstruction = torch.randn(2, 10, 16)
        target_classification = torch.randint(0, 4, (2,))
        
        # Forward pass
        reconstructed, classification, losses = model(x, coords)
        
        # Compute losses
        recon_loss = F.mse_loss(reconstructed, target_reconstruction)
        class_loss = F.cross_entropy(classification, target_classification)
        
        total_loss = recon_loss + class_loss
        for loss_value in losses.values():
            if isinstance(loss_value, torch.Tensor):
                total_loss += 0.1 * loss_value
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert total_loss.item() >= 0
    
    def test_model_state_dict_save_load(self):
        """Test model state dict saving and loading."""
        model1 = AGTNO(input_dim=6, hidden_dim=32, output_dim=8, n_classes=3)
        
        # Save state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state dict
        model2 = AGTNO(input_dim=6, hidden_dim=32, output_dim=8, n_classes=3)
        model2.load_state_dict(state_dict)
        
        # Test that models produce same output
        x = torch.randn(1, 5, 6)
        coords = torch.randn(1, 5, 2)
        
        with torch.no_grad():
            out1, class1, _ = model1(x, coords)
            out2, class2, _ = model2(x, coords)
        
        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(class1, class2, atol=1e-6)
    
    def test_model_performance_metrics(self):
        """Test model performance measurement."""
        model = create_motor_agtno(input_dim=10, n_classes=5)
        
        # Measure inference time
        x = torch.randn(1, 20, 10)
        coords = torch.randn(1, 20, 2)
        
        import time
        
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _, _, _ = model(x, coords)
            end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10
        
        # Should be reasonably fast (less than 100ms for this small example)
        assert avg_inference_time < 0.1
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable


if __name__ == "__main__":
    pytest.main([__file__])