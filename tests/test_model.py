import pytest
import torch
import numpy as np
from src.model import HDBPricePredictor
import os

def test_model_initialization():
    """Test model can be initialized with correct architecture."""
    embedding_sizes = [(26, 15), (7, 6), (17, 10), (21, 12)]
    n_continuous = 5
    
    model = HDBPricePredictor(
        embedding_sizes=embedding_sizes,
        n_continuous=n_continuous,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.2
    )
    
    # Check model parameters exist
    assert sum(p.numel() for p in model.parameters()) > 0
    assert len(model.embeddings) == 4

def test_model_forward_pass():
    """Test model can process input tensors."""
    model = HDBPricePredictor(
        embedding_sizes=[(26, 15), (7, 6), (17, 10), (21, 12)],
        n_continuous=5
    )
    
    # Create dummy input
    batch_size = 32
    categorical = torch.cat([
        torch.randint(0, 26, (batch_size, 1)),  # For (26, 15)
        torch.randint(0, 7, (batch_size, 1)),   # For (7, 6)
        torch.randint(0, 17, (batch_size, 1)),  # For (17, 10)
        torch.randint(0, 21, (batch_size, 1))   # For (21, 12)
    ], dim=1)
    continuous = torch.randn(batch_size, 5)
    
    # Forward pass
    output = model(categorical, continuous)
    
    assert output.shape == (batch_size,)
    assert not torch.isnan(output).any()

def test_model_save_load():
    """Test model can be saved and loaded."""
    model = HDBPricePredictor(
        embedding_sizes=[(26, 15), (7, 6), (17, 10), (21, 12)],
        n_continuous=5
    )
    
   # Use a temporary directory for the test file
    temp_dir = 'test_checkpoints'
    temp_file = os.path.join(temp_dir, 'test_model.pth')
    
    try:
        # Save model
        model.save_model(temp_file, epoch=1, metrics={'rmse': 50000})
        
        # Load model
        loaded_model, checkpoint = HDBPricePredictor.load_model(temp_file)
        
        # Assertions
        assert checkpoint['epoch'] == 1
        assert checkpoint['metrics']['rmse'] == 50000
        assert loaded_model.embedding_sizes == model.embedding_sizes
    
    finally:
        # Clean up the test file and directory
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    pytest.main([__file__])