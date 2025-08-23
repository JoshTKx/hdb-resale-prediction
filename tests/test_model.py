import pytest
import torch
import numpy as np
from src.model import HDBPricePredictor

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
    categorical = torch.randint(0, 10, (batch_size, 4))
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
    
    # Save model
    model.save_model('test_model.pth', epoch=1, metrics={'rmse': 50000})
    
    # Load model
    loaded_model, checkpoint = HDBPricePredictor.load_model('test_model.pth')
    
    assert checkpoint['epoch'] == 1
    assert checkpoint['metrics']['rmse'] == 50000
    assert loaded_model.embedding_sizes == model.embedding_sizes

if __name__ == "__main__":
    pytest.main([__file__])