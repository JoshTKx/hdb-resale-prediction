import torch
import torch.nn as nn
import os

class HDBPricePredictor(nn.Module):
  def __init__(self, embedding_sizes, n_continuous, hidden_sizes=[64, 32], dropout_rate=0.2):
    super().__init__()

    # Store architecture info for debugging
    self.embedding_sizes = embedding_sizes
    self.n_continuous = n_continuous
    self.hidden_sizes = hidden_sizes
    self.dropout_rate = dropout_rate

    self.embeddings = nn.ModuleList([
        nn.Embedding(categories, size) for categories, size in embedding_sizes
    ])

    total_embedding_dim = sum(size for _, size in embedding_sizes)
    input_size = total_embedding_dim + n_continuous

    layers = []
    prev_size = input_size

    for hidden_size in self.hidden_sizes:
      layers.extend([
        nn.Linear(prev_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(self.dropout_rate)
      ])
      prev_size = hidden_size

    layers.append(nn.Linear(prev_size, 1))

    self.main_network = nn.Sequential(*layers)

    print(f"Model created:")
    print(f"  - Embedding dimensions: {total_embedding_dim}")
    print(f"  - Continuous features: {n_continuous}")
    print(f"  - Total input size: {input_size}")
    print(f"  - Hidden layers: {hidden_sizes}")
    print(f"  - Dropout rate: {dropout_rate}")

  def forward(self, categorical, continuous):
    # 1. Get embeddings for each categorical feature
    embedded_features = []
    for i, embedding_layer in enumerate(self.embeddings):
      embedded = embedding_layer(categorical[:, i])
      embedded_features.append(embedded)

    # 2. Concatenate all embeddings
    if embedded_features:
      embedded_concat = torch.cat(embedded_features, dim=1)
      # 3. Concatenate with continuous features
      combined = torch.cat([embedded_concat, continuous], dim=1)
    else:
      combined = continuous

    # 4. Pass through main network
    output = self.main_network(combined)

    return output.squeeze()  # Remove extra dimension

  def save_model(self, filepath, optimizer=None, epoch=None, metrics=None):
    """Save complete model checkpoint with architecture info"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': self.state_dict(),
        'model_config': {
            'embedding_sizes': self.embedding_sizes,
            'n_continuous': self.n_continuous,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate
        },
        'metrics': metrics or {},
        'epoch': epoch
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")

  @classmethod
  def load_model(cls, filepath, device=None):
    """Load model from checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['model_config']

    model = cls(
        embedding_sizes=config['embedding_sizes'],
        n_continuous=config['n_continuous'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint