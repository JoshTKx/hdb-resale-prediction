"""Configuration settings for HDB price prediction model."""

# Model Architecture
MODEL_CONFIG = {
    'embedding_sizes': [
        (26, 15),  # town
        (7, 6),    # flat_type  
        (17, 10),  # storey_range
        (21, 12)   # flat_model
    ],
    'n_continuous': 5,
    'hidden_sizes': [128, 64, 32],
    'dropout_rate': 0.2
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 512,
    'learning_rate': 0.01,
    'epochs': 100,
    'patience': 15,
    'weight_decay': 1e-5
}

# Data Configuration
DATA_CONFIG = {
    'categorical_features': ['town', 'flat_type', 'storey_range', 'flat_model'],
    'continuous_features': ['remaining_lease_months', 'floor_area_sqm', 'year', 'month_num', 'building_age'],
    'target': 'resale_price',
    'train_years': list(range(2017, 2023)),
    'val_years': [2023],
    'test_years': [2024]
}

# Paths
PATHS = {
    'raw_data': 'data/sample_hdb_data.csv',
    'models': 'models/',
    'results': 'results/'
}