# Technical Methodology

## Model Architecture Design

### Why Neural Networks for Tabular Data?

Traditional regression models assume linear relationships, but HDB pricing exhibits complex non-linear patterns:
- **Remaining Lease**: Multiple price peaks at different lease durations
- **Location Effects**: Neighborhood similarities not captured by simple encoding
- **Temporal Patterns**: Policy changes create non-linear time effects

### Categorical Embeddings Approach

Instead of one-hot encoding (sparse, high-dimensional), we use learned embeddings:

**Traditional One-Hot Encoding:**
- Town: 26 binary features (mostly zeros)
- No relationship learning between similar areas

**Embedding Approach:**
- Town: 15-dimensional dense vectors
- Similar neighborhoods learn similar representations
- Reduces dimensionality while capturing relationships

### Architecture Decisions

**Input Layer Design:**
Categorical Features → Embeddings → Concatenate
Continuous Features → Normalization → Concatenate
Combined Features → Dense Network

**Hidden Layer Configuration:**
- **[128, 64, 32]**: Funnel architecture for feature compression
- **ReLU Activation**: Handles non-linear patterns
- **Dropout (0.2)**: Prevents overfitting on 200K+ dataset

**Output Layer:**
- Single neuron for price regression
- No activation (linear output for price prediction)

## Training Strategy

### Time-Based Validation
**Why Not Random Split?**
- Real estate has temporal trends
- Random splits create data leakage (future predicts past)
- Time-based splits simulate real deployment

**Split Strategy:**
- **Train**: 2017-2022 (143K samples) - Historical patterns
- **Validation**: 2023 (26K samples) - Recent validation
- **Test**: 2024 (28K samples) - Future prediction

### Optimization Details

**Adam Optimizer:**
- Adaptive learning rates per parameter
- Better convergence on sparse categorical data
- Learning rate: 0.01 → 0.005 → 0.0025 (scheduled reduction)

**Learning Rate Scheduling:**
- ReduceLROnPlateau: Cut LR when validation plateaus
- Patience: 7 epochs before reduction
- Factor: 0.5 reduction

**Early Stopping:**
- Monitor validation loss (not training loss)
- Patience: 15 epochs
- Saves best model weights automatically

### Regularization Techniques

**Dropout (0.2):**
- Applied after each hidden layer
- Prevents complex co-adaptations
- Essential for 200K+ dataset

**L2 Weight Decay (1e-5):**
- Penalizes large weights
- Prevents overfitting to outliers
- Built into Adam optimizer

## Feature Engineering Philosophy

### Categorical Feature Selection
**Included:**
- `town`: Geographic premium effects
- `flat_type`: Size and layout impact
- `storey_range`: Height preference patterns
- `flat_model`: Design and age proxy

**Excluded:**
- `block`: Too granular (2,743 unique values)
- `street_name`: Redundant with town information

### Continuous Feature Engineering
**Temporal Features:**
- `year`: Market trends and policy effects
- `month_num`: Seasonal buying patterns

**Property Features:**
- `floor_area_sqm`: Primary size determinant
- `building_age`: Depreciation effects
- `remaining_lease_months`: Complex non-linear relationship

### Normalization Strategy
**StandardScaler for Continuous Features:**
- Mean centering: Improves gradient descent
- Unit variance: Equal feature importance initially
- Maintains distribution shape (unlike Min-Max scaling)

## Model Evaluation Framework

### Metric Selection
**Primary Metrics:**
- **R² Score**: Variance explained (interpretability)
- **RMSE**: Penalizes large errors (important for expensive outliers)
- **MAPE**: Percentage error (relative performance across price ranges)

**Why Multiple Metrics?**
- R² shows model quality vs baseline
- RMSE gives dollar-amount interpretability
- MAPE normalizes across price ranges

### Validation Strategy
**Hold-Out Validation:**
- Clean temporal split (no data leakage)
- Large validation set (26K samples)
- Representative of deployment conditions

**Performance Benchmarks:**
- R² > 0.85: Excellent (industry standard)
- MAPE < 10%: Very good for real estate
- RMSE < 15% of mean price: Acceptable error range

## Implementation Considerations

### Computational Efficiency
**Batch Size (512):**
- GPU memory optimization
- Stable gradient estimates
- Fast convergence

**DataLoader Configuration:**
- Pin memory: Faster GPU transfer
- Multiple workers: Parallel data loading
- Shuffle training: Prevents order bias

### Reproducibility
**Fixed Random Seeds:**
- PyTorch, NumPy, Python random
- Deterministic GPU operations
- Consistent results across runs

**Model Checkpointing:**
- Save best validation performance
- Enable model recovery
- Prevent training loss from crashes

## Future Architecture Improvements

### Potential Enhancements
1. **Attention Mechanisms**: Weight important features dynamically
2. **Residual Connections**: Enable deeper networks
3. **Batch Normalization**: Stabilize deep network training
4. **Ensemble Methods**: Combine multiple model predictions

### Advanced Techniques
1. **Feature Selection**: Automated feature importance ranking
2. **Hyperparameter Optimization**: Grid search or Bayesian optimization
3. **Cross-Validation**: Time series cross-validation for robust evaluation
4. **Model Interpretation**: SHAP values for feature importance analysis