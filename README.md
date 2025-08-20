# hdb-resale-prediction
Deep learning model predicting Singapore HDB resale prices with 90% accuracy

# Singapore HDB Resale Price Prediction

A deep learning project predicting Singapore HDB (public housing) resale prices using PyTorch neural networks with categorical embeddings.

## Project Overview

This project builds a production-ready neural network to predict HDB resale prices in Singapore with **90% accuracy (R²)** and **8% mean absolute percentage error**. The model uses advanced categorical embeddings and continuous features to capture complex market dynamics.

**Key Achievement**: Developed a deep learning model that outperforms traditional regression approaches by handling non-linear relationships in Singapore's unique public housing market.

## Dataset

- **Size**: 213,883 HDB transactions (2017-2025)
- **Coverage**: All 26 HDB towns and 7 flat types
- **Features**: 11 original variables including location, size, lease terms, and transaction timing
- **Quality**: Clean dataset with no missing values

## Technical Architecture

### Model Design
- **Framework**: PyTorch with custom neural network architecture
- **Approach**: Categorical embeddings + continuous features
- **Architecture**: 
 - Input: 43 embedding dimensions + 5 continuous features
 - Hidden: [128 → 64 → 32] with ReLU activation and dropout (0.2)
 - Output: Single price prediction

### Feature Engineering
**Categorical Features (Embeddings):**
- Town (26 categories → 15-dim embedding)
- Flat Type (7 categories → 6-dim embedding)
- Storey Range (17 categories → 10-dim embedding)
- Flat Model (21 categories → 12-dim embedding)

**Continuous Features (Normalized):**
- Floor area, remaining lease duration, building age, transaction year/month

### Training Strategy
- **Data Split**: Time-based (2017-2022 train, 2023 validation, 2024 test)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Early stopping + L2 weight decay
- **Batch Size**: 512 for optimal GPU utilization

## Model Performance

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **R² Score** | 0.900 | > 0.85 (Excellent) |
| **RMSE** | $59,000 | ~10-12% of avg price |
| **MAPE** | 8.0% | < 10% (Very Good) |
| **MAE** | $45,000 | Competitive |

## Project Structure

**Google Colab Notebooks:**
- `HDB_resale_analysis.ipynb` - Exploratory data analysis and market insights
- `HDB_resale_preprocessing.ipynb` - Feature engineering and data preparation  
- `HDB_resale_model.ipynb` - PyTorch model development and training

**Data:**
- HDB resale price dataset (2017-present) stored in Google Drive
- Preprocessed arrays saved as .npy files for efficient loading

## Key Features

### Advanced ML Techniques
- **Categorical Embeddings**: Learned dense representations for categorical features
- **Custom PyTorch Architecture**: Hand-designed network optimized for tabular data  
- **Time-based Validation**: Realistic evaluation using temporal splits
- **Automated Model Selection**: Early stopping with best model checkpointing

### Engineering Best Practices
- **Google Colab Integration**: Seamless cloud-based development environment
- **Drive Integration**: Persistent data storage across sessions
- **Reproducible Results**: Fixed random seeds and documented hyperparameters
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Memory Efficient**: Optimized data loading with PyTorch DataLoaders

## Requirements
`
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
`

## Quick Start

1. **Open in Google Colab**
   - Upload notebooks to Google Drive
   - Open with Google Colab
   - Mount Google Drive for data access

2. **Run Notebooks in Order**
   - Start with `HDB_resale_analysis.ipynb` [![Open Analysis](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoshTKx/hdb-resale-prediction/blob/main/notebooks/HDB_resale_analysis.ipynb) for data exploration 
   - Continue with `HDB_resale_preprocessing.ipynb` [![Open Preprocessing](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoshTKx/hdb-resale-prediction/blob/main/notebooks/HDB_resale_preprocessing.ipynb) for feature engineering
   - Finish with `HDB_resale_model.ipynb` [![Open Model Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoshTKx/hdb-resale-prediction/blob/main/notebooks/HDB_resale_model.ipynb) for model training

3. **Data Setup**
   - Upload HDB dataset to Google Drive
   - Update file paths in notebooks to match your Drive structure
   
   
## Key Insights

### Market Analysis Findings
- **Location Premium**: Bukit Timah commands highest prices, Yishun lowest
- **Size-Price Relationship**: Strong positive correlation with floor area
- **Age Impact**: Non-linear relationship - newer buildings (post-2010) show price premium
- **Policy Effects**: 2018-2019 price corrections visible from cooling measures

### Model Insights
- **Complex Lease Patterns**: Multiple price peaks justify neural network over linear models
- **Embedding Effectiveness**: Categorical embeddings capture neighborhood similarities
- **Seasonal Patterns**: Monthly features improve prediction accuracy

## Future Enhancements

- Geospatial Features: Integration of MRT distance, schools, amenities
- Model Comparison: Benchmark against XGBoost/LightGBM
- Ensemble Methods: Combine multiple model predictions
- Model Interpretability: SHAP analysis for feature importance
- Real-time Deployment: API endpoint for live predictions

## Technical Skills Demonstrated

- **Deep Learning**: PyTorch, neural network design, embedding layers
- **Data Science**: EDA, feature engineering, time series validation
- **Cloud Computing**: Google Colab, Drive integration, GPU utilization
- **MLOps**: Model checkpointing, hyperparameter tuning, evaluation pipelines

## Development Environment

**Platform**: Google Colab Pro (recommended for GPU access)
**Runtime**: Python 3.x with GPU acceleration
**Storage**: Google Drive for persistent data storage
**Dependencies**: Automatically installed via pip in notebook cells

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

*This project demonstrates advanced machine learning techniques applied to real-world Singapore housing market data, showcasing both technical depth and practical business value.*