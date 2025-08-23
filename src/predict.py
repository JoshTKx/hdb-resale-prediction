import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.model import HDBPricePredictor
from src.utils import convert_remaining_lease_to_months, get_device
import json
import os



class HDBPricePredictor_Inference:
    def __init__(self, model_path, params_dir='../data'):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to saved PyTorch model
            params_dir: Directory containing scaler_params.json and encoder_mappings.json
        """
        self.device = torch.device('cpu')
        self.model, _ = HDBPricePredictor.load_model(model_path, self.device)
        self.model.eval()
        
        # Load actual parameters from training
        try:
            with open(f'{params_dir}/scaler_params.json', 'r') as f:
                self.scaler_params = json.load(f)
            
            with open(f'{params_dir}/encoder_mappings.json', 'r') as f:
                self.encoder_mappings = json.load(f)
                
            print("Successfully loaded preprocessing parameters from training")
            print(f"Scaler parameters loaded for: {list(self.scaler_params.keys())}")
            print(f"Encoder mappings loaded for: {list(self.encoder_mappings.keys())}")
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load preprocessing parameters: {e}")
            print("Using default parameters - predictions may be inaccurate!")
    
        
    def list_available_options(self):
        """Show all available options for categorical features."""
        print("Available options for prediction:")
        for feature, mapping in self.encoder_mappings.items():
            print(f"\n{feature.upper()}:")
            options = list(mapping.keys())
            if len(options) <= 10:
                for option in sorted(options):
                    print(f"  - '{option}'")
            else:
                for option in sorted(options)[:10]:
                    print(f"  - '{option}'")
                print(f"  ... and {len(options)-10} more")
    
    def preprocess_input(self, town, flat_type, floor_area, remaining_lease, 
                        storey_range, flat_model, year=2024, month=6):
        """Preprocess single input for prediction."""
        
        # Convert remaining lease to months
        lease_months = convert_remaining_lease_to_months(remaining_lease)
        
        # Calculate building age (rough estimate)
        # If lease started with ~99 years, original start year = current_year - (99 - remaining_years)
        remaining_years = lease_months / 12
        estimated_lease_start = year - (99 - remaining_years)  # Assuming 99-year leases
        building_age = year - estimated_lease_start
        
        # Encode categorical features using actual mappings
        try:
            categorical_features = np.array([
                self.encoder_mappings['town'][town],
                self.encoder_mappings['flat_type'][flat_type],
                self.encoder_mappings['storey_range'][storey_range],
                self.encoder_mappings['flat_model'][flat_model]
            ], dtype=int)
        except KeyError as e:
            print(f"Error: Unknown category {e}")
            print("\nUse predictor.list_available_options() to see valid options")
            raise ValueError(f"Invalid input: {e}")
        
        # Scale continuous features using actual scaler parameters
        continuous_raw = [lease_months, floor_area, year, month, building_age]
        continuous_features = []
        
        feature_names = ['remaining_lease_months', 'floor_area_sqm', 'year', 'month_num', 'building_age']
        for i, feature_name in enumerate(feature_names):
            mean = self.scaler_params[feature_name]['mean']
            std = self.scaler_params[feature_name]['std']
            scaled_value = (continuous_raw[i] - mean) / std
            continuous_features.append(scaled_value)
        
        return categorical_features, np.array(continuous_features)
    
    def predict_price(self, town, flat_type, floor_area, remaining_lease, 
                 storey_range='04 TO 06', flat_model='Improved', year=2024, month=6):
        """
        Predict HDB resale price for given parameters.
        
        Args:
            town: HDB town name (must match training data exactly)
            flat_type: Type of flat ('3 ROOM', '4 ROOM', etc.)
            floor_area: Floor area in sqm
            remaining_lease: Remaining lease string ('65 years 3 months')
            storey_range: Storey range ('04 TO 06')
            flat_model: HDB flat model ('Improved', etc.)
            year: Transaction year
            month: Transaction month
            
        Returns:
            Predicted price in SGD
        """
        
        try:
            cat_features, cont_features = self.preprocess_input(
                town, flat_type, floor_area, remaining_lease, 
                storey_range, flat_model, year, month
            )
            
            # For MPS compatibility, create tensors on CPU first, then move to device
            cat_tensor = torch.tensor(cat_features, dtype=torch.long).unsqueeze(0)
            cont_tensor = torch.tensor(cont_features, dtype=torch.float32).unsqueeze(0)
            
            # Move to device after creation
            cat_tensor = cat_tensor.to(self.device)
            cont_tensor = cont_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(cat_tensor, cont_tensor)
                
            return float(prediction.cpu().numpy())
        
        except Exception as e:
            print(f"Prediction failed: {e}")
            print(f"Debug info:")
            print(f"  Device: {self.device}")
            print(f"  Cat features shape: {cat_features.shape if 'cat_features' in locals() else 'Not created'}")
            print(f"  Cont features shape: {cont_features.shape if 'cont_features' in locals() else 'Not created'}")
            return None

# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = HDBPricePredictor_Inference(
        model_path='../models/best_hdb_model.pth',
        params_dir='../data'
    )
    
    print("=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)
    
    # Test with known good values
    test_cases = [
        {
            'name': 'Bishan 4-Room',
            'town': 'BISHAN',
            'flat_type': '4 ROOM',
            'floor_area': 90,
            'remaining_lease': '65 years 3 months',
            'storey_range': '07 TO 09',
            'flat_model': 'Improved'
        },
        {
            'name': 'Bukit Timah Premium',
            'town': 'BUKIT TIMAH',
            'flat_type': '4 ROOM', 
            'floor_area': 100,
            'remaining_lease': '70 years 6 months',
            'storey_range': '10 TO 12',
            'flat_model': 'Improved'
        },
        {
            'name': 'Ang Mo Kio Budget',
            'town': 'ANG MO KIO',
            'flat_type': '3 ROOM',
            'floor_area': 75,
            'remaining_lease': '60 years 0 months', 
            'storey_range': '04 TO 06',
            'flat_model': 'Improved'
        }
    ]
    
    for case in test_cases:
        name = case.pop('name')
        try:
            price = predictor.predict_price(**case)
            if price:
                print(f"{name}: ${price:,.0f}")
                
                # Show input details
                print(f"  📍 {case['town']}, {case['flat_type']}")
                print(f"  📐 {case['floor_area']}sqm, {case['remaining_lease']}")
                print(f"  🏢 {case['storey_range']}, {case['flat_model']}")
                print()
            else:
                print(f"{name}: Prediction failed")
        except Exception as e:
            print(f"{name}: Error - {e}")