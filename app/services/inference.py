import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import RobustScaler

class InferenceService:
    """Inference service for predictions"""
    
    def __init__(self, model, scaler: RobustScaler = None):
        self.model = model
        self.scaler = scaler
        self.feature_columns = [
            'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
            'pool', 'fireplace', 'garage_spaces', 'condition_score',
            'age', 'age_squared', 'sqft_bedroom_ratio', 'bath_bedroom_ratio'
        ]
    
    def preprocess(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw input data into features"""
        
        # Extract features
        features = {
            'square_feet': raw_data.get('square_feet', 0),
            'bedrooms': raw_data.get('bedrooms', 0),
            'bathrooms': raw_data.get('bathrooms', 0),
            'year_built': raw_data.get('year_built', 2000),
            'lot_size': raw_data.get('lot_size', raw_data.get('square_feet', 0) * 0.2),
            'pool': 1 if raw_data.get('pool', False) else 0,
            'fireplace': 1 if raw_data.get('fireplace', False) else 0,
            'garage_spaces': raw_data.get('garage_spaces', 0),
            'condition_score': raw_data.get('condition_score', 5),
        }
        
        # Derived features
        features['age'] = 2024 - features['year_built']
        features['age_squared'] = features['age'] ** 2
        features['sqft_bedroom_ratio'] = features['square_feet'] / (features['bedrooms'] + 1)
        features['bath_bedroom_ratio'] = features['bathrooms'] / (features['bedrooms'] + 1)
        
        # Create feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns])
        
        # Scale if scaler is available
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return feature_vector
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """Make prediction with confidence intervals"""
        
        # Log transformation prediction
        pred_log = self.model.predict(features.reshape(1, -1))[0]
        
        # Inverse transform
        price = np.expm1(pred_log)
        
        # Simple confidence interval (based on prediction variance)
        ci_width = price * 0.15  # 15% confidence interval
        
        return {
            'price': price,
            'ci_lower': price - ci_width,
            'ci_upper': price + ci_width
        }