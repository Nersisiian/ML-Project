import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import RobustScaler

class Preprocessor:
    """Feature preprocessing for inference"""
    
    def __init__(self, scaler: RobustScaler = None):
        self.scaler = scaler
        self.fitted = scaler is not None
        
    def transform(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Transform raw input to feature vector"""
        
        # Extract base features
        features = {
            'square_feet': float(raw_data.get('square_feet', 0)),
            'bedrooms': float(raw_data.get('bedrooms', 0)),
            'bathrooms': float(raw_data.get('bathrooms', 0)),
            'year_built': float(raw_data.get('year_built', 2000)),
            'lot_size': float(raw_data.get('lot_size', 0)),
            'pool': 1.0 if raw_data.get('pool', False) else 0.0,
            'fireplace': 1.0 if raw_data.get('fireplace', False) else 0.0,
            'garage_spaces': float(raw_data.get('garage_spaces', 0)),
            'condition_score': float(raw_data.get('condition_score', 5.0)),
        }
        
        # Create derived features
        current_year = 2024
        features['property_age'] = current_year - features['year_built']
        features['property_age_squared'] = features['property_age'] ** 2
        features['sqft_per_bedroom'] = features['square_feet'] / (features['bedrooms'] + 1)
        features['bed_bath_ratio'] = features['bedrooms'] / (features['bathrooms'] + 1)
        features['log_sqft'] = np.log1p(features['square_feet'])
        features['is_new'] = 1.0 if features['property_age'] < 5 else 0.0
        features['is_old'] = 1.0 if features['property_age'] > 50 else 0.0
        
        # Create feature vector
        feature_columns = [
            'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
            'pool', 'fireplace', 'garage_spaces', 'condition_score',
            'property_age', 'property_age_squared', 'sqft_per_bedroom',
            'bed_bath_ratio', 'log_sqft', 'is_new', 'is_old'
        ]
        
        feature_vector = np.array([features[col] for col in feature_columns])
        
        # Scale if fitted
        if self.fitted and self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return feature_vector
    
    def fit(self, X: np.ndarray):
        """Fit scaler on training data"""
        self.scaler = RobustScaler()
        self.scaler.fit(X)
        self.fitted = True