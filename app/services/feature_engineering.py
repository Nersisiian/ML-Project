import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering service for real estate predictions"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        self.feature_columns = [
            'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
            'pool', 'fireplace', 'garage_spaces', 'condition_score',
            'property_age', 'property_age_squared', 'sqft_per_bedroom',
            'bath_bedroom_ratio', 'log_sqft', 'log_lot', 'is_new_construction',
            'is_old_property', 'is_luxury', 'sqft_bedroom_interaction',
            'age_condition_interaction'
        ]
    
    async def transform(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Transform raw input to feature vector"""
        
        features = self._extract_base_features(raw_data)
        features = self._create_derived_features(features)
        features = self._create_interaction_features(features)
        features = self._create_categorical_features(features)
        
        # Create feature vector in correct order
        feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
        
        # Handle any NaN or infinite values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_vector
    
    async def batch_transform(self, batch_data: List[Dict[str, Any]]) -> np.ndarray:
        """Transform multiple inputs at once"""
        
        feature_vectors = []
        for data in batch_data:
            features = await self.transform(data)
            feature_vectors.append(features)
        
        return np.vstack(feature_vectors)
    
    def _extract_base_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract base features from input"""
        
        return {
            'square_feet': float(data.get('square_feet', 0)),
            'bedrooms': float(data.get('bedrooms', 0)),
            'bathrooms': float(data.get('bathrooms', 0)),
            'year_built': float(data.get('year_built', 2000)),
            'lot_size': float(data.get('lot_size', data.get('square_feet', 0) * 0.2)),
            'pool': 1.0 if data.get('pool', False) else 0.0,
            'fireplace': 1.0 if data.get('fireplace', False) else 0.0,
            'garage_spaces': float(data.get('garage_spaces', 0)),
            'condition_score': float(data.get('condition_score', 5.0))
        }
    
    def _create_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create derived features"""
        
        # Age features
        features['property_age'] = self.current_year - features['year_built']
        features['property_age_squared'] = features['property_age'] ** 2
        
        # Ratio features
        features['sqft_per_bedroom'] = features['square_feet'] / (features['bedrooms'] + 1)
        features['bath_bedroom_ratio'] = features['bathrooms'] / (features['bedrooms'] + 1)
        
        # Log transformations
        features['log_sqft'] = np.log1p(features['square_feet'])
        features['log_lot'] = np.log1p(features['lot_size'])
        
        # Binary flags
        features['is_new_construction'] = 1.0 if features['property_age'] < 5 else 0.0
        features['is_old_property'] = 1.0 if features['property_age'] > 50 else 0.0
        features['is_luxury'] = 1.0 if (
            features['square_feet'] > 3000 and 
            features['bedrooms'] >= 4 and
            features['pool'] == 1
        ) else 0.0
        
        return features
    
    def _create_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features"""
        
        features['sqft_bedroom_interaction'] = features['square_feet'] * features['bedrooms']
        features['age_condition_interaction'] = features['property_age'] * features['condition_score']
        
        # Polynomial features
        features['sqft_squared'] = features['square_feet'] ** 2
        features['bedrooms_squared'] = features['bedrooms'] ** 2
        
        return features
    
    def _create_categorical_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create categorical encodings"""
        
        # Condition score bins
        if features['condition_score'] >= 8:
            features['condition_excellent'] = 1.0
            features['condition_good'] = 0.0
            features['condition_fair'] = 0.0
        elif features['condition_score'] >= 5:
            features['condition_excellent'] = 0.0
            features['condition_good'] = 1.0
            features['condition_fair'] = 0.0
        else:
            features['condition_excellent'] = 0.0
            features['condition_good'] = 0.0
            features['condition_fair'] = 1.0
        
        # Age bins
        if features['property_age'] < 10:
            features['age_new'] = 1.0
            features['age_medium'] = 0.0
            features['age_old'] = 0.0
        elif features['property_age'] < 30:
            features['age_new'] = 0.0
            features['age_medium'] = 1.0
            features['age_old'] = 0.0
        else:
            features['age_new'] = 0.0
            features['age_medium'] = 0.0
            features['age_old'] = 1.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

class FeatureValidator:
    """Validate features before inference"""
    
    def __init__(self):
        self.ranges = {
            'square_feet': (100, 50000),
            'bedrooms': (0, 10),
            'bathrooms': (0, 10),
            'year_built': (1800, 2024),
            'lot_size': (0, 100000),
            'garage_spaces': (0, 10),
            'condition_score': (1, 10)
        }
    
    def validate(self, features: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate feature values"""
        
        errors = []
        
        for feature, (min_val, max_val) in self.ranges.items():
            if feature in features:
                value = features[feature]
                if value < min_val or value > max_val:
                    errors.append(
                        f"{feature} must be between {min_val} and {max_val}, got {value}"
                    )
        
        # Check required fields
        required_fields = ['square_feet', 'bedrooms', 'bathrooms', 'year_built']
        for field in required_fields:
            if field not in features:
                errors.append(f"Missing required field: {field}")
        
        return len(errors) == 0, errors