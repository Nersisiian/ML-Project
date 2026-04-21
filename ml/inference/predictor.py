import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class InferencePredictor:
    """High-performance inference predictor"""
    
    def __init__(self, model, scaler, feature_columns: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.cache = {}
    
    async def predict_single(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Single prediction with caching"""
        
        # Create cache key
        cache_key = self._create_cache_key(features)
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Preprocess
        feature_vector = self._preprocess(features)
        
        # Predict
        pred_log = self.model.predict(feature_vector.reshape(1, -1))[0]
        price = np.expm1(pred_log)
        
        result = {
            'price': float(price),
            'confidence_lower': float(price * 0.85),
            'confidence_upper': float(price * 1.15)
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Batch prediction for multiple inputs"""
        
        # Preprocess all
        feature_vectors = [self._preprocess(f) for f in features_list]
        X = np.vstack(feature_vectors)
        
        # Batch predict
        predictions_log = self.model.predict(X)
        prices = np.expm1(predictions_log)
        
        return prices.tolist()
    
    def _preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for model input"""
        
        # Create feature vector in correct order
        feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
        
        # Scale
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return feature_vector
    
    def _create_cache_key(self, features: Dict[str, Any]) -> str:
        """Create cache key from features"""
        import hashlib
        feature_str = str(sorted(features.items()))
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
        logger.info("Cache cleared")