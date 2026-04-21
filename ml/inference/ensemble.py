import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Ensemble of multiple models for better predictions"""
    
    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        # Normalize weights
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
        logger.info(f"Ensemble initialized with {len(models)} models, weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average prediction"""
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
    
    def predict_with_variance(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict with uncertainty estimation"""
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        variance = np.var(predictions, axis=0)
        
        return {
            'mean': mean_pred,
            'variance': variance,
            'std': np.sqrt(variance),
            'lower': mean_pred - 1.96 * np.sqrt(variance),
            'upper': mean_pred + 1.96 * np.sqrt(variance)
        }
    
    def add_model(self, model: Any, weight: float = None):
        """Add model to ensemble"""
        self.models.append(model)
        
        if weight:
            self.weights = np.append(self.weights, weight)
        else:
            self.weights = np.append(self.weights, 1.0 / len(self.models))
        
        # Re-normalize
        self.weights = self.weights / np.sum(self.weights)
        
        logger.info(f"Added model to ensemble. Now {len(self.models)} models")