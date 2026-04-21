import json
import yaml
import joblib
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelUtils:
    """Utilities for model management"""
    
    @staticmethod
    def save_model(model, path: str, metadata: Optional[Dict] = None):
        """Save model with metadata"""
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if path.suffix == '.pkl':
            joblib.dump(model, path)
        elif path.suffix == '.joblib':
            joblib.dump(model, path)
        elif path.suffix == '.pth':
            import torch
            torch.save(model.state_dict(), path)
        else:
            joblib.dump(model, path)
        
        # Save metadata
        if metadata:
            metadata_path = path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, model_type: str = 'joblib'):
        """Load model from disk"""
        
        if model_type == 'joblib':
            model = joblib.load(path)
        elif model_type == 'pickle':
            with open(path, 'rb') as f:
                model = pickle.load(f)
        elif model_type == 'pytorch':
            import torch
            model = torch.load(path)
        else:
            model = joblib.load(path)
        
        logger.info(f"Model loaded from {path}")
        return model
    
    @staticmethod
    def save_config(config: Dict[str, Any], path: str):
        """Save configuration to file"""
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                return json.load(f)
    
    @staticmethod
    def get_model_size(path: str) -> int:
        """Get model file size in MB"""
        size_bytes = Path(path).stat().st_size
        return size_bytes / (1024 * 1024)
    
    @staticmethod
    def validate_model(model, X_sample: np.ndarray) -> bool:
        """Validate model can predict"""
        try:
            predictions = model.predict(X_sample)
            return len(predictions) == len(X_sample)
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False