import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelVersioning:
    """Model versioning and tracking system"""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load model registry from file"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'models': {}, 'versions': {}}
    
    def _save_registry(self):
        """Save model registry to file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def compute_model_hash(self, model_params: Dict) -> str:
        """Compute hash of model parameters"""
        param_str = json.dumps(model_params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> str:
        """Register new model version"""
        
        model_hash = self.compute_model_hash(params)
        version = f"v{len(self.registry['models'].get(model_name, [])) + 1}"
        
        model_info = {
            'version': version,
            'hash': model_hash,
            'path': model_path,
            'metrics': metrics,
            'params': params,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'stage': 'staging'
        }
        
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = []
        
        self.registry['models'][model_name].append(model_info)
        self.registry['versions'][model_hash] = model_info
        
        self._save_registry()
        logger.info(f"Registered {model_name} {version}")
        
        return version
    
    def promote_to_production(self, model_name: str, version: str):
        """Promote model version to production"""
        
        for model in self.registry['models'].get(model_name, []):
            if model['version'] == version:
                # Demote current production
                for m in self.registry['models'][model_name]:
                    if m['stage'] == 'production':
                        m['stage'] = 'archived'
                
                model['stage'] = 'production'
                model['promoted_at'] = datetime.now().isoformat()
                self._save_registry()
                logger.info(f"Promoted {model_name} {version} to production")
                return True
        
        raise ValueError(f"Model {model_name} version {version} not found")
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production model"""
        for model in self.registry['models'].get(model_name, []):
            if model['stage'] == 'production':
                return model
        return None
    
    def rollback(self, model_name: str) -> Optional[Dict]:
        """Rollback to previous version"""
        versions = self.registry['models'].get(model_name, [])
        prod_idx = None
        
        for i, model in enumerate(versions):
            if model['stage'] == 'production':
                prod_idx = i
                break
        
        if prod_idx is not None and prod_idx > 0:
            prev_version = versions[prod_idx - 1]
            self.promote_to_production(model_name, prev_version['version'])
            logger.info(f"Rolled back {model_name} to {prev_version['version']}")
            return prev_version
        
        return None