import lightgbm as lgb
import numpy as np
import joblib
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LightGBMModel:
    """LightGBM model wrapper with training and inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_importance = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """Train LightGBM model"""
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.get('num_leaves', 255),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'feature_fraction': self.config.get('feature_fraction', 0.8),
            'bagging_fraction': self.config.get('bagging_fraction', 0.8),
            'bagging_freq': self.config.get('bagging_freq', 5),
            'min_child_samples': self.config.get('min_child_samples', 20),
            'reg_alpha': self.config.get('reg_alpha', 0.0),
            'reg_lambda': self.config.get('reg_lambda', 0.0),
            'n_estimators': self.config.get('n_estimators', 1000),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1,
            'verbose': -1
        }
        
        logger.info(f"Training LightGBM with params: {params}")
        
        if X_val is not None and y_val is not None:
            # Train with validation for early stopping
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )
        else:
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration_}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk"""
        model = cls.__new__(cls)
        model.model = joblib.load(path)
        model.config = {}
        return model
    
    def get_feature_importance(self, feature_names: list) -> dict:
        """Get feature importance mapping"""
        if self.feature_importance is None:
            return {}
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))