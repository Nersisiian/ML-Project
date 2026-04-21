import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Dict, Any, Optional, Union
import logging
import joblib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass

class LightGBMModel(BaseModel):
    """LightGBM implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_importance_ = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        
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
        
        self.model = lgb.LGBMRegressor(**params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.feature_importance_ = self.model.feature_importances_
        logger.info(f"LightGBM training complete. Best iteration: {self.model.best_iteration_}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.config = {}
        return instance

class XGBoostModel(BaseModel):
    """XGBoost implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': self.config.get('max_depth', 8),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'n_estimators': self.config.get('n_estimators', 1000),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1,
            'verbose': 0
        }
        
        self.model = xgb.XGBRegressor(**params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info(f"XGBoost training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, path: str):
        self.model.save_model(path)
    
    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance.model = xgb.XGBRegressor()
        instance.model.load_model(path)
        return instance

class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None):
        
        params = {
            'n_estimators': self.config.get('n_estimators', 500),
            'max_depth': self.config.get('max_depth', 20),
            'min_samples_split': self.config.get('min_samples_split', 5),
            'min_samples_leaf': self.config.get('min_samples_leaf', 2),
            'max_features': self.config.get('max_features', 'sqrt'),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1
        }
        
        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)
        
        logger.info(f"Random Forest training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str):
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        return instance

def get_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
    """Factory method to get model by name"""
    models = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'randomforest': RandomForestModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](config)