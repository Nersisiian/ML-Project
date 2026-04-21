import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Any, Callable
import logging

logger = logging.getLogger(__name__)

class TimeSeriesCrossValidator:
    """Time series cross-validation for temporal data"""
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.results = []
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: Any,
        model_params: Dict[str, Any],
        train_func: Callable = None
    ) -> Dict[str, List[float]]:
        """Run time series cross-validation"""
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(self.tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            if train_func:
                model = train_func(X_train, y_train, X_val, y_val, model_params)
            else:
                model = model_class(**model_params)
                model.fit(X_train, y_train)
            
            # Predict
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_val)
            
            # Calculate metrics
            cv_scores['mae'].append(mean_absolute_error(y_true, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            cv_scores['r2'].append(r2_score(y_true, y_pred))
            cv_scores['mape'].append(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            
            logger.info(f"Fold {fold + 1} - MAE: ${cv_scores['mae'][-1]:.2f}, R²: {cv_scores['r2'][-1]:.3f}")
            
            self.results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': {
                    'mae': cv_scores['mae'][-1],
                    'rmse': cv_sodes['rmse'][-1],
                    'r2': cv_scores['r2'][-1],
                    'mape': cv_scores['mape'][-1]
                }
            })
        
        # Summary statistics
        summary = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for metric, scores in cv_scores.items()
        }
        
        logger.info(f"CV Results - MAE: {summary['mae']['mean']:.2f} ± {summary['mae']['std']:.2f}")
        
        return summary
    
    def expanding_window_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: Any,
        model_params: Dict[str, Any],
        initial_train_size: int = 1000
    ) -> Dict[str, List[float]]:
        """Expanding window validation (growing training set)"""
        
        n_samples = len(X)
        scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for test_start in range(initial_train_size, n_samples, 100):
            train_idx = list(range(test_start))
            val_idx = list(range(test_start, min(test_start + 200, n_samples)))
            
            if len(val_idx) == 0:
                break
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            y_pred = np.expm1(model.predict(X_val))
            y_true = np.expm1(y_val)
            
            scores['mae'].append(mean_absolute_error(y_true, y_pred))
            scores['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            scores['r2'].append(r2_score(y_true, y_pred))
            
            logger.info(f"Window {test_start}: MAE=${scores['mae'][-1]:.2f}")
        
        return scores