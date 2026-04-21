import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Dict, Any

class HyperparameterTuner:
    """Hyperparameter optimization with Optuna"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'verbose': -1,
        }
        
        # Train with early stopping
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(self.X_val)
        mae = mean_absolute_error(
            np.expm1(self.y_val), 
            np.expm1(y_pred)
        )
        
        return mae
    
    def run(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params