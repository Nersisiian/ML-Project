# ml/training/train.py
import mlflow
import mlflow.lightgbm
import optuna
from optuna.integration.mlflow import MLflowCallback
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealEstateTrainer:
    """Production training pipeline with hyperparameter optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        mlflow.set_experiment(config['experiment_name'])
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate training data"""
        # Load from feature store
        import feast
        store = feast.FeatureStore(repo_path=self.config['feast_repo_path'])
        
        # Retrieve features for training
        entity_df = pd.read_parquet(self.config['training_entity_path'])
        
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "property_characteristics:square_feet",
                "property_characteristics:bedrooms",
                "property_characteristics:bathrooms",
                "property_characteristics:year_built",
                "property_characteristics:lot_size",
                "location_features:median_income",
                "location_features:crime_rate",
                "location_features:school_rating",
                "location_features:walk_score",
                "temporal_features:price_momentum_7d",
                "temporal_features:seasonal_index",
            ]
        ).to_df()
        
        # Prepare features and target
        X = training_df.drop(columns=['price', 'sale_date', 'property_id'])
        y = np.log1p(training_df['price'])  # Log transform for skewed target
        
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def objective(self, trial: optuna.Trial, X_train, X_val, y_train, y_val):
        """Optuna objective function for hyperparameter tuning"""
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
            'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
            'early_stopping_rounds': 50,
            'verbose': -1,
        }
        
        # Train with early stopping
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred))
        
        return mae
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Run hyperparameter optimization with Optuna"""
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Add MLflow callback for tracking
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config['mlflow_tracking_uri'],
            metric_name='val_mae'
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            callbacks=[mlflow_callback],
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_value = study.best_value
        
        mlflow.log_params(best_params)
        mlflow.log_metric('best_val_mae', best_value)
        
        return best_params
    
    def cross_validate(self, X, y, n_splits=5):
        """Time series cross-validation for realistic evaluation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = lgb.LGBMRegressor(
                **self.config['model_params'],
                n_estimators=2000,
                random_state=42,
                verbose=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predict (inverse log transform)
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_val)
            
            # Calculate metrics
            cv_scores['mae'].append(mean_absolute_error(y_true, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            cv_scores['r2'].append(r2_score(y_true, y_pred))
            cv_scores['mape'].append(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            
            mlflow.log_metrics({
                f'fold_{fold}_mae': cv_scores['mae'][-1],
                f'fold_{fold}_rmse': cv_scores['rmse'][-1],
                f'fold_{fold}_r2': cv_scores['r2'][-1],
                f'fold_{fold}_mape': cv_scores['mape'][-1],
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
        
        return summary
    
    def train_final_model(self, X, y, best_params):
        """Train final model on all data with best parameters"""
        final_model = lgb.LGBMRegressor(
            **best_params,
            n_estimators=5000,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
        
        # Train with validation set from last fold
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
        )
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance as artifact
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        return final_model
    
    def run(self):
        """Main training pipeline"""
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log config
            mlflow.log_params(self.config)
            
            # Load data
            X, y = self.load_data()
            mlflow.log_metric('num_samples', len(X))
            mlflow.log_metric('num_features', X.shape[1])
            
            # Split data
            split_idx = int(len(X) * 0.7)
            X_train, X_temp = X[:split_idx], X[split_idx:]
            y_train, y_temp = y[:split_idx], y[split_idx:]
            
            split_val_idx = int(len(X_temp) * 0.5)
            X_val, X_test = X_temp[:split_val_idx], X_temp[split_val_idx:]
            y_val, y_test = y_temp[:split_val_idx], y_temp[split_val_idx:]
            
            # Hyperparameter tuning
            best_params = self.hyperparameter_tuning(
                X_train, y_train, X_val, y_val,
                n_trials=self.config.get('n_trials', 100)
            )
            
            # Cross-validation
            cv_results = self.cross_validate(X, y)
            mlflow.log_metrics({
                f'cv_{metric}_mean': cv_results[metric]['mean']
                for metric in cv_results
            })
            
            # Train final model
            final_model = self.train_final_model(X, y, best_params)
            
            # Evaluate on test set
            y_pred_log = final_model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test)
            
            test_metrics = {
                'test_mae': mean_absolute_error(y_true, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'test_r2': r2_score(y_true, y_pred),
                'test_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            mlflow.log_metrics(test_metrics)
            
            # Save model
            mlflow.lightgbm.log_model(
                final_model,
                "model",
                signature=mlflow.models.infer_signature(X_test, y_pred_log),
                input_example=X_test[:5]
            )
            
            # Save scaler
            joblib.dump(self.scaler, 'scaler.pkl')
            mlflow.log_artifact('scaler.pkl')
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "real_estate_price_predictor")
            
            return final_model, test_metrics