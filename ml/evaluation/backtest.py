import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtesting framework for time series models"""
    
    def __init__(self, model, feature_columns: List[str]):
        self.model = model
        self.feature_columns = feature_columns
        self.results = []
    
    def run(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        retrain_frequency: str = 'monthly'
    ) -> Dict[str, Any]:
        """Run backtest over time period"""
        
        current_date = start_date
        predictions = []
        actuals = []
        
        while current_date <= end_date:
            # Get training data (all data before current_date)
            train_df = df[df['sale_date'] < current_date]
            
            # Get test data (current period)
            if retrain_frequency == 'monthly':
                next_date = current_date + timedelta(days=30)
            elif retrain_frequency == 'weekly':
                next_date = current_date + timedelta(days=7)
            else:
                next_date = end_date
            
            test_df = df[
                (df['sale_date'] >= current_date) & 
                (df['sale_date'] < next_date)
            ]
            
            if len(train_df) > 100 and len(test_df) > 0:
                # Train model on historical data
                X_train = train_df[self.feature_columns].values
                y_train = np.log1p(train_df['price'].values)
                
                self.model.fit(X_train, y_train)
                
                # Predict on test period
                X_test = test_df[self.feature_columns].values
                y_pred_log = self.model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                
                predictions.extend(y_pred)
                actuals.extend(test_df['price'].values)
                
                # Record period results
                self.results.append({
                    'period_start': current_date,
                    'period_end': next_date,
                    'n_train': len(train_df),
                    'n_test': len(test_df),
                    'predictions': y_pred.tolist(),
                    'actuals': test_df['price'].values.tolist()
                })
                
                logger.info(f"Period {current_date.date()} - {next_date.date()}: {len(test_df)} predictions")
            
            current_date = next_date
        
        # Calculate overall metrics
        from ml.evaluation.metrics import calculate_metrics
        metrics = calculate_metrics(np.array(actuals), np.array(predictions))
        
        return {
            'metrics': metrics,
            'results': self.results,
            'n_periods': len(self.results),
            'total_predictions': len(predictions)
        }