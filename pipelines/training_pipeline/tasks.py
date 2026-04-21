"""
Individual tasks for training pipeline
"""

from prefect import task
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

@task
def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@task
def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality metrics"""
    
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Check for data freshness
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        quality_metrics['max_date'] = df['sale_date'].max().isoformat()
        quality_metrics['min_date'] = df['sale_date'].min().isoformat()
    
    return quality_metrics

@task
def split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """Split data into train/val/test sets"""
    
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }

@task
def save_artifacts(
    model,
    scaler,
    feature_names: List[str],
    metrics: Dict[str, float],
    output_dir: str
):
    """Save training artifacts"""
    import joblib
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path / 'model.pkl')
    
    # Save scaler
    joblib.dump(scaler, output_path / 'scaler.pkl')
    
    # Save feature names
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Artifacts saved to {output_dir}")

@task
def send_notification(message: str, channel: str = 'slack'):
    """Send notification about pipeline status"""
    # Implement Slack/Email notification
    logger.info(f"Notification: {message}")