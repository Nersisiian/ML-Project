import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics"""
    
    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Percentage metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Error distribution
    errors = y_pred - y_true
    error_std = np.std(errors)
    error_mean = np.mean(errors)
    
    # Quantile errors
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_errors = {f'q{int(q*100)}': np.quantile(errors, q) for q in quantiles}
    
    # Accuracy within thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    accuracy_within = {}
    
    for threshold in thresholds:
        within = np.mean(np.abs(errors / y_true) <= threshold) * 100
        accuracy_within[f'within_{int(threshold*100)}pct'] = within
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'error_mean': error_mean,
        'error_std': error_std,
        **quantile_errors,
        **accuracy_within
    }
    
    return metrics

def calculate_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics by segment (e.g., by zipcode)"""
    
    unique_segments = np.unique(segments)
    segment_metrics = {}
    
    for segment in unique_segments:
        mask = segments == segment
        if np.sum(mask) > 10:  # Minimum samples
            metrics = calculate_metrics(y_true[mask], y_pred[mask])
            segment_metrics[str(segment)] = metrics
    
    return segment_metrics