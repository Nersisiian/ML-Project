"""
Prefect workflow for ML training pipeline
"""

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.logging import get_run_logger
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import mlflow
import joblib

@task(retries=3, retry_delay_seconds=60, timeout_seconds=3600)
def extract_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data from sources"""
    logger = get_run_logger()
    logger.info("Starting data extraction")
    
    from pipelines.data_pipeline.ingestion.postgres_ingestor import PostgresIngestor
    from pipelines.data_pipeline.ingestion.s3_loader import S3Loader
    
    # Extract from S3
    s3_loader = S3Loader(config['s3'])
    df = s3_loader.load_parquet(config['data_path'])
    
    logger.info(f"Extracted {len(df)} rows")
    
    return {
        'data': df,
        'extraction_time': datetime.datetime.now().isoformat(),
        'row_count': len(df)
    }

@task(retries=2, timeout_seconds=7200)
def validate_data(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted data"""
    logger = get_run_logger()
    logger.info("Starting data validation")
    
    from pipelines.data_pipeline.validation.validator import DataValidator
    
    df = data_dict['data']
    validator = DataValidator()
    validation_results = validator.run_all_checks(df)
    
    if not validation_results['overall_valid']:
        logger.warning(f"Validation issues: {validation_results['schema_errors']}")
    
    return {
        **data_dict,
        'validation_results': validation_results
    }

@task(retries=2, timeout_seconds=3600)
def preprocess_data(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess and clean data"""
    logger = get_run_logger()
    logger.info("Starting data preprocessing")
    
    from ml.training.feature_pipeline import FeaturePipeline
    
    df = data_dict['data']
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['property_id'])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Filter outliers
    for col in ['square_feet', 'price']:
        q99 = df[col].quantile(0.99)
        df = df[df[col] <= q99]
    
    logger.info(f"Preprocessed data: {len(df)} rows")
    
    return {
        **data_dict,
        'processed_data': df,
        'preprocessing_time': datetime.datetime.now().isoformat()
    }

@task(retries=2, timeout_seconds=7200)
def create_features(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create features for training"""
    logger = get_run_logger()
    logger.info("Starting feature engineering")
    
    from ml.training.feature_pipeline import FeaturePipeline
    
    df = data_dict['processed_data']
    feature_pipeline = FeaturePipeline(config={})
    
    # Create features
    df = feature_pipeline.create_derived_features(df)
    df = feature_pipeline.create_temporal_features(df)
    df = feature_pipeline.create_aggregate_features(df)
    
    # Prepare X and y
    exclude_cols = ['price', 'property_id', 'sale_date', 'created_at']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = np.log1p(df['price'].values)
    
    # Scale features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Created {len(feature_cols)} features")
    
    return {
        **data_dict,
        'X': X_scaled,
        'y': y,
        'feature_names': feature_cols,
        'scaler': scaler,
        'n_features': len(feature_cols),
        'n_samples': len(X_scaled)
    }

@task(retries=2, timeout_seconds=10800)
def train_model(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Train ML model"""
    logger = get_run_logger()
    logger.info("Starting model training")
    
    from ml.training.train import RealEstateTrainer
    from ml.training.hyperparameter_tuning import HyperparameterTuner
    from ml.training.cross_validation import TimeSeriesCrossValidator
    
    X = data_dict['X']
    y = data_dict['y']
    
    # Split data chronologically
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    split_val_idx = int(len(X_train) * 0.8)
    X_train, X_val = X_train[:split_val_idx], X_train[split_val_idx:]
    y_train, y_val = y_train[:split_val_idx], y_train[split_val_idx:]
    
    # Hyperparameter tuning
    tuner = HyperparameterTuner(X_train, y_train, X_val, y_val)
    best_params = tuner.run(n_trials=data_dict.get('n_trials', 50))
    
    logger.info(f"Best params: {best_params}")
    
    # Cross validation
    cv = TimeSeriesCrossValidator(n_splits=5)
    cv_results = cv.validate(X, y, None, best_params)
    
    logger.info(f"CV results: MAE={cv_results['mae']['mean']:.2f}")
    
    # Train final model
    import lightgbm as lgb
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # Evaluate on test set
    y_pred_log = final_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    metrics = {
        'test_mae': mean_absolute_error(y_true, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'test_r2': r2_score(y_true, y_pred),
        'test_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    logger.info(f"Test metrics: {metrics}")
    
    return {
        **data_dict,
        'model': final_model,
        'best_params': best_params,
        'cv_results': cv_results,
        'test_metrics': metrics,
        'training_time': datetime.datetime.now().isoformat()
    }

@task
def register_model(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Register model in MLflow"""
    logger = get_run_logger()
    logger.info("Registering model")
    
    import mlflow
    import mlflow.lightgbm
    
    model = data_dict['model']
    scaler = data_dict['scaler']
    feature_names = data_dict['feature_names']
    metrics = data_dict['test_metrics']
    params = data_dict['best_params']
    
    with mlflow.start_run(run_name=f"training_flow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        # Log scaler as artifact
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            joblib.dump(scaler, tmp.name)
            mlflow.log_artifact(tmp.name, "scaler.pkl")
            os.unlink(tmp.name)
        
        # Log feature names
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write('\n'.join(feature_names))
            mlflow.log_artifact(tmp.name, "feature_names.txt")
            os.unlink(tmp.name)
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "real_estate_predictor")
        
        data_dict['model_uri'] = model_uri
        data_dict['run_id'] = mlflow.active_run().info.run_id
    
    return data_dict

@task
def cleanup(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Cleanup temporary files"""
    logger = get_run_logger()
    logger.info("Cleaning up")
    
    # Remove large objects from memory
    for key in ['data', 'processed_data', 'X', 'y']:
        if key in data_dict:
            data_dict[key] = None
    
    return data_dict

@flow(name="ML Training Pipeline", task_runner=ConcurrentTaskRunner())
def training_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main training pipeline flow"""
    logger = get_run_logger()
    logger.info("Starting training pipeline")
    
    # Run pipeline
    data = extract_data(config)
    data = validate_data(data)
    data = preprocess_data(data)
    data = create_features(data)
    data = train_model(data)
    data = register_model(data)
    data = cleanup(data)
    
    logger.info("Training pipeline completed successfully")
    
    return {
        'status': 'success',
        'test_metrics': data.get('test_metrics', {}),
        'model_uri': data.get('model_uri', ''),
        'run_id': data.get('run_id', ''),
        'completion_time': datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Example config
    config = {
        's3': {
            'bucket': 'real-estate-data',
            'aws_access_key_id': 'your_key',
            'aws_secret_access_key': 'your_secret'
        },
        'data_path': 's3://real-estate-data/raw/properties.parquet',
        'n_trials': 50
    }
    
    result = training_pipeline(config)
    print(f"Pipeline result: {result}")