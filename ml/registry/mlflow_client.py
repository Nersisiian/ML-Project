import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowClientWrapper:
    """Wrapper for MLflow tracking and model registry"""
    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri
        
    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None):
        """Log parameters to current run"""
        if run_id:
            for key, value in params.items():
                self.client.log_param(run_id, key, value)
        else:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], run_id: Optional[str] = None):
        """Log metrics to current run"""
        if run_id:
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, value)
        else:
            mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log model to MLflow"""
        mlflow.lightgbm.log_model(model, artifact_path)
    
    def register_model(self, model_uri: str, model_name: str) -> str:
        """Register model in model registry"""
        version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {version.version}")
        return version.version
    
    def transition_model_stage(
        self, 
        model_name: str, 
        version: str, 
        stage: str
    ):
        """Transition model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model {model_name} v{version} moved to {stage}")
    
    def get_production_model(self, model_name: str):
        """Get production model"""
        model = self.client.get_registered_model(model_name)
        for version in model.latest_versions:
            if version.current_stage == "Production":
                model_uri = f"models:/{model_name}/Production"
                return mlflow.lightgbm.load_model(model_uri)
        raise ValueError(f"No production model found for {model_name}")
    
    def compare_models(self, model_name: str, metric: str = "mae"):
        """Compare model versions by metric"""
        model = self.client.get_registered_model(model_name)
        versions = []
        
        for version in model.latest_versions:
            run = self.client.get_run(version.run_id)
            if metric in run.data.metrics:
                versions.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    metric: run.data.metrics[metric],
                    'run_id': version.run_id
                })
        
        return sorted(versions, key=lambda x: x[metric])