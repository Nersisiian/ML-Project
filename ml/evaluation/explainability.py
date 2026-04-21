import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Model explainability using SHAP values"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def explain(self, X: np.ndarray, background_samples: int = 100):
        """Generate SHAP explanations"""
        
        logger.info("Generating SHAP explanations...")
        
        # Use a subset of data as background
        if len(X) > background_samples:
            background = X[np.random.choice(len(X), background_samples, replace=False)]
        else:
            background = X
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model, background)
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X)
        
        logger.info(f"SHAP values computed: shape {self.shap_values.shape}")
        
        return self.shap_values
    
    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        """Create summary plot"""
        
        if self.shap_values is None:
            self.explain(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_importance(self, save_path: Optional[str] = None):
        """Plot feature importance based on SHAP"""
        
        if self.shap_values is None:
            raise ValueError("Run explain() first")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return importance_df
    
    def explain_prediction(self, X: np.ndarray, index: int) -> Dict[str, Any]:
        """Explain a single prediction"""
        
        if self.shap_values is None:
            self.explain(X)
        
        # Get prediction
        prediction = self.model.predict(X[index:index+1])[0]
        
        # Get SHAP values for this prediction
        shap_row = self.shap_values[index]
        
        # Create explanation
        explanation = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_row)):
            if abs(shap_val) > 0.01:
                explanation.append({
                    'feature': feature,
                    'value': X[index, i],
                    'shap_value': float(shap_val),
                    'contribution': 'positive' if shap_val > 0 else 'negative'
                })
        
        # Sort by absolute SHAP value
        explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'prediction': float(prediction),
            'top_factors': explanation[:10],
            'base_value': float(self.explainer.expected_value)
        }
    
    def get_feature_interactions(self, X: np.ndarray, feature1: str, feature2: str) -> np.ndarray:
        """Get SHAP interaction values between two features"""
        
        if self.shap_values is None:
            self.explain(X)
        
        # Get interaction explainer
        interaction_explainer = shap.TreeExplainer(self.model)
        shap_interaction = interaction_explainer.shap_interaction_values(X)
        
        # Get indices of features
        idx1 = self.feature_names.index(feature1)
        idx2 = self.feature_names.index(feature2)
        
        # Return interaction values
        return shap_interaction[:, idx1, idx2]

class PartialDependencePlot:
    """Partial dependence plots for feature analysis"""
    
    def __init__(self, model, X: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X = X
        self.feature_names = feature_names
    
    def compute_pdp(self, feature_idx: int, grid_points: int = 50) -> Dict[str, np.ndarray]:
        """Compute partial dependence for a feature"""
        
        feature_values = np.linspace(
            self.X[:, feature_idx].min(),
            self.X[:, feature_idx].max(),
            grid_points
        )
        
        predictions = []
        X_temp = self.X.copy()
        
        for value in feature_values:
            X_temp[:, feature_idx] = value
            pred = self.model.predict(X_temp)
            predictions.append(pred.mean())
        
        return {
            'feature_values': feature_values,
            'predictions': np.array(predictions)
        }
    
    def plot_pdp(self, feature_name: str, save_path: Optional[str] = None):
        """Plot partial dependence for a feature"""
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_idx = self.feature_names.index(feature_name)
        pdp = self.compute_pdp(feature_idx)
        
        plt.figure(figsize=(8, 5))
        plt.plot(pdp['feature_values'], pdp['predictions'], linewidth=2)
        plt.xlabel(feature_name)
        plt.ylabel('Average Prediction')
        plt.title(f'Partial Dependence Plot: {feature_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()