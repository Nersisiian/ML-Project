import pytest
import numpy as np
import lightgbm as lgb
from ml.models.lightgbm_model import LightGBMModel
from ml.evaluation.metrics import calculate_metrics

class TestModel:
    """Unit tests for ML model"""
    
    @pytest.fixture
    def model_config(self):
        return {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'n_estimators': 100,
            'random_state': 42
        }
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 20)
        y = np.random.randn(n_samples) * 100 + 500
        return X, y
    
    def test_model_initialization(self, model_config):
        """Test model initialization"""
        model = LightGBMModel(model_config)
        
        assert model.config == model_config
        assert model.model is None
    
    def test_model_training(self, model_config, sample_data):
        """Test model training"""
        X, y = sample_data
        model = LightGBMModel(model_config)
        
        model.train(X, y, X_val=X[:100], y_val=y[:100])
        
        assert model.model is not None
        assert isinstance(model.model, lgb.LGBMRegressor)
    
    def test_model_prediction(self, model_config, sample_data):
        """Test model prediction"""
        X, y = sample_data
        model = LightGBMModel(model_config)
        
        model.train(X, y, X_val=X[:100], y_val=y[:100])
        predictions = model.predict(X[:10])
        
        assert predictions.shape[0] == 10
        assert predictions.dtype == np.float64
    
    def test_model_save_load(self, model_config, sample_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        
        # Save model
        save_path = tmp_path / "model.pkl"
        model.save(str(save_path))
        assert save_path.exists()
        
        # Load model
        loaded_model = LightGBMModel.load(str(save_path))
        assert loaded_model.model is not None
        
        # Compare predictions
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0