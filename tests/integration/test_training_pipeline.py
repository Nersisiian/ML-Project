import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'property_id': [f'prop_{i}' for i in range(n_samples)],
            'square_feet': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'year_built': np.random.randint(1950, 2024, n_samples),
            'lot_size': np.random.normal(5000, 2000, n_samples),
            'zipcode': np.random.choice(['94105', '94103', '94123'], n_samples),
            'price': np.random.normal(800000, 200000, n_samples),
            'sale_date': pd.date_range('2020-01-01', periods=n_samples, freq='D')
        }
        
        df = pd.DataFrame(data)
        
        # Ensure price is positive
        df['price'] = df['price'].abs()
        
        return df
    
    def test_end_to_end_training(self, sample_data):
        """Test complete training pipeline"""
        
        from ml.training.train import RealEstateTrainer
        from ml.training.feature_pipeline import FeaturePipeline
        
        # Feature engineering
        feature_pipeline = FeaturePipeline(config={})
        X = feature_pipeline.fit_transform(sample_data)
        y = np.log1p(sample_data['price'].values)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        split_val_idx = int(len(X_train) * 0.8)
        X_train, X_val = X_train[:split_val_idx], X_train[split_val_idx:]
        y_train, y_val = y_train[:split_val_idx], y_train[split_val_idx:]
        
        # Train model
        import lightgbm as lgb
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
        
        assert mae < 100000  # MAE should be reasonable
        assert model is not None
    
    def test_model_persistence(self, sample_data):
        """Test model saving and loading"""
        
        import lightgbm as lgb
        import joblib
        
        # Train model
        X = sample_data[['square_feet', 'bedrooms', 'bathrooms', 'year_built']].values
        y = sample_data['price'].values
        
        model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X, y)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            joblib.dump(model, tmp.name)
            
            # Load model
            loaded_model = joblib.load(tmp.name)
            
            # Compare predictions
            pred1 = model.predict(X[:10])
            pred2 = loaded_model.predict(X[:10])
            
            np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_feature_pipeline_consistency(self, sample_data):
        """Test feature pipeline produces consistent results"""
        
        from ml.training.feature_pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline(config={})
        
        # First fit
        X1 = pipeline.fit_transform(sample_data)
        
        # Second transform (should be same)
        X2 = pipeline.transform(sample_data)
        
        np.testing.assert_array_almost_equal(X1, X2)