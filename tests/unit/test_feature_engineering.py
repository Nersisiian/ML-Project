import pytest
import numpy as np
import pandas as pd
from app.services.feature_engineering import FeatureEngineer

class TestFeatureEngineering:
    """Unit tests for feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_input(self):
        return {
            'square_feet': 2200,
            'bedrooms': 4,
            'bathrooms': 2.5,
            'year_built': 2010,
            'lot_size': 6000,
            'pool': True,
            'fireplace': True,
            'garage_spaces': 2,
            'condition_score': 8,
            'zipcode': '94105',
            'latitude': 37.7749,
            'longitude': -122.4194
        }
    
    @pytest.mark.asyncio
    async def test_transform_returns_correct_shape(self, feature_engineer, sample_input):
        """Test that transform returns correct feature shape"""
        features = await feature_engineer.transform(sample_input)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 50  # 50 features
    
    @pytest.mark.asyncio
    async def test_transform_handles_missing_values(self, feature_engineer):
        """Test transform handles missing values gracefully"""
        input_with_missing = {
            'square_feet': 2200,
            'bedrooms': 4,
            'bathrooms': 2.5,
            'year_built': 2010,
            # missing lot_size, pool, fireplace
            'zipcode': '94105'
        }
        
        features = await feature_engineer.transform(input_with_missing)
        assert not np.isnan(features).any()
    
    def test_create_derived_features(self, feature_engineer):
        """Test derived feature creation"""
        base_features = {
            'square_feet': 2200,
            'bedrooms': 4,
            'bathrooms': 2.5,
            'year_built': 2010
        }
        
        derived = feature_engineer._create_derived_features(base_features)
        
        assert 'age' in derived
        assert derived['age'] == 14  # 2024 - 2010
        assert 'sqft_bedroom_ratio' in derived
        assert derived['sqft_bedroom_ratio'] == 2200 / 5  # square_feet / (bedrooms + 1)
    
    def test_encode_categorical(self, feature_engineer):
        """Test categorical encoding"""
        zipcode = '94105'
        encoded = feature_engineer._encode_categorical(zipcode)
        
        assert isinstance(encoded, list)
        assert len(encoded) == 10  # One-hot encoding for top 10 zipcodes
    
    @pytest.mark.asyncio
    async def test_batch_transform(self, feature_engineer, sample_input):
        """Test batch transformation"""
        batch_inputs = [sample_input] * 10
        features_batch = await feature_engineer.batch_transform(batch_inputs)
        
        assert features_batch.shape[0] == 10
        assert features_batch.shape[1] == 50