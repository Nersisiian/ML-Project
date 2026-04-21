import pytest
import pandas as pd
from pipelines.data_pipeline.ingestion.postgres_ingestor import PostgresIngestor
from pipelines.data_pipeline.validation.validator import DataValidator

class TestDataPipeline:
    """Integration tests for data pipeline"""
    
    @pytest.fixture
    def postgres_ingestor(self):
        return PostgresIngestor(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'property_id': ['p1', 'p2', 'p3'],
            'square_feet': [2000, 1500, 3000],
            'bedrooms': [3, 2, 4],
            'bathrooms': [2.0, 1.5, 3.0],
            'year_built': [2010, 2005, 2015],
            'price': [500000, 400000, 700000]
        })
    
    def test_data_validation_passes(self, validator, sample_dataframe):
        """Test data validation passes for good data"""
        is_valid = validator.validate_schema(sample_dataframe)
        assert is_valid is True
    
    def test_data_validation_fails_missing_columns(self, validator):
        """Test data validation fails for missing columns"""
        invalid_df = pd.DataFrame({
            'property_id': ['p1', 'p2'],
            'square_feet': [2000, 1500]
        })
        
        is_valid = validator.validate_schema(invalid_df)
        assert is_valid is False
    
    def test_data_validation_fails_null_values(self, validator, sample_dataframe):
        """Test data validation fails for null values"""
        sample_dataframe.loc[0, 'price'] = None
        is_valid = validator.validate_no_nulls(sample_dataframe)
        assert is_valid is False
    
    def test_data_validation_fails_outliers(self, validator, sample_dataframe):
        """Test data validation fails for outliers"""
        sample_dataframe.loc[0, 'square_feet'] = 100000  # Too large
        
        is_valid = validator.validate_outliers(sample_dataframe, 'square_feet', max_value=50000)
        assert is_valid is False