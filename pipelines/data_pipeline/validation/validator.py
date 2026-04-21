import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality checks"""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or self._default_schema()
    
    def _default_schema(self) -> Dict[str, Any]:
        """Default schema for real estate data"""
        return {
            'property_id': {'type': 'string', 'required': True, 'unique': True},
            'square_feet': {'type': 'float', 'required': True, 'min': 100, 'max': 50000},
            'bedrooms': {'type': 'int', 'required': True, 'min': 0, 'max': 10},
            'bathrooms': {'type': 'float', 'required': True, 'min': 0, 'max': 10},
            'year_built': {'type': 'int', 'required': True, 'min': 1800, 'max': 2024},
            'lot_size': {'type': 'float', 'required': False, 'min': 0, 'max': 100000},
            'price': {'type': 'float', 'required': True, 'min': 10000, 'max': 10000000},
            'zipcode': {'type': 'string', 'required': True, 'pattern': r'^\d{5}$'},
            'latitude': {'type': 'float', 'required': False, 'min': -90, 'max': 90},
            'longitude': {'type': 'float', 'required': False, 'min': -180, 'max': 180}
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataframe schema"""
        
        errors = []
        
        # Check required columns
        for col, rules in self.schema.items():
            if rules.get('required', False) and col not in df.columns:
                errors.append(f"Required column missing: {col}")
        
        # Check data types
        for col, rules in self.schema.items():
            if col in df.columns:
                expected_type = rules.get('type')
                if expected_type == 'float' and not pd.api.types.is_float_dtype(df[col]):
                    errors.append(f"Column {col} should be float")
                elif expected_type == 'int' and not pd.api.types.is_integer_dtype(df[col]):
                    errors.append(f"Column {col} should be integer")
                elif expected_type == 'string' and not pd.api.types.is_string_dtype(df[col]):
                    errors.append(f"Column {col} should be string")
        
        return len(errors) == 0, errors
    
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate value ranges"""
        
        errors = []
        
        for col, rules in self.schema.items():
            if col in df.columns and 'min' in rules and 'max' in rules:
                min_val = rules['min']
                max_val = rules['max']
                
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    errors.append(
                        f"Column {col} has {len(out_of_range)} values outside [{min_val}, {max_val}]"
                    )
        
        return len(errors) == 0, errors
    
    def validate_uniqueness(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate unique constraints"""
        
        errors = []
        
        for col, rules in self.schema.items():
            if rules.get('unique', False) and col in df.columns:
                if df[col].duplicated().any():
                    duplicates = df[col].duplicated().sum()
                    errors.append(f"Column {col} has {duplicates} duplicate values")
        
        return len(errors) == 0, errors
    
    def detect_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr', 
                       threshold: float = 3.0) -> Dict[str, List]:
        """Detect outliers using IQR or Z-score"""
        
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > threshold
            
            outliers[col] = df[outlier_mask].index.tolist()
        
        return outliers
    
    def check_missing_values(self, df: pd.DataFrame, 
                            threshold: float = 0.3) -> Dict[str, float]:
        """Check missing values percentage"""
        
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        high_missing = missing_pct[missing_pct > threshold]
        
        if len(high_missing) > 0:
            logger.warning(f"Columns with >{threshold*100}% missing: {high_missing.to_dict()}")
        
        return missing_pct.to_dict()
    
    def validate_distribution(self, df: pd.DataFrame, 
                             reference_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Validate data distribution using KS test"""
        
        if reference_df is None:
            return {}
        
        drift_scores = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_df.columns:
                ks_stat, p_value = stats.ks_2samp(
                    df[col].dropna(), 
                    reference_df[col].dropna()
                )
                drift_scores[col] = ks_stat
        
        return drift_scores
    
    def run_all_checks(self, df: pd.DataFrame, 
                       reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run all validation checks"""
        
        results = {
            'schema_valid': None,
            'schema_errors': [],
            'range_valid': None,
            'range_errors': [],
            'unique_valid': None,
            'unique_errors': [],
            'missing_values': {},
            'outliers': {},
            'drift_scores': {}
        }
        
        # Schema validation
        results['schema_valid'], results['schema_errors'] = self.validate_schema(df)
        
        # Range validation
        results['range_valid'], results['range_errors'] = self.validate_ranges(df)
        
        # Uniqueness validation
        results['unique_valid'], results['unique_errors'] = self.validate_uniqueness(df)
        
        # Missing values
        results['missing_values'] = self.check_missing_values(df)
        
        # Outliers
        results['outliers'] = self.detect_outliers(df)
        
        # Distribution drift
        if reference_df is not None:
            results['drift_scores'] = self.validate_distribution(df, reference_df)
        
        # Overall status
        results['overall_valid'] = all([
            results['schema_valid'],
            results['range_valid'],
            results['unique_valid']
        ])
        
        logger.info(f"Validation complete. Overall valid: {results['overall_valid']}")
        
        return results