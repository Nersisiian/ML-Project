import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Advanced feature engineering pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = RobustScaler()
        self.pca = None
        self.feature_columns = None
        self.fitted = False
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from base features"""
        df_copy = df.copy()
        
        # Age features
        current_year = 2024
        df_copy['property_age'] = current_year - df_copy['year_built']
        df_copy['property_age_squared'] = df_copy['property_age'] ** 2
        df_copy['property_age_log'] = np.log1p(df_copy['property_age'])
        
        # Ratio features
        df_copy['bed_bath_ratio'] = df_copy['bedrooms'] / (df_copy['bathrooms'] + 1)
        df_copy['sqft_per_bedroom'] = df_copy['square_feet'] / (df_copy['bedrooms'] + 1)
        df_copy['sqft_per_room'] = df_copy['square_feet'] / (df_copy['bedrooms'] + df_copy['bathrooms'] + 1)
        
        # Interaction features
        df_copy['size_age_interaction'] = df_copy['square_feet'] * df_copy['property_age']
        df_copy['bed_age_interaction'] = df_copy['bedrooms'] * df_copy['property_age']
        
        # Polynomial features (for top features)
        df_copy['sqft_squared'] = df_copy['square_feet'] ** 2
        df_copy['lot_sqrt'] = np.sqrt(df_copy['lot_size'])
        
        # Log transformations
        df_copy['log_sqft'] = np.log1p(df_copy['square_feet'])
        df_copy['log_lot'] = np.log1p(df_copy['lot_size'])
        
        # Condition-based features
        df_copy['is_new_construction'] = (df_copy['property_age'] < 5).astype(int)
        df_copy['is_old'] = (df_copy['property_age'] > 50).astype(int)
        df_copy['is_luxury'] = (
            (df_copy['square_feet'] > 3000) & 
            (df_copy['bedrooms'] > 4) & 
            (df_copy['pool'] == 1)
        ).astype(int)
        
        # Location-based features (if coordinates available)
        if 'latitude' in df_copy.columns and 'longitude' in df_copy.columns:
            df_copy['lat_lon_interaction'] = df_copy['latitude'] * df_copy['longitude']
        
        return df_copy
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df_copy = df.copy()
        
        if 'sale_date' in df_copy.columns:
            df_copy['sale_date'] = pd.to_datetime(df_copy['sale_date'])
            df_copy['sale_year'] = df_copy['sale_date'].dt.year
            df_copy['sale_month'] = df_copy['sale_date'].dt.month
            df_copy['sale_quarter'] = df_copy['sale_date'].dt.quarter
            df_copy['sale_day_of_week'] = df_copy['sale_date'].dt.dayofweek
            
            # Cyclical encoding
            df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['sale_month'] / 12)
            df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['sale_month'] / 12)
            df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['sale_day_of_week'] / 7)
            df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['sale_day_of_week'] / 7)
        
        return df_copy
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features by zipcode"""
        df_copy = df.copy()
        
        # Group by zipcode
        zipcode_stats = df_copy.groupby('zipcode').agg({
            'price': ['mean', 'median', 'std'],
            'square_feet': 'mean',
            'property_age': 'mean'
        }).round(2)
        
        zipcode_stats.columns = ['_'.join(col).strip() for col in zipcode_stats.columns.values]
        zipcode_stats = zipcode_stats.reset_index()
        
        # Merge back
        df_copy = df_copy.merge(zipcode_stats, on='zipcode', how='left')
        
        # Price relative to zipcode average
        df_copy['price_vs_zip_avg'] = df_copy['price'] / df_copy['price_mean']
        df_copy['price_vs_zip_median'] = df_copy['price'] / df_copy['price_median']
        
        return df_copy
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale features using RobustScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted yet")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def apply_pca(self, X: np.ndarray, n_components: int = 50, fit: bool = True) -> np.ndarray:
        """Apply PCA for dimensionality reduction"""
        if fit:
            self.pca = PCA(n_components=min(n_components, X.shape[1]))
            X_pca = self.pca.fit_transform(X)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted yet")
            X_pca = self.pca.transform(X)
        
        return X_pca
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit pipeline and transform data"""
        # Create all features
        df = self.create_derived_features(df)
        df = self.create_temporal_features(df)
        df = self.create_aggregate_features(df)
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_columns = list(numeric_cols)
        
        # Convert to numpy
        X = df[numeric_cols].values
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        
        # Apply PCA if configured
        if self.config.get('use_pca', False):
            X_scaled = self.apply_pca(X_scaled, fit=True)
        
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline"""
        if not self.fitted:
            raise ValueError("Pipeline not fitted yet. Call fit_transform first.")
        
        # Create features
        df = self.create_derived_features(df)
        df = self.create_temporal_features(df)
        df = self.create_aggregate_features(df)
        
        # Select features
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale
        X_scaled = self.scale_features(X, fit=False)
        
        # Apply PCA
        if self.config.get('use_pca', False):
            X_scaled = self.apply_pca(X_scaled, fit=False)
        
        return X_scaled