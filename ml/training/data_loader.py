import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from feast import FeatureStore
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataLoader:
    """Production data loader with Feast integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_store = None
        self._init_feature_store()
    
    def _init_feature_store(self):
        """Initialize Feast feature store"""
        try:
            self.feature_store = FeatureStore(
                repo_path=self.config.get('feast_repo_path', 'feature_repo')
            )
            logger.info("Feature store initialized")
        except Exception as e:
            logger.warning(f"Feature store not available: {e}")
            self.feature_store = None
    
    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """Load data from parquet file"""
        logger.info(f"Loading data from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_from_feast(
        self, 
        entity_df: pd.DataFrame,
        feature_refs: list
    ) -> pd.DataFrame:
        """Load features from Feast feature store"""
        if not self.feature_store:
            raise ValueError("Feature store not initialized")
        
        logger.info(f"Retrieving {len(feature_refs)} features from Feast")
        
        training_df = self.feature_store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        logger.info(f"Retrieved {len(training_df)} rows")
        return training_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Define feature columns (excluding target and metadata)
        exclude_cols = ['price', 'sale_date', 'property_id', 'created_at']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = np.log1p(df['price'].values)  # Log transform target
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets"""
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_relative_size, 
            random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def create_entity_df(
        self, 
        property_ids: list, 
        event_timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Create entity dataframe for Feast"""
        return pd.DataFrame({
            'property_id': property_ids,
            'event_timestamp': event_timestamp
        })