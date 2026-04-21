#!/usr/bin/env python3
"""
Backfill features for historical data
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBackfiller:
    """Backfill features for historical data"""
    
    def __init__(self, feature_pipeline, batch_size: int = 10000):
        self.feature_pipeline = feature_pipeline
        self.batch_size = batch_size
    
    def backfill(self, df: pd.DataFrame, output_path: str):
        """Backfill features for entire dataframe"""
        
        logger.info(f"Starting backfill for {len(df)} rows")
        
        all_features = []
        
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch = df.iloc[i:i+self.batch_size]
            
            # Create features for batch
            features = self.feature_pipeline.fit_transform(batch)
            all_features.append(features)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}")
        
        # Combine all features
        X = np.vstack(all_features)
        
        # Save features
        np.save(output_path, X)
        logger.info(f"Saved features to {output_path}, shape: {X.shape}")
        
        return X
    
    def backfill_incremental(
        self, 
        df: pd.DataFrame, 
        last_timestamp: datetime,
        timestamp_column: str = 'sale_date'
    ):
        """Backfill only new data"""
        
        df_new = df[df[timestamp_column] > last_timestamp]
        
        if len(df_new) > 0:
            logger.info(f"Backfilling {len(df_new)} new rows")
            return self.backfill(df_new, f"features_new_{datetime.now().strftime('%Y%m%d')}.npy")
        else:
            logger.info("No new data to backfill")
            return None

def main():
    parser = argparse.ArgumentParser(description='Backfill features')
    parser.add_argument('--input', required=True, help='Input parquet file')
    parser.add_argument('--output', required=True, help='Output numpy file')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")
    
    # Create feature pipeline
    from ml.training.feature_pipeline import FeaturePipeline
    feature_pipeline = FeaturePipeline(config={})
    
    # Backfill
    backfiller = FeatureBackfiller(feature_pipeline, args.batch_size)
    backfiller.backfill(df, args.output)

if __name__ == "__main__":
    main()