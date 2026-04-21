import boto3
import pandas as pd
from typing import List, Optional, Dict, Any
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class S3Loader:
    """S3 data loader for batch processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key'),
            region_name=config.get('region_name', 'us-west-2'),
            endpoint_url=config.get('endpoint_url')  # For MinIO
        )
        self.bucket = config['bucket']
    
    def load_parquet(self, key: str) -> pd.DataFrame:
        """Load parquet file from S3"""
        
        logger.info(f"Loading parquet from s3://{self.bucket}/{key}")
        
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        logger.info(f"Loaded {len(df)} rows from {key}")
        return df
    
    def load_csv(self, key: str, **kwargs) -> pd.DataFrame:
        """Load CSV file from S3"""
        
        logger.info(f"Loading CSV from s3://{self.bucket}/{key}")
        
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_csv(BytesIO(response['Body'].read()), **kwargs)
        
        logger.info(f"Loaded {len(df)} rows from {key}")
        return df
    
    def load_multiple(self, prefix: str, file_type: str = 'parquet') -> pd.DataFrame:
        """Load multiple files with given prefix"""
        
        logger.info(f"Loading files with prefix: {prefix}")
        
        # List all files
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        
        dfs = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith(file_type):
                        if file_type == 'parquet':
                            df = self.load_parquet(key)
                        else:
                            df = self.load_csv(key)
                        dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(result)} rows from {len(dfs)} files")
            return result
        else:
            logger.warning(f"No files found with prefix {prefix}")
            return pd.DataFrame()
    
    def save_parquet(self, df: pd.DataFrame, key: str):
        """Save dataframe as parquet to S3"""
        
        logger.info(f"Saving parquet to s3://{self.bucket}/{key}")
        
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer
        )
        
        logger.info(f"Saved {len(df)} rows to {key}")
    
    def file_exists(self, key: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
    
    def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix"""
        
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    files.append(obj['Key'])
        
        return files