# app/services/storage.py
import boto3
from botocore.config import Config

# Настрой Cloudflare R2 (бесплатно!)
r2_config = {
    'endpoint_url': 'https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com',
    'aws_access_key_id': 'YOUR_R2_ACCESS_KEY',
    'aws_secret_access_key': 'YOUR_R2_SECRET_KEY',
    'region_name': 'auto'
}

s3_client = boto3.client('s3', **r2_config)

# Загружай модели как обычно!
s3_client.upload_file('model.pkl', 'ml-models', 'model.pkl')