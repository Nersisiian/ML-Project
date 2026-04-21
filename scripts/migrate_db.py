#!/usr/bin/env python3
"""
Database migration script
"""

import argparse
import psycopg2
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handle database migrations"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
    
    def create_mlflow_tables(self):
        """Create MLflow tracking tables"""
        
        logger.info("Creating MLflow tables...")
        
        queries = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY,
                name VARCHAR(256) NOT NULL,
                artifact_location VARCHAR(256),
                lifecycle_stage VARCHAR(32),
                creation_time BIGINT,
                last_update_time BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR(32) PRIMARY KEY,
                experiment_id INTEGER REFERENCES experiments(experiment_id),
                status VARCHAR(9),
                source_type VARCHAR(20),
                source_name VARCHAR(500),
                user_id VARCHAR(256),
                start_time BIGINT,
                end_time BIGINT,
                source_version VARCHAR(50),
                lifecycle_stage VARCHAR(20),
                artifact_uri VARCHAR(200)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS metrics (
                key VARCHAR(250),
                value DOUBLE PRECISION,
                timestamp BIGINT,
                run_id VARCHAR(32) REFERENCES runs(run_id),
                step BIGINT,
                is_nan BOOLEAN,
                PRIMARY KEY (key, timestamp, step, run_id, value, is_nan)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS params (
                key VARCHAR(500),
                value VARCHAR(500),
                run_id VARCHAR(32) REFERENCES runs(run_id),
                PRIMARY KEY (key, run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tags (
                key VARCHAR(250),
                value VARCHAR(5000),
                run_id VARCHAR(32) REFERENCES runs(run_id),
                PRIMARY KEY (key, run_id)
            )
            """
        ]
        
        with self.engine.connect() as conn:
            for query in queries:
                conn.execute(text(query))
                conn.commit()
        
        logger.info("MLflow tables created")
    
    def create_feast_tables(self):
        """Create Feast feature store tables"""
        
        logger.info("Creating Feast tables...")
        
        queries = [
            """
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_view_name VARCHAR(256) PRIMARY KEY,
                entities TEXT,
                features TEXT,
                source TEXT,
                ttl BIGINT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS online_features (
                feature_name VARCHAR(256),
                entity_key VARCHAR(256),
                feature_value TEXT,
                event_timestamp TIMESTAMP,
                created_timestamp TIMESTAMP,
                PRIMARY KEY (feature_name, entity_key)
            )
            """
        ]
        
        with self.engine.connect() as conn:
            for query in queries:
                conn.execute(text(query))
                conn.commit()
        
        logger.info("Feast tables created")
    
    def create_metadata_tables(self):
        """Create custom metadata tables"""
        
        logger.info("Creating metadata tables...")
        
        queries = [
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(256),
                model_version VARCHAR(50),
                training_date TIMESTAMP,
                metrics JSONB,
                parameters JSONB,
                feature_names TEXT[],
                status VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                request_id VARCHAR(64),
                model_version VARCHAR(50),
                features JSONB,
                prediction FLOAT,
                actual_price FLOAT,
                latency_ms FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_model_version (model_version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS data_drift_metrics (
                id SERIAL PRIMARY KEY,
                feature_name VARCHAR(256),
                drift_score FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_feature (feature_name)
            )
            """
        ]
        
        with self.engine.connect() as conn:
            for query in queries:
                conn.execute(text(query))
                conn.commit()
        
        logger.info("Metadata tables created")
    
    def run_migrations(self):
        """Run all migrations"""
        
        logger.info("Starting database migrations...")
        
        self.create_mlflow_tables()
        self.create_feast_tables()
        self.create_metadata_tables()
        
        logger.info("All migrations completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Run database migrations')
    parser.add_argument('--environment', default='development', help='Environment')
    parser.add_argument('--connection-string', help='Database connection string')
    
    args = parser.parse_args()
    
    # Get connection string
    if args.connection_string:
        conn_string = args.connection_string
    else:
        # Load from environment
        import os
        from dotenv import load_dotenv
        
        load_dotenv(f'.env.{args.environment}')
        
        conn_string = os.getenv('POSTGRES_URL')
        if not conn_string:
            logger.error("POSTGRES_URL not found in environment")
            return
    
    # Run migrations
    migrator = DatabaseMigrator(conn_string)
    migrator.run_migrations()

if __name__ == "__main__":
    main()