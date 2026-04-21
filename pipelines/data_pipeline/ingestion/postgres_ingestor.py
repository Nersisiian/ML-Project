import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PostgresIngestor:
    """PostgreSQL data ingestor for batch processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Create database connection"""
        try:
            connection_string = (
                f"postgresql://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config.get('port', 5432)}"
                f"/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def ingest_table(self, table_name: str, 
                     columns: Optional[List[str]] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
        """Ingest data from a table"""
        
        query = f"SELECT * FROM {table_name}"
        
        if columns:
            query = f"SELECT {', '.join(columns)} FROM {table_name}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query: {query}")
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"Ingested {len(df)} rows from {table_name}")
        
        return df
    
    def ingest_custom_query(self, query: str) -> pd.DataFrame:
        """Ingest data using custom SQL query"""
        
        logger.info(f"Executing custom query")
        df = pd.read_sql(query, self.engine)
        logger.info(f"Ingested {len(df)} rows")
        
        return df
    
    def ingest_incremental(self, table_name: str, 
                          timestamp_column: str,
                          last_timestamp: datetime) -> pd.DataFrame:
        """Ingest incremental data since last timestamp"""
        
        query = f"""
            SELECT * FROM {table_name}
            WHERE {timestamp_column} > '{last_timestamp}'
            ORDER BY {timestamp_column}
        """
        
        df = pd.read_sql(query, self.engine)
        logger.info(f"Ingested {len(df)} incremental rows")
        
        return df
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get table schema information"""
        
        query = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """
        
        return pd.read_sql(query, self.engine)
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")