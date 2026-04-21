from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import logging

logger = logging.getLogger(__name__)

class SparkDataTransformer:
    """Spark-based data transformation for large-scale processing"""
    
    def __init__(self, app_name: str = "RealEstateDataPipeline"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        
        logger.info("Spark session created")
    
    def load_parquet(self, path: str):
        """Load parquet files"""
        df = self.spark.read.parquet(path)
        logger.info(f"Loaded {df.count()} rows from {path}")
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        
        # Remove duplicates
        df = df.dropDuplicates(['property_id'])
        
        # Handle missing values
        df = df.fillna({
            'square_feet': 0,
            'bedrooms': 2,
            'bathrooms': 1.5,
            'year_built': 2000,
            'lot_size': 0,
            'condition_score': 5
        })
        
        # Filter outliers
        df = df.filter(
            (col('square_feet') > 100) & 
            (col('square_feet') < 50000) &
            (col('bedrooms') <= 10) &
            (col('price') > 10000) &
            (col('price') < 10000000)
        )
        
        logger.info(f"Data cleaned. Remaining rows: {df.count()}")
        return df
    
    def create_features(self, df):
        """Create derived features using Spark SQL"""
        
        df = df.withColumn("property_age", lit(2024) - col("year_built"))
        df = df.withColumn("property_age_squared", col("property_age") ** 2)
        df = df.withColumn("price_per_sqft", col("price") / col("square_feet"))
        df = df.withColumn("rooms_total", col("bedrooms") + col("bathrooms"))
        df = df.withColumn("sqft_per_room", col("square_feet") / col("rooms_total"))
        
        # Log transformations
        df = df.withColumn("log_price", log1p(col("price")))
        df = df.withColumn("log_sqft", log1p(col("square_feet")))
        
        # Categorical encoding for zipcode
        from pyspark.ml.feature import StringIndexer, OneHotEncoder
        
        indexer = StringIndexer(inputCol="zipcode", outputCol="zipcode_index")
        df = indexer.fit(df).transform(df)
        
        encoder = OneHotEncoder(inputCol="zipcode_index", outputCol="zipcode_encoded")
        df = encoder.fit(df).transform(df)
        
        logger.info(f"Feature engineering complete")
        return df
    
    def create_aggregations(self, df):
        """Create aggregated features by zipcode"""
        
        # Window for rolling averages
        window_spec = Window.partitionBy("zipcode").orderBy("sale_date").rowsBetween(-30, -1)
        
        df = df.withColumn(
            "rolling_avg_price_30d",
            avg("price").over(window_spec)
        )
        
        df = df.withColumn(
            "rolling_avg_price_90d",
            avg("price").over(Window.partitionBy("zipcode").rowsBetween(-90, -1))
        )
        
        # Price rank within zipcode
        df = df.withColumn(
            "price_rank_zipcode",
            percent_rank().over(Window.partitionBy("zipcode").orderBy("price"))
        )
        
        return df
    
    def save_parquet(self, df, path: str, mode: str = 'overwrite'):
        """Save dataframe to parquet"""
        df.write.mode(mode).parquet(path)
        logger.info(f"Saved data to {path}")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
        logger.info("Spark session stopped")