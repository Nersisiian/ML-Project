# pipelines/data_pipeline/transformation/feature_store.py
from feast import FeatureStore, Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String, Array
from datetime import timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import hashlib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array, struct, when
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructType, StructField

# ============= FEATURE STORE DEFINITION =============

property_entity = Entity(
    name="property_id",
    value_type=ValueType.STRING,
    description="Unique property identifier",
)

location_entity = Entity(
    name="location_id",
    value_type=ValueType.STRING,
    description="Location hash for geographic features",
)

# Source configuration
batch_source = FileSource(
    path="s3://real-estate-feast/features/*.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature View 1: Property Characteristics
property_features = FeatureView(
    name="property_characteristics",
    entities=[property_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="square_feet", dtype=Float32),
        Field(name="bedrooms", dtype=Int64),
        Field(name="bathrooms", dtype=Float32),
        Field(name="year_built", dtype=Int64),
        Field(name="lot_size", dtype=Float32),
        Field(name="garage_spaces", dtype=Int64),
        Field(name="pool", dtype=Int64),
        Field(name="fireplace", dtype=Int64),
        Field(name="basement", dtype=Int64),
        Field(name="condition_score", dtype=Float32),
    ],
    source=batch_source,
    tags={"category": "property", "update_frequency": "daily"},
)

# Feature View 2: Location-based features
location_features = FeatureView(
    name="location_features",
    entities=[location_entity],
    ttl=timedelta(days=180),
    schema=[
        Field(name="median_income", dtype=Float32),
        Field(name="crime_rate", dtype=Float32),
        Field(name="school_rating", dtype=Float32),
        Field(name="walk_score", dtype=Int64),
        Field(name="transit_score", dtype=Int64),
        Field(name="distance_to_downtown", dtype=Float32),
        Field(name="distance_to_park", dtype=Float32),
        Field(name="density_population", dtype=Float32),
        Field(name="employment_rate", dtype=Float32),
    ],
    source=batch_source,
    tags={"category": "location", "update_frequency": "weekly"},
)

# Feature View 3: Temporal features (rolling windows)
temporal_features = FeatureView(
    name="temporal_features",
    entities=[property_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="price_per_sqft_rolling_30d", dtype=Float32),
        Field(name="days_on_market_rolling_30d", dtype=Float32),
        Field(name="price_momentum_7d", dtype=Float32),
        Field(name="seasonal_index", dtype=Float32),
        Field(name="year_over_year_growth", dtype=Float32),
    ],
    source=batch_source,
    tags={"category": "temporal", "update_frequency": "daily"},
)

# ============= FEATURE ENGINEERING WITH SPARK =============

class FeatureEngineeringPipeline:
    """Production feature engineering with Spark and Feast"""
    
    def __init__(self, spark: SparkSession, feast_store_path: str):
        self.spark = spark
        self.feast_store = FeatureStore(repo_path=feast_store_path)
        
    def create_complex_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and derived features"""
        # Polynomial features
        df['sqft_bedroom_ratio'] = df['square_feet'] / (df['bedrooms'] + 1)
        df['bath_bedroom_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
        df['age'] = 2024 - df['year_built']
        df['age_squared'] = df['age'] ** 2
        
        # Log transformations for skewed features
        df['log_square_feet'] = np.log1p(df['square_feet'])
        df['log_lot_size'] = np.log1p(df['lot_size'])
        
        # Interaction features
        df['location_quality_score'] = (
            df['school_rating'] * 0.4 + 
            (100 - df['crime_rate']) * 0.3 + 
            df['walk_score'] * 0.3
        )
        
        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def process_batch(self, input_path: str, output_path: str):
        """Process batch data with Spark for scalability"""
        
        # Read raw data
        df = self.spark.read.parquet(input_path)
        
        # Handle missing values with sophisticated imputation
        df = self._impute_missing_values(df)
        
        # Feature engineering at scale
        df = df.withColumn("price_per_sqft", col("price") / col("square_feet"))
        df = df.withColumn("rooms_total", col("bedrooms") + col("bathrooms"))
        
        # Rolling window aggregations (Window functions)
        from pyspark.sql.window import Window
        window_spec = Window.partitionBy("property_id").orderBy("sale_date").rowsBetween(-30, -1)
        
        df = df.withColumn(
            "rolling_avg_price_30d", 
            avg(col("price")).over(window_spec)
        )
        
        # Geographic clustering
        df = self._add_geo_clusters(df)
        
        # Write processed features
        df.write.mode("overwrite").parquet(output_path)
        
        # Materialize to Feast
        self._materialize_to_feast(df)
        
        return df
    
    def _impute_missing_values(self, df):
        """Smart imputation using median by location group"""
        from pyspark.sql.functions import median, col
        
        # Group by zipcode and impute
        zipcode_medians = df.groupBy("zipcode").agg(
            median("square_feet").alias("median_sqft"),
            median("year_built").alias("median_year")
        )
        
        df = df.join(zipcode_medians, on="zipcode", how="left")
        df = df.fillna({
            "square_feet": col("median_sqft"),
            "year_built": col("median_year"),
            "bedrooms": 2,
            "bathrooms": 1.5,
        })
        
        return df
    
    def _add_geo_clusters(self, df):
        """Add H3 geospatial indices for location clustering"""
        from pyspark.sql.functions import udf
        import h3
        
        @udf(StringType())
        def h3_index(lat, lon, resolution=7):
            return h3.geo_to_h3(lat, lon, resolution)
        
        df = df.withColumn("h3_index_7", h3_index(col("latitude"), col("longitude")))
        df = df.withColumn("h3_index_9", h3_index(col("latitude"), col("longitude"), 9))
        
        return df
    
    def _materialize_to_feast(self, df):
        """Push features to Feast online store for low-latency serving"""
        # Convert to pandas for Feast
        pandas_df = df.select(
            "property_id", "square_feet", "bedrooms", "bathrooms",
            "year_built", "location_id", "median_income", "crime_rate"
        ).toPandas()
        
        # Materialize to online store
        self.feast_store.materialize_incremental(
            end_date=pd.Timestamp.now()
        )