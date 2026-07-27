from typing import Tuple
from pyspark.sql import DataFrame, SparkSession
from config.tables.table_config import TABLE_CONFIG
from pyspark.sql.window import Window
import pyspark.sql.functions as F

class DataProcessor:
    """
    ETL pipeline for processing iFood transaction, offer, and profile data.
    Functional approach - methods return DataFrames directly.
    
    Attributes:
        spark: SparkSession instance
        table_config: Configuration object with table names
    """
    
    def __init__(self, spark_session: SparkSession) -> None:
        """
        Initialize the DataProcessor.
        
        Args:
            spark_session: Active Spark session
            table_config: Table configuration object
        """
        self.spark: SparkSession = spark_session
    
    def load_raw_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Load raw data from configured tables.
        
        Returns:
            tuple: (offers_raw, profile_raw, transactions_raw)
        """
        offers_raw: DataFrame = self.spark.table(TABLE_CONFIG.OFFERS)
        profile_raw: DataFrame = self.spark.table(TABLE_CONFIG.PROFILE)
        transactions_raw: DataFrame = self.spark.table(TABLE_CONFIG.TRASACTIONS)
        return offers_raw, profile_raw, transactions_raw
    
    def fill_missing_transactions_offers(self, transactions_df: DataFrame) -> DataFrame:
        """
        Fill missing offer_id values using window function.
        Uses lag to propagate previous offer_id within same account_id and time window.
        
        Args:
            transactions_df: Input transactions DataFrame
        
        Returns:
            DataFrame: Transactions with filled offer_id values
        """
        w = Window.partitionBy('account_id', 'time_since_test_start') \
                  .orderBy('account_id', 'time_since_test_start', 'event')
        
        return transactions_df \
            .withColumn('time_since_test_start_lag', F.lag('offer_id').over(w)) \
            .withColumn('offer_id', 
                F.when(F.col('offer_id').isNull(), F.col('time_since_test_start_lag'))
                 .otherwise(F.col('offer_id'))
            )
    
    def clean_offers(self, offers_df: DataFrame) -> DataFrame:
        """
        Clean and transform offers data.
        Extracts channel flags and renames id column.
        
        Args:
            offers_df: Input offers DataFrame
        
        Returns:
            DataFrame: Cleaned offers with channel flags
        """
        return offers_df \
            .withColumn("has_web", F.col("channels").contains("web").cast("int")) \
            .withColumn("has_email", F.col("channels").contains("email").cast("int")) \
            .withColumn("has_mobile", F.col("channels").contains("mobile").cast("int")) \
            .withColumn("has_social", F.col("channels").contains("social").cast("int")) \
            .drop("channels") \
            .withColumnRenamed("id", "offer_id")
    
    def extract_transactions(self, transactions_df: DataFrame) -> DataFrame:
        """
        Extract and select relevant transaction columns.
        
        Args:
            transactions_df: Input transactions DataFrame
        
        Returns:
            DataFrame: Extracted transaction columns
        """
        return transactions_df \
            .select(
                "account_id",
                "event",
                "time_since_test_start",
                "offer_id",
                F.col("amount").alias("amount"),
                F.col("reward").alias("reward")
            )
    
    def create_offer_events(self, transactions_extracted_df: DataFrame) -> DataFrame:
        """
        Aggregate transaction events by account and offer.
        Creates conversion flag based on view and completion.
        
        Args:
            transactions_extracted_df: Extracted transactions DataFrame
        
        Returns:
            DataFrame: Aggregated offer events with conversion flag
        """
        return transactions_extracted_df \
            .fillna('no offer', subset=['offer_id']) \
            .fillna(0, subset=['reward']) \
            .filter(F.col("offer_id").isNotNull()) \
            .groupBy("account_id", "offer_id") \
            .agg(
                F.sum(F.when(F.col("event") == "offer received", 1).otherwise(0)).alias("cnt_received"),
                F.sum(F.when(F.col("event") == "offer viewed", 1).otherwise(0)).alias("cnt_viewed"),
                F.sum(F.when(F.col("event") == "offer completed", 1).otherwise(0)).alias("cnt_completed")
            ) \
            .withColumn("is_converted_valid", 
                F.when((F.col("cnt_viewed") > 0) & (F.col("cnt_completed") > 0), 1).otherwise(0)
            )
    
    def calculate_user_spend_stats(self, transactions_extracted_df: DataFrame) -> DataFrame:
        """
        Calculate aggregate spending statistics per user.
        
        Args:
            transactions_extracted_df: Extracted transactions DataFrame
        
        Returns:
            DataFrame: User spending statistics
        """
        return transactions_extracted_df \
            .filter(F.col("event") == "transaction") \
            .groupBy("account_id") \
            .agg(
                F.count("amount").alias("total_transactions"),
                F.sum("amount").alias("total_spent"),
                F.avg("amount").alias("avg_ticket"),
                F.max("amount").alias("max_spent")
            )
    
    def create_final_dataset(self, offer_events_df: DataFrame, profile_df: DataFrame, 
                             offers_clean_df: DataFrame, user_spend_stats_df: DataFrame) -> DataFrame:
        """
        Join all processed data sources into final consolidated dataset.
        
        Args:
            offer_events_df: Aggregated offer events
            profile_df: User profile data
            offers_clean_df: Cleaned offers data
            user_spend_stats_df: User spending statistics
        
        Returns:
            DataFrame: Final consolidated dataset
        """
        offer_events_df = offer_events_df \
            .join(profile_df, on="account_id", how="inner") \
            .join(offers_clean_df, on="offer_id", how="left") \
            .join(user_spend_stats_df, on="account_id", how="left") \
            .na.fill({
                "total_transactions": 0, 
                "total_spent": 0.0, 
                "avg_ticket": 0.0, 
                "max_spent": 0.0
            })
        return offer_events_df
    
    def save(self, df: DataFrame, mode: str = 'overwrite') -> None:
        """
        Save DataFrame to configured table.
        
        Args:
            df: DataFrame to save
            mode: Write mode (default: 'overwrite')
        """
        df.write.mode(mode).option('overwriteSchema', True).saveAsTable(TABLE_CONFIG.DATA_PROCESSING)
    
    def process(self) -> DataFrame:
        """
        Execute the full ETL pipeline.
        
        Args:
            save: Whether to save the result (default: True)
        
        Returns:
            DataFrame: Final processed dataset
        """
        # Load raw data
        offers_raw: DataFrame
        profile_raw: DataFrame
        transactions_raw: DataFrame
        offers_raw, profile_raw, transactions_raw = self.load_raw_data()
        
        # Transform data
        transactions_filled: DataFrame = self.fill_missing_transactions_offers(transactions_raw)
        offers_clean: DataFrame = self.clean_offers(offers_raw)
        transactions_extracted: DataFrame = self.extract_transactions(transactions_filled)
        offer_events: DataFrame = self.create_offer_events(transactions_extracted)
        user_spend_stats: DataFrame = self.calculate_user_spend_stats(transactions_extracted)
        final_dataset: DataFrame = self.create_final_dataset(offer_events, profile_raw, offers_clean, user_spend_stats)

        self.save(final_dataset)
        
        return final_dataset