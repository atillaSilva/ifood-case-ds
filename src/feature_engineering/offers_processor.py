import pandas as pd
from pyspark.sql import SparkSession
from config.tables.table_config import TABLE_CONFIG
import pyspark.sql.functions as F

class OffersProcessor:
    """
    A class to process offers data from JSON and save to Spark table.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize the OffersProcessor.
        
        Args:
            json_path: Path to the offers JSON file
            table_name: Fully qualified table name for saving (catalog.schema.table)
        """
        self.json_path = json_path
        self.table_name = TABLE_CONFIG.OFFERS
        self.spark = SparkSession.builder.getOrCreate()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load offers data from JSON file.
        
        Returns:
            pandas DataFrame with offers data
        """
        df_offers = self.spark.read.json(self.json_path)
        return df_offers
    
    def explode_channels(self, df_offers: pd.DataFrame) -> pd.DataFrame:
        """
        Explode the channels column to create one row per channel.
        
        Returns:
            pandas DataFrame with exploded channels
        """
        df_offers = df_offers.withColumn('channels', F.explode('channels').alias('channels_1'))
        return df_offers
    
    def formated_df(self, df_offers: pd.DataFrame) -> pd.DataFrame:
        """
        Explode the channels column to create one row per channel.
        
        Returns:
            pandas DataFrame with exploded channels
        """
        df_offers = df_offers.withColumnRenamed('id', 'offer_id')
        return df_offers
    
    def save_to_table(self, df_offers: pd.DataFrame, mode: str = "overwrite") -> None:
        """
        Convert pandas DataFrame to Spark DataFrame and save as table.
        
        Args:
            mode: Write mode ('overwrite', 'append', etc.)
        """
        df_offers.write.mode(mode).saveAsTable(self.table_name)
        print(f"Data saved to table: {self.table_name}")
    
    def process(self) -> None:
        """
        Execute the complete pipeline: load, optionally explode, and save.
        
        Args:
            explode: Whether to explode the channels column
        """
        df_offers = self.load_data()
        df_offers = self.explode_channels(df_offers)
        df_formated = self.formated_df(df_offers)
        df_formated = self.formated_df(df_offers)
        self.save_to_table(df_formated)
        print("Processing complete!")