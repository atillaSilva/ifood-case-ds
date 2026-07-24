from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

class TransactionsTableProcessor:
    """
    Process transaction data from JSON and save to Unity Catalog table.
    Pure PySpark implementation - no pandas dependencies.
    """
    
    def __init__(self, spark_session: SparkSession, 
                 json_path: str,
                 output_table: str = "ifood_case.default.transactions") -> None:
        """
        Initialize the TransactionsTableProcessor.
        
        Args:
            spark_session: Active Spark session
            json_path: Path to the transactions JSON file
            output_table: Full table name for output (catalog.schema.table)
        """
        self.spark: SparkSession = spark_session
        self.json_path: str = json_path
        self.output_table: str = output_table
    
    def load_data(self) -> DataFrame:
        """
        Load transaction data from JSON file.
        
        Returns:
            DataFrame: Raw transaction data
        """
        df: DataFrame = self.spark.read.json(self.json_path)
        return df
    
    def flatten_value_column(self, df: DataFrame) -> DataFrame:
        """
        Flatten nested 'value' column into separate columns.
        
        Args:
            df: Input DataFrame with nested 'value' column
        
        Returns:
            DataFrame: Flattened DataFrame with value fields as top-level columns
        """
        # Select all columns except 'value', then expand value struct
        value_cols = df.select("value.*").columns
        
        # Build select expression: keep non-value columns and expand value struct
        base_cols = [col for col in df.columns if col != 'value']
        select_expr = base_cols + [F.col(f"value.{c}").alias(c) for c in value_cols]
        
        return df.select(*select_expr)
    
    def fill_missing_offer_ids(self, df: DataFrame) -> DataFrame:
        """
        Fill missing offer_id values using 'offer id' column.
        
        Args:
            df: Input DataFrame with potentially missing offer_id values
        
        Returns:
            DataFrame: DataFrame with filled offer_id values
        """
        if 'offer id' in df.columns:
            df = df.withColumn(
                'offer_id',
                F.when(F.col('offer_id').isNull(), F.col('offer id'))
                 .otherwise(F.col('offer_id'))
            )
            df = df.drop('offer id')
        
        return df
    
    def save_to_table(self, df: DataFrame, mode: str = "overwrite") -> None:
        """
        Save DataFrame to Unity Catalog table.
        
        Args:
            df: DataFrame to save
            mode: Write mode (default: 'overwrite')
        """
        df.write.mode(mode).option('overwriteSchema', True).saveAsTable(self.output_table)
        print(f"Data saved to table: {self.output_table}")
        print(f"Total rows: {df.count()}")
    
    def process(self, save: bool = True) -> DataFrame:
        """
        Execute the complete pipeline: load, flatten, clean, and save.
        
        Args:
            save: Whether to save the result to table (default: True)
        
        Returns:
            DataFrame: Processed transaction data
        """
        print("Loading transaction data...")
        df: DataFrame = self.load_data()
        
        print("Flattening nested value column...")
        df = self.flatten_value_column(df)
        
        print("Filling missing offer IDs...")
        df = self.fill_missing_offer_ids(df)
        
        if save:
            print("Saving to table...")
            self.save_to_table(df)
        
        print("Processing complete!")
        return df