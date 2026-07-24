# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup Path
import sys
sys.path.append('./../..')

# COMMAND ----------

# DBTITLE 1,Import DataProcessor
from src.feature_engineering.data_processor import DataProcessor

# COMMAND ----------

# DBTITLE 1,Run ETL Pipeline
# Initialize and run the ETL pipeline
processor = DataProcessor(spark)
final_df = processor.process()