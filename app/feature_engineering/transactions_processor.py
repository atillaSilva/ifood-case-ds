# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup Path
import sys
sys.path.append('./../..')

# COMMAND ----------

from src.feature_engineering.transactions_table_processor import TransactionsTableProcessor

# COMMAND ----------

# Initialize and run the processor
processor = TransactionsTableProcessor(
    spark_session=spark,
    json_path='file:/Workspace/Users/atillaa4@gmail.com/iFood - Case/src/data/raw/transactions.json',
    output_table='ifood_case.default.transactions'
)

df_transactions = processor.process()