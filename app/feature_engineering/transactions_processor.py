# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup Path
import sys
sys.path.append('./../..')

# COMMAND ----------

from src.feature_engineering.transactions_processor import TransactionsTableProcessor

# COMMAND ----------

# Initialize and run the processor
processor = TransactionsTableProcessor(
    json_path='file:/Workspace/Users/atillaa4@gmail.com/ifood-case-ds/src/data/raw/transactions.json',
)

df_transactions = processor.process()