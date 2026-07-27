# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Setup Path
import sys
sys.path.append('./../..')

# COMMAND ----------

# DBTITLE 1,Import ProfileProcessor
from src.feature_engineering.profile_processor import ProfileProcessor

# COMMAND ----------

# Process profile data
processor = ProfileProcessor()
processor.process('file:/Workspace/Users/atillaa4@gmail.com/ifood-case-ds/src/data/raw/profile.json')

# COMMAND ----------

spark.read.json('file:/Workspace/Users/atillaa4@gmail.com/ifood-case-ds/src/data/raw/profile.json')