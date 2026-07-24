# Databricks notebook source
# DBTITLE 1,Setup Path
import sys
sys.path.append('./../..')

# COMMAND ----------

# DBTITLE 1,Import ProfileProcessor
from src.feature_engineering.profile_processor import ProfileProcessor

# COMMAND ----------

# DBTITLE 1,Process Profile Data
# Process profile data
processor = ProfileProcessor()
processor.process('file:/Workspace/Users/atillaa4@gmail.com/iFood - Case/src/data/raw/profile.json')

# COMMAND ----------

