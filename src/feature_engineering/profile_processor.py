from config.tables.table_config import TABLE_CONFIG
import pyspark.sql.functions as f
import pyspark.sql as DataFrame

class ProfileProcessor:
    def __init__(self):
        self.input_table = TABLE_CONFIG.PROFILE

    def load_profile(self, json_path: str) -> DataFrame:
        df_profile = spark.read.json(json_path)
        return df_profile
    
    def formated_df(self, df_profile: DataFrame) -> DataFrame:
        df_profile = df_profile.withColumnRenamed('id', 'account_id')
        return df_profile
    
    def convert_str_to_date(self, df_profile: DataFrame) -> DataFrame:
        df_profile = df_profile.withColumn('registered_on', f.to_date("registered_on", 'yyyyMMdd'))
        return  df_profile
    
    def remove_null_gender_values(self, df_profile: DataFrame) -> DataFrame:
        return df_profile.filter(f.col('gender').isNotNull())
    
    def create_registered_on_days(self, df_profile: DataFrame) -> DataFrame:
        df_profile = df_profile.withColumn('registered_on_days', f.date_diff(f.current_date(), 'registered_on'))
        return df_profile

    def save_profile(self, df_profile: DataFrame, mode: str="overwrite"):
        df_profile.write.mode(mode).option('overwriteSchema', True).saveAsTable(self.input_table)

    def process(self, json_path:str) -> DataFrame:
        df_profile = self.load_profile(json_path)
        df_profile = self.convert_str_to_date(df_profile)
        df_profile = self.remove_null_gender_values(df_profile)
        df_profile = self.create_registered_on_days(df_profile)
        df_profile = self.formated_df(df_profile)
        self.save_profile(df_profile)

# Example usage:
# processor = ProfileProcessor(config)
# df_profile = processor.load_profile('../raw/profile.json')
# processor.save_profile(df_profile)