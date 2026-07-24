from typing import Dict, List, Tuple, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from config.tables.table_config import TABLE_CONFIG
import mlflow
import mlflow.spark
from mlflow.models import infer_signature


class ModelTrainer:
    """
    Machine learning pipeline for training offer-specific models.
    Trains separate RandomForest models for each offer type.
    
    Attributes:
        spark: SparkSession instance
        table_config: Configuration object with table names
        feature_cols: List of feature column names
        models: Dictionary storing trained models by offer type
    """
    
    def __init__(self, spark_session: SparkSession) -> None:
        """
        Initialize the ModelTrainer.
        
        Args:
            spark_session: Active Spark session
            table_config: Table configuration object
        """
        self.spark: SparkSession = spark_session
        self.feature_cols: List[str] = [
            "age",
            "gender_idx",
            "credit_card_limit",
            "registered_on_days",
            "total_transactions",
            "avg_ticket",
            "max_spent",
            "min_value",
            "duration",
            "discount_value",
            "has_web",
            "has_email",
            "has_mobile",
            "has_social"
        ]
        self.offer_types = ['informational', 'bogo', 'discount', 'no_offer']
        self.models: Dict[str, RandomForestRegressionModel] = {}
    
    def load_data(self) -> DataFrame:
        """
        Load processed data from configured table.
        
        Returns:
            DataFrame: Processed data ready for feature engineering
        """
        df: DataFrame = self.spark.table(TABLE_CONFIG.DATA_PROCESSING)
        return df
    
    def encode_gender(self, df: DataFrame) -> DataFrame:
        """
        Encode gender column using manual mapping.
        Avoids Spark Connect model size limit.
        
        Args:
            df: Input DataFrame with gender column
        
        Returns:
            DataFrame: DataFrame with gender_idx column
        """
        return df.withColumn(
            "gender_idx",
            F.when(F.col("gender") == "F", 0.0)
             .when(F.col("gender") == "M", 1.0)
             .when(F.col("gender") == "O", 2.0)
        )
    
    def prepare_features(self, df: DataFrame) -> DataFrame:
        """
        Prepare features for modeling: fill nulls and assemble feature vector.
        
        Args:
            df: Input DataFrame with encoded features
        
        Returns:
            DataFrame: DataFrame with assembled features vector
        """
        # Fill missing values
        df = df.fillna(0, subset=["has_web", "has_email", "has_mobile", "has_social", 
                                  "discount_value", "duration", "min_value"])
        df = df.fillna('no_offer', ['offer_type'])
        
        # Assemble features
        assembler: VectorAssembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        
        return assembler.transform(df)
    
    def split_and_save(self, df: DataFrame, offer_type: str, 
                      train_ratio: float = 0.8, seed: int = 42) -> Tuple[DataFrame, DataFrame]:
        """
        Split data for specific offer type and save to tables.
        
        Args:
            df: Full dataset
            offer_type: Offer type to filter
            train_ratio: Training data ratio (default: 0.8)
            seed: Random seed for reproducibility (default: 42)
        
        Returns:
            tuple: (train_df, test_df)
        """
        # Filter for offer type
        offer_df: DataFrame = df.filter(F.col("offer_type") == offer_type)
        
        # Split data
        train_df: DataFrame
        test_df: DataFrame
        train_df, test_df = offer_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
        
        # Save to tables
        train_df.write.mode("overwrite").saveAsTable(f"ifood_case.default.train_{offer_type}")
        test_df.write.mode("overwrite").saveAsTable(f"ifood_case.default.test_{offer_type}")
        
        return train_df, test_df
    
    def train_model(self, train_df: DataFrame, num_trees: int = 100, 
                   max_depth: int = 15) -> RandomForestRegressionModel:
        """
        Train a RandomForest regression model.
        
        Args:
            train_df: Training DataFrame
            num_trees: Number of trees (default: 100)
            max_depth: Maximum tree depth (default: 15)
        
        Returns:
            RandomForestRegressionModel: Trained model
        """
        model: RandomForestRegressionModel = RandomForestRegressor(
            featuresCol="features",
            labelCol="total_spent",
            numTrees=num_trees,
            maxDepth=max_depth
        ).fit(train_df)
        
        return model
    
    def evaluate_model(self, model: RandomForestRegressionModel, 
                      test_df: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
        
        Returns:
            dict: Dictionary with MAE, RMSE, and R2 scores
        """
        # Make predictions
        predictions: DataFrame = model.transform(test_df)
        
        # Calculate metrics
        predictions = predictions.withColumn(
            "abs_error", 
            F.abs(F.col("total_spent") - F.col("prediction"))
        ).withColumn(
            "squared_error",
            F.pow(F.col("total_spent") - F.col("prediction"), 2)
        )
        
        metrics_row = predictions.agg(
            F.avg("abs_error").alias("MAE"),
            F.sqrt(F.avg("squared_error")).alias("RMSE"),
            F.corr("total_spent", "prediction").alias("correlation")
        ).collect()[0]
        
        metrics: Dict[str, float] = {
            "mae": metrics_row['MAE'],
            "rmse": metrics_row['RMSE'],
            "r2": metrics_row['correlation'] ** 2 if metrics_row['correlation'] else 0.0
        }
        
        return metrics
    
    def save_model_to_uc(self, model: RandomForestRegressionModel, 
                        offer_type: str, 
                        metrics: Dict[str, float],
                        train_df: DataFrame,
                        catalog: str = "ifood_case",
                        schema: str = "default") -> str:
        """
        Save model to Unity Catalog with MLflow.
        
        Args:
            model: Trained model to save
            offer_type: Offer type identifier
            metrics: Dictionary of evaluation metrics
            train_df: Training DataFrame for signature inference
            catalog: UC catalog name (default: "ifood_case")
            schema: UC schema name (default: "default")
        
        Returns:
            str: Registered model URI
        """
        model_name = f"{catalog}.{schema}.rf_model_{offer_type}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"rf_{offer_type}") as run:
            # Log parameters
            mlflow.log_param("offer_type", offer_type)
            mlflow.log_param("num_trees", model.getNumTrees)
            mlflow.log_param("max_depth", model.getMaxDepth())
            mlflow.log_param("num_features", len(self.feature_cols))
            
            # Log metrics
            mlflow.log_metric("test_mae", metrics["mae"])
            mlflow.log_metric("test_rmse", metrics["rmse"])
            mlflow.log_metric("test_r2", metrics["r2"])
            
            # Infer signature from training data
            sample_input = train_df.select("features").limit(10).toPandas()
            sample_output = model.transform(train_df.limit(10)).select("prediction").toPandas()
            signature = infer_signature(sample_input, sample_output)
            
            # Log model to UC
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )
            
            print(f"Model saved to UC: {model_name}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R²: {metrics['r2']:.4f}")
        
        return model_name
    
    def train_all_models(self, df: DataFrame, save_to_uc: bool = False,
                        catalog: str = "ifood_case",
                        schema: str = "default") -> Tuple[Dict[str, RandomForestRegressionModel], Dict[str, Dict[str, float]]]:
        """
        Train models for all offer types.
        
        Args:
            df: Prepared DataFrame with features
            save_to_uc: Whether to save models to Unity Catalog (default: False)
            catalog: UC catalog name (default: "ifood_case")
            schema: UC schema name (default: "default")
        
        Returns:
            tuple: (models_dict, metrics_dict)
        """
        all_metrics: Dict[str, Dict[str, float]] = {}
        
        for offer in self.offer_types:
            print(f"\n{'='*50}")
            print(f"Training model for: {offer}")
            print(f"{'='*50}")
            
            train_df, test_df = self.split_and_save(df, offer)
            model = self.train_model(train_df)
            
            # Evaluate model
            metrics = self.evaluate_model(model, test_df)
            all_metrics[offer] = metrics
            
            print(f"Metrics - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            
            # Save to UC if requested
            if save_to_uc:
                self.save_model_to_uc(model, offer, metrics, train_df, catalog, schema)
            
            # Store model only after saving to UC (avoid cache buildup)
            self.models[offer] = model
            
            # Clear references to free cache for Spark Connect
            del model, train_df, test_df
        
        return self.models, all_metrics
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Returns:
            dict: Dictionary of metrics by offer type
        """
        results: Dict[str, Dict[str, float]] = {}
        
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60)
        
        for offer in self.offer_types:
            test_df: DataFrame = self.spark.table(f'ifood_case.default.test_{offer}')
            metrics: Dict[str, float] = self.evaluate_model(self.models[offer], test_df)
            results[offer] = metrics
            print(f'\n{offer}:')
            print(f'  MAE:  {metrics["mae"]:.2f}')
            print(f'  RMSE: {metrics["rmse"]:.2f}')
            print(f'  R²:   {metrics["r2"]:.4f}')
        
        return results
    
    def run_pipeline(self, train: bool = True, evaluate: bool = True,
                    save_to_uc: bool = False,
                    catalog: str = "ifood_case",
                    schema: str = "default") -> Tuple[Dict[str, RandomForestRegressionModel], Dict[str, Dict[str, float]]]:
        """
        Execute the full ML pipeline.
        
        Args:
            train: Whether to train models (default: True)
            evaluate: Whether to evaluate models (default: True)
            save_to_uc: Whether to save models to Unity Catalog (default: False)
            catalog: UC catalog name (default: "ifood_case")
            schema: UC schema name (default: "default")
        
        Returns:
            tuple: (models_dict, metrics_dict)
        """
        # Load and prepare data
        print("Loading data...")
        df: DataFrame = self.load_data()
        
        print("Encoding features...")
        df = self.encode_gender(df)
        
        print("Preparing features...")
        df = self.prepare_features(df)
        
        # Train models
        models: Dict[str, RandomForestRegressionModel] = {}
        train_metrics: Dict[str, Dict[str, float]] = {}
        
        if train:
            print("\nTraining models...")
            models, train_metrics = self.train_all_models(df, save_to_uc=save_to_uc, catalog=catalog, schema=schema)
        
        # Evaluate models (only if not already evaluated during training)
        results: Dict[str, Dict[str, float]] = {}
        if evaluate and self.models and not train_metrics:
            print("\nEvaluating models...")
            results = self.evaluate_all_models()
        else:
            results = train_metrics
        
        return models, results