import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import logging

# Create log directory if not exists
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging to log both to a file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/pipeline.log"),
        logging.StreamHandler()
    ]
)

# Set MLflow tracking URI to the root directory
mlflow.set_tracking_uri("file:///E:/Kiffya_10_acc/Week%208-9/Fraud-Detection/mlruns")


class ModelPipeline:
    """Class to handle data loading, splitting, model training, evaluation, and logging."""

    def __init__(self, dataset_type, path):
        """
        Initialize the pipeline with dataset type and file path.
        
        dataset_type: A string ('creditcard' or 'fraud') to indicate dataset type.
        path: File path for the dataset.
        """
        self.dataset_type = dataset_type
        self.path = path
        self.data = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Set different experiments based on the dataset
        if self.dataset_type == 'creditcard':
            self.experiment_name = 'creditcard_experiment'
        elif self.dataset_type == 'fraud':
            self.experiment_name = 'Fraud_Detection_Experiment'
        else:
            raise ValueError("Invalid dataset_type! Must be 'creditcard' or 'fraud'")

        # Set the experiment for MLflow
        mlflow.set_experiment(self.experiment_name)

    def load_data(self):
        """Load data based on the dataset type."""
        if self.dataset_type == 'creditcard':
            logging.info(f"Loading credit card data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'Class'  # Target column for creditcard dataset

        elif self.dataset_type == 'fraud':
            logging.info(f"Loading fraud data from {self.path}...")
            self.data = pd.read_csv(self.path)
            self.target = 'class'  # Target column for fraud dataset

        else:
            raise ValueError("Invalid dataset_type! Must be 'creditcard' or 'fraud'")
        
        logging.info("Data loading complete.")

    def split_data(self, test_size=0.2, random_state=42):
        """Split the loaded data into training and test sets."""
        if self.data is not None:
            X = self.data.drop(columns=[self.target])
            y = self.data[self.target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logging.info("Data has been split into train and test sets.")
        else:
            raise ValueError("Data not loaded. Please load the data first.")
        
    def apply_smote(self):
        """Apply SMOTE to balance the training data."""
        if self.X_train is not None and self.y_train is not None:
            logging.info("Applying SMOTE to the training data...")
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logging.info("SMOTE applied to training data. Classes have been balanced.")
        else:
            raise ValueError("Training data is not available. Please split the data first.")
        
    def train_model(self, model, model_name):
        """Train the model with the training data."""
        logging.info(f"Training {model_name} on {self.dataset_type} dataset...")
        model.fit(self.X_train, self.y_train)
        logging.info(f"{model_name} training complete.")

    def evaluate_model(self, model, model_name):
        """Evaluate the model using the test data and return the classification report."""
        logging.info(f"Evaluating {model_name} on {self.dataset_type} dataset...")
        y_pred = model.predict(self.X_test)
        
        report = classification_report(self.y_test, y_pred, output_dict=True)
        logging.info(f"{model_name} evaluation report:\n{classification_report(self.y_test, y_pred)}")
        return report

    def log_model(self, model, model_name, report):
        """Log the model, performance metrics, and save the model artifact to MLflow."""
        logging.info(f"Logging {model_name} to MLflow...")

        # Create the model save path
        model_dir = "../saved_models"
        os.makedirs(model_dir, exist_ok=True)

        # Determine the next version number
        version = 1
        model_path = os.path.join(model_dir, f"{self.dataset_type}_{model_name}_v{version}.pkl")
        while os.path.exists(model_path):
            version += 1
            model_path = os.path.join(model_dir, f"{self.dataset_type}_{model_name}_v{version}.pkl")

        # Save the model locally
        mlflow.sklearn.save_model(model, model_path)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log model parameters if available
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())

            # Log classification metrics
            mlflow.log_metrics({
                "precision": report['1']['precision'],
                "recall": report['1']['recall'],
                "f1-score": report['1']['f1-score'],
                "accuracy": report['accuracy']
            })
            
            # Log the saved model to MLflow
            mlflow.sklearn.log_model(model, f"{self.dataset_type}_{model_name}_model")
            mlflow.log_artifact(model_path)  # Save the model artifact for future use
            
            logging.info(f"{model_name} has been logged and saved in MLflow as version {version}.")

    def run_pipeline(self):
        """Run the entire pipeline from loading data to training and logging models."""
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split data
        self.split_data()
        self.apply_smote()
        # Step 3: Train and evaluate multiple models
        models = [
            (LogisticRegression(), 'Logistic Regression'),
            (DecisionTreeClassifier(), 'Decision Tree'),
            (RandomForestClassifier(), 'Random Forest'),
            (GradientBoostingClassifier(), 'Gradient Boosting')
        ]
        
        for model, name in models:
            self.train_model(model, name)
            report = self.evaluate_model(model, name)
            self.log_model(model, name, report)
