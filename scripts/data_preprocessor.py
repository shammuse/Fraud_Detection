import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logg = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path1, file_path2, file_path3):
        """
        Initialize the DataPreprocessor with file paths for three datasets.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.data = None
        self.data1 = None
        self.data2 = None

    def load_data(self):
        """
        Load data from the provided file paths into Pandas DataFrames.
        """
        try:
            self.data = pd.read_csv(self.file_path1)
            self.data1 = pd.read_csv(self.file_path2)
            self.data2 = pd.read_csv(self.file_path3)
            logg.info("Data loaded successfully!")
            return self.data, self.data1, self.data2
        
        except Exception as e:
            logg.error(f"An error occurred while loading data: {e}")
            return None, None, None

    def explore_data(self):
        """
        Display the first few rows of the main dataset for exploration purposes.
        """
        logg.info("Exploring data:")
        logg.info(f"\n{self.data.head()}")

    def check_missing_values(self, data):
        """
        Check for missing values in the provided dataset.
        """
        missing_values = data.isnull().sum()
        logg.info(f"Missing values: \n{missing_values}")
        return missing_values

    def handle_missing_values(self):
        """
        Handle missing values in numerical columns by imputing with the column mean.
        """
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.data[num_cols].isnull().sum().any():
            self.data[num_cols] = self.data[num_cols].fillna(self.data[num_cols].mean())
            logg.info("Missing values in numerical columns handled with mean imputation.")
        else:
            logg.info("No missing values found in numerical columns.")
        return self.data

    def normalize_data(self, data, columns):
        """
        Normalize specified numerical columns of the dataset using Min-Max scaling.
        """
        logg.info("Normalizing data...")
        if all(col in data.columns for col in columns):
            scaler = MinMaxScaler()
            data[columns] = scaler.fit_transform(data[columns])
            logg.info("Dataset normalized successfully using Min-Max scaling.")
            return data
        else:
            missing_cols = [col for col in columns if col not in data.columns]
            logg.error(f"Missing columns for normalization: {missing_cols}")
            return data
        
    def feature_engineering(self, merged_data):
        """
        Perform feature engineering on the dataset by creating new features based on existing columns.
        """
        logg.info("Performing feature engineering...")

        # Ensure 'purchase_time' is a valid datetime
        merged_data['purchase_time'] = pd.to_datetime(merged_data['purchase_time'], errors='coerce')

        # Create 'hour_of_day' and 'day_of_week' features
        merged_data['hour_of_day'] = merged_data['purchase_time'].dt.hour
        merged_data['day_of_week'] = merged_data['purchase_time'].dt.dayofweek  # 0=Monday, 6=Sunday

        logg.info("Feature engineering completed: added 'hour_of_day' and 'day_of_week'.")
        
        # Calculate transaction frequency and velocity
        # self.calculate_transaction_features(merged_data)
        
        return merged_data

    def calculate_transaction_features(self, df):
        """
        Calculate transaction frequency and velocity for each user.
        """
        logg.info("Calculating transaction frequency and velocity...")
        
        # Transaction frequency
        transaction_frequency = df['user_id'].value_counts().reset_index()
        transaction_frequency.columns = ['user_id', 'transaction_frequency']
        
        # Calculate time differences
        df = df.sort_values(by=['user_id', 'purchase_time'])
        df['time_diff'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
        
        # Handle NaN values in time_diff by filling with 0 or forward filling
        df['time_diff'].fillna(0, inplace=True)  # Filling NaN with 0 for first transaction

        # Transaction velocity (average time between transactions)
        transaction_velocity = df.groupby('user_id')['time_diff'].mean().reset_index()
        transaction_velocity.columns = ['user_id', 'average_velocity']
        
        # Merge frequency and velocity back to the main DataFrame
        df = df.merge(transaction_frequency, on='user_id', how='left')
        df = df.merge(transaction_velocity, on='user_id', how='left')

        # Handle NaN values in average_velocity
        # Fill NaN values with the overall mean or any other preferred method
        overall_mean_velocity = df['average_velocity'].mean()
        df['average_velocity'].fillna(overall_mean_velocity, inplace=True)

        logg.info("Transaction features calculated and merged.")
        return df


    def encode_categorical_data(self, df, cat_columns):
        """
        Perform one-hot encoding on specified categorical columns.
        """
        logg.info("Encoding categorical data...")
        
        if all(col in df.columns for col in cat_columns):
            encoded_data = pd.get_dummies(df, columns=cat_columns, drop_first=True)
            logg.info("Categorical data encoded successfully!")

            # Convert only the newly created one-hot encoded columns to integers
            one_hot_cols = [col for col in encoded_data.columns if col.startswith(tuple(cat_columns))]
            encoded_data[one_hot_cols] = encoded_data[one_hot_cols].astype(int)
            
            return encoded_data
        else:
            missing_cols = [col for col in cat_columns if col not in df.columns]
            logg.error(f"Missing columns for encoding: {missing_cols}")
            return df  # Return unchanged data if categorical columns are missing
