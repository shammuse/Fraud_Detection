# model.py

import joblib
import pandas as pd

class FraudModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        # Predefined mapping for countries
        self.country_mapping = {
            'USA': 0,
            'Canada': 1,
            'UK': 2,
            # Add more countries as needed
        }

        # Keep track of one-hot encoded column names for consistency
        self.required_columns = []

    def encode_country(self, country):
        return self.country_mapping.get(country, -1)  # Return -1 for unknown countries

    def preprocess_input(self, input_data):
        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # Encode the country
        input_df['country_encoded'] = self.encode_country(input_data['country'])

        # One-hot encode 'source' and 'browser'
        input_df = pd.get_dummies(input_df, columns=['source', 'browser'], prefix=['source', 'browser'])

        # Extract time features from purchase_time and signup_time
        input_df['signup_time'] = pd.to_datetime(input_df['signup_time'])
        input_df['purchase_time'] = pd.to_datetime(input_df['purchase_time'])

        # Extract hour and day of the week
        input_df['hour_of_day'] = input_df['purchase_time'].dt.hour
        input_df['day_of_week'] = input_df['purchase_time'].dt.dayofweek

        # Drop original time columns as they are no longer needed
        input_df.drop(columns=['signup_time', 'purchase_time'], inplace=True)

        # Set required columns dynamically based on trained model
        if not self.required_columns:
            # Retrieve the feature names used during model training
            self.required_columns = self.model.feature_names_in_.tolist()

        # Create missing columns and fill with 0
        for column in self.required_columns:
            if column not in input_df.columns:
                input_df[column] = 0  # Fill missing columns with default values

        return input_df[self.required_columns]

    def predict(self, input_data):
        input_df = self.preprocess_input(input_data)
        prediction = self.model.predict(input_df)
        return prediction[0]  # Return the first prediction
