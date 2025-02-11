import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

class FraudDetectionInterpretability:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.shap_explainer = None

    def load_and_split_data(self, test_size=0.2):
        """Load the dataset, split into features and target, and divide into training and testing sets."""
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=['class'])  # Features
        y = data['class']  # Target variable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    def train_model(self):
        """Train the Random Forest model on the training data."""
        self.model.fit(self.X_train, self.y_train)

    def shap_summary_plot(self):
        """Generate SHAP summary plot to visualize feature importance."""
        if not self.shap_explainer:
            self.shap_explainer = shap.Explainer(self.model,self.X_train)
        
        X_test_sample = self.X_test.sample(100, random_state=42)
        shap_values = self.shap_explainer.shap_values(X_test_sample, check_additivity=False)
        
        # Summary plot for global feature importance
        shap.summary_plot(shap_values[..., 1], X_test_sample)
        return shap_values

    def shap_force_plot(self, shap_values, instance_index=0):
        """Generate SHAP force plot for a specific instance to explain the individual prediction."""
        shap.initjs()
        return shap.force_plot(
            self.shap_explainer.expected_value[1], 
            shap_values[instance_index, :], 
            self.X_test_sample.iloc[instance_index]
        )

    def lime_explanation(self, instance_index=0):
        """Generate LIME explanation for a single test instance to interpret individual prediction."""
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values, 
            feature_names=self.X_train.columns.tolist(),
            class_names=['0', '1'],
            mode='classification'
        )
        
        explanation = lime_explainer.explain_instance(
            data_row=self.X_test.values[instance_index], 
            predict_fn=self.model.predict_proba
        )
        explanation.show_in_notebook(show_table=True)
        return explanation
