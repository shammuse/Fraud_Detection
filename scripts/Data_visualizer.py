import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging as logg

logg.basicConfig(level=logg.INFO)

class DataVisualizer:
    """
    A class for visualizing data using various plotting techniques.

    Attributes:
        data (pd.DataFrame): The input DataFrame containing the dataset to be visualized.
    """

    def __init__(self, data):
        """
        Initializes the DataVisualizer with the provided dataset.

        Args:
            data (pd.DataFrame): The dataset to visualize.
        """
        self.data = data

    def visualize_data(self):
        """Visualizes the data using a pairplot to show relationships between features."""
        sns.pairplot(self.data)
        plt.show()

    def plot_histogram(self, numerical_features):
        """Plots histograms for the specified numerical features.

        Args:
            numerical_features (list): List of numerical feature names to plot.
        """
        plt.figure(figsize=(16, 5))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(1, len(numerical_features), i)
            sns.histplot(self.data[feature], bins=20, kde=True)
            plt.title(f'Histogram for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        logg.info("Histograms plotted successfully!")

    def plot_bar_chart(self, categorical_features: list):
        """Plots bar charts for each specified categorical feature in a grid layout.

        Args:
            categorical_features (list): List of categorical feature names to plot.
        """
        try:
            num_features = len(categorical_features)
            num_cols = 2  # We want 2 columns
            num_rows = (num_features + num_cols - 1) // num_cols
            
            plt.figure(figsize=(num_cols * 6, num_rows * 4))

            for i, feature in enumerate(categorical_features, 1):
                if feature not in self.data.columns:
                    logg.error(f"Feature '{feature}' not found in data!")
                    continue

                plt.subplot(num_rows, num_cols, i)
                sns.barplot(
                    x=self.data[feature].value_counts().index,
                    y=self.data[feature].value_counts().values,
                    palette='viridis'
                )
                plt.title(f'Bar Chart for {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

            logg.info("Bar charts plotted successfully!")
        except Exception as e:
            logg.error(f"An error occurred while plotting bar charts: {e}")

    def plot_scatter_matrix(self, numerical_features):
        """Plots a scatter matrix for the specified numerical features.

        Args:
            numerical_features (list): List of numerical feature names to plot.
        """
        plt.figure(figsize=(16, 10))
        sns.pairplot(self.data[numerical_features], palette='viridis')
        plt.title('Scatter Matrix')
        plt.tight_layout()
        plt.show()
        logg.info("Scatter matrix plotted successfully!")

    def scatter_plot(self, x_feature, y_feature):
        """Plots a scatter plot for two specified features.

        Args:
            x_feature (str): The feature for the x-axis.
            y_feature (str): The feature for the y-axis.
        """
        logg.info("Plotting scatter plot...")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.data[x_feature], y=self.data[y_feature], palette='viridis')
        plt.title(f'Scatter Plot: {x_feature} vs {y_feature}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()

    def plot_box_plot(self, numerical_features):
        """Plots box plots for the specified numerical features.

        Args:
            numerical_features (list): List of numerical feature names to plot.
        """
        plt.figure(figsize=(16, 5))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(1, len(numerical_features), i)
            sns.boxplot(self.data[feature])
            plt.title(f'Box Plot for {feature}')
            plt.xlabel(feature)

    def plot_correlation_matrix(self, numerical_features):
        """Plots a heatmap of the correlation matrix for specified numerical features.

        Args:
            numerical_features (list): List of numerical feature names to include in the correlation matrix.
        """
        corr_matrix = self.data[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
        logg.info("Correlation matrix plotted successfully!")

    def plot_distribution_with_class(self, features: list, target: str, feature_type: str = 'numerical'):
        """Plots the distribution of specified features with respect to the target class.

        Args:
            features (list): List of features to plot.
            target (str): The target variable (class).
            feature_type (str): Type of features ('numerical' or 'categorical'). Defaults to 'numerical'.
        """
        try:
            num_features = len(features)
            num_cols = min(num_features, 2)
            num_rows = (num_features + num_cols - 1) // num_cols

            plt.figure(figsize=(num_cols * 5, num_rows * 4))

            for i, feature in enumerate(features, 1):
                if feature not in self.data.columns or target not in self.data.columns:
                    logg.error(f"Feature '{feature}' or target '{target}' not found in data!")
                    continue

                plt.subplot(num_rows, num_cols, i)

                if feature_type == 'numerical':
                    sns.histplot(data=self.data, x=feature, hue=target, bins=20, kde=True, palette="viridis", element="step")
                    plt.title(f'Distribution of {feature} by {target}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')

                elif feature_type == 'categorical':
                    sns.countplot(x=feature, hue=target, data=self.data, palette="viridis")
                    plt.title(f'Distribution of {feature} by {target}')
                    plt.xlabel(feature)
                    plt.ylabel('Count')

            plt.tight_layout()
            plt.show()

            logg.info("Feature distributions with respect to target class plotted successfully!")

        except Exception as e:
            logg.error(f"An error occurred while plotting feature distributions: {e}")
