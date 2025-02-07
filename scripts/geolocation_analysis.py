import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging configuration
log_file = '../logs/geolocation_fraud.log'

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='w'),
                            logging.StreamHandler()
                        ])

class GeolocationFraudAnalysis:
    def __init__(self, data):
        """
        Initialize with dataset.
        """
        self.data = data
        logging.info("GeolocationFraudAnalysis initialized with country data.")

    def analyze_fraud_by_country(self):
        """
        Analyze and report fraud distribution across countries.
        """
        logging.info("Analyzing fraud distribution by country.")
        
        # Calculate fraud rate per country (percentage of fraudulent transactions)
        fraud_rate_by_country = self.data[self.data['class'] == 1].groupby('country').size() / self.data.groupby('country').size()
        fraud_rate_by_country = fraud_rate_by_country.reset_index(name='fraud_rate')
        
        logging.info(f"Fraud analysis by country completed. Data: \n{fraud_rate_by_country.head()}")
        return fraud_rate_by_country

    def visualize_top_10_fraud_by_country(self, fraud_rate_by_country):
        """
        Visualize the top 10 countries with the highest fraud rates using a bar chart.
        """
        logging.info("Visualizing top 10 countries by fraud rate.")
        
        top_10_fraud = fraud_rate_by_country.nlargest(10, 'fraud_rate')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10_fraud, x='country', y='fraud_rate', palette='viridis')
        plt.title('Top 10 Countries with Highest Fraud Rates')
        plt.xlabel('Country')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        logging.info("Top 10 countries by fraud rate bar chart displayed.")

    def visualize_fraud_distribution(self, fraud_rate_by_country):
        """
        Visualize overall fraud distribution using a pie chart.
        """
        logging.info("Visualizing overall fraud distribution.")
        
        # Calculate total fraud and non-fraud counts
        fraud_counts = self.data['class'].value_counts()
        labels = ['Non-Fraud', 'Fraud']
        
        plt.figure(figsize=(8, 8))
        plt.pie(fraud_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
        plt.title('Overall Fraud Distribution')
        plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
        plt.show()

        logging.info("Overall fraud distribution pie chart displayed.")

    def run_geolocation_fraud_analysis(self):
        """
        Orchestrate the geolocation analysis and visualize fraud distribution.
        """
        logging.info("Starting fraud analysis pipeline.")
        
        # Analyze fraud by country
        fraud_rate_by_country = self.analyze_fraud_by_country()
        
        # Visualize top 10 fraud rates by country
        self.visualize_top_10_fraud_by_country(fraud_rate_by_country)

        # Visualize overall fraud distribution
        self.visualize_fraud_distribution(fraud_rate_by_country)
        
        logging.info("Fraud analysis pipeline completed.")