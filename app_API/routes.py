# routes.py

from flask import Blueprint, request, jsonify, render_template
from model import FraudModel
import pandas as pd

# Create a Blueprint for routes
routes = Blueprint('routes', __name__)

# Load the model
model = FraudModel('model.pkl')  # Update with the correct path to your model

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Perform prediction using the model
        prediction = model.predict(data)
        
        # Create a descriptive message based on the prediction
        if prediction == 1:
            message = "The transaction is classified as **Fraud**."
        else:
            message = "The transaction is classified as **Not Fraud**."

        # Return a structured JSON response
        return jsonify({'prediction': int(prediction), 'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@routes.route('/fraud-trends', methods=['GET'])
def fraud_trends():
    try:
        # Load the fraud data from a CSV file
        fraud_data = pd.read_csv('../data/merged_fraud_data.csv')
        data= jsonify(fraud_data.to_dict(orient='records'))
        return data
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
