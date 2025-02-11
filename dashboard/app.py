import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import requests
from layouts import create_layout
from callbacks import register_callbacks
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data from the API (flask backend)
def load_fraud_data():
    response = requests.get("http://localhost:5000/fraud-trends")
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)  # Convert JSON response to a pandas DataFrame
    else:
        return pd.DataFrame()  # Return empty DataFrame on error

# Define layout
# Set the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app, load_fraud_data)


# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
