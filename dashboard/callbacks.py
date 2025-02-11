from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

def register_callbacks(app, load_fraud_data):
    @app.callback(
        [Output("total-transactions", "children"),
         Output("fraud-cases", "children"),
         Output("fraud-percentage", "children"),
         Output("fraud-over-time", "figure"),
         Output("fraud-by-browser", "figure"),
         Output("fraud-by-sex", "figure"),
         Output("fraud_by_country_map", "figure"),
         Output("fraud-class", "figure"),
        Output("fraud-by-age-bin", "figure")],  # Added Output for age bin chart
        [Input("fraud-over-time", "id")]  # Dummy input to trigger the callback
    )
    def update_dashboard(_):
        # Load fraud data
        data = load_fraud_data()
        
        # Creating age bins
        age_bins = [0, 18, 35, 50, 65, 100]  # Define the bins
        age_labels = ['0-18', '19-35', '36-50', '51-65', '66+']  # Define labels for the bins
        data['age_bin'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)  # Create age bins
        
        total_transactions = data.shape[0]
        fraud_cases = data[data['class'] == 1].shape[0]
        fraud_percentage = (fraud_cases / total_transactions) * 100

        # Convert 'purchase_time' to datetime format
        data['purchase_time'] = pd.to_datetime(data['purchase_time'],errors='coerce')

        # Group the data by purchase date
        fraud_case = data[data['class'] == 1].groupby(data['purchase_time'].dt.date).size()
        non_fraud_case = data[data['class'] == 0].groupby(data['purchase_time'].dt.date).size()

        # Create a DataFrame for easier plotting
        trend_data = pd.DataFrame({
            'Fraud Cases': fraud_case,
            'Non-Fraud Cases': non_fraud_case
        }).fillna(0)  # Fill missing values with 0

        # Fraud and Non-Fraud Cases Over Time
        fraud_over_time = {
            'data': [
                {
                    'x': trend_data.index,
                    'y': trend_data['Fraud Cases'],
                    'type': 'line',
                    'name': 'Fraud Cases',
                    'line': {'color': 'red'}
                },
                {
                    'x': trend_data.index,
                    'y': trend_data['Non-Fraud Cases'],
                    'type': 'line',
                    'name': 'Non-Fraud Cases',
                    'line': {'color': 'blue'}
                }
            ],
            'layout': {
                'title': 'Fraud and Non-Fraud Cases Over Time',
                'xaxis': {
                    'title': 'Purchase Date',
                    'type': 'date'
                },
                'yaxis': {
                    'title': 'Number of Cases',
                    'range': [0, trend_data[['Fraud Cases', 'Non-Fraud Cases']].max().max() + 10]
                },
                'hovermode': 'x unified'
            }
        }

        # Fraud by Browser
        fraud_by_browser = {
            'data': [
                {
                    'x': data['browser'].unique(),
                    'y': data.groupby('browser').size(),
                    'type': 'bar',
                    'name': 'Fraud by Browser',
                    'marker': {
                        'color': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                        'line': {
                            'width': 1.5,
                            'color': '#000'
                        }
                    }
                }
            ],
            'layout': {
                'title': 'Fraud Cases by Browser',
                'xaxis': {
                    'title': 'Browser',
                    'tickangle': -45
                },
                'yaxis': {
                    'title': 'Number of Fraud Cases'
                },
                'barmode': 'group'
            }
        }

        # Fraud by Sex
        fraud_by_sex = {
            'data': [
                {
                    'x': data['sex'].unique(),
                    'y': data.groupby('sex').size(),
                    'type': 'bar',
                    'name': 'Fraud by Sex',
                    'marker': {
                        'color': ['#17becf', '#bcbd22'],
                        'line': {
                            'width': 1.5,
                            'color': '#000'
                        }
                    }
                }
            ],
            'layout': {
                'title': 'Fraud Cases by Sex',
                'xaxis': {
                    'title': 'Sex'
                },
                'yaxis': {
                    'title': 'Number of Fraud Cases'
                },
                'barmode': 'group'
            }
        }

        # Fraud Counts by Country
        fraud_counts_by_country = data[data['class'] == 1].groupby('country').size().reset_index(name='fraud_count')

        # Create a choropleth map for fraud by country
        fraud_by_country_map = px.choropleth(
            fraud_counts_by_country,
            locations='country',
            locationmode='country names',  # Use country names
            color='fraud_count',
            hover_name='country',
            color_continuous_scale=px.colors.sequential.Viridis,
            title='Fraud Cases by Country',
            labels={'fraud_count': 'Number of Fraud Cases'}
        )
        # Update layout to make the map fit better
        fraud_by_country_map.update_layout(
            height=600,  # Set height for better visibility
            margin=dict(l=10, r=10, t=40, b=20)  # Adjust margins to fit the map better
        )
        # Fraud Class Analysis
        fraud_class = {
            'data': [
                {
                    'x': data['class'].unique(),
                    'y': data.groupby('class').size(),
                    'type': 'bar',
                    'name': 'Fraud by Class',
                    'marker': {
                        'color': ['#17becf', '#bcbd22'],
                        'line': {
                            'width': 1.5,
                            'color': '#000'
                        }
                    }
                }
            ],
            'layout': {
                'title': 'Fraud Cases by Class',
                'xaxis': {
                    'title': 'Fraud Class'
                },
                'yaxis': {
                    'title': 'Number of Fraud Cases'
                },
                'barmode': 'group'
            }
        }
        
        # Age Bin Analysis
        age_bin_counts = data.groupby('age_bin').size().reset_index(name='fraud_count')
        
        # Create a bar chart for fraud by age bin
        fraud_by_age_bin = {
            'data': [
                {
                    'x': age_bin_counts['age_bin'],
                    'y': age_bin_counts['fraud_count'],
                    'type': 'bar',
                    'name': 'Fraud by Age Bin',
                    'marker': {
                        'color': '#17becf',
                        'line': {
                            'width': 1.5,
                            'color': '#000'
                        }
                    }
                }
            ],
            'layout': {
                'title': 'Fraud Cases by Age Bin',
                'xaxis': {
                    'title': 'Age Bin'
                },
                'yaxis': {
                    'title': 'Number of Fraud Cases'
                },
                'barmode': 'group'
            }
        }
        return (
            str(total_transactions),
            str(fraud_cases),
            f"{fraud_percentage:.2f}%",
            fraud_over_time,
            fraud_by_browser,
            fraud_by_sex,
            fraud_by_country_map,  # Updated to return the choropleth map
            fraud_class,
            fraud_by_age_bin
        )
