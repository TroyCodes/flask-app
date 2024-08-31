from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
import xgboost as xgb

app = Flask(__name__)

# Load the original XGBoost model and the updated tuned XGBoost model
original_xgb_model = joblib.load('original_xgb_model.pkl')
tuned_xgb_model = joblib.load('tuned_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# BetsAPI key (replace with your actual key)
API_KEY = "201236-lgg61IvF4XDXE2"

# Function to get live odds from BetsAPI
def get_live_odds(event_id):
    url = f"https://api.betsapi.com/v1/bet365/event?token={API_KEY}&FI={event_id}"
    response = requests.get(url)
    data = response.json()
    try:
        odds = data['results'][0]['odds']['full_time']['home']  # Example: fetching home win odds
    except (KeyError, IndexError):
        odds = None
    return odds

# Function to calculate implied probability from odds
def implied_probability(odds):
    return 1 / odds

# Function to calculate expected value
def expected_value(model_probability, odds):
    return (model_probability * odds) - (1 - model_probability)

# Function to determine if the bet is favorable
def is_bet_favorable(model_probability, odds, threshold=0.0):
    ev = expected_value(model_probability, odds)
    return ev > threshold, ev

@app.route('/')
def index():
    return "Hello, this is your Flask app running with betting predictions!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    event_id = data['event_id']  # Event ID for fetching live odds
    
    # Get live odds
    odds = get_live_odds(event_id)
    
    if odds is None:
        return jsonify({"error": "Could not fetch odds"}), 400
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions with each model
    rf_preds = original_xgb_model.predict_proba(input_data_scaled)[:, 1]
    gb_preds = tuned_xgb_model.predict_proba(input_data_scaled)[:, 1]
    
    # Combine predictions
    final_preds = 0.7 * rf_preds + 0.3 * gb_preds
    
    # Determine if the bet is favorable
    model_probability = final_preds[0]
    favorable, ev = is_bet_favorable(model_probability, odds)
    
    result = {
        'model_probability': model_probability,
        'implied_probability': implied_probability(odds),
        'expected_value': ev,
        'favorable': favorable
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
