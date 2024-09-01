from flask import Flask, render_template, request
import requests
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load pre-trained models and scaler
original_xgb_model = joblib.load('original_xgb_model.pkl')
tuned_xgb_model = joblib.load('tuned_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# API configuration
API_TOKEN = '201236-lgg61IvF4XDXE2'  # Replace with your actual token
API_BASE_URL = 'https://api.b365api.com/v3'

# Function to fetch live tennis matches
def get_live_tennis_matches():
    endpoint = f"{API_BASE_URL}/events/upcoming?sport_id=13&token={API_TOKEN}"
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        if data['success']:
            return data['results']
    return []

# Function to fetch live odds for a given event
def get_live_odds(event_id):
    endpoint = f"{API_BASE_URL}/event/odds?token={API_TOKEN}&event_id={event_id}"
    response = requests.get(endpoint)

    if response.status_code == 200:
        data = response.json()
        if data['success']:
            return data['results']['odds']
    return None

# Function to check if a bet is favorable
def is_bet_favorable(model_probability, odds):
    implied_probability = 1 / odds
    expected_value = model_probability * odds - 1
    return expected_value > 0, expected_value

# Function to extract features from the match data
def extract_features(match):
    # Example feature extraction (modify according to your data needs)
    home_ranking = match['home'].get('ranking', -1)
    away_ranking = match['away'].get('ranking', -1)
    
    # Add more feature extraction logic here based on your model's requirements
    features = [
        home_ranking,
        away_ranking,
        # Add other features here...
    ]

    return features

@app.route('/favorable_bets')
def show_favorable_bets():
    try:
        matches = get_live_tennis_matches()
        favorable_bets = []

        for match in matches:
            event_id = match['id']

            # Extract features for prediction
            features = extract_features(match)

            # Ensure the correct number of features for your model
            if len(features) != 28:
                continue

            # Standardize the input data
            input_data_scaled = scaler.transform([features])

            # Get live odds
            odds = get_live_odds(event_id)
            if odds is None:
                continue

            # Make predictions with each model
            rf_preds = original_xgb_model.predict_proba(input_data_scaled)[:, 1]
            gb_preds = tuned_xgb_model.predict_proba(input_data_scaled)[:, 1]

            # Combine predictions
            final_preds = 0.7 * rf_preds + 0.3 * gb_preds

            # Determine if the bet is favorable
            model_probability = final_preds[0]
            favorable, ev = is_bet_favorable(model_probability, odds)

            if favorable:
                favorable_bets.append({
                    'event_id': event_id,
                    'match': match['home']['name'] + ' vs ' + match['away']['name'],
                    'model_probability': model_probability,
                    'implied_probability': 1 / odds,
                    'odds': odds,
                    'expected_value': ev
                })

        return render_template('favorable_bets.html', favorable_bets=favorable_bets)

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while processing your request: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
