from flask import Flask, render_template
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

# Function to get live tennis matches from BetsAPI
def get_live_tennis_matches():
    url = f"https://api.betsapi.com/v1/bet365/inplay_filter?sport_id=13&token={API_KEY}"
    response = requests.get(url)
    data = response.json()
    matches = []
    if 'results' in data:
        matches = data['results']
    return matches

# Function to get live odds from BetsAPI for a specific event
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
    return '''
    <h1>Welcome to the Betting App</h1>
    <p><a href="/favorable_bets">Check Favorable Tennis Bets</a></p>
    '''
def extract_features(match):
    try:
        # Extract features with safe access to 'ranking'
        home_ranking = match['home'].get('ranking', -1)  # Use -1 or another default if 'ranking' is missing
        away_ranking = match['away'].get('ranking', -1)  # Use -1 or another default if 'ranking' is missing

        # Continue extracting other features...
        # Assuming you have more features to extract, handle them similarly
        features = [
            home_ranking,
            away_ranking,
            # Add other features here
        ]

        if len(features) != 28:  # Adjust this based on your feature count
            raise ValueError(f"Expected 28 features, but got {len(features)}")

        return features

    except KeyError as e:
        print(f"KeyError: Missing key {e}")
        raise ValueError(f"Missing key in match data: {e}")

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise

@app.route('/favorable_bets')
def show_favorable_bets():
    try:
        matches = get_live_tennis_matches()
        favorable_bets = []

        for match in matches:
            event_id = match['id']

            # Safely extract features
            features = extract_features(match)

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
                    'implied_probability': implied_probability(odds),
                    'odds': odds,
                    'expected_value': ev
                })

        return render_template('favorable_bets.html', favorable_bets=favorable_bets)

    except ValueError as ve:
        error_message = f"ValueError: {ve}"
        print(error_message)
        return f"An error occurred: {ve}", 500  # Return a more detailed error to the user
    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        return f"An error occurred while processing your request: {e}", 500  # Return a more detailed error to the user

if __name__ == '__main__':
    app.run(debug=True)
