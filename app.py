from flask import Flask, render_template, request
import requests
import joblib
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

app = Flask(__name__)

# Load pre-trained models and scaler
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
original_xgb_model = joblib.load(os.path.join(MODEL_DIR, 'original_xgb_model.pkl'))
tuned_xgb_model = joblib.load(os.path.join(MODEL_DIR, 'tuned_xgb_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# API configuration
API_TOKEN = '201236-lgg61IvF4XDXE2'
API_BASE_URL = 'https://api.b365api.com/v1'

# Function to fetch upcoming tennis matches
def get_upcoming_tennis_matches():
    endpoint = f"{API_BASE_URL}/bet365/upcoming"
    params = {
        'sport_id': 13,
        'token': API_TOKEN,
        'page': 1
    }
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            return data.get('results', [])
    return []

# Function to fetch odds for a given event
def get_event_odds(event_id):
    endpoint = f"{API_BASE_URL}/bet365/prematch"
    params = {
        'token': API_TOKEN,
        'FI': event_id
    }
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            return data.get('results', {}).get('odds', {})
    return {}

# Function to check if a bet is favorable
def is_bet_favorable(model_probability, odds):
    try:
        implied_probability = 1 / float(odds)
        expected_value = model_probability * float(odds) - 1
        return expected_value > 0, round(expected_value, 4)
    except (ZeroDivisionError, ValueError, TypeError):
        return False, 0

# Function to extract features from the match data
def extract_features(match):
    home_ranking = match.get('home', {}).get('ranking', 0)
    away_ranking = match.get('away', {}).get('ranking', 0)

    features = [
        home_ranking,
        away_ranking,
    ]

    return features

# Custom Jinja filter to convert Unix timestamp to datetime
@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime_filter(timestamp):
    try:
        timestamp = int(timestamp)
        return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return timestamp

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upcoming_matches')
def show_upcoming_matches():
    try:
        matches = get_upcoming_tennis_matches()
        return render_template('upcoming_matches.html', matches=matches)
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while processing your request: {e}", 500

@app.route('/favorable_bets')
def show_favorable_bets():
    try:
        matches = get_upcoming_tennis_matches()
        print(f"Fetched Matches: {matches}")  # Debugging line
        
        all_bets = []
        favorable_bets = []

        for match in matches:
            event_id = match.get('id')
            if not event_id:
                continue

            # Ensure required fields are present
            required_fields = ['home', 'away', 'league']
            missing_fields = [field for field in required_fields if field not in match or not match[field]]
            if missing_fields:
                print(f"Missing fields {missing_fields} for Event ID {event_id}")
                continue

            # Extract features for prediction
            features = extract_features(match)
            print(f"Features for Event ID {event_id}: {features}")  # Debugging line

            EXPECTED_FEATURES = 2  # Adjust this to match your feature extraction
            if len(features) != EXPECTED_FEATURES:
                print(f"Expected {EXPECTED_FEATURES} features, got {len(features)} for Event ID {event_id}")
                continue

            # Standardize the input data
            input_data_scaled = scaler.transform([features])

            # Get live odds
            odds_data = get_event_odds(event_id)
            print(f"Fetched Odds for Event ID {event_id}: {odds_data}")  # Debugging line
            if not odds_data:
                continue

            odds = odds_data.get('odds', None)
            if not odds:
                print(f"No odds data for Event ID {event_id}")
                continue

            # Make predictions with each model
            rf_preds = original_xgb_model.predict_proba(input_data_scaled)[:, 1]
            gb_preds = tuned_xgb_model.predict_proba(input_data_scaled)[:, 1]

            final_preds = 0.7 * rf_preds + 0.3 * gb_preds

            model_probability = final_preds[0]
            favorable, ev = is_bet_favorable(model_probability, odds)

            bet_info = {
                'event_id': event_id,
                'match': f"{match.get('home', {}).get('name', 'Home')} vs {match.get('away', {}).get('name', 'Away')}",
                'model_probability': round(model_probability, 4),
                'implied_probability': round(1 / float(odds), 4),
                'odds': odds,
                'expected_value': ev
            }

            all_bets.append(bet_info)

            if favorable:
                favorable_bets.append(bet_info)

        print(f"All Bets: {all_bets}")  # Debugging line
        print(f"Favorable Bets: {favorable_bets}")  # Debugging line

        return render_template('favorable_bets.html', all_bets=all_bets, favorable_bets=favorable_bets)

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while processing your request: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
