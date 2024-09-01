from flask import Flask, render_template, request
import requests
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load pre-trained models and scaler
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
original_xgb_model = joblib.load(os.path.join(MODEL_DIR, 'original_xgb_model.pkl'))
tuned_xgb_model = joblib.load(os.path.join(MODEL_DIR, 'tuned_xgb_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# API configuration
API_TOKEN = '201236-lgg61IvF4XDXE2'  # Replace with your actual BetsAPI token
API_BASE_URL = 'https://api.b365api.com/v3'

# Function to fetch upcoming tennis matches
def get_upcoming_tennis_matches():
    endpoint = f"{API_BASE_URL}/bet365/upcoming"
    params = {
        'sport_id': 13,  # Assuming 13 is the SPORT_ID for Tennis; verify with API docs
        'token': API_TOKEN
    }
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        print("API Response Data:", data)  # Debugging print
        if data.get('success'):
            return data.get('results', [])
        else:
            print("API Response was not successful")
    else:
        print(f"Failed to fetch matches, status code: {response.status_code}")
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
        print("Odds Data:", data)  # Debugging print
        if data.get('success'):
            return data.get('results', {}).get('main', {}).get('odds', {})
        else:
            print("Failed to fetch odds")
    else:
        print(f"Failed to fetch odds, status code: {response.status_code}")
    return {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favorable_bets')
def show_favorable_bets():
    try:
        matches = get_upcoming_tennis_matches()
        print("Fetched Matches:", matches)  # Debugging print
        all_bets = []
        favorable_bets = []

        if not matches:
            print("No matches found")  # Debugging print

        for match in matches:
            event_id = match.get('id')
            if not event_id:
                continue

            features = extract_features(match)
            EXPECTED_FEATURES = 28  # Update this based on your model's feature count
            if len(features) != EXPECTED_FEATURES:
                continue

            input_data_scaled = scaler.transform([features])
            odds_data = get_event_odds(event_id)
            if not odds_data:
                continue

            odds = odds_data.get('odds', None)
            if not odds:
                continue

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

        return render_template('favorable_bets.html', all_bets=all_bets, favorable_bets=favorable_bets)

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while processing your request: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
