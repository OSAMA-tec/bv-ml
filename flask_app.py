from flask import Flask, render_template, request, jsonify
import joblib
from house_price_predictor import HousePricePredictor
import pandas as pd

app = Flask(__name__)

# Load the trained model and related data
model_path = 'saved_model/model.joblib'
model_data = joblib.load(model_path)
predictor = HousePricePredictor()
predictor.best_model = model_data['model']
predictor.scaler = model_data['scaler']
predictor.label_encoders = model_data['label_encoders']
predictor.feature_columns = model_data['feature_columns']

@app.route('/')
def home():
    cities = predictor.label_encoders['city'].classes_
    locations = predictor.label_encoders['location'].classes_
    return render_template('index.html', cities=cities, locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'city': request.form['city'],
            'location': request.form['location'],
            'bedrooms': float(request.form['bedrooms']),
            'baths': float(request.form['baths']),
            'size': float(request.form['size']),
            'price_per_sqft': float(request.form['price_per_sqft'])
        }
        
        df = pd.DataFrame([data])[predictor.feature_columns]
        
        for col in ['city', 'location']:
            df[col] = predictor.label_encoders[col].transform(df[col])
        
        predictions = predictor.predict_future_prices(df, years=2)
        
        formatted_predictions = {
            'current_price': {
                'pkr': f"₨{predictions['2024']:,.2f}",
                'usd': f"${predictions['2024']/282:,.2f}"
            },
            'future_prices': [
                {
                    'year': year,
                    'pkr': f"₨{price:,.2f}",
                    'usd': f"${price/282:,.2f}",
                    'growth': f"{((price/predictions['2024'] - 1) * 100):,.1f}%"
                }
                for year, price in predictions.items()
                if year != '2024'
            ],
            'price_per_sqft': f"₨{predictions['2024']/float(data['size']):,.2f}"
        }
        
        return jsonify({'success': True, 'predictions': formatted_predictions})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/locations/<city>')
def get_locations(city):
    try:
        all_locations = predictor.label_encoders['location'].classes_
        city_locations = [loc for loc in all_locations if city.lower() in loc.lower()]
        return jsonify({'success': True, 'locations': sorted(city_locations)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 