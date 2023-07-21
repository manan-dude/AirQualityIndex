import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model using pickle
with open('xgboost.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(request.form[f]) for f in ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    print(final_features)

    aqi = round(prediction[0], 2)
    print(aqi)
    def get_aqi_category(aqi):
            if aqi <= 50:
                return "Good"
            elif 50 < aqi <= 100:
                return "Moderate"
            elif 100 < aqi <= 200:
                return "Poor"
            elif 200 < aqi <= 300:
                return "Unhealthy"
            elif 300 < aqi <= 400:
                return "Very Unhealthy"
            else:
                return "Hazardous"

        # Get AQI category based on AQI value
    aqi_category = get_aqi_category(aqi)

    return render_template('result.html', prediction_text='Air Quality Index (AQI) Category: {} - Prediction: {}'.format(aqi_category, round(prediction[0], 2))
)

if __name__ == "__main__":
    app.run(port=5000)
