from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)


# Load the trained model
loaded_model = joblib.load("xgb_classifier.pkl")

@app.route("/")
def home():
    return render_template('base.html')

def ValuePredictor1(to_predict_dict):
    # Make a prediction using the loaded model
    result = loaded_model.predict(pd.DataFrame([to_predict_dict]))
    return result[0]
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert input values to float
        weather = int(request.form['weather'])
        city = int(request.form['city'])
        day = int(request.form['day'])
        temperature = float(request.form['temperature'])
        dew_point = float(request.form['dew_point'])
        wind_speed = float(request.form['wind_speed'])
        

        print("weather", weather)
    except ValueError:
        return render_template('result.html', prediction='Invalid input. Please enter valid values.')

    to_predict_dict = {
        'WEATHER': weather,
        'CITY': city,
        'DAY': day,
        'Temperature': temperature,
        'Dew_point': dew_point,
        'Wind_speed': wind_speed
    }

    # Make a prediction
    prediction = ValuePredictor1(to_predict_dict)
    print(prediction)

    if prediction == 1:
        prediction = "Person Sustaining  Injuries"
    else :
        prediction = "Person Without Injuries"

    return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)