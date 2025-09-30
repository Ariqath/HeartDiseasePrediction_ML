
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = int(request.form['Age'])
        Sex = int(request.form['Sex'])
        ChestPainType = int(request.form['ChestPainType'])
        RestingBP = float(request.form['RestingBP'])
        Cholesterol = float(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        RestingECG = int(request.form['RestingECG'])
        MaxHR = float(request.form['MaxHR'])
        ExerciseAngina = int(request.form['ExerciseAngina'])
        Oldpeak = float(request.form['Oldpeak'])
        ST_Slope = int(request.form['ST_Slope'])

        input_data = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol,
                                FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
        input_scaled = scaler.transform(input_data)

        prediction_proba = model.predict_proba(input_scaled)[0][1]
        prediction_percent = prediction_proba * 100

        return render_template('index.html', prediction_text=f'Heart Disease Risk: {prediction_percent}%')

    except Exception as e:
        import traceback
        return render_template('index.html', prediction_text=f'Error: {traceback.format_exc()}')

if __name__ == '__main__':
    app.run(debug=True)
