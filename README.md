# Heart Disease Prediction

A machine learning web application to predict the risk of heart disease based on medical attributes.  
This project uses **Python (Flask)** for the web backend, **scikit-learn** for ML models, and a clean **HTML/CSS** frontend interface.

---

## 📂 Project Structure
├── app.py # Flask backend for serving the web app
├── index.html # Frontend (user form for inputs)
├── HeartDisease.py # Model training & preprocessing script
├── Heart Disease.ipynb # Jupyter notebook for EDA & training
├── heart.csv # Dataset
├── model.pkl # Trained Random Forest model
├── scaler.pkl # Scaler used for preprocessing


---

## 🚀 Features
- Clean and responsive **web interface**.
- Multiple medical attributes for prediction:
  - Age, Sex, Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol Level
  - Fasting Blood Sugar
  - Resting ECG
  - Max Heart Rate
  - Exercise Induced Angina
  - Oldpeak
  - ST Slope
- **Machine Learning Models** tested: Logistic Regression, SVM, KNN, Random Forest.
- Final model: **Random Forest Classifier** with scaler preprocessing.
- Outputs **Heart Disease Risk in %**.

## 📊 Dataset
The dataset used: heart.csv
It contains medical features such as age, blood pressure, cholesterol, etc. The target variable is HeartDisease (0 = No, 1 = Yes).

---

## 🔮 Future Improvements

- Deploy app to Heroku / Vercel / Render.
- Improve UI/UX with React or Tailwind.
- Add more ML models & hyperparameter tuning.
- API endpoint for external usage.

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/username/heart-disease-prediction.git
cd heart-disease-prediction

2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows

3. Install dependencies
pip install -r requirements.txt
(Make sure to include requirements.txt with Flask, numpy, pandas, scikit-learn, seaborn, plotly, matplotlib, joblib)

4. Run the Flask app
python app.py

API endpoint for external usage.
