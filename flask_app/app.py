from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load(r"E:\Machine Learning\flask_app\xgb_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    input_data = {
        'SeniorCitizen': int(form_data['SeniorCitizen']),
        'gender': form_data['gender'],
        'Partner': form_data['Partner'],
        'Dependents': form_data['Dependents'],
        'tenure': float(form_data['tenure']),
        'PhoneService': form_data['PhoneService'],
        'MultipleLines': form_data['MultipleLines'],
        'InternetService': form_data['InternetService'],
        'OnlineSecurity': form_data['OnlineSecurity'],
        'OnlineBackup': form_data['OnlineBackup'],
        'DeviceProtection': form_data['DeviceProtection'],
        'TechSupport': form_data['TechSupport'],
        'StreamingTV': form_data['StreamingTV'],
        'StreamingMovies': form_data['StreamingMovies'],
        'Contract': form_data['Contract'],
        'PaperlessBilling': form_data['PaperlessBilling'],
        'PaymentMethod': form_data['PaymentMethod'],
        'MonthlyCharges': float(form_data['MonthlyCharges']),
        'TotalCharges': float(form_data['TotalCharges'])
    }
    
    # Convert input_data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing as was done during model training
    input_df = model.named_steps['preprocessor'].transform(input_df)

    # Make prediction
    prediction = model.named_steps['classifier'].predict(input_df)
    prediction_prob = model.named_steps['classifier'].predict_proba(input_df)
    
    return render_template('result.html', prediction=prediction[0], probability=prediction_prob[0][1])

if __name__ == '__main__':
    app.run(debug=True)
