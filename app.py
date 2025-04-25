from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    model = joblib.load("ckd_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_classes = joblib.load("feature_classes.pkl")
except Exception as e:
    print("Error loading model files. Please run cdktrain.py first to train the model.")
    raise e

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all form fields
        patient_data = {
            'age': float(request.form['age']),
            'bp': float(request.form['bp']),
            'sg': float(request.form['sg']),
            'al': float(request.form['al']),
            'su': float(request.form['su']),
            'bgr': float(request.form['bgr']),
            'bu': float(request.form['bu']),
            'sc': float(request.form['sc']),
            'sod': float(request.form['sod']),
            'pot': float(request.form['pot']),
            'hemo': float(request.form['hemo']),
            'pcv': float(request.form['pcv']),
            'wc': float(request.form['wc']),
            'rc': float(request.form['rc']),
            'rbc': request.form['rbc'],
            'pc': request.form['pc'],
            'pcc': request.form['pcc'],
            'ba': request.form['ba'],
            'htn': request.form['htn'],
            'dm': request.form['dm'],
            'cad': request.form['cad'],
            'appet': request.form['appet'],
            'pe': request.form['pe'],
            'ane': request.form['ane']
        }
        
        # Create DataFrame and handle missing values
        df = pd.DataFrame([patient_data])
        
        # Prepare features in same order as training
        selected_features = joblib.load("selected_features.pkl")
        X = df[selected_features].copy()
        
        # Scale numeric features
        numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                         'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        X[numeric_columns] = scaler.transform(X[numeric_columns])
        
        # Encode categorical features
        categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in categorical_features:
            if col in label_encoders:
                X[col] = label_encoders[col].transform(X[col])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
