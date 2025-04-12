import os
import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define model as a global variable
model = None

# Add this to your Flask app
preprocessor = None

# Update load_model function
def load_model():
    global model, preprocessor
    try:
        model_path = 'model/svm_model.pkl'
        preprocessor_path = 'model/preprocessor.pkl'
        
        # Load the model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"{model_path} not found")
        
        # Load the preprocessor
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print("Preprocessor loaded successfully.")
        else:
            print(f"Preprocessor file not found at: {preprocessor_path}")
            raise FileNotFoundError(f"{preprocessor_path} not found")

    except Exception as e:
        print(f"Loading failed: {e}")

# Load the model at startup
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, preprocessor
    if model is None or preprocessor is None:
        return render_template('error.html', error="Model or preprocessor not loaded. Please check server logs.")
    
    try:
        # Extract features from form data
        features = {
            'Age': float(request.form['Age']),
            'Weight': float(request.form['Weight']),
            'Height': float(request.form['Height']),
            'BMI': float(request.form['Weight']) / ((float(request.form['Height'])/100) ** 2),  # BMI is uppercase here
            'WaistCircumference': float(request.form['WaistCircumference']),
            'HipCircumference': float(request.form['HipCircumference']),
            'ArmCircumference': float(request.form['ArmCircumference']),
            'waist_hip_ratio': float(request.form['WaistCircumference']) / float(request.form['HipCircumference']),
            'bmi_age_factor': float(request.form['Age']) * (float(request.form['Weight']) / ((float(request.form['Height'])/100) ** 2))
        }
        
        # Add missing categorical features with default values
        features['Gender'] = request.form.get('Gender', 'Male')
        features['Ethnicity'] = request.form.get('Ethnicity', 'White')
        features['Education'] = request.form.get('Education', 'High School')
        
        # Print features to debug
        print("Features before preprocessing:", features)
        
        # Convert features into a DataFrame for preprocessing
        input_df = pd.DataFrame([features])
        
        # Print column names to debug
        print("DataFrame columns:", input_df.columns.tolist())
        
        # Apply the same preprocessing steps as during training
        input_processed = preprocessor.transform(input_df)
        
        # Make prediction using the processed features
        prediction = model.predict(input_processed)
        
        # Get prediction probability
        try:
            prediction_proba = model.predict_proba(input_processed)[0][1]
        except AttributeError:
            prediction_proba = 0.5 if prediction[0] == 1 else 0.2
        
        # Determine risk level
        risk_level = "Low"
        if prediction_proba > 0.7:
            risk_level = "High"
        elif prediction_proba > 0.4:
            risk_level = "Medium"
            
        result = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba),
            'risk_level': risk_level
        }
        
        return render_template('result.html', result=result, features=features)
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR: {error_traceback}")
        return render_template('error.html', error=f"An error occurred: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
