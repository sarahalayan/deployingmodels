from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle  # For loading scaling statistics

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('model/model.h5')

# Load the scaling statistics (mean and std)
try:
    with open('model/scaling_stats.pkl', 'rb') as f:  # Load from pickle file
        scaling_stats = pickle.load(f)
except FileNotFoundError:
    print("Error: scaling_stats.pkl not found. Make sure it's in the correct directory. You need to run the notebook first.")
    exit()
# Define the features used by the model
model_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                  'gender_Female', 'gender_Male', 'ever_married_No',
                  'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                  'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                  'Residence_type_Rural', 'Residence_type_Urban',
                  'smoking_status_Unknown', 'smoking_status_formerly smoked',
                  'smoking_status_never smoked', 'smoking_status_smokes']



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Preprocessing (MATCH TRAINING DATA PREPROCESSING EXACTLY)

        # Handle Categorical Features (One-Hot Encoding)
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            all_values = ['Female', 'Male', 'Yes', 'No', 'Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children', 'Rural', 'Urban', 'Unknown', 'formerly smoked', 'never smoked', 'smokes'] # All possible values
            if col in input_data.columns:
                for val in all_values:
                    input_data[f'{col}_{val}'] = (input_data[col] == val).astype(int)
                input_data.drop(col, axis=1, inplace=True)
            else: # Handle missing categorical columns in input data
                for val in all_values:
                    input_data[f'{col}_{val}'] = 0 # add the dummies with 0 values


        # Ensure all required features are present (handle missing ones)
        for feature in model_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[model_features]  # Order columns

        # Scale numerical features (using loaded statistics)
        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        for feature in numerical_features:
            if feature in input_data.columns:
                try:
                    mean = scaling_stats[feature]['mean']
                    std = scaling_stats[feature]['std']
                    input_data[feature] = (input_data[feature] - mean) / std
                except KeyError:
                    return jsonify({'error': f"Feature '{feature}' not found in scaling statistics. Make sure the training data and model match."}), 500
            else: # Handle missing numerical columns in input data
                input_data[feature] = 0 # add the missing feature with 0 value

        prediction = model.predict(input_data)
        probability = prediction[0][0]
        predicted_class = 1 if probability > 0.5 else 0

        response = {
            'probability': float(probability),
            'prediction': int(predicted_class)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)