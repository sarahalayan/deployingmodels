import gradio as gr
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load your scaling statistics
with open('model/scaling_stats.pkl', 'rb') as f:
    scaling_stats = pickle.load(f)

# Load your Keras/TensorFlow model (.h5 file)
model = keras.models.load_model('model/model.h5')

def predict(age, hypertension, heart_disease, avg_glucose_level, bmi, gender, ever_married, work_type, residence_type, smoking_status):
    # 1. Create a Pandas DataFrame
    data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status,
        'stroke': 0  # Dummy stroke column
    }
    df = pd.DataFrame([data])

    # 2. One-hot encode using pd.get_dummies (same as training)
    df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

    # 3. Scale numerical features
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    for feature in numerical_features:
        mean = scaling_stats[feature]['mean']
        std = scaling_stats[feature]['std']
        df[feature] = (df[feature] - mean) / std
        df[feature] = df[feature].astype(np.float32) # Explicitly cast to float32

    # 4. Make the prediction (and ensure correct data type for input)
    input_data = df.values.astype(np.float32)

    # 4. Make the prediction
    input_data = df.drop('stroke', axis=1).values  # Drop dummy stroke column
    prediction = model.predict(input_data)

    # 5. Process the prediction
    predicted_class = np.argmax(prediction, axis=1)[0] if prediction.shape[1] > 1 else round(prediction[0][0])

    return int(predicted_class)
# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown([0, 1], label="Hypertension"),
        gr.Dropdown([0, 1], label="Heart Disease"),
        gr.Number(label="Average Glucose Level"),
        gr.Number(label="BMI"),
        gr.Dropdown([0, 1], label="Gender"),  # Assuming 0/1 encoding
        gr.Dropdown([0, 1], label="Ever Married"), # Assuming 0/1 encoding
        gr.Dropdown([0, 1, 2, 3, 4], label="Work Type"), # Assuming numerical encoding
        gr.Dropdown([0, 1], label="Residence Type"), # Assuming 0/1 encoding
        gr.Dropdown([0, 1, 2, 3], label="Smoking Status"), # Assuming numerical encoding
    ],
    outputs=gr.Textbox(label="Prediction (0 or 1)"),
    title="Stroke Prediction",
    description="Predicts the likelihood of stroke based on input features."
)


iface.launch(share=True)