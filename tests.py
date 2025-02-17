import requests
import json

data = {
    "age": 60,  # Changed age
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 120,
    "bmi": 30,  # Changed bmi
    "gender": "Female",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "formerly smoked"
}

headers = {'Content-Type': 'application/json'}
url = 'http://127.0.0.1:5000/predict'  # Your Flask app URL

response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    print(response.json())
else:
    print(response.text)  # Print the error message