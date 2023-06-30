import requests
import json

# Define the input data
data = {
    'Age': 24,
    'family_members': 8.0,
    'parents': 3,
    'Fare': 21.0750,
    'Pclass_2': 0,
    'Pclass_3': 1,
    'Sex_male': 0,
    'port_Q': 0,
    'port_S': 1
}

# Send a POST request to the running Flask app
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# Process the response
if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print("Error:", response.text)
