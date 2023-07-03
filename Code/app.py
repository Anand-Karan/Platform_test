from flask import Flask, jsonify, request
import pandas as pd
from model import prep_data, load_model

# Load the trained logistic regression model from the pickle file
model = load_model('logreg_model.pkl')

# Create a Flask application instance
app = Flask(__name__)

# Define a route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the request
    data = request.json

    # Preprocess the input data
    score = prep_data(pd.DataFrame(data))

    # Make predictions using the loaded model
    preds = model.predict(score)

    # Map the predictions to meaningful labels
    labels = ['Not Survived', 'Survived']
    predicted_labels = [labels[prediction] for prediction in preds]

    # Return the predictions as a JSON response
    return jsonify({'predictions': predicted_labels})

# Run the Flask application
if __name__ == '__main__':
    app.run()
