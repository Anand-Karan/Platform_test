from flask import Flask, jsonify, request
import pickle

# Load the trained logistic regression model from the pickle file
with open('logreg_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask application instance
app = Flask(__name__)

# Define a route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the request
    data = flask.request.json

    # Make predictions using the loaded model
    predictions = model.predict(data)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

# Run the Flask application
if __name__ == '__main__':
    app.run()