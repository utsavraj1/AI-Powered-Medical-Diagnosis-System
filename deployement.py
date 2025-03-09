from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('medical_diagnosis_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Extract features from the JSON data
    features = np.array(data['features']).reshape(1, -1)
    
    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
