from flask import Flask, request, jsonify
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the trained model
model = joblib.load('diagnosis_model.pkl')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    symptoms = data['symptoms']  # Expecting a list of symptoms
    # Preprocess the input as needed (this may vary based on your model)
    # For example, you might need to convert symptoms to a numerical format
    # Here, we assume the model can directly accept the symptoms as input
    prediction = model.predict([symptoms])
    return jsonify({'diagnosis': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
