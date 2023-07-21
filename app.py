import joblib

# Assuming 'model' is your trained RandomForestClassifier model
joblib.dump(model, 'model.pkl')
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pickled model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    length = data['Length1']
    width = data['Width1']
    height = data['Height']
    weight = data['Weight']

    # Prepare input data for prediction
    input_data = pd.DataFrame([[length, width, height, weight]], columns=['Length1', 'Width1', 'Height', 'Weight'])

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    # Convert prediction to species name (if needed)
    species_map = {0: 'Bream', 1: 'Parkki', 2: 'Perch', 3: 'Pike', 4: 'Roach', 5: 'Smelt', 6: 'Whitefish'}
    predicted_species = species_map.get(prediction, 'Unknown')

    return jsonify({'species': predicted_species})

if __name__ == '__main__':
    app.run(debug=True)
