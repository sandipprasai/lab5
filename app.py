from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Correctly load the model 
model_path = os.path.join(app.root_path, 'fish_weight_predictor_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")  # Debug print

        # Validate and convert inputs
        try:
            species = float(data['species'])
            length1 = float(data['length1'])
            length2 = float(data['length2'])
            length3 = float(data['length3'])
            height = float(data['height'])
            width = float(data['width'])
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid input data: {e}")

        features = np.array([[species, length1, length2, length3, height, width]])
        app.logger.info(f"Features array: {features}")  # Debug print

        prediction = model.predict(features)
        app.logger.info(f"Prediction: {prediction[0]}")  # Debug print
        return jsonify(weight=prediction[0])
    except Exception as e:
        app.logger.error(f"Error: {e}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)