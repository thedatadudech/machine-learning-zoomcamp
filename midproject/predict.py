import pandas as pd
from flask import Flask, request, jsonify
import pickle

# Define column names as used in train.py
column_names = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Roof_Area",
    "Overall_Height",
    "Orientation",
    "Glazing_Area",
    "Glazing_Area_Distribution",
]

# Load the trained model pipeline
with open("best_heating_load_model_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# Create a Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print(data)

        # Convert features to a DataFrame with specified column names
        features_df = pd.DataFrame([data["features"]], columns=column_names)
        print(features_df)

        # Predict heating load
        prediction = model_pipeline.predict(features_df)[0]
        return jsonify({"heating_load": prediction})
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5100)
