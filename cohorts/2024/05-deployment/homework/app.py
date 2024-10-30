import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import threading

# Load the DictVectorizer and the model
with open("dv.bin", "rb") as dv_file:
    dv = pickle.load(dv_file)

with open("model1.bin", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    client_data = request.get_json()
    # Transform client data
    X = dv.transform([client_data])
    # Predict probability
    probability = model.predict_proba(X)[0, 1]
    # Return the probability as JSON
    return jsonify({"credit_probability": probability})


# Function to run the Flask app in a thread
def run_app():
    app.run(host="0.0.0.0", port=9696, debug=False, use_reloader=False)


# Start Flask in a background thread
thread = threading.Thread(target=run_app)
thread.start()
