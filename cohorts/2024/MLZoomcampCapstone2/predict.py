import torch
from flask import Flask, request, jsonify
from PIL import Image
from src.preprocessing import preprocess_image
from src.model import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('models/best_model.pth')
model.eval()

# Class labels
CLASSES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file = request.files['file']
        image_tensor = preprocess_image(file)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return jsonify({
            'predicted_class': CLASSES[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(CLASSES, probabilities[0].tolist())
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)