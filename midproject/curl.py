import requests

import requests

# URL of the API
url = "http://127.0.0.1:5100/predict"

# Example input data with column names from train.py
data = {
    "features": {
        "Relative_Compactness": 0.85,
        "Surface_Area": 500,
        "Wall_Area": 300,
        "Roof_Area": 200,
        "Overall_Height": 5,
        "Orientation": 3,
        "Glazing_Area": 0.2,
        "Glazing_Area_Distribution": 2,
    }
}

try:
    # Send POST request
    response = requests.post(url, json=data)
    # Print the response
    print("Response:", response.json())
except requests.exceptions.RequestException as e:
    # Print the error
    print("An error occurred:", e)
