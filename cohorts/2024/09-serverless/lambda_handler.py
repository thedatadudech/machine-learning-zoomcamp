import numpy as np
import tensorflow as tf
from PIL import Image
from urllib import request
from io import BytesIO


# Function to download an image from a URL
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


# Function to prepare the image
def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


# Function to preprocess the image
def preprocess_image(img):
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizing the image
    return np.expand_dims(img_array, axis=0)  # Expanding dims to add batch size


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]


# Handler function to process the image and predict using the model
def handler(event, context):
    # Assuming 'event' contains the image URL
    image_url = event["image_url"]
    img = download_image(image_url)

    # Prepare and preprocess the image
    prepared_img = prepare_image(img, (input_shape[1], input_shape[2]))
    preprocessed_img = preprocess_image(prepared_img)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], preprocessed_img)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data.tolist()  # Returning the output as a list


# You can customize this part as per your actual event and context structure
if __name__ == "__main__":
    # Example test event with an image URL
    test_event = {
        "image_url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    }
    test_context = {}
    print(handler(test_event, test_context))
