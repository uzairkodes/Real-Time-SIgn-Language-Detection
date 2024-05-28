# Modift for grey scale ACL model
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64

app = Flask(__name__, static_folder='static')

# Load the trained model
model = load_model('ASL.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(image):
    resized_image = cv2.resize(image, (28, 28))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    processed_image = np.array(grayscale_image) / 255.0
    return processed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.data.split(b',')[1]
        decoded_image = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image.reshape(-1, 28, 28, 1))
        predicted_sign = chr(np.argmax(prediction) + ord('A'))
        return jsonify({'predicted_sign': predicted_sign, 'confidence_scores': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 4000

if __name__ == '__main__':
    app.run(debug=True)
