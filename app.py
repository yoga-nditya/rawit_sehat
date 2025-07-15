import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time
import threading

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_vgg16.h5'
MODEL_URL = 'https://drive.google.com/uc?id=14FKuNO91ESYq5M5w_9ulfjk287ga-qxw'
labels = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellowish']

model = None

def download_and_load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Loading model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded!")

# Preprocess image
def preprocess_image_from_memory(image_file, target_size=(48, 48)):
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded yet'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    try:
        start_time = time.time()
        image = preprocess_image_from_memory(image_file)

        preprocess_time = time.time()
        prediction = model.predict(image)

        predict_time = time.time()
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        predicted_label = labels[predicted_index]

        total_time = time.time() - start_time
        print(f"⏱️ Preprocess: {preprocess_time - start_time:.3f}s | Predict: {predict_time - preprocess_time:.3f}s | Total: {total_time:.3f}s")

        return jsonify({
            'predicted_class': predicted_label,
            'confidence': round(confidence, 2),
            'class_index': predicted_index,
            'time_taken': round(total_time, 3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Jalankan model download dan load di background, supaya app langsung hidup
if __name__ == '__main__':
    threading.Thread(target=download_and_load_model).start()
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
