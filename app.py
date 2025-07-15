from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time
import os
import gdown

app = Flask(__name__)
CORS(app)

# ======== Konfigurasi Model ========
MODEL_PATH = 'model_vgg16.h5'
GDRIVE_FILE_ID = '14FKuNO91ESYq5M5w_9ulfjk287ga-qxw'

# ======== Download model jika belum ada ========
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

# ======== Load model =========
model = load_model(MODEL_PATH)

labels = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellowish']

# ======== Preprocessing Gambar =========
def preprocess_image_from_memory(image_file, target_size=(48, 48)):
    image = Image.open(image_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# ======== API Predict =========
@app.route('/predict', methods=['POST'])
def predict():
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

# ======== Jalankan Server ========
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=8)
