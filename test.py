from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('brain_best.keras')
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return "Brain Tumor Detection Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']  
        img = Image.open(file.stream)
        img_array = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            result = 'Tumor Present'
        else:
            result = 'No Tumor'

        return jsonify({'prediction': result, 'probability': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
