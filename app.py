from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained ResNet50 model (Downloads weights on first run)
model = ResNet50(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # Save file temporarily
    img_path = "temp_image.jpg"
    file.save(img_path)

    # Preprocess image for ResNet
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    # Convert predictions to human-readable labels
    decoded = decode_predictions(preds, top=1)[0][0]
    result_text = f"{decoded[1].replace('_', ' ').title()}"
    confidence = f"{decoded[2]*100:.2f}%"

    return f"""
    <html>
        <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
            <h1>Result: {result_text}</h1>
            <p>Confidence: {confidence}</p>
            <a href="/">Back to Upload</a>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, port=5000)