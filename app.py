from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os
import cv2

def load_and_preprocess_image(img_path):
    """Load and preprocess an image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_array = img / 255.0  # Scale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load models
pneumonia_model = load_model('models/pneumonia_classification_model1.keras')
brain_tumor_model = load_model('models/brain_mri_model.keras')
lung_cancer_model = load_model('models/lung_cancer_model.keras')

# Define target sizes for each model
pneumonia_target_size = (150, 150)
brain_tumor_target_size = (150, 150)
lung_cancer_target_size = (128, 128)

def predict(model, img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    return prediction.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict(pneumonia_model, filepath, pneumonia_target_size)
            if prediction[0][0] > 0.5:
                result = 'PNEUMONIA'
            else:
                result = 'NORMAL'
            return render_template('pneumonia.html', prediction=result, img_path=url_for('static', filename='uploads/' + filename))
    return render_template('pneumonia.html')

@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
            img_array = load_and_preprocess_image(filepath)
            predictions = brain_tumor_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            return render_template('brain_tumor.html', prediction=predicted_class_name, img_path=url_for('static', filename='uploads/' + filename))
    return render_template('brain_tumor.html')

@app.route('/lung_cancer', methods=['GET', 'POST'])
def lung_cancer():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            binary_prediction = lung_cancer_model.predict(img_array)
            if binary_prediction > 0.5:
                prediction = "Normal"
            else:
                prediction = "Cancerous"
            return render_template('lung_cancer.html', prediction=prediction, img_path=url_for('static', filename='uploads/' + filename))
    return render_template('lung_cancer.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
